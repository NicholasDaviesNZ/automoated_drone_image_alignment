import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps
from create_datasets import pil_to_tensor, pad_image
import os

def live_plot(iteration, loss, epoch_list, val_loss_list):
    plt.figure(figsize=(10, 5))
    plt.plot(iteration, loss, marker='o', linestyle='-', color='b')
    plt.plot(epoch_list, val_loss_list, marker='o', linestyle='-', color='r', label='Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    # Save plot to file
    plt.savefig('training_loss.jpg')
    plt.close() 

class STN(nn.Module):
    """
    Spatial Transformer Network model class. it takes in the image input size and 
    batch size to calculate the size of the output of the localization network 
    so that we know the input size of the ST network.
    self.localisation takes in an image and learns features in a smaller space
    Both the reference and transofrm image are run though this network
    self.affine regressor is the network which takes in the output of both localisation 
    networks as a single vector - concat the outputs of the two localisations 
    for input to the affine_regressor, the output is the affine matrix to 
    align image 1 with image 2. Note image 1 is the base image, and image 2 is 
    the one assocated with the output affine matrix.
    """
    def __init__(self, padded_image_size, batch_size):
        super(STN, self).__init__()
        # inputs needed to calculate the input shape to the regressor
        self.padded_image_size = padded_image_size
        self.batch_size = batch_size
        
        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Dropout(p=0.3),
            nn.Conv2d(8, 12, kernel_size=5),
            nn.BatchNorm2d(12),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Dropout(p=0.3),
            nn.Conv2d(12, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # caclulate the out size of the localization layer, so that it cna be passed into the regressor for the affine
        dummy_input = torch.zeros(self.batch_size, 3, self.padded_image_size[0], self.padded_image_size[1])  
        dummy_output = self.localization(dummy_input)
        dummy_xs = dummy_output.view(dummy_output.size(0), -1)
        dummy_size = dummy_xs.shape[1]*2

        # Regressor for the 3 * 2 affine matrix
        self.affine_regressor = nn.Sequential(
            nn.Linear(dummy_size, 256),
            nn.ReLU(True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Dropout(p=0.3),
            nn.Linear(32, 6)
        )


    def forward(self, img1, img2):
        # pass both the georeferenced base images, img1 and the unreferenced img2 through the localisation network

        img1_xs = self.localization(img1)
        img2_xs = self.localization(img2)

        img1_xs = img1_xs.view(img1_xs.size(0), -1)
        img2_xs = img2_xs.view(img2_xs.size(0), -1)
        
        # pass both localised images to the regressor to estimate the affine matrix to convert img2 to the same georeference as img1
        combined_xs = torch.cat((img1_xs, img2_xs), dim=1)
        affine = self.affine_regressor(combined_xs)
        affine = affine.view(-1, 6)

        return affine
    
def load_model(model, model_file):
    model.load_state_dict(torch.load(model_file))
    model.eval()  # Set the model to evaluation mode
    return model

def train_stn(dataloader, dataloader_val, num_epochs=10, learning_rate=0.001, padded_image_size=(1280, 1280), batch_size=16, best_model_file = 'best_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Initialize the STN model
    stn = STN(padded_image_size, batch_size).to(device)
    
    if os.path.exists("/workspaces/automoated_drone_image_alignment/best_model.pth"):
        stn = load_model(stn, best_model_file)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(stn.parameters(), lr=learning_rate)
    
    iteration_list = []
    loss_list = []
    epoch_list = []
    val_loss_list = []
    # Initialize a counter for iterations
    iteration = 0

    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        
        for i, (img1, img2, affine) in enumerate(dataloader):
            img1, img2, affine = img1.to(device), img2.to(device), affine.to(device)
            
            # Forward pass through STN
            predicted_affine = stn(img1, img2)
            
            # Calculate the loss between the transformed image and the georeferenced image
            loss = criterion(predicted_affine, affine)
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            iteration_list.append(iteration)
            loss_list.append(loss.item())
            iteration += 1
            live_plot(iteration_list, loss_list, epoch_list, val_loss_list)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')
    
        stn.eval()  # Set the model to evaluation mode
        
        with torch.no_grad():
            val_loss = 0
            for img1, img2, affine in dataloader_val:
                img1, img2, affine = img1.to(device), img2.to(device), affine.to(device)

                predicted_affine = stn(img1, img2)                
                
                loss = criterion(predicted_affine, affine)
                val_loss += loss.item()
    
            val_loss /= len(dataloader_val)  
            val_loss_list.append(val_loss)
            epoch_list.append(epoch*(len(dataloader))+1)
        
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(stn.state_dict(), best_model_file)
            
            live_plot(iteration_list, loss_list, epoch_list, val_loss_list)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}')
    
    print("Training complete!")
    return stn

    
def inference(base_image_path, new_image_path, padded_image_size=(1280, 1280), output_res = (1000,1000), best_model_file = 'best_model.pth'):
    model = STN(padded_image_size, 1)
    model.load_state_dict(torch.load(best_model_file))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    img1 = Image.open(base_image_path).convert('RGB')
    img1 = img1.resize(output_res, Image.LANCZOS)
    
    img2 = Image.open(new_image_path).convert('RGB')
    img2 = img2.resize(output_res, Image.LANCZOS)
        
    img1_padded = pad_image(img1, padding = padded_image_size)
    img2_padded = pad_image(img2, padding = padded_image_size)
    
    base_img = pil_to_tensor(img1_padded).unsqueeze(0)
    new_img = pil_to_tensor(img2_padded).unsqueeze(0)
    
    base_img = base_img.to(device)
    new_img = new_img.to(device)
    with torch.no_grad(): 
        affine_matrix = model(base_img, new_img)
    
    affine_matrix = affine_matrix.cpu().numpy().flatten().tolist()
    return affine_matrix



