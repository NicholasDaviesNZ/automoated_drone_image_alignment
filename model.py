import os
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
from create_datasets import get_dataloader
torch.cuda.empty_cache()

image_pair_folder = '/workspaces/automoated_drone_image_alignment/georeferenced_image_pairs/'
list_of_trial_folders = os.listdir(image_pair_folder)
batch_size = 16
num_epochs = 100
learning_rate = 0.0002
training_set_size = 500
val_set_size = 20
padded_image_size = (1280, 1280)
output_res = (1024,1024)

dataloader = get_dataloader(image_pair_folder, list_of_trial_folders, batch_size, n=training_set_size, padded_size=padded_image_size, output_res=output_res)
dataloader_val = get_dataloader(image_pair_folder, list_of_trial_folders, batch_size, n=val_set_size, padded_size=padded_image_size, output_res=output_res)



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
    def __init__(self, padded_image_size, batch_size):
        super(STN, self).__init__()
        # inputs needed to calculate the input shape to the regressor
        self.padded_image_size = padded_image_size
        self.batch_size = batch_size
        
        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=9),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Dropout(p=0.3),
            nn.Conv2d(8, 10, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Dropout(p=0.3),
            nn.Conv2d(10, 12, kernel_size=5),
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
            nn.Linear(dummy_size, 128), 
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
        #print("Shape of img1_xs after localization:", img1_xs.shape) 
        #print("Shape of img2_xs after localization:", img2_xs.shape)

        img1_xs = img1_xs.view(img1_xs.size(0), -1)
        img2_xs = img2_xs.view(img2_xs.size(0), -1)
        
        # pass both localised images to the regressor to estimate the affine matrix to convert img2 to the same georeference as img1
        combined_xs = torch.cat((img1_xs, img2_xs), dim=1)
        affine = self.affine_regressor(combined_xs)
        affine = affine.view(-1, 6)

        return affine
    
    

def train_stn(dataloader, num_epochs=num_epochs, learning_rate=learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Initialize the STN model
    stn = STN(padded_image_size, batch_size).to(device)
    
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
            #print(affine.shape)
            #print(predicted_affine.shape)
            
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
                print(val_loss)
    
            val_loss /= len(dataloader_val)  
            val_loss_list.append(val_loss)
            epoch_list.append(epoch*(len(dataloader))+1)
        
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(stn.state_dict(), 'best_model.pth')
            
            live_plot(iteration_list, loss_list, epoch_list, val_loss_list)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}')
    
    print("Training complete!")
    return stn

# Example usage:
# Assuming `dataloader` is defined and provides batches of (img1, img2, affine)
stn_model = train_stn(dataloader, num_epochs=num_epochs, learning_rate=learning_rate)