import os
import torch
from create_datasets import get_dataloader
from model import train_stn
torch.cuda.empty_cache()

image_pair_folder = '/workspaces/automoated_drone_image_alignment/georeferenced_image_pairs/'
list_of_trial_folders = os.listdir(image_pair_folder)
best_model_file = 'best_model.pth'
batch_size = 8
num_epochs = 500
learning_rate = 0.0002
training_set_size = 1024
val_set_size = 256
padded_image_size = (1280, 1280)
output_res = (1024,1024)

dataloader = get_dataloader(image_pair_folder, list_of_trial_folders, batch_size, n=training_set_size, padded_size=padded_image_size, output_res=output_res)
dataloader_val = get_dataloader(image_pair_folder, list_of_trial_folders, batch_size, n=val_set_size, padded_size=padded_image_size, output_res=output_res)


stn_model = train_stn(dataloader, dataloader_val, num_epochs=num_epochs, learning_rate=learning_rate, padded_image_size=padded_image_size, batch_size=batch_size, best_model_file=best_model_file)