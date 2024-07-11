
import os
import math
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image, ImageOps
import random
import torch
image_pair_folder = '/workspaces/automoated_drone_image_alignment/georeferenced_image_pairs/'
list_of_trial_folders = os.listdir(image_pair_folder)
batch_size = 4




def invert_affine_transform(affine_matrix):
    """
    Takes in the affine matrix used to convert the georeferenced image to a modified on for training
    Returns the inverse - the inverse affine is the matrix used to convert the modified image back to the georeferenced image ie the one we are trying to predict
    """
    
    a, b, c, d, e, f = affine_matrix

    # Calculate the determinant
    det = a * e - b * d
    
    if det == 0:
        return # add error here
    
    # Calculate the inverse affine transformation matrix
    a_inv = e / det
    b_inv = -b / det
    c_inv = (b * f - e * c) / det
    d_inv = -d / det
    e_inv = a / det
    f_inv = (d * c - a * f) / det
    
    inv_affine = (a_inv, b_inv, c_inv, d_inv, e_inv, f_inv)
    
    return inv_affine


def combined_affine_transform(image, rotation_range=(-5, 5), translation=(0.05, 0.05), scale=(0.95, 1.05)):
    
    """
    Function to do a random affine transform on an image,
    ARGS:
    input is the PIL image to transofrm
    limits of rotation (degrees)
    translation (as proportion of image size)
    scale, 1 is origonal size
    OUTPUT:
    the transormed image
    the inverse affine matrix - the matrix which will get the transformed image back to the origonal ie what we want to predict
    """
    
    width, height = image.size

    # rotation
    rotation = random.uniform(rotation_range[0], rotation_range[1])
    theta = math.radians(rotation)
    
    # translation
    max_dx = translation[0] * width
    max_dy = translation[1] * height
    dx = random.uniform(-max_dx, max_dx)
    dy = random.uniform(-max_dy, max_dy)
    
    # scale
    scale_factor_x = random.uniform(scale[0], scale[1])
    scale_factor_y = random.uniform(scale[0], scale[1])
    
    # affine transformation matrix
    a = scale_factor_x * math.cos(theta)
    b = -scale_factor_y * math.sin(theta)
    c = dx
    d = scale_factor_x * math.sin(theta)
    e = scale_factor_y * math.cos(theta)
    f = dy
    
    affine = (a, b, c, d, e, f)
    inv_affine = invert_affine_transform(affine)
    
    # adding padding to the image so that it doesnt cut off anything during the transform - note in the absolute extremes the padding does nto acount for rotation so may crop a little
    pad_width = int((image.width + max_dx)*(scale[1]-1))
    pad_height = int((image.height + max_dy)*(scale[1]-1))
    padded_image = ImageOps.expand(image, border=(pad_width, pad_height, pad_width, pad_height), fill=0)

    # apply affine transformation
    transformed_image = padded_image.transform(
        padded_image.size,
        Image.AFFINE,
        affine,
        resample=Image.BILINEAR
    )
    
    return transformed_image, inv_affine



class ImagePairDataset(Dataset):
    """
    This dataset is used to generate training/testing data. It takes in the folder where the trial image 
    folders are and a list of those trial folders we want to sample from along with n, the number of image pairs we want to generate
    for each n, it selects 2 random images from the same trial folder, it takes one of them and applies a random transofmation to it
    and calculates the inverse of the affine matrix used to cause the transform (this is the affine to transform back to the origonal)
    The output is the base image (untransformed), the transformed image, and the inv_affine matrix which takes the transoformed image back to the georeferenced image
    """
    def __init__(self, image_pair_folder, list_of_trial_folders, n=1000, transform=None):
        self.image_pair_folder = image_pair_folder
        self.list_of_trial_folders = list_of_trial_folders
        self.transform = transform
        self.n = n
        self.image_pairs = self._load_image_pairs()

    def _load_image_pairs(self):
        image_pairs = []
        
        random_trial_folder = random.choice(self.list_of_trial_folders) # note can add weights here if wanted but have to think about how it will effect data balance
        folder_path = os.path.join(self.image_pair_folder, random_trial_folder)
        images = os.listdir(folder_path)

        if len(images) < 2:
            return  # add error all folders should have at lest 2 images

        # Randomly sample two images
        random_images = random.sample(images, 2)
        image_paths = [os.path.join(folder_path, img) for img in random_images]

        # Load images
        img1 = Image.open(image_paths[0]).convert('RGB')
        img2 = Image.open(image_paths[1]).convert('RGB')

        image_pairs.append((img1, img2))

        return image_pairs

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # load the image pair
        img1, img2 = self.image_pairs[0]
        # send img2 to get randomly transformed - expect back img2_transformed and the affine matrix
        img2_transformed, affine_matrix = combined_affine_transform(img2)

        if self.transform:
            img1 = self.transform(img1)
            img2_transformed = self.transform(img2_transformed)

        img1 = self.pil_to_tensor(img1)
        img2_transformed = self.pil_to_tensor(img2_transformed)
        affine_tensor = torch.FloatTensor(affine_matrix)
        
        return img1, img2_transformed, affine_tensor
    
    def pil_to_tensor(self, img):
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() # get into pytorch order
        return img_tensor / 255.0  # Normalize to [0, 1]
    
    
    


# dataloader
dataset = ImagePairDataset(image_pair_folder, list_of_trial_folders, n=10)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for batch_idx, batch in enumerate(dataloader):
    img1_batch, img2_transformed_batch, affine_batch = batch

    # Print batch information
    print(f"Batch {batch_idx}:")
    print(f"img1_batch.shape: {img1_batch.shape}")
    print(f"img2_transformed_batch.shape: {img2_transformed_batch.shape}")
    print(f"affine_batch.shape: {affine_batch.shape}")