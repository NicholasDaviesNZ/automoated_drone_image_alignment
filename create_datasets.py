
import os
import math
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image, ImageOps
import random
import torch


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


def pad_image(img, padding = (1280, 1280)):
    pad_width, pad_height = padding
    img_width, img_height = img.size
    
    if img_width > pad_width:
        #print("Warning: padding width is smaller than image width, cropping image to fit the padding size.")
        left = (img_width - pad_width) // 2
        right = left + pad_width
        img = img.crop((left, 0, right, img_height))

    # Check if the image height is greater than the final height
    if img_height > pad_height:
        #print("Warning: padding height is smaller than image height, cropping image to fit the padding size.")
        top = (img_height - pad_height) // 2
        bottom = top + pad_height
        img = img.crop((0, top, img_width, bottom))
        
    # adding padding to the image so that it doesnt cut off anything during the transform - note in the absolute extremes the padding does nto acount for rotation so may crop a little
    padded_img = ImageOps.pad(img, (pad_width, pad_height), color=0)

    return padded_img

def combined_affine_transform(image, rotation_range=(-5, 5), translation=(0.05, 0.05), scale=(0.95, 1.05), padded_size = (1280, 1280)):
    
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
    

    padded_image = pad_image(image, padding = padded_size)
    # apply affine transformation
    transformed_image = padded_image.transform(
        padded_image.size,
        Image.AFFINE,
        affine,
        resample=Image.BILINEAR
    )
    
    return transformed_image, inv_affine

def pil_to_tensor(img):
    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() # get into pytorch order
    return img_tensor / 255.0  # Normalize to [0, 1]

class ImagePairDataset(Dataset):
    """
    This dataset is used to generate training/testing data. It takes in the folder where the trial image 
    folders are and a list of those trial folders we want to sample from along with n, the number of image pairs we want to generate
    for each n, it selects 2 random images from the same trial folder, it takes one of them and applies a random transofmation to it
    and calculates the inverse of the affine matrix used to cause the transform (this is the affine to transform back to the origonal)
    The output is the base image (untransformed), the transformed image, and the inv_affine matrix which takes the transoformed image back to the georeferenced image
    """
    def __init__(self, image_pair_folder, list_of_trial_folders, n=1000, padded_size = (1280, 1280), output_res = (1024,1024), transform=None):
        self.image_pair_folder = image_pair_folder
        self.list_of_trial_folders = list_of_trial_folders
        self.transform = transform
        self.n = n
        self.padded_size = padded_size
        self.output_res = output_res

    def _load_image_pairs(self):
        random_trial_folder = random.choice(self.list_of_trial_folders) # note can add weights here if wanted but have to think about how it will effect data balance
        folder_path = os.path.join(self.image_pair_folder, random_trial_folder)
        images = os.listdir(folder_path)

        if len(images) < 2:
            print(f"not enough images in folder {random_trial_folder}")
            return  # add error all folders should have at lest 2 images

        # Randomly sample two images
        random_images = random.sample(images, 2)
        image_paths = [os.path.join(folder_path, img) for img in random_images]
        # Load images
        img1 = Image.open(image_paths[0]).convert('RGB')
        img1 = img1.resize(self.output_res, Image.LANCZOS)
        img2 = Image.open(image_paths[1]).convert('RGB')
        img2 = img2.resize(self.output_res, Image.LANCZOS)
        return img1, img2
    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # load the image pair
        img1, img2 = self._load_image_pairs()
        
        img1_padded = pad_image(img1, padding = self.padded_size)
        # send img2 to get randomly transformed - expect back img2_transformed and the affine matrix
        img2_transformed, affine_matrix = combined_affine_transform(img2, padded_size=self.padded_size)

        if self.transform:
            img1_padded = self.transform(img1_padded)
            img2_transformed = self.transform(img2_transformed)

        img1_padded = pil_to_tensor(img1_padded)
        img2_transformed = pil_to_tensor(img2_transformed)
        affine_tensor = torch.FloatTensor(affine_matrix)
        
        return img1_padded, img2_transformed, affine_tensor
    

    
    
    


# dataloader interface for other py files
def get_dataloader(image_pair_folder, list_of_trial_folders, batch_size, n=1000, padded_size=(1280, 1280), output_res = (1024, 1024)):
    dataset = ImagePairDataset(image_pair_folder, list_of_trial_folders, n=n, padded_size=padded_size, output_res=output_res)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

