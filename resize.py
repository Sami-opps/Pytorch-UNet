import cv
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import os

def preprocess_image_and_mask(image_path, mask_path, size=250, save_image_path=None, save_mask_path=None):
    # Load the grayscale image and its corresponding mask
    img = Image.open(image_path).convert('L')
    mask = Image.open(mask_path).convert('L')

    # Convert the images to NumPy arrays
    img = np.array(img)
    mask = np.array(mask)

    # Resize the images to the desired size
    img = cv.resize(img, (size, size), interpolation=cv.INTER_LINEAR)
    mask = cv.resize(mask, (size, size), interpolation=cv.INTER_NEAREST)

    # Convert the NumPy arrays back to PIL images
    img = Image.fromarray(img)
    mask = Image.fromarray(mask)

    # Convert the PIL images to PyTorch tensors
    transform = transforms.ToTensor()
    img = transform(img)
    mask = transform(mask)

    # Save the preprocessed images and masks to separate file folders
    if save_image_path is not None:
        if not os.path.exists(save_image_path):
            os.makedirs(save_image_path)
        save_image_filename = os.path.join(save_image_path, os.path.basename(image_path))
        torch.save(img, save_image_filename)

    if save_mask_path is not None:
        if not os.path.exists(save_mask_path):
            os.makedirs(save_mask_path)
        save_mask_filename = os.path.join(save_mask_path, os.path.basename(mask_path))
        torch.save(mask, save_mask_filename)

    return img, mask


# Define the paths to the dataset
data_dir = '/data/'
image_dir = 'C:\Users\samue\Downloads\Senior_Project\Pytorch-UNet\data\imgs'
mask_dir = './data/masks'
print(os.getcwd())
# Iterate over the files in the dataset
for filename in os.listdir(image_dir):
    # Construct the paths to the image and mask files
    image_path = os.path.join(image_dir, filename)
    mask_path = os.path.join(mask_dir, filename)

    # Preprocess the image and mask
    img, mask = preprocess_image_and_mask(image_path, mask_path, size=250)


    # Do something with the preprocessed image and mask
    # ...