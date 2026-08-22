import os
import argparse
import numpy as np
import cv2
from skimage import exposure, img_as_ubyte
from skimage.io import imread, imsave
from tqdm import tqdm
from scipy.ndimage import gaussian_laplace, gaussian_filter
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import imageio.v3 as iio 

def get_detectors(gabor_size, discreteness=4, device='cpu'):

    orientations = torch.linspace(0, np.pi*(discreteness-1)/discreteness, discreteness, device=device)
    lambd = 10.0
    sigma = 2.5
    gamma = 1

    # Create a meshgrid for Gabor function
    x, y = torch.meshgrid(
    	torch.linspace(-gabor_size//2, gabor_size//2, gabor_size, device=device), 
        torch.linspace(-gabor_size//2, gabor_size//2, gabor_size, device=device), 
        indexing='ij'
        )

    orientations = orientations.view(discreteness, 1, 1, 1)

    x_theta = x * torch.cos(orientations) + y * torch.sin(orientations)
    y_theta = -x * torch.sin(orientations) + y * torch.cos(orientations)
    
    gb = torch.exp(-.5 * (x_theta**2 + gamma**2 * y_theta**2) / sigma**2) \
    	* torch.cos(2 * np.pi * x_theta / lambd)
    
    return gb - gb.mean([2,3],keepdim=True) # (discreteness, 1, gabor_size, gabor_size)

def process_image(image_path):

    # Read the image
    img = imread(image_path)
    
    # Convert to grayscale if necessary
    if img.ndim >= 3:
        gray_img = rgb2gray(img[:,:,:3])  # using skimage's rgb2gray for simplicity
    else:
        gray_img = img
    
    gray_img = np.array(gray_img, dtype=float)
    
    # Apply Laplacian of Gaussian edge detection
    #log_img = gaussian_laplace(gray_img, sigma=1.2)
    log_img = gaussian_filter(gray_img, sigma=1.2) - gaussian_filter(gray_img, sigma=4.8)
    inv_log_img = -log_img  # Inverse polarity for the OFF channel
    
    #threshold = 0
    #log_img[log_img < threshold] = 0
    #log_img= torch.tensor(log_img[None,None], dtype=torch.float)
    #log_img = F.conv2d(log_img, gabors, padding=gabors.shape[-1]//2)
    
    # Normalize the ON and OFF channels
    on_channel = normalize_channel(log_img)   
    off_channel = normalize_channel(inv_log_img)
    
    # Create a dummy third channel (black)
    dummy_channel = np.zeros_like(on_channel)
    
    # Stack channels to form a 3-channel image
    processed_img = np.stack((on_channel, off_channel, dummy_channel), axis=-1)
    
    return processed_img
    
# Normalize each channel
def normalize_channel(channel):
	threshold = 0
	channel[channel < threshold] = 0

	# Gaussian filter parameters
	sigma = 4

	# Local normalization
	neighborhood_avg = gaussian_filter(channel, sigma=sigma, mode='constant')
	eps = 0.01
	saturation = 0.3
	cgc_img = np.tanh((channel / (neighborhood_avg + eps)) * saturation)

	# Convert to 8-bit
	return img_as_ubyte(cgc_img)

def process_images_in_folder(root_folder, output_folder):
	
    # Get all image paths first to calculate progress
    image_paths = []
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                image_paths.append(os.path.join(subdir, file))
                
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process images with progress bar
    for file_path in tqdm(image_paths, desc="Processing images"):
        processed_img = process_image(file_path)
        
        # Change file extension to '.png' for the output
        output_filename = os.path.splitext(os.path.basename(file_path))[0] + '.png'
        output_path = os.path.join(output_folder, output_filename)
        
        # Save the processed image in PNG format
        iio.imwrite(output_path, processed_img)   # drop format='png'

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess natural images into NeuralSheet stimuli.')
    parser.add_argument('root_folder')
    parser.add_argument('output_folder')
    args = parser.parse_args()
    process_images_in_folder(args.root_folder, args.output_folder)


if __name__ == '__main__':
    main()

