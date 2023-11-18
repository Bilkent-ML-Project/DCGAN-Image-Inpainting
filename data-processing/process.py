from mask import process_image
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def process_directory(directory, square_size=50):

    processed_dir = "./processed/"
    masks_dir = "./processed/masks/"
    ims_dir = "./processed/ims/"
    os.makedirs(processed_dir, exist_ok=True)  # Create the processed directory if it doesn't exist

    count = 0  # Initialize a counter to track progress

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            img_tensor, mask_tensor = process_image(image_path, square_size=square_size)

            # Save the processed image tensor as an image
            img_np = img_tensor.numpy().transpose(1, 2, 0)
            plt.imsave(os.path.join(ims_dir, filename), img_np)

            # Save the mask tensor as a grayscale image
            mask = mask_tensor.numpy()
            plt.imsave(os.path.join(masks_dir, filename), mask, cmap="gray")

            count += 1
            if count % 1000 == 0:
                print(f"Processed {count} images")

# Specify the directory containing images
input_directory = "./archive/img_align_celeba/img_align_celeba"

# Process the images in the directory
process_directory(input_directory)
