import cv2
import numpy as np
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

seed_value = 42
np.random.seed(seed_value)

def process_image(image_path, square_size=50):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get dimensions
    height, width = image.shape[:2]

    # Random coordinates for the square
    upper_left_x = np.random.randint(0, width - square_size)
    upper_left_y = np.random.randint(0, height - square_size)

    # Apply the cutout to the image
    image_cutout = image.copy()
    image_cutout[upper_left_y:upper_left_y+square_size, upper_left_x:upper_left_x+square_size] = 0

    # Create a mask corresponding to the cutout
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[upper_left_y:upper_left_y+square_size, upper_left_x:upper_left_x+square_size] = 1

    # Convert to PIL Images and then to tensors
    img_tensor = transforms.ToTensor()(Image.fromarray(image_cutout))
    mask_tensor = torch.tensor(mask, dtype=torch.float32)

    return img_tensor, mask_tensor

def process_directory(directory, square_size=50):
    images = []
    masks = []

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            img_tensor, mask_tensor = process_image(image_path, square_size=square_size)
            images.append(img_tensor)
            masks.append(mask_tensor)

    return images, masks

# Example usage
# image_tensors, mask_tensors = process_directory('D:/mlproject/test_images', square_size=50)


def visualize_images_and_masks(image_tensors, mask_tensors, num_images=5):
    fig, axs = plt.subplots(num_images, 2, figsize=(10, num_images * 3))

    for i in range(num_images):
        img = image_tensors[i].numpy().transpose(1, 2, 0)
        mask = mask_tensors[i].numpy()

        axs[i, 0].imshow(img)
        axs[i, 0].set_title('Processed Image')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(mask, cmap='gray')
        axs[i, 1].set_title('Mask')
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

def show_im(in_im):
    # Convert the PyTorch tensor to a NumPy array
    im = in_im.numpy()

    # Display the array using Matplotlib
    plt.imshow(im.transpose(1, 2, 0))  # Transpose to (height, width, channels) for RGB image
    plt.show()

# Use the function to visualize the processed images and masks
# visualize_images_and_masks(image_tensors, mask_tensors, num_images=5)

