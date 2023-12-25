import torch
import torch.nn as nn
import torch.optim as optim
from dcganv2 import Generator, Discriminator
from PIL import Image
from torchvision.transforms import ToTensor
import cv2
import matplotlib.pyplot as plt


image_path = '../processed/ims/000001.jpg'  # Replace with your image path
masked_image = Image.open(image_path)
# Convert the image to a PyTorch tensor
to_tensor = ToTensor()
masked_image = to_tensor(masked_image)

mask_path = '../processed/masks/000001.jpg'  # Replace with your mask path
mask = Image.open(mask_path)
# Convert the mask to a PyTorch tensor
to_tensor = ToTensor()
mask = to_tensor(mask)

# Load the pre-trained models
generator = Generator(1)
generator.load_state_dict(torch.load('./run1/generator.pth'))
generator.eval()

discriminator = Discriminator(1)
discriminator.load_state_dict(torch.load('./run1/discriminator.pth'))
discriminator.eval()

# Generate inpainting candidates
num_candidates = 2000
latent_dim = 100  # Adjust based on your generator's input size
noise = torch.randn(num_candidates, latent_dim, 1, 1)
generated_ims = generator(noise)

# Inpainted images
inpainting_candidates = generated_ims * mask
# Use cv2 to save the first inpainting candidate, save it as RGB
# plt.imsave('inpainting_candidate.jpg', inpainting_candidates[0].permute(1, 2, 0).detach().numpy())
candidates = inpainting_candidates + masked_image

with torch.no_grad():
    # Calculate the discriminator scores for each candidate
    scores = discriminator(candidates)


# Select the candidate with the lowest content loss
best_candidate_index = torch.argmin(scores)

# Retrieve the best inpainting candidate
best_candidate = candidates[best_candidate_index]

# Visualize the original image, mask, and the best inpainting candidate
fig, axs = plt.subplots(1, 3, figsize=(15, 15))
axs[0].imshow(masked_image.permute(1, 2, 0))
axs[0].set_title("Original Image")
axs[1].imshow(mask.permute(1, 2, 0))
axs[1].set_title("Mask")
axs[2].imshow(best_candidate.permute(1, 2, 0).detach().numpy())
axs[2].set_title("Best Inpainting Candidate")

plt.show()