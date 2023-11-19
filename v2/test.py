import torch
import torch.nn as nn
import torch.optim as optim
from dcganv2 import Generator, Discriminator
from PIL import Image
from torchvision.transforms import ToTensor
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
num_candidates = 1000
latent_dim = 100  # Adjust based on your generator's input size
noise = torch.randn(num_candidates, latent_dim, 1, 1)
inpainting_candidates = generator(noise)

# Assuming masked_image is your original image
content_loss_fn = nn.MSELoss()

print(inpainting_candidates[0].shape)
print(masked_image.shape)
# Calculate content loss for each candidate
content_losses = [content_loss_fn(candidate, masked_image) for candidate in inpainting_candidates]

# Select the candidate with the lowest content loss
best_candidate_index = torch.argmin(torch.tensor(content_losses))

# Retrieve the best inpainting candidate
best_candidate = inpainting_candidates[best_candidate_index]

# Inpaint the masked image with the best candidate
inpainted_image = masked_image * (1- mask) + best_candidate *  mask

# Visualize the original image, mask, and the best inpainting candidate
fig, axs = plt.subplots(1, 3, figsize=(15, 15))
axs[0].imshow(masked_image.permute(1, 2, 0))
axs[0].set_title("Original Image")
axs[1].imshow(mask.permute(1, 2, 0))
axs[1].set_title("Mask")
axs[2].imshow(inpainted_image.permute(1, 2, 0).detach().numpy())
axs[2].set_title("Inpainting Candidate")

plt.show()