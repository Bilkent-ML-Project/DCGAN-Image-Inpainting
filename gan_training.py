import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
from dcgan import Generator, Discriminator, init_weights
import os
import numpy as np
import torchvision
from torchvision.transforms import ToTensor

run_name = "faces_64_3"

to_tensor = ToTensor()

def calculate_mean_and_std(dataset):
    """
    Calculate the mean and standard deviation of a dataset.

    Parameters:
    - dataset: A numpy array or a PyTorch dataset containing the data.

    Returns:
    - mean: Mean value for each feature/channel.
    - std: Standard deviation for each feature/channel.
    """
    if isinstance(dataset, np.ndarray):
        # If the dataset is a numpy array
        mean = np.mean(dataset, axis=(0, 1, 2))  # Calculate mean along each channel
        std = np.std(dataset, axis=(0, 1, 2))    # Calculate standard deviation along each channel
    else:
        # If the dataset is a PyTorch dataset
        # For the first 10000 images, calculate the mean and standard deviation along each channel
        mean = torch.zeros(3)
        std = torch.zeros(3)
        for i in range(50000):
            if i % 1000 == 0:
                print(f"Processing image {i}/50000...")
            image, _ = dataset[i]
            image_tensor = to_tensor(image)
            mean += torch.mean(image_tensor, dim=(1, 2))
            std += torch.std(image_tensor, dim=(1, 2))

        mean /= 50000
        std /= 50000


    return mean, std

# 10k images mean and std
# Mean: tensor([0.5066, 0.4261, 0.3836])
# Std: tensor([0.2663, 0.2456, 0.2416])

# 50k images mean and std



def save_checkpoint(epoch, generator, discriminator, optimizer_g, optimizer_d):
    checkpoint_path_latest = f"{run_name}/checkpoint_latest.pth"
    checkpoint_path_epoch = f"{run_name}/checkpoint_epoch_{epoch - 1}.pth"

    if epoch > 1:
        # Rename the previous checkpoint file from latest
        os.rename(checkpoint_path_latest, checkpoint_path_epoch)

    # Save the latest checkpoint
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
    }, checkpoint_path_latest)

def load_checkpoint(generator, discriminator, optimizer_g, optimizer_d, checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    except FileNotFoundError:
        log("Checkpoint file not found. Starting from epoch 0.")
        epoch = 0
    return epoch

def main():
    # Set device
    # Check if CUDA is available
    # if torch.cuda.is_available():
    #     # Get the number of available CUDA devices
    #     num_cuda_devices = torch.cuda.device_count()

    #     log(f"Number of CUDA devices: {num_cuda_devices}")

    #     # Iterate over each CUDA device and log its properties
    #     for i in range(num_cuda_devices):
    #         device = torch.cuda.get_device_properties(i)
    #         log(f"CUDA Device {i}:", device)
    #     return
    # else:
    #     log("CUDA is not available on this system.")
    
    os.makedirs(run_name, exist_ok=True)
    # Create log.txt at run_name directory
    with open(f"{run_name}/log.txt", "w") as f:
        f.write("")
   
    
    # Set a specific seed for reproducibility
    torch.manual_seed(42)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    dataset_path = "./data/img_align_celeba"  # Replace with the actual path to your CelebA dataset
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5060, 0.4255, 0.3828), (0.2659, 0.2454, 0.2414)),
    ])
    celeba_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    
    # Use random_split with a specific seed to split the dataset into training and validation sets
    torch.manual_seed(42)
    subset_indices = torch.randperm(len(celeba_dataset))[:int(0.1 * len(celeba_dataset))]
    subset_dataset = torch.utils.data.Subset(celeba_dataset, subset_indices)
        
    # Define the ratio for splitting the dataset (e.g., 80% training, 20% validation)
    train_ratio = 0.8
    num_train = int(len(subset_dataset) * train_ratio)
    num_val = len(subset_dataset) - num_train

    # Use random_split to split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(subset_dataset, [num_train, num_val])
    # Create DataLoaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Model, criterion, and optimizer
    z_dim = 100
    num_gf = 64
    num_df = 64
    num_channels = 3  # RGB images
    generator = Generator(z_dim, num_gf, num_channels).to(device)
    discriminator = Discriminator(num_channels, num_df).to(device)

    log("Assigned networks to devices.")

    # Weight initialization
    generator.apply(init_weights)
    discriminator.apply(init_weights)

    # Loss function and optimizers
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Load checkpoint if available
    checkpoint_path = f"{run_name}/checkpoint_latest.pth"  # Change to the desired checkpoint file
    start_epoch = 0

    start_epoch = load_checkpoint(generator, discriminator, optimizer_g, optimizer_d, checkpoint_path)
    if start_epoch > 0:
        log(f"Checkpoint loaded. Resuming training from epoch {start_epoch + 1}.")

    # Training loop
    num_epochs = 50

    for epoch in range(start_epoch, start_epoch + num_epochs):
        log(f"Epoch {epoch}/{start_epoch + num_epochs}")
        
        # Training phase.
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            real_images, _ = data
            real_images = real_images.to(device)

            # Train Discriminator
            discriminator.zero_grad()
            real_labels = torch.full((real_images.size(0),), 1, device=device, dtype=torch.float32)
            fake_labels = torch.full((real_images.size(0),), 0, device=device, dtype=torch.float32)

            # Forward pass real batch through Discriminator
            output_real = discriminator(real_images)
            loss_real = criterion(output_real, real_labels)
            loss_real.backward()

            # Generate fake images
            noise = torch.randn(real_images.size(0), z_dim, 1, 1, device=device)
            fake_images = generator(noise)

            # Forward pass fake batch through Discriminator
            output_fake = discriminator(fake_images.detach())
            loss_fake = criterion(output_fake, fake_labels)
            loss_fake.backward()

            optimizer_d.step()

            # Train Generator
            generator.zero_grad()
            output_generated = discriminator(fake_images)
            loss_generator = criterion(output_generated, real_labels)
            loss_generator.backward()
            optimizer_g.step()

            if i % 100 == 0:
                # Save generated images
                torchvision.utils.save_image(fake_images[0], f"fake_images_{epoch}_{i}.png")


            # log statistics
            if i % 100 == 0:
                log(f"[Epoch {epoch}/{start_epoch + num_epochs}] [Batch {i}/{len(train_loader)}] "
                        f"[D Loss Real: {loss_real.item():.4f}, D Loss Fake: {loss_fake.item():.4f}] "
                        f"[G Loss: {loss_generator.item():.4f}]")
            
            
        # Validation phase
        generator.eval()  # Set the generator to evaluation mode
        with torch.no_grad():
            val_loss_d_real, val_loss_d_fake, val_loss_g = 0.0, 0.0, 0.0
            num_batches_val = len(val_loader)

            for i, data in tqdm(enumerate(val_loader), total=num_batches_val):
                real_images_val, _ = data
                real_images_val = real_images_val.to(device)

                # Validation for Discriminator
                output_real_val = discriminator(real_images_val)
                loss_real_val = criterion(output_real_val, real_labels)
                val_loss_d_real += loss_real_val.item()

                noise_val = torch.randn(real_images_val.size(0), z_dim, 1, 1, device=device)
                fake_images_val = generator(noise_val)
                output_fake_val = discriminator(fake_images_val)
                loss_fake_val = criterion(output_fake_val, fake_labels)
                val_loss_d_fake += loss_fake_val.item()

                # Validation for Generator
                output_generated_val = discriminator(fake_images_val)
                loss_generator_val = criterion(output_generated_val, real_labels)
                val_loss_g += loss_generator_val.item()

            # Average losses over the validation set
            avg_val_loss_d_real = val_loss_d_real / num_batches_val
            avg_val_loss_d_fake = val_loss_d_fake / num_batches_val
            avg_val_loss_g = val_loss_g / num_batches_val

            # Print and log validation losses
            log(f"[Validation] [Epoch {epoch}/{start_epoch + num_epochs}] "
                f"[D Loss Real: {avg_val_loss_d_real:.4f}, D Loss Fake: {avg_val_loss_d_fake:.4f}] "
                f"[G Loss: {avg_val_loss_g:.4f}]")

        generator.train()  # Set the generator back to training mode

        # Save checkpoint at the end of each epoch
        save_checkpoint(epoch, generator, discriminator, optimizer_g, optimizer_d)

    # Save models
    torch.save(generator.state_dict(), f"{run_name}/generator.pth")
    torch.save(discriminator.state_dict(), f"{run_name}/discriminator.pth")


def log(message):
    print(message)
    with open(f"{run_name}/log.txt", "a") as f:
        f.write(f"{message}\n")


if __name__ == "__main__":
    main()