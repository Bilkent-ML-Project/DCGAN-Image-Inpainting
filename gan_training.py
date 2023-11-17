import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from dcgan import Generator, Discriminator, init_weights
import os

def save_checkpoint(epoch, generator, discriminator, optimizer_g, optimizer_d):
    checkpoint_path_latest = "checkpoint_latest.pth"
    checkpoint_path_epoch = f"checkpoint_epoch_{epoch - 1}.pth"

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
        print("Checkpoint file not found. Starting from epoch 0.")
        epoch = 0
    return epoch

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    dataset_path = "/Users/tolgaozgun/Downloads/archive/img_align_celeba"  # Replace with the actual path to your CelebA dataset
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    celeba_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    train_loader = DataLoader(celeba_dataset, batch_size=64, shuffle=True, num_workers=4)

    # Model, criterion, and optimizer
    z_dim = 100
    num_gf = 64
    num_df = 64
    num_channels = 3  # RGB images
    generator = Generator(z_dim, num_gf, num_channels).to(device)
    discriminator = Discriminator(num_channels, num_df).to(device)

    print("Assigned networks to devices.")

    # Weight initialization
    generator.apply(init_weights)
    discriminator.apply(init_weights)

    # Loss function and optimizers
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Load checkpoint if available
    checkpoint_path = "checkpoint_latest.pth"  # Change to the desired checkpoint file
    start_epoch = 0

    start_epoch = load_checkpoint(generator, discriminator, optimizer_g, optimizer_d, checkpoint_path)
    if start_epoch > 0:
        print(f"Checkpoint loaded. Resuming training from epoch {start_epoch + 1}.")

    # Training loop
    num_epochs = 20

    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(f"Epoch {epoch}/{start_epoch + num_epochs}")
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

            # Print statistics
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{start_epoch + num_epochs}] [Batch {i}/{len(train_loader)}] "
                        f"[D Loss Real: {loss_real.item():.4f}, D Loss Fake: {loss_fake.item():.4f}] "
                        f"[G Loss: {loss_generator.item():.4f}]")

        # Save checkpoint at the end of each epoch
        save_checkpoint(epoch, generator, discriminator, optimizer_g, optimizer_d)

    # Save models
    torch.save(generator.state_dict(), "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")


if __name__ == "__main__":
    main()
