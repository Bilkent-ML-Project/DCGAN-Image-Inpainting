import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import ImageFolder
from dcganv2 import Generator, weights_init, Discriminator
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os
from torch.utils.data import random_split

# Root directory for dataset
dataroot = "../data/img_align_celeba"
workers = 2
batch_size = 128
image_size = 64
nz = 100
num_epochs = 20
lr = 0.0002
beta1 = 0.5
ngpu = 1
run_name = "run1"

torch.manual_seed(42)

def save_checkpoint(epoch, generator, discriminator, optimizer_g, optimizer_d):
    os.makedirs(run_name, exist_ok=True)

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
        print("Checkpoint file not found. Starting from epoch 0.")
        epoch = 0
    return epoch

def load_data(subset_size=0.2, train_ratio=0.8):
    # Create the dataset by loading images from the folder
    print("Loading data...")
    celeba_dataset = ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), 
                                                     (0.5, 0.5, 0.5)),
                            ]))
    
    
    celeba_dataset = ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))

    # Create a subset of the dataset with only {subset_size} of the data
    print(f"Creating a subset of the dataset with only {subset_size} of the data...")
    subset_indices = torch.randperm(len(celeba_dataset))[:int(subset_size * len(celeba_dataset))]
    subset_dataset = torch.utils.data.Subset(celeba_dataset, subset_indices)
        
    # Split the dataset into training and validation sets 
    print(f"Splitting the dataset into training and validation sets with a ratio of {train_ratio}...")
    num_train = int(len(subset_dataset) * train_ratio)
    num_val = len(subset_dataset) - num_train

    # Use random_split to split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(subset_dataset, [num_train, num_val])
    return train_dataset, val_dataset


def main():
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print("Using device: ", device)

    train_dataset, val_dataset = load_data(subset_size=0.2, train_ratio=0.8)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    # Decide which device we want to run on

    # Create the generator and discriminator
    print("Initializing generator and discriminator...")
    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)

    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    real_label = 1
    fake_label = 0

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    checkpoint_path = f"{run_name}/checkpoint_latest.pth"  # Change to the desired checkpoint file
    start_epoch = load_checkpoint(netG, netD, optimizerG, optimizerD, checkpoint_path)

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(start_epoch, num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(train_loader, 0):

            ## Train with all-real batch
            netD.zero_grad()
            real_image = data[0].to(device)
            b_size = real_image.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            # Forward pass
            output = netD(real_image).view(-1)
            # Calculate loss on all-real batch
            lossD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            lossD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = lossD_real + errD_fake
            # Update D
            optimizerD.step()

            # (2) Update G network
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(train_loader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(train_loader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

        save_checkpoint(epoch, netG, netD, optimizerG, optimizerD)
        

    # Save models
    torch.save(netG.state_dict(), f"{run_name}/generator.pth")
    torch.save(netD.state_dict(), f"{run_name}/discriminator.pth")

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


    # Grab a batch of real images from the train_loader
    real_batch = next(iter(train_loader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imsave(f"{run_name}/real_images.png", np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imsave(f"{run_name}/fake_images.png", np.transpose(img_list[-1],(1,2,0)))

if __name__ == "__main__":
    main()