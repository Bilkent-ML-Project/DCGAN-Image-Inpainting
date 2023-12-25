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
from utils import load_checkpoint, save_checkpoint
from tqdm import tqdm

# Root directory for dataset
dataroot = "../data/"
workers = 2
batch_size = 128
image_size = 64
nz = 100
num_epochs = 1
lr = 0.0002
beta1 = 0.5
ngpu = 1
run_name = "run1"

torch.manual_seed(42)

directory = None


def create_cur_run_dir():
    current_directory = os.getcwd()
    run_directories = [d for d in os.listdir(current_directory) if os.path.isdir(os.path.join(current_directory, d)) and d.startswith("run_")]

    if run_directories:
        # Extract the run numbers and find the maximum
        run_numbers = [int(d.split("_")[1]) for d in run_directories]
        highest_run_number = max(run_numbers)
        directory_name = f"run_{highest_run_number + 1}"
    else:
        directory_name = f"run_0"
        

    directory_path = os.path.join(current_directory, directory_name)

    log("Directory path is" + directory_path)

    os.mkdir(directory_path)

    # Create log.txt in directory path
    with open(f"{directory_path}/log.txt", "w") as f:
        f.write("")



    return directory_path


def load_data(subset_size=0.2, train_ratio=0.8):
    # Create the dataset by loading images from the folder
    log("Loading data...")
    celeba_dataset = ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), 
                                                     (0.5, 0.5, 0.5)),
                            ]))

    # Create a subset of the dataset with only {subset_size} of the data
    log(f"Creating a subset of the dataset with only {subset_size} of the data...")
    subset_indices = torch.randperm(len(celeba_dataset))[:int(subset_size * len(celeba_dataset))]
    subset_dataset = torch.utils.data.Subset(celeba_dataset, subset_indices)
        
    # Split the dataset into training and validation sets 
    log(f"Splitting the dataset into training and validation sets with a ratio of {train_ratio}...")
    num_train = int(len(subset_dataset) * train_ratio)
    num_val = len(subset_dataset) - num_train

    # Use random_split to split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(subset_dataset, [num_train, num_val])
    return train_dataset, val_dataset


def main():

    # List cuda devices
    log("Available devices: " + str(torch.cuda.device_count()))

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Create the run directory
    global directory
    directory = create_cur_run_dir()

    log("Using device: " + str(device))
    
    train_dataset, val_dataset = load_data(subset_size=0.2, train_ratio=0.8)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4)

    # Decide which device we want to run on

    # Create the generator and discriminator
    log("Initializing generator and discriminator...")
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

    checkpoint_path = f"{directory}/checkpoint_latest.pth"  # Change to the desired checkpoint file
    start_epoch = load_checkpoint(netG, netD, optimizerG, optimizerD, checkpoint_path)

    log("Starting Training Loop...")
    # For each epoch
    for epoch in range(start_epoch, num_epochs):
        # For each batch in the dataloader
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):

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
                log('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
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
        netG.eval()
        netD.eval()

        val_loss_d = 0.0
        val_loss_g = 0.0

        with torch.no_grad():
            for i, data in tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation"):
                real_image = data[0].to(device)
                b_size = real_image.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

                # Forward pass real batch through Discriminator
                output = netD(real_image).view(-1)
                lossD_real = criterion(output, label)
                val_loss_d += lossD_real.item()

                # Generate batch of latent vectors
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                # Generate fake image batch with G
                fake = netG(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                val_loss_d += errD_fake.item()

                # Generator loss on validation not used for training generator
                label.fill_(real_label)
                output = netD(fake).view(-1)
                errG = criterion(output, label)
                val_loss_g += errG.item()

                
                # log statistics
                if i % 100 == 0:
                    log('[Validation][%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, num_epochs, i, len(val_loader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        val_loss_d /= len(val_loader)
        val_loss_g /= len(val_loader)
        
        log(f"[Epoch {epoch}/{start_epoch + num_epochs}] [Validation END]"
                f"[D Avg Loss {val_loss_d:4f}] "
                f"[G Avg Loss: {val_loss_g:.4f}]")
        save_checkpoint(epoch, netG, netD, optimizerG, optimizerD, directory)
        

    # Save models
    torch.save(netG.state_dict(), f"{directory}/generator.pth")
    torch.save(netD.state_dict(), f"{directory}/discriminator.pth")

    # plt.figure(figsize=(10,5))
    # plt.title("Generator and Discriminator Loss During Training")
    # plt.plot(G_losses,label="G")
    # plt.plot(D_losses,label="D")
    # plt.xlabel("iterations")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()


    # Grab a batch of real images from the train_loader
    # real_batch = next(iter(train_loader))

    # # Plot the real images
    # plt.figure(figsize=(15,15))
    # plt.subplot(1,2,1)
    # plt.axis("off")
    # plt.title("Real Images")
    # plt.imsave(f"{directory}/real_images.png", np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # # Plot the fake images from the last epoch
    # plt.subplot(1,2,2)
    # plt.axis("off")
    # plt.title("Fake Images")
    # plt.imsave(f"{directory}/fake_images.png", np.transpose(img_list[-1],(1,2,0)))


def log(message):
    print(message)
    if directory:
        with open(f"{directory}/log.txt", "a") as f:
            f.write(f"{message}\n")

if __name__ == "__main__":
    main()