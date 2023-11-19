import os
import torch

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


def save_checkpoint(epoch, generator, discriminator, optimizer_g, optimizer_d, directory):

    checkpoint_path_latest = f"{directory}/checkpoint_latest.pth"
    checkpoint_path_epoch = f"{directory}/checkpoint_epoch_{epoch - 1}.pth"

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

