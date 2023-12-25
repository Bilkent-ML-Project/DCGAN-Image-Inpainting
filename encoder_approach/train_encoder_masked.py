#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for Encoder network with pre-trained DCGAN components.
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from dcgan import DCGAN, Generator, Discriminator, Encoder  # Import Encoder class here
import utils
import argparse
import os
import train
from torch.utils.data import DataLoader
import logging
import time
import numpy as np


def create_run_dir(output_dir, subfolder):
    # If output_dir is None, set it to the current directory
    if output_dir is None:
        output_dir = os.getcwd()

    # Get the list of directories in the folder that has the syntax "run_***" where *** is a number
    # Get the latest run number
    # Calculate the current run number by incrementing the latest run number by 1
    # Create a directory with the name "run_***" where *** is the current run number
    # Return the current run number
    print("Calculating run number...")

    run_number = 0

    for file in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, file)):
            if file.startswith("train_run_"):
                # Compare the current run number with the run number in the file name
                # If the run number in the file name is greater than the current run number, set the current run number to the run number in the file name
                if int(file[-4:]) > run_number:
                    run_number = int(file[-4:])
    run_number += 1
    # Set the directory name to "run_***" where *** is the current run number in 3 digits
    main_dir = os.path.join(output_dir, f"train_run_{run_number:04d}")
    os.mkdir(main_dir)
    # If subfolder is not None, create a subfolder in the run directory
    if subfolder is not None:
        run_dir = os.path.join(main_dir, subfolder)
        os.mkdir(run_dir)
    else:
        run_dir = main_dir

    print(f"Run number: {run_number:04d}")
    print(f"Run directory: {run_dir}")
    return run_number, run_dir, main_dir


run_number, run_dir, main_dir = create_run_dir(os.getcwd(), "encoder_masked")
log_file = os.path.join(run_dir, "log.txt")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Create a logger
logger = logging.getLogger("analyze.py")


def log(message: str):
    global logger
    # Log the message with current time
    # The format of the time is "DD-MM-YYYY HH:MM:SS"
    # The format of the message is "[DD-MM-YYYY HH:MM:SS] message"
    # Use the logger
    encoded_msg = f"[{time.strftime('%d-%m-%Y %H:%M:%S')}] {message}"
    logger.info(encoded_msg)
    print(f"{encoded_msg}")


def apply_mask(images, mask_size=(16, 16), mask_value=0):
    """
    Apply a square mask to the given batch of images.
    The mask will have a size of `mask_size` and will be placed randomly on each image.
    The pixels under the mask will be set to `mask_value`.
    """
    masked_images = images.clone()
    n, c, h, w = images.shape
    mask_h, mask_w = mask_size

    for i in range(n):
        top = np.random.randint(0, h - mask_h)
        left = np.random.randint(0, w - mask_w)
        masked_images[i, :, top : top + mask_h, left : left + mask_w] = mask_value

    return masked_images


class EncoderTrainer(object):
    """Class to train the Encoder with a pre-trained DCGAN"""

    def __init__(self, train_params, ckpt_params, gan_params):
        # ... (Retain initialization code from CelebA class)

        # Initialize DCGAN and Encoder
        self.gan = DCGAN(
            gan_params["gan_type"],
            gan_params["latent_dim"],
            train_params["batch_size"],
            train_params["use_cuda"],
        )
        self.encoder = Encoder(gan_params["latent_dim"])

        # Load pre-trained models
        self.load_pretrained_models(ckpt_params["ckpt_path"])

        # Optimizer for the encoder
        self.encoder_optimizer = optim.Adam(
            self.encoder.parameters(), lr=train_params["learning_rate"]
        )

        if train_params["use_cuda"] and torch.cuda.is_available():
            self.encoder = self.encoder.cuda()

    def load_pretrained_models(self, ckpt_path):
        # Load pre-trained generator and discriminator weights
        self.gan.G.load_state_dict(torch.load(os.path.join(ckpt_path, "gan-gen.pt")))
        self.gan.D.load_state_dict(torch.load(os.path.join(ckpt_path, "gan-disc.pt")))

    def save_encoder_state(self, filename):
        torch.save(self.encoder.state_dict(), filename)
        log(f"Encoder state saved to {filename}")

    def load_encoder_state(self, filename):
        self.encoder.load_state_dict(torch.load(filename))
        log(f"Encoder state loaded from {filename}")

    def validate_encoder(self, data_loader):
        self.encoder.eval()  # Set encoder to evaluation mode
        total_loss = 0
        with torch.no_grad():  # No need to track gradients
            for i, (x, _) in enumerate(data_loader):
                x = Variable(x)
                if self.gan.use_cuda:
                    x = x.cuda()

                z = self.encoder(x)
                reconstructed_x = self.gan.G(z)
                loss = nn.MSELoss()(reconstructed_x, x)
                total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        self.encoder.train()  # Set encoder back to train mode
        return avg_loss

    def train_encoder(self, data_loader, val_loader, nb_epochs):
        if self.gan.use_cuda:
            self.encoder.cuda()
            self.gan.G.cuda()

        # Training loop for the encoder with masked images
        for epoch in range(nb_epochs):
            for i, (x, _) in enumerate(data_loader):
                # Apply mask to input images
                masked_x = apply_mask(x)

                masked_x = Variable(masked_x)
                x = Variable(x)  # Original images as target
                if self.gan.use_cuda:
                    masked_x = masked_x.cuda()
                    x = x.cuda()

                # Forward pass through encoder and pre-trained generator
                z = self.encoder(masked_x)
                reconstructed_x = self.gan.G(z)

                # Reconstruction loss with original image as target
                loss = nn.MSELoss()(reconstructed_x, x)

                # Backward pass and optimization
                self.encoder_optimizer.zero_grad()
                loss.backward()
                self.encoder_optimizer.step()

                if i % 100 == 0:
                    log(
                        f"Epoch [{epoch+1}/{nb_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item()}"
                    )

            val_loss = self.validate_encoder(val_loader)
            log(f"Epoch {epoch+1}, Validation Loss: {val_loss}")
            self.save_encoder_state(os.path.join(run_dir, f"encoder_epoch_{epoch}.pt"))


# Argument parser
parser = argparse.ArgumentParser(
    description="Generative adversarial network (GAN) implementation in PyTorch"
)
parser.add_argument(
    "-d",
    "--ckpt",
    help="checkpoint path",
    metavar="PATH",
    default="./checkpoints/trained_gan",
)
parser.add_argument(
    "-t",
    "--type",
    help="model type",
    action="store",
    choices=["gan", "wgan", "lsgan"],
    default="gan",
    type=str,
)
parser.add_argument(
    "-r", "--redux", help="train on smaller dataset with 10k faces", action="store_true"
)
parser.add_argument(
    "-o",
    "--optimizer",
    help="sgd optimizer",
    choices=["adam", "rmsprop"],
    default="adam",
    type=str,
)
parser.add_argument(
    "-lr", "--learning-rate", help="learning rate", default=0.0004, type=float
)
parser.add_argument(
    "-bs", "--batch-size", help="sgd minibatch size", default=64, type=int
)
parser.add_argument("-n", "--nb-epochs", help="number of epochs", default=50, type=int)
parser.add_argument(
    "-c", "--critic", help="d/g update ratio (critic)", default=1, type=int
)
parser.add_argument("-s", "--seed", help="random seed for debugging", type=int)
parser.add_argument("-gpu", "--cuda", help="use cuda", action="store_true")
args = parser.parse_args()

# Training parameters (saving directory, learning rate, optimizer, etc.)
train_params = {
    "root_dir": "./processed/",
    "gen_dir": "../generated",
    "batch_size": 128,
    "train_len": 12800 if args.redux else 202599,
    "learning_rate": args.learning_rate,
    "momentum": (0.5, 0.999),
    "optim": args.optimizer,
    "use_cuda": args.cuda,
}

# Checkpoint parameters (report interval size, directories)
ckpt_params = {
    "batch_report_interval": 100,
    "ckpt_path": args.ckpt,
    "save_stats_interval": 500,
}

# GAN parameters (type and latent dimension size)
gan_params = {"gan_type": args.type, "latent_dim": 100, "n_critic": args.critic}

# Main execution
if __name__ == "__main__":
    # (Retain argument parsing code from the original script)
    log("Training started")

    # Ready to train
    encoder_trainer = EncoderTrainer(train_params, ckpt_params, gan_params)
    train_dataset, val_dataset, test_dataset = utils.load_dataset(
        train_params["root_dir"], train_params["batch_size"]
    )

    data_loader = DataLoader(
        train_dataset, batch_size=train_params["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_params["batch_size"], shuffle=True
    )
    # Train the encoder
    encoder_trainer.train_encoder(data_loader, val_loader, args.nb_epochs)
    encoder_trainer.save_encoder_state(os.path.join(run_dir, "encoder_latest.pt"))
    log("Training completed")
