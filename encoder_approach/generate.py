from train_encoder import EncoderTrainer
import utils
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)

train_params = {
    "root_dir": "./processed/img_align_celeba",
    "gen_dir": "../generated",
    "batch_size": 128,
    "train_len": 12800,
    "learning_rate": 0.0004,
    "momentum": (0.5, 0.999),
    "optim": "adam",
    "use_cuda": True,
}

# Checkpoint parameters (report interval size, directories)
ckpt_params = {
    "batch_report_interval": 100,
    "ckpt_path": "./checkpoints/trained_gan",
    "save_stats_interval": 500,
}

# GAN parameters (type and latent dimension size)
gan_params = {"gan_type": "gan", "latent_dim": 100, "n_critic": 1}

train_dataset, val_dataset, test_dataset = utils.load_dataset(
    train_params["root_dir"], train_params["batch_size"]
)

test_dataset = DataLoader(test_dataset, batch_size=train_params["batch_size"])

encoder_trainer = EncoderTrainer(train_params, ckpt_params, gan_params)
# encoder_trainer.load_encoder_state("./train_run_0001/encoder_masked/encoder_latest.pt")
encoder_trainer.load_encoder_state("./checkpoints/trained_gan/erkin_233_epoch.pt")

# Call encode for first batch
for x, _ in test_dataset:
    masked_ims = x
    output = encoder_trainer.encode(x)
    break

first_sixteen = output[:16]
masked_ims = masked_ims[:16]
# Detach the tensors from gpu to cpu
first_sixteen = first_sixteen.cpu().detach()
generated_ims = encoder_trainer.gan.G(first_sixteen)

# Create a grid from the generated images tensor
grid_generated = make_grid(generated_ims, nrow=4, padding=2, normalize=True)
grid_generated_np = grid_generated.permute(1, 2, 0).numpy()

# Create a grid from the original masked images tensor
grid_masked = make_grid(masked_ims, nrow=4, padding=2, normalize=True)
grid_masked_np = grid_masked.permute(1, 2, 0).numpy()

# Concatenate the two grids horizontally
concatenated_grid = np.concatenate([grid_generated_np, grid_masked_np], axis=1)

# Display the concatenated grid of images
plt.imshow(concatenated_grid)
plt.axis("off")

plt.savefig("concatenated_grid.png")
