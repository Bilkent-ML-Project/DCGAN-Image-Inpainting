from train_encoder_masked import EncoderTrainer
import utils
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import lpips
from skimage.metrics import structural_similarity as ssim
import numpy as np
import time
import os
import logging

torch.manual_seed(42)


def create_test_run_dir(output_dir, subfolder):
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
            if file.startswith("run_"):
                # Compare the current run number with the run number in the file name
                # If the run number in the file name is greater than the current run number, set the current run number to the run number in the file name
                if int(file[4:]) > run_number:
                    run_number = int(file[4:])
    run_number += 1
    # Set the directory name to "run_***" where *** is the current run number in 3 digits
    main_dir = os.path.join(output_dir, f"run_{run_number:04d}")
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


run_number, run_dir, main_dir = create_test_run_dir(os.getcwd(), "encoder_masked")
log_file = os.path.join(run_dir, "log.txt")
logging.basicConfig(filename=log_file, level=logging.INFO)
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


train_params = {
    "root_dir": "data/img_align_celeba/img_align_celeba",
    "gen_dir": "../generated",
    "batch_size": 128,
    "train_len": 12800,
    "learning_rate": 0.0004,
    "momentum": (0.5, 0.999),
    "optim": "adam",
    "use_cuda": False,
}

# Checkpoint parameters (report interval size, directories)
ckpt_params = {
    "batch_report_interval": 100,
    "ckpt_path": "./checkpoints/trained_gan",
    "save_stats_interval": 500,
}

# GAN parameters (type and latent dimension size)
gan_params = {"gan_type": "gan", "latent_dim": 100, "n_critic": 1}


encoder_trainer = EncoderTrainer(train_params, ckpt_params, gan_params)
encoder_trainer.load_encoder_state("lr_00002/encoder_epoch_49.pt")

train_dataset, test_dataset, test_dataset = utils.load_dataset(
    train_params["root_dir"], train_params["batch_size"]
)


test_loader = DataLoader(
    test_dataset, batch_size=train_params["batch_size"], shuffle=False
)

# Initialize LPIPS model
lpips_model = lpips.LPIPS(net="alex")


# Function to apply mask
def apply_mask(images, mask_size=(16, 16), mask_value=0):
    masked_images = images.clone()
    n, c, h, w = images.shape
    mask_h, mask_w = mask_size

    for i in range(n):
        top = np.random.randint(0, h - mask_h)
        left = np.random.randint(0, w - mask_w)
        masked_images[i, :, top : top + mask_h, left : left + mask_w] = mask_value

    return masked_images


"""
def calculate_ssim(original, generated):
    ssim_scores = []
    for o, g in zip(original, generated):
        score = ssim(o.numpy(), g.detach().cpu().numpy(), multichannel=True)
        ssim_scores.append(score)
    return ssim_scores
"""


def calculate_ssim(original, generated):
    ssim_scores = []
    for o, g in zip(original, generated):
        # Pass win_size explicitly with an odd value less than or equal to the smaller side of the images
        win_size = min(o.shape[0], o.shape[1], g.shape[0], g.shape[1])
        win_size = win_size if win_size % 2 == 1 else win_size - 1  # Ensure it's odd

        if not isinstance(o, np.ndarray):
            o = o.numpy()

        if not isinstance(g, np.ndarray):
            g = g.numpy()

        score = ssim(o, g, win_size=win_size, multichannel=True, data_range=255)
        ssim_scores.append(score)
    return ssim_scores


def calculate_lpips(original, generated, model):
    lpips_scores = []
    for o, g in zip(original, generated):
        if isinstance(o, np.ndarray):
            o = torch.from_numpy(o)

        if isinstance(g, np.ndarray):
            g = torch.from_numpy(g)
        o = o.unsqueeze(0)
        g = g.unsqueeze(0)
        score = model(o, g)
        lpips_scores.append(score.item())
    return lpips_scores


# Placeholder for real images
real_images = []
masked_images = []
generated_images = []


# Function to calculate average SSIM and LPIPS scores
def calculate_average_scores(ssim_scores, lpips_scores):
    avg_ssim = sum(ssim_scores) / len(ssim_scores)
    avg_lpips = sum(lpips_scores) / len(lpips_scores)
    return avg_ssim, avg_lpips


# Placeholder for SSIM and LPIPS scores
all_ssim_scores = []
all_lpips_scores = []

composite_images = []
composite_ssim_scores = []
composite_lpips_scores = []


def create_composite_image(original, generated):
    """
    Combine the unmasked part of the original image with the inpainted part of the generated image.
    Assumes masked regions in the original image are set to 0.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if not isinstance(original, np.ndarray):
        original = original.numpy()
    if not isinstance(generated, np.ndarray):
        generated = generated.numpy()

    # Create a mask based on where the original image is 0 (assumed to be the masked area)
    mask = original == 0

    # Use the mask to blend the original and generated images
    composite = np.where(mask, generated, original)

    return composite


log("Starting the test...")
# Encode and generate images with masks
for i, (x, _) in enumerate(test_loader):
    if train_params["use_cuda"] and torch.cuda.is_available():
        encoder_trainer.encoder.cuda()
        encoder_trainer.gan.G.cuda()

    masked_x = apply_mask(x)  # Apply mask to the input images

    if train_params["use_cuda"] and torch.cuda.is_available():
        masked_x = masked_x.cuda()
        x = x.cuda()

    output = encoder_trainer.encoder(masked_x)

    if train_params["use_cuda"] and torch.cuda.is_available():
        output = output.cuda()

    generated_ims = encoder_trainer.gan.G(output)

    # Move tensors to CPU for SSIM and LPIPS calculation
    x_cpu = x.cpu()
    generated_ims_cpu = generated_ims.detach().cpu()

    if train_params["use_cuda"] and torch.cuda.is_available():
        generated_ims = generated_ims.cuda()

    composite_img = create_composite_image(masked_x, generated_ims.detach().cpu())
    composite_img_tensor = torch.from_numpy(composite_img)
    composite_images.extend(composite_img_tensor)
    real_images.extend(x)
    masked_images.extend(masked_x.cpu())
    generated_images.extend(generated_ims.detach().cpu())

    # Calculate and accumulate scores
    ssim_scores = calculate_ssim(x_cpu, generated_ims_cpu)
    lpips_scores = calculate_lpips(x_cpu, generated_ims_cpu, lpips_model)

    composite_ssim_score = calculate_ssim(x_cpu, composite_img_tensor)
    composite_lpips_score = calculate_lpips(x_cpu, composite_img_tensor, lpips_model)

    all_ssim_scores.extend(ssim_scores)
    all_lpips_scores.extend(lpips_scores)

    composite_ssim_scores.extend(composite_ssim_score)
    composite_lpips_scores.extend(composite_lpips_score)

# Detach and move tensors to CPU
real_images = torch.stack(real_images).cpu()
generated_ims = generated_ims.cpu().detach()

# Sort images based on SSIM scores
lowest_ssim_indices = np.argsort(all_ssim_scores)[:16]
highest_ssim_indices = np.argsort(all_ssim_scores)[-16:]

# Sort images based on LPIPS scores
lowest_lpips_indices = np.argsort(all_lpips_scores)[:16]
highest_lpips_indices = np.argsort(all_lpips_scores)[-16:]

# Convert to tensor
expected_batch_size = 128  # Set your expected batch size
current_batch_size = len(composite_images)

# if current_batch_size < expected_batch_size:
#     # Calculate the difference in size
#     size_difference = expected_batch_size - current_batch_size

#     # Assuming the composite images have 3 color channels and a size of 64x64
#     # Adjust the dimensions according to your specific use case
#     zero_filled_tensors = torch.zeros(size_difference, 3, 64, 64)

#     # Append zero-filled tensors to composite_images
#     composite_images.extend(zero_filled_tensors)

# Convert to tensor
composite_images = torch.stack(composite_images)

def create_grid_image(originals, masked, generated, composite, indices, title):
    images_per_set = 4  # Four images per set (original, masked, generated, composite)
    total_images = len(indices) * images_per_set
    grid = torch.zeros(
        (total_images, originals.size(1), originals.size(2), originals.size(3))
    )

    for i, idx in enumerate(indices):
        grid[i * images_per_set] = originals[idx]
        grid[i * images_per_set + 1] = masked[idx]
        grid[i * images_per_set + 2] = generated[idx]
        grid[i * images_per_set + 3] = composite[idx]  # Composite image

    grid_image = make_grid(grid, nrow=images_per_set, padding=2, normalize=True)

    plt.figure(figsize=(10, len(indices) * 10))
    plt.imshow(np.transpose(grid_image.numpy(), (1, 2, 0)))
    plt.title(title)
    plt.axis("off")
    plt.savefig(f"{run_dir}/{title}.png")


log("Creating images...")
# Create and save images for lowest and highest SSIM scores
create_grid_image(
    real_images,
    masked_images,
    generated_images,
    composite_images,
    lowest_ssim_indices,
    "Lowest SSIM Scores",
)
create_grid_image(
    real_images,
    masked_images,
    generated_images,
    composite_images,
    highest_ssim_indices,
    "Highest SSIM Scores",
)

# Create and save images for lowest and highest LPIPS scores
create_grid_image(
    real_images,
    masked_images,
    generated_images,
    composite_images,
    lowest_lpips_indices,
    "Lowest LPIPS Scores",
)
create_grid_image(
    real_images,
    masked_images,
    generated_images,
    composite_images,
    highest_lpips_indices,
    "Highest LPIPS Scores",
)

# Sort composite images based on SSIM scores
composite_lowest_ssim_indices = np.argsort(composite_ssim_scores)[:16]
composite_highest_ssim_indices = np.argsort(composite_ssim_scores)[-16:]

# Sort composite images based on LPIPS scores
composite_lowest_lpips_indices = np.argsort(composite_lpips_scores)[:16]
composite_highest_lpips_indices = np.argsort(composite_lpips_scores)[-16:]

# Create and save images for lowest and highest SSIM scores for composite images
create_grid_image(
    real_images,
    masked_images,
    generated_images,
    composite_images,
    composite_lowest_ssim_indices,
    "Composite Lowest SSIM Scores",
)
create_grid_image(
    real_images,
    masked_images,
    generated_images,
    composite_images,
    composite_highest_ssim_indices,
    "Composite Highest SSIM Scores",
)

# Create and save images for lowest and highest LPIPS scores for composite images
create_grid_image(
    real_images,
    masked_images,
    generated_images,
    composite_images,
    composite_lowest_lpips_indices,
    "Composite Lowest LPIPS Scores",
)
create_grid_image(
    real_images,
    masked_images,
    generated_images,
    composite_images,
    composite_highest_lpips_indices,
    "Composite Highest LPIPS Scores",
)


log("Calculating average scores...")
# Calculate average scores
avg_ssim_score, avg_lpips_score = calculate_average_scores(
    all_ssim_scores, all_lpips_scores
)
log(f"Average SSIM Score: {avg_ssim_score}")
log(f"Average LPIPS Score: {avg_lpips_score}")


com_avg_ssim_score, com_avg_lpips_score = calculate_average_scores(
    composite_ssim_scores, composite_lpips_scores
)
log(f"Average Composite SSIM Score: {com_avg_ssim_score}")
log(f"Average Composite LPIPS Score: {com_avg_lpips_score}")
