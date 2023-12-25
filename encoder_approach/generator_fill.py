from PIL import Image
from torchvision.transforms import ToTensor
from train_encoder import EncoderTrainer


image_path = "processed/img_align_celeba/ims/000135.png"  # Replace with your image path
masked_image = Image.open(image_path)
# Convert the image to a PyTorch tensor
to_tensor = ToTensor()
masked_image = to_tensor(masked_image)
print(masked_image.shape)

mask_path = "masks/000135.png"  # Replace with your mask path
mask = Image.open(mask_path)
# Convert the mask to a PyTorch tensor
to_tensor = ToTensor()
mask = to_tensor(mask)


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


encoder_trainer = EncoderTrainer(train_params, ckpt_params, gan_params)
# encoder_trainer.load_encoder_state("./train_run_0001/encoder_masked/encoder_latest.pt")
encoder_trainer.load_encoder_state("./checkpoints/trained_gan/erkin_233_epoch.pt")
encoded_image = encoder_trainer.encode(masked_image)

encoded_image = encoded_image.cpu().detach()
generated_ims = encoder_trainer.gan.G(encoded_image)

# Inpainted images
inpainting_candidates = generated_ims * mask
# Use cv2 to save the first inpainting candidate, save it as RGB
# plt.imsave('inpainting_candidate.jpg', inpainting_candidates[0].permute(1, 2, 0).detach().numpy())
candidates = inpainting_candidates + masked_image
print(candidates)
