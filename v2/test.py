import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from dcganv2 import Generator
from PIL import Image

# Root directory for dataset
nz = 100
ngpu = 1
dataroot = "../processed/ims"
image_size = 64
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    return image

def load_generator(state_dict_path, ngpu):
    generator = Generator(ngpu)  
    state_dict = torch.load(state_dict_path)
    generator.load_state_dict(state_dict)
    return generator


def test_generator(generator, input_image):
    with torch.no_grad():
        output = generator(input_image)

    return output

def display_images(input_image, generated_image):
    input_image = input_image.squeeze(0).permute(1, 2, 0).numpy()
    generated_image = generated_image.squeeze(0).permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(input_image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(generated_image)
    axes[1].set_title("Generated Image")
    axes[1].axis("off")

    plt.show()


def main():
    # Load an image from the dataset
    image_path = dataroot + "/000001.jpg"

    # Load image
    img = load_image(image_path)
    netG = load_generator("./run1/generator.pth", ngpu)
    output = test_generator(netG, img)
    display_images(img, output)

if __name__ == "__main__":
    main()