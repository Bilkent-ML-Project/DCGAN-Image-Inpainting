#!-*- coding:utf-8 -*-
"""image processing utils for pytorch"""
from imageio import imread
import numpy as np
import torchvision.utils as tvls
import torchvision.transforms as transforms

def get_image_from_path(path):
    """
    Read a single image from the input path and return a PIL Image object.
    """
    img = imread(path)
    return img

def image_to_tensor(image):
    """
    Convert the input PIL Image object to a PyTorch Tensor object, in the format channel x image_height x image_width.
    """
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return trans(image)

def get_tensor_image(path):
    img = get_image_from_path(path)
    return image_to_tensor(img)

def save_tensor_images(images, filename, nrow=None, normalize=True):
    """
    Save the input array of image tensors as an image. The default value for nrow is 8, meaning 8 images will be displayed in one row.
    """
    if not nrow:
        tvls.save_image(images, filename, normalize=normalize)
    else:
        tvls.save_image(images, filename, normalize=normalize, nrow=nrow)
