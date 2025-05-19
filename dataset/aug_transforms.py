import os
import random
import json
import cv2
import numpy as np
import torch
from tqdm import trange
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps


# Define custom transformations to replace `imgaug` transformations
class Equalize:
    def __call__(self, img):
        return ImageOps.equalize(img)

class Autocontrast:
    def __call__(self, img):
        return ImageOps.autocontrast(img)

class EnhanceColor:
    def __call__(self, img):
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(random.uniform(0.5, 1.5))

class Sharpen:
    def __call__(self, img):
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(random.uniform(0.0, 1.0))

class ChangeColorTemperature:
    def __init__(self, kelvin_range=(1100, 5000)):
        self.kelvin_range = kelvin_range

    def __call__(self, img):
        kelvin = random.randint(*self.kelvin_range)
        return adjust_temperature(img, kelvin)

def adjust_temperature(img, kelvin):
    temp_scale = kelvin / 5000
    red_channel = int(255 * temp_scale)
    blue_channel = int(255 / temp_scale)
    r, g, b = img.split()
    r = r.point(lambda i: min(255, int(i * red_channel / 255)))
    b = b.point(lambda i: min(255, int(i * blue_channel / 255)))
    return Image.merge('RGB', (r, g, b))

# Define the list of augmentations
aug_type = [
    EnhanceColor(),                   # adjust color enhancement
    Sharpen(),                        # apply sharpening
    ChangeColorTemperature(),         # change color temperature
    Equalize(),            # histogram equalization
    transforms.RandomSolarize(0.5),   # invert colors
    transforms.ColorJitter(brightness=(0.8, 1.2)),  # brightness adjustment
    Autocontrast(),        # adjust contrast automatically
    transforms.RandomGrayscale(p=0.5) # random grayscale
]


