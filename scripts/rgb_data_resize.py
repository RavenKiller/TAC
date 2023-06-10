from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


def crop_to_square(img):
    width, height = img.size  # Get dimensions
    if width == height:
        return img
    if width > height:
        new_width = height
        new_height = height
    else:
        new_width = width
        new_height = width

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    # Crop the center of the image
    img = img.crop((left, top, right, bottom))
    return img


def resize_square(img, size, path):
    width, height = img.size  # Get dimensions
    assert width == height
    if width == size:
        return img
    img = img.resize((size, size), resample=Image.BICUBIC)
    img.save(path)
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kw", required=True, help="Keyword in the sequence names.")
    args = parser.parse_args()
    FOLDER = "/root/TAC/data/sun3d/data"
    all_paths = []
    for root, dirs, files in os.walk(FOLDER):
        if args.kw in root and files:
            for fname in files:
                if ".png" in fname or ".jpg" in fname:
                    path = os.path.join(root, fname)
                    all_paths.append(path)
    for path in tqdm(all_paths):
        img = Image.open(path)
        img = crop_to_square(img)
        img = resize_square(img, 224, path)
