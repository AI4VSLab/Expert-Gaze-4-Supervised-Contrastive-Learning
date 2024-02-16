from torchvision.io import read_image
import cv2
import torch
import pandas as pd


def read_img(p):
    image = read_image(p)
    # if 4 channels need to convert colour space
    if image.shape[0] != 3:
        image = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        image = torch.tensor(image).permute((2, 0, 1))
    return image
