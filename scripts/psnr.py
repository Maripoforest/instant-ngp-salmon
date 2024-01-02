import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import io
from skimage.metrics import structural_similarity as compare_ssim
from lpips import LPIPS
import torch
import json


def compute_lpips():
    nf = 25
    images = list()
    image_truth = cv2.imread(f"./gt/0008.jpg")
    for i in range(nf):
        n = i + 1
        path = f"./snapshots/video0008/{n}.png"
        p = cv2.imread(path)
        images.append(p)

    if images[0].shape != image_truth.shape:
        raise ValueError("Image shape does not match")

    for i in range(nf):
        images[i] = images[i][900:1900, 900:1550]
    image_truth = image_truth[900:1900, 900:1550]

    lpips_model = LPIPS(net='alex')
    lpips_truth = torch.tensor(image_truth.astype('float32') / 255).permute(2, 0, 1)
    lpips = list()
    for i in range(nf):
        lpips_image = torch.tensor(images[i].astype('float32') / 255).permute(2, 0, 1)
        lpips.append(lpips_model.forward(lpips_image, lpips_truth).flatten().detach().numpy().tolist()[0])
    data = dict()
    min_i = 0
    min_l = 100
    for i in range(nf):
        if lpips[i] < min_l:
            min_i = i
            min_l = lpips[i]
    data["time"] = 25 - min_i
    data["lpips"] = 1 - min(lpips)
    with open("./temp/tmp.json", "w+") as f:
        json.dump(data, f)

if __name__ == "__main__":
    compute_lpips()
    