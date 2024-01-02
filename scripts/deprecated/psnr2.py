import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import io
from skimage.metrics import structural_similarity as compare_ssim
from lpips import LPIPS
import torch
import json

nf = 25

images = list()
image_truth = cv2.imread(f"./gt/0012.jpg")
for i in range(nf):
    n = i + 1
    path = f"./snapshots/video0012/{n}.png"
    p = cv2.imread(path)
    images.append(p)


if images[0].shape != image_truth.shape:
    raise ValueError("Image shape does not match")


for i in range(nf):
    images[i] = images[i][700:1700, 800:1500]
image_truth = image_truth[700:1700, 800:1500]

# cv2.imshow('Bottom Left Part', image_truth)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

mse = list()
for i in range(nf):
    mse.append(np.mean((images[i] - image_truth) ** 2))
max_pixel_value = 255

psnr = list()
for i in range(nf):
    psnr.append(10 * np.log10((max_pixel_value ** 2) / mse[i]))
print(psnr)


# Convert the images to grayscale (if they are not already)
gray = list()
for i in range(nf):
    gray.append(cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY))
t_gray = cv2.cvtColor(image_truth, cv2.COLOR_BGR2GRAY)

ssim = list()
for i in range(nf):
    ssim.append(compare_ssim(gray[i], t_gray))
print(ssim)

lpips_model = LPIPS(net='alex')
lpips_truth = torch.tensor(image_truth.astype('float32') / 255).permute(2, 0, 1)
lpips = list()
for i in range(nf):
    lpips_image = torch.tensor(images[i].astype('float32') / 255).permute(2, 0, 1)
    lpips.append(lpips_model.forward(lpips_image, lpips_truth).flatten().detach().numpy().tolist()[0])
print(lpips)

plt.plot(psnr, label="NeRF psnr per image")
plt.show()
plt.plot(ssim, label="NeRF ssim per image")
plt.show()
plt.plot(lpips, label="NeRF lpips per image")
plt.show()

data = dict()
data["psnr"] = psnr 
data["ssim"] = ssim
data["lpips"] = lpips
with open("./results/evaluation12.json", "w") as f:
    json.dump(data, f)


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

# cv2.imshow('Bottom Left Part', image_truth)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

mse = list()
for i in range(nf):
    mse.append(np.mean((images[i] - image_truth) ** 2))
max_pixel_value = 255

psnr = list()
for i in range(nf):
    psnr.append(10 * np.log10((max_pixel_value ** 2) / mse[i]))
print(psnr)


# Convert the images to grayscale (if they are not already)
gray = list()
for i in range(nf):
    gray.append(cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY))
t_gray = cv2.cvtColor(image_truth, cv2.COLOR_BGR2GRAY)

ssim = list()
for i in range(nf):
    ssim.append(compare_ssim(gray[i], t_gray))
print(ssim)

lpips_model = LPIPS(net='alex')
lpips_truth = torch.tensor(image_truth.astype('float32') / 255).permute(2, 0, 1)
lpips = list()
for i in range(nf):
    lpips_image = torch.tensor(images[i].astype('float32') / 255).permute(2, 0, 1)
    lpips.append(lpips_model.forward(lpips_image, lpips_truth).flatten().detach().numpy().tolist()[0])
print(lpips)

plt.plot(psnr, label="NeRF psnr per image")
plt.show()
plt.plot(ssim, label="NeRF ssim per image")
plt.show()
plt.plot(lpips, label="NeRF lpips per image")
plt.show()

data = dict()
data["psnr"] = psnr 
data["ssim"] = ssim
data["lpips"] = lpips
with open("./results/evaluation08.json", "w") as f:
    json.dump(data, f)