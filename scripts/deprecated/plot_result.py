import json
import os
import shutil
import matplotlib.pyplot as plt

n = 6

for i in range(6):
    filename = f"./results/evaluation{i}.json"
    with open(filename) as f:
        data = json.load(f)
        # plt.plot(data["lpips"])
        # plt.show()
    if i == 0:
        lpips = data["lpips"]
        ssim = data["ssim"]
        psnr = data["psnr"]
    else:
        for i in range(len(lpips)):
            ssim[i] += data["ssim"][i]
            lpips[i] += data["lpips"][i]
            psnr[i] += data["psnr"][i]
for i in range(len(lpips)):
    lpips[i] = lpips[i]/6
    ssim[i] = ssim[i]/6
    psnr[i] = psnr[i]/6
lpips = lpips[1:]
ssim = ssim[1:]
psnr = psnr[1:]
plt.plot(lpips)
plt.show()
plt.plot(ssim)
plt.show()
plt.plot(psnr)
plt.show()