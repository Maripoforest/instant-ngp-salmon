import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

motherpath = "./bad/video0010/"
goodpath = "./good/video0010/"
truth = "./truth/"
gp = list()
bp = list()

for i in range(30):
    bpath = motherpath + str(i) + ".png"
    print(bpath)
    if not os.path.exists(bpath):
        continue
    else:
        gpath = goodpath + str(i)+ ".png"
        tpath = truth + str(i).zfill(4) + "/0010.jpg"
        image_bad = cv2.imread(bpath)
        image_good = cv2.imread(gpath)
        image_truth = cv2.imread(tpath)

        if image_bad.shape != image_truth.shape:
            raise ValueError("Image shape does not match")

        gmse = np.mean((image_good - image_truth) ** 2)
        bmse = np.mean((image_bad - image_truth) ** 2)

        max_pixel_value = 255

        gpsnr = 10 * np.log10((max_pixel_value ** 2) / gmse)
        bpsnr = 10 * np.log10((max_pixel_value ** 2) / bmse)

        gp.append(gpsnr)
        bp.append(bpsnr)
print(gp)
print(bp)
plt.plot(gp, label="NeRF psnr per image")
plt.plot(bp, label="Delayed NeRF psnr per image")
plt.legend()
plt.show()