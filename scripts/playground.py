# import threading
# from pipeline import NeRFPipeline

# ppl = NeRFPipeline()
# ppl.delay = True
# pipeline_thread = threading.Thread(target=ppl.spinning)
# pipeline_thread.start()

# import numpy as np
# import matplotlib.pyplot as plt
# import json

# with open("./result.json", "r") as f:
#     delay = json.load(f)
# with open("./result_no.json", "r") as f:
#     no_delay = json.load(f)

# plt.plot(delay, label="delay")
# plt.plot(no_delay, label="no delay")
# plt.legend()
# plt.show()

# for i in range(50):
#     if i < 10:

#         continue
#     print("continue")
#     print(i)

# import json
# with open('./data/nerf/salmon/transforms.json', 'r') as f:
#     transforms = json.load(f)

# print(transforms["frames"][6])

import os
import cv2
import tqdm

# image_folder = './data/nerf/salmon/video0011/'
image_folder = './snapshots/video0010/'
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
pbar = tqdm.tqdm(total=len(images))
images.sort(key=lambda x: int(x.split(".")[0]))
frame_rate = 30
image = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = image.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec (e.g., 'XVID' for AVI)
video = cv2.VideoWriter('output.mp4', fourcc, frame_rate, (width, height))
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)
    pbar.update(1)
video.release()
cv2.destroyAllWindows()

# import scipy.stats as stats

# class PoissonData:
#     def __init__(self, mean):
#         self.mean = mean
#         self.poisson_distribution = stats.poisson(self.mean)
#         self.scale = mean * 2

#     def get_number(self):
#         return self.poisson_distribution.rvs()

# # Example usage
# mean = 35
# poisson_data = PoissonData(mean)

# # Generate numbers from the Poisson distribution
# for _ in range(10):
#     number = poisson_data.get_number()
#     print(number)


# import numpy as np

# x = list()
# y = list()
# for i in range(10000):
#     a = np.random.exponential(scale=1/10) # Scale beta = 1/lam
#     b = np.random.exponential(scale=1/4000)
#     # print(a)
#     x.append(a)
#     y.append(b)


# import random
# import matplotlib.pyplot as plt
# import numpy as np

# a = random.uniform(50, 100)
# x = list()
# y = list()
# for i in range(10000):
#     a = random.uniform(100, 150)
#     b = random.uniform(0, 50)
#     x.append(a)
#     y.append(b)

# plt.hist(x, bins=100, edgecolor='k')  # Adjust the number of bins as needed
# plt.hist(y, bins=100, edgecolor='g')
# plt.title('Distribution of Data')
# plt.xlabel('Values')
# plt.ylabel('Frequency')
# plt.show()


