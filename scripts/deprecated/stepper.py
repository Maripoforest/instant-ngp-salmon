import threading
import io
import numpy as np
import os
import shutil
import time
import random
import json
from pipeline import NeRFPipeline

with open("./camera_frameshot.json") as f:
    frameshots = json.load(f)
n = 12
cameras_to_skip = [8, 12]
destination_folder = "./data/nerf/steak/frames/"
parent_folder = "./data/nerf/steak/allframes/frames/"
if os.path.exists(destination_folder):
    shutil.rmtree(destination_folder)
    os.makedirs(destination_folder)
while n >= 1:
    last_name = str(n+1).zfill(4)
     
    last_folder = destination_folder + last_name     
    destination_image_folder = os.path.join(destination_folder, f"{n:04d}")
    if os.path.exists(last_folder) and not os.path.exists(destination_image_folder):
        print(last_folder)
        print(destination_image_folder)
        shutil.copytree(last_folder, destination_image_folder)  
    elif not os.path.exists(destination_image_folder):
        os.makedirs(destination_image_folder) 
    for i in range(21):
        n_to_transfer = int(frameshots["frames"][n+94][i])
        frame_name = str(n_to_transfer).zfill(4)
        source_folder = parent_folder + frame_name
        source_path = source_folder
        if i in cameras_to_skip:
            continue
        elif n_to_transfer == 0:
            continue
        else:
            if os.path.isdir(source_path):
                source_image_path = os.path.join(source_path, f"{i:04d}.jpg")
                if os.path.exists(source_image_path):
                    
                    destination_image_path = os.path.join(destination_image_folder, f"{i:04d}.jpg")
                    shutil.copy(source_image_path, destination_image_path)
    n -= 1