import threading
import io
import numpy as np
import os
import shutil
import time
import random
import json


class NeRFPipeline:
    def __init__(self, delay_probability = 0.4) -> None:
        self.frame_n = 1
        self.dataset = "salmon"
        self.cameras_to_skip = [8, 12]
        self.destination_folder = "./data/nerf/" + self.dataset + "/images/"
        self.num_images_to_transfer = 21
        self.parent_folder = "./data/nerf/" + self.dataset + "/frames/"
        self.source_folder = self.parent_folder + str(self.frame_n).zfill(4)
        with open("./camera_frameshot.json") as f:
            self.frameshots = json.load(f)
        self.stop = False
        self.delay_probability = delay_probability
        self.delay = False
        self.sampling_interval = 1  # 0.25 miu
        self.intentional_delay = 3  # 0.15 
        # self.service_time = 0.2
        self.list_p = 0
        self.service_time = np.random.normal(0.5, 0.1, 200)
        self._use_old_data = False
        self.eof = False 
        self.init_images()

    def init_images(self):
        source_path = os.path.join(self.parent_folder, str(1).zfill(4))
        if os.path.exists(self.destination_folder):
            shutil.rmtree(self.destination_folder)
        os.makedirs(self.destination_folder)
        if self.delay:
            for i in range(2):
                    if i in self.cameras_to_skip:
                        continue
                    else:
                        source_image_path = os.path.join(source_path, f"{i:04d}.jpg")
                        if os.path.exists(source_image_path):
                            destination_image_path = os.path.join(self.destination_folder, f"{i:04d}.jpg")
                            if os.path.exists(destination_image_path):
                                shutil.os.remove(destination_image_path)
                            shutil.copy(source_image_path, destination_image_path)
        else:
            frame_name = str(self.frame_n).zfill(4)    
            self.source_folder = os.path.join(self.parent_folder + frame_name) 
            print(self.source_folder)
            if not os.path.exists(self.source_folder):
                print("source_folder does not exist")
            print("normal transmission")
            self.transfer_images()
        self.frame_n += 1
    
    def set_dataset(self, dataset):
        self.dataset = dataset
        self.destination_folder = "./data/nerf/" + self.dataset + "/images/"
        self.parent_folder = "./data/nerf/" + self.dataset + "/frames/"
        print(self.dataset)
        print("dataset: ", self.dataset)
        self.init_images()


    def transfer_images(self):
        source_path = self.source_folder
        if os.path.isdir(source_path):
            for i in range(self.num_images_to_transfer):
                if i in self.cameras_to_skip:
                    continue
                else:
                    source_image_path = os.path.join(source_path, f"{i:04d}.jpg")
                    if os.path.exists(source_image_path):
                        destination_image_path = os.path.join(self.destination_folder, f"{i:04d}.jpg")
                        # if os.path.exists(destination_image_path):
                        #     shutil.os.remove(destination_image_path)
                        shutil.copy(source_image_path, destination_image_path)
                        # print(source_image_path)

    def step(self):
        if not self.eof:
            if self.delay: 
                print("delayed transmission")
                if os.path.exists(self.destination_folder):
                    shutil.rmtree(self.destination_folder)
                os.makedirs(self.destination_folder)
                for i in range(self.num_images_to_transfer):
                    n_to_transfer = int(self.frameshots["frames"][self.frame_n][i])
                    if i in self.cameras_to_skip:
                        continue
                    elif n_to_transfer == 0:
                        continue
                    else:
                        frame_name = str(n_to_transfer).zfill(4)
                        self.source_folder = self.parent_folder + frame_name      
                        source_path = self.source_folder
                        if os.path.isdir(source_path):
                        # The no-data method
                            source_image_path = os.path.join(source_path, f"{i:04d}.jpg")
                            if os.path.exists(source_image_path):
                                destination_image_path = os.path.join(self.destination_folder, f"{i:04d}.jpg")
                                # The old-data method
                                # if os.path.exists(destination_image_path):
                                #     shutil.os.remove(destination_image_path)            
                                shutil.copy(source_image_path, destination_image_path)

            else:
                frame_name = str(self.frame_n).zfill(4)    
                self.source_folder = os.path.join(self.parent_folder + frame_name) 
                if os.path.exists(self.destination_folder):
                    shutil.rmtree(self.destination_folder)
                os.makedirs(self.destination_folder)
                print(self.source_folder)
                if not os.path.exists(self.source_folder):
                    print("f{self.source_folder} does not exist")
                print("normal transmission")
                self.transfer_images()
            self.frame_n += 1
        else:
            print(self.source_folder)
            print("EOF Stop")

    
if __name__ == '__main__':
    ppl = NeRFPipeline()
    ppl.delay = False

    # 1. Per camera comparison at same frame 
    # 2. Object with different speed
    # 3. Training interval
    # 4. Use old data or no data
    # 5. Blurring in detail (screenshot)
    # 6. Delay components (which part caused most delay? I/O, training, etc.) Realtime analysis
    # 7. Paper: Uncertainty guided policy for active robotic
    # 8. Paper: Resource allocation and Quality of Service Evaluation for Wireless Communication Systems
    # 9. Bursty traffic: Burstiness Aware Bandwidth Reservation for Ultra-Reliable and Low-Latency Communications
    # 10. Orthodoxy Markovian delay
    # 11. Yibu Diaoyong 
    # 12. Instant NGP paper

    # min AOI(pi)
    # s.t. sampling interval < n
  
