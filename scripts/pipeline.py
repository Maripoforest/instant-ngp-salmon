import threading
import io
import numpy as np
import os
import shutil
import time
import random

# import tqdm

class NeRFPipeline:
    def __init__(self, delay_probability = 0.4) -> None:
        self.frame_n = 1
        self.destination_folder = "./data/nerf/salmon/images"
        self.num_images_to_transfer = 19
        self.source_folder = "./data/nerf/salmon/frames/" + str(self.frame_n).zfill(4)
        self.stop = False
        self.delay_probability = delay_probability
        self.delay = True
        self.sampling_interval = 1  # 0.25 miu
        self.intentional_delay = 3  # 0.15 
        # self.service_time = 0.2
        # np.random.seed(0)
        self.list_p = 0
        self.service_time = np.random.normal(0.5, 0.1, 200)
        self._use_old_data = False


    def transfer_images(self):
        # if os.path.exists(self.destination_folder):
        #     shutil.rmtree(self.destination_folder)
        # os.makedirs(self.destination_folder)
        source_path = self.source_folder
        if os.path.isdir(source_path):

            time.sleep(self.service_time[self.list_p])
            self.list_p += 1
            if self.list_p >= len(self.service_time):
                self.list_p = 0
                
            for i in range(self.num_images_to_transfer):
                source_image_path = os.path.join(source_path, f"{i:04d}.jpg")
                if os.path.exists(source_image_path):
                    destination_image_path = os.path.join(self.destination_folder, f"{i:04d}.jpg")
                    if os.path.exists(destination_image_path):
                        shutil.os.remove(destination_image_path)
                    shutil.copy(source_image_path, destination_image_path)
                    
    def transfer_images_with_delay(self):
        # _t = time.time()
        # if os.path.exists(self.destination_folder):
        #     shutil.rmtree(self.destination_folder)
        # os.makedirs(self.destination_folder)
        source_path = self.source_folder
        to_delay = list()

        if os.path.isdir(source_path):

            # TODO: Markovian delay added here Birth and death process 
            time.sleep(self.service_time[self.list_p])
            self.list_p += 1
            if self.list_p >= len(self.service_time):
                self.list_p = 0

            for i in range(self.num_images_to_transfer):
                if random.random() < self.delay_probability:
                    to_delay.append(i)
                else:
                    source_image_path = os.path.join(source_path, f"{i:04d}.jpg")
                    if os.path.exists(source_image_path):
                        destination_image_path = os.path.join(self.destination_folder, f"{i:04d}.jpg")
                        if os.path.exists(destination_image_path):
                            shutil.os.remove(destination_image_path)
                        shutil.copy(source_image_path, destination_image_path) 
                        # print(destination_image_path)

            # print("Consumed Time: ", time.time() - _t)
            _t = time.time()
            # print(to_delay)

            if len(to_delay) > 0:
                time.sleep(self.intentional_delay)
                for i in to_delay:   
                    source_image_path = os.path.join(source_path, f"{i:04d}.jpg")
                    if os.path.exists(source_image_path):
                        destination_image_path = os.path.join(self.destination_folder, f"{i:04d}.jpg")
                        if os.path.exists(destination_image_path):
                            shutil.os.remove(destination_image_path)
                        # print(destination_image_path)
                        shutil.copy(source_image_path, destination_image_path)

    def get_image_batch_cb(self):
        frame_name = str(self.frame_n).zfill(4)    
        self.source_folder = "./data/nerf/salmon/frames/" + frame_name    
        if self.delay:   
            # print("delayed transmission")
            self.transfer_images_with_delay()
        else:
            # print("NO DELAY")
            self.transfer_images()
        self.frame_n += 1
      
    def step(self):
        if self.delay:  
            print("delayed transmission")
            frame_name = str(self.frame_n).zfill(4)    
            self.source_folder = "./data/nerf/salmon/frames/" + frame_name       
            source_path = self.source_folder
            if os.path.isdir(source_path):
                # The no-data method
                if os.path.exists(self.destination_folder):
                    shutil.rmtree(self.destination_folder)
                os.makedirs(self.destination_folder)

                for i in range(self.num_images_to_transfer):
                    if random.random() < self.delay_probability:
                        continue
                    source_image_path = os.path.join(source_path, f"{i:04d}.jpg")
                    if os.path.exists(source_image_path):
                        destination_image_path = os.path.join(self.destination_folder, f"{i:04d}.jpg")
                        # The old-data method
                        # if os.path.exists(destination_image_path):
                        #     shutil.os.remove(destination_image_path)            
                        shutil.copy(source_image_path, destination_image_path)

                
        else:
            # TODO: Markovian delay added here Birth and death process 
            print("normal transmission")
            self.transfer_images()
        self.frame_n += 1

    def spinning(self):
        while not self.stop:
            time.sleep(self.sampling_interval)
            self.get_image_batch_cb()        

    

if __name__ == '__main__':
    ppl = NeRFPipeline()
    ppl.delay = False
    ppl.spinning()

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
    #      

    # 0.00318