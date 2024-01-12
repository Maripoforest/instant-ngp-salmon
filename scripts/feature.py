import os
import shutil
import numpy as np
import json
import time
from tqdm import tqdm
from channel import GE_Channel
import random

from rl_step import main_ngp, parse_args
from pipeline import NeRFPipeline
from lpips import LPIPS

class FeatureEvaluator:
    def __init__(self, dataset = "steak"):
        self.cameras_to_skip = [8, 12]
        self.dataset = dataset
        self.set_dataset(dataset)
        self.num_images_to_transfer = 21
        self.lpips = LPIPS(net='alex')
        self.length = 25

    def set_dataset(self, dataset):
        self.dataset = dataset
        self.destination_folder = "./data/nerf/" + self.dataset + "/images/"
        self.parent_folder = "./data/nerf/" + self.dataset + "/frames/"

    def build_channel(self, 
                    lam: float = 0.5, 
                    miu: float = 0.5, 
                    l1: int = 10,
                    l2: int = 10,
                    l3: int = 20,
                    l4: int = 20,
                    g: int = 10):
        lam = 0.5
        miu = 0.5
        camera_list = list()
        for i in range(21):
            ge = GE_Channel(lam, miu, l1, l2, l3, l4)
            camera_list.append(ge)
        # M/M/1/1
        receive_series = np.zeros((21, 1))
        send_series = np.zeros((21, 1))
        states = dict()
        last_dt = 0
        start = True
        for i in range(100):
            send_frame = list()
            receive_frame = list()
            for ge in camera_list:
                g, d, s = ge.step()
                if start:
                    g += random.randint(10, 50)
                    start = False
                states[str(i)] = s
                pdt = int(last_dt/g)
                while pdt > 1:
                    next_g = ge.sample()
                    g += next_g
                    pdt = int(last_dt/g)
                send_frame.append(g)
                receive_frame.append(g+d)
                last_dt = d
            send_frame = np.array(send_frame)
            receive_frame = np.array(receive_frame)
            send_frame = send_frame + send_series[:, -1].reshape(-1, 1).T
            receive_frame = receive_frame + send_series[:, -1].reshape(-1, 1).T
            send_series = np.hstack((send_series, send_frame.T))
            receive_series = np.hstack((receive_series, receive_frame.T))
        receive_series = receive_series.T
        send_series = send_series.T
        l = int(np.max(receive_series)) + 1
        frames = np.zeros((l, 21))
        # last_frame = np.zeros(21)
        for i in range(len(receive_series)):
            for j in range(21):
                frames[int(receive_series[i, j]), j] = send_series[i, j]
        camera_frameshot = frames.tolist()
        # int_list_2d = [[int(x) for x in inner_list] for inner_list in camera_frameshot]
        channel_sim = dict()
        channel_sim["frames"] = camera_frameshot
        channel_sim["states"] = states
        self.camera_frameshot = channel_sim
        with open('./camera_frameshot.json', 'w') as f:
            json.dump(channel_sim, f)
            print("saved")

    def stepper(self) -> int:
        frameshots = self.camera_frameshot
        n = self.length
        cameras_to_skip = self.cameras_to_skip
        destination_folder = f"/home/socialai/instant-ngp-salmon/data/nerf/{self.dataset}/frames/"
        parent_folder = f"/home/socialai/instant-ngp-salmon/data/nerf/{self.dataset}/allframes/frames/"
        if os.path.exists(destination_folder):
            shutil.rmtree(destination_folder)
            os.makedirs(destination_folder)
        frame_counter = 0
        while n >= 1:
            last_name = str(n+1).zfill(4)
            last_folder = destination_folder + last_name     
            destination_image_folder = os.path.join(destination_folder, f"{n:04d}")
            if os.path.exists(last_folder) and not os.path.exists(destination_image_folder):
                shutil.copytree(last_folder, destination_image_folder)  
            elif not os.path.exists(destination_image_folder):
                os.makedirs(destination_image_folder) 
            for i in range(21):
                n_to_transfer = int(frameshots["frames"][n+86][i])
                frame_name = str(n_to_transfer).zfill(4)
                source_folder = parent_folder + frame_name
                source_path = source_folder
                if i in cameras_to_skip:
                    continue
                elif n_to_transfer == 0:
                    continue
                else:
                    frame_counter += 1
                    if os.path.isdir(source_path):
                        
                        source_image_path = os.path.join(source_path, f"{i:04d}.jpg")
                        # print("moving", source_image_path)
                        # print(source_image_path)
                        if os.path.exists(source_image_path):
                            destination_image_path = os.path.join(destination_image_folder, f"{i:04d}.jpg")
                            if os.path.exists(destination_image_path):
                                shutil.os.remove(destination_image_path)
                                # print("removed", destination_image_path)
                                # print("replaced by", source_image_path)
                            shutil.copy(source_image_path, destination_image_path)
            n -= 1
        return frame_counter

    def wait(self, action):
        source_path = os.path.join(self.parent_folder, str(action).zfill(4))
        if os.path.exists(self.destination_folder):
            shutil.rmtree(self.destination_folder)
        os.makedirs(self.destination_folder)
        frame_name = str(action).zfill(4)    
        source_path = os.path.join(self.parent_folder + frame_name) 
        if not os.path.exists(source_path):
            print("source_folder does not exist")
            print(source_path)
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

    def count_images(self):
        folder_path = f"data/nerf/{self.dataset}/images"
        if not os.path.exists(folder_path):
            print(f"Error: Folder '{folder_path}' not found.")
            return 0
        files = os.listdir(folder_path)
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = [file for file in files if any(file.lower().endswith(ext) for ext in image_extensions)]
        num_images = len(image_files)
        return num_images
    
    def train(self, episode_length: int = 20):
        output = list()
        for i in range(int(episode_length)):
            self.build_channel(l3=20, l4=20, l1=10, l2=10, g=15)
            self.stepper()
            episode_reward = list()
            for action in range(self.length):
                self.wait(action+1)
                if self.count_images() == 0:
                    reward = [0, 0, 1]
                else:
                    # reward = 1
                    reward = main_ngp(dataset=self.dataset, lpips_model=self.lpips, rl=False, render = None)
                episode_reward.append(reward)
            t = time.time()
            with open(f"./results/reward-{t}.json", "w") as f:
                json.dump(episode_reward, f)
    
class FeatureEvaluator:
    def __init__(self, dataset = "steak"):
        self.cameras_to_skip = [8, 12]
        self.dataset = dataset
        self.set_dataset(dataset)
        self.num_images_to_transfer = 21
        self.lpips = LPIPS(net='alex')
        self.length = 25

    def set_dataset(self, dataset):
        self.dataset = dataset
        self.destination_folder = "./data/nerf/" + self.dataset + "/images/"
        self.parent_folder = "./data/nerf/" + self.dataset + "/frames/"
        self.stepper_folder = "./data/nerf/" + self.dataset + "/allframes/frames/"

    def build_channel(self, 
                    lam: float = 0.5, 
                    miu: float = 0.5, 
                    l1: int = 10,
                    l2: int = 10,
                    l3: int = 20,
                    l4: int = 20,
                    g: int = 10):
        lam = 0.5
        miu = 0.5
        camera_list = list()
        for i in range(21):
            ge = GE_Channel(lam, miu, l1, l2, l3, l4)
            camera_list.append(ge)
        # M/M/1/1
        receive_series = np.zeros((21, 1))
        send_series = np.zeros((21, 1))
        states = dict()
        last_dt = 0
        start = True
        for i in range(100):
            send_frame = list()
            receive_frame = list()
            for ge in camera_list:
                g, d, s = ge.step()
                if start:
                    g += random.randint(10, 50)
                    start = False
                states[str(i)] = s
                pdt = int(last_dt/g)
                while pdt > 1:
                    next_g = ge.sample()
                    g += next_g
                    pdt = int(last_dt/g)
                send_frame.append(g)
                receive_frame.append(g+d)
                last_dt = d
            send_frame = np.array(send_frame)
            receive_frame = np.array(receive_frame)
            send_frame = send_frame + send_series[:, -1].reshape(-1, 1).T
            receive_frame = receive_frame + send_series[:, -1].reshape(-1, 1).T
            send_series = np.hstack((send_series, send_frame.T))
            receive_series = np.hstack((receive_series, receive_frame.T))
        receive_series = receive_series.T
        send_series = send_series.T
        l = int(np.max(receive_series)) + 1
        frames = np.zeros((l, 21))
        # last_frame = np.zeros(21)
        for i in range(len(receive_series)):
            for j in range(21):
                frames[int(receive_series[i, j]), j] = send_series[i, j]
        camera_frameshot = frames.tolist()
        # int_list_2d = [[int(x) for x in inner_list] for inner_list in camera_frameshot]
        channel_sim = dict()
        channel_sim["frames"] = camera_frameshot
        channel_sim["states"] = states
        self.camera_frameshot = channel_sim
        # with open('./camera_frameshot.json', 'w') as f:
        #     json.dump(channel_sim, f)
        #     print("saved")

    def stepper(self) -> int:
        frameshots = self.camera_frameshot
        n = self.length
        cameras_to_skip = self.cameras_to_skip
        destination_folder = f"/home/socialai/instant-ngp-salmon/data/nerf/{self.dataset}/frames/"
        parent_folder = f"/home/socialai/instant-ngp-salmon/data/nerf/{self.dataset}/allframes/frames/"
        if os.path.exists(destination_folder):
            shutil.rmtree(destination_folder)
            os.makedirs(destination_folder)
        frame_counter = 0
        while n >= 1:
            last_name = str(n+1).zfill(4)
            last_folder = destination_folder + last_name     
            destination_image_folder = os.path.join(destination_folder, f"{n:04d}")
            if os.path.exists(last_folder) and not os.path.exists(destination_image_folder):
                shutil.copytree(last_folder, destination_image_folder)  
            elif not os.path.exists(destination_image_folder):
                os.makedirs(destination_image_folder) 
            for i in range(21):
                n_to_transfer = int(frameshots["frames"][n+86][i])
                frame_name = str(n_to_transfer).zfill(4)
                source_folder = parent_folder + frame_name
                source_path = source_folder
                if i in cameras_to_skip:
                    continue
                elif n_to_transfer == 0:
                    continue
                else:
                    frame_counter += 1
                    if os.path.isdir(source_path):
                        
                        source_image_path = os.path.join(source_path, f"{i:04d}.jpg")
                        # print("moving", source_image_path)
                        # print(source_image_path)
                        if os.path.exists(source_image_path):
                            destination_image_path = os.path.join(destination_image_folder, f"{i:04d}.jpg")
                            if os.path.exists(destination_image_path):
                                shutil.os.remove(destination_image_path)
                                # print("removed", destination_image_path)
                                # print("replaced by", source_image_path)
                            shutil.copy(source_image_path, destination_image_path)
            n -= 1
        return frame_counter

    def wait(self, action):
        source_path = os.path.join(self.parent_folder, str(action).zfill(4))
        if os.path.exists(self.destination_folder):
            shutil.rmtree(self.destination_folder)
        os.makedirs(self.destination_folder)
        frame_name = str(action).zfill(4)    
        source_path = os.path.join(self.parent_folder + frame_name) 
        if not os.path.exists(source_path):
            print("source_folder does not exist")
            print(source_path)
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

    def aoistepper(self) -> int:
        k = 0   
        no_image = False
        self.task = 111
        state = np.zeros(19)
        for i in range(21):
            if i in self.cameras_to_skip:
                continue
            else:
                j = 0
                while int(self.camera_frameshot["frames"][self.task-j][i]) == 0:
                    j += 1
                    if self.task - j < 0:
                        no_image = True
                        break
                if not no_image:
                    state[k] = self.task - self.camera_frameshot["frames"][self.task-j][i]
                    k += 1 
                else:
                    state[k] = 0
                    k += 1
                    no_image = True   
        print(state)  
        self.state = state
        return state
    
    def aoiwait(self, action):
        if os.path.exists(self.destination_folder):
            shutil.rmtree(self.destination_folder)
        os.makedirs(self.destination_folder)
        for i in range(19):
            if self.state[i] >= action:
                continue
            else:
                frame = int(self.task - self.state[i])
                source_path = os.path.join(self.stepper_folder, str(frame).zfill(4))      
                if i < 8:
                    k = i
                elif i >= 8 and i < 12:
                    k = i + 1
                else:
                    k = i + 2
                source_image_path = os.path.join(source_path, f"{k:04d}.jpg")
                destination_image_path = os.path.join(self.destination_folder, f"{k:04d}.jpg")
                if os.path.exists(source_image_path):
                    shutil.copy(source_image_path, destination_image_path)



    def count_images(self):
        folder_path = f"data/nerf/{self.dataset}/images"
        if not os.path.exists(folder_path):
            print(f"Error: Folder '{folder_path}' not found.")
            return 0
        files = os.listdir(folder_path)
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = [file for file in files if any(file.lower().endswith(ext) for ext in image_extensions)]
        num_images = len(image_files)
        return num_images
    
    def train(self, episode_length: int = 20):
        output = list()
        for i in range(int(episode_length)):
            self.build_channel(l3=20, l4=20, l1=10, l2=10, g=15)
            self.aoistepper()
            episode_reward = list()
            for i in range(self.length):
                action = i * 2 + 1
                self.aoiwait(action)
                if self.count_images() == 0:
                    reward = [0, 0, 1]
                else:
                    # reward = 1
                    reward = main_ngp(dataset=self.dataset, lpips_model=self.lpips, rl=False, render = None)
                episode_reward.append(reward)
            t = time.time()
            with open(f"./results/reward-{t}.json", "w") as f:
                json.dump(episode_reward, f)
    

if __name__ == "__main__":
    fe = FeatureEvaluator(dataset="steak")
    fe.train(60)