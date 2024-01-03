import gym
import numpy as np
from gym import spaces
import json
import time
import os
import shutil

from channel import GE_Channel
from rl_step import main_ngp
import sys
from contextlib import redirect_stdout
from lpips import LPIPS

class CountEnv(gym.Env):
    def __init__(self, dataset = "steak"):
        self.observation_space = spaces.Box(low=0, high=300, dtype=int)
        self.action_space = spaces.Discrete(25)
        self.cameras_to_skip = [8, 12]
        self.dataset = dataset
        self.state = None
        self.set_dataset(dataset)
        self.num_images_to_transfer = 21

    def set_dataset(self, dataset):
        self.dataset = dataset
        self.destination_folder = "./data/nerf/" + self.dataset + "/images/"
        self.parent_folder = "./data/nerf/" + self.dataset + "/frames/"

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

    def build_channel(self, 
                      lam: float = 0.5, 
                      miu: float = 0.5, 
                      l1: int = 30,
                      l2: int = 10,
                      l3: int = 30,
                      l4: int = 15):
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
        for i in range(100):
            send_frame = list()
            receive_frame = list()
            for ge in camera_list:
                g, d, s = ge.step()
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

    def stepper(self) -> int:
        frameshots = self.camera_frameshot
        n = 25
        cameras_to_skip = self.cameras_to_skip
        destination_folder = "/home/socialai/instant-ngp-salmon/data/nerf/steak/frames/"
        parent_folder = "/home/socialai/instant-ngp-salmon/data/nerf/steak/allframes/frames/"
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
                n_to_transfer = int(frameshots["frames"][n+200][i])
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
                        if os.path.exists(source_image_path):
                            
                            destination_image_path = os.path.join(destination_image_folder, f"{i:04d}.jpg")
                            if os.path.exists(destination_image_path):
                                shutil.os.remove(destination_image_path)
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

    def reset(self):
        self.build_channel()
        self.state = np.array(self.stepper())
        return self.state

    def step(self, action):
        print("state: ", self.state)
        print("action: ", action)
        self.wait(action)
        if self.count_images() == 0:
            reward = -1
        else:
            reward = 0  # Initialize reward
            with open('nul' if sys.platform == 'win32' else '/dev/null', 'w') as null_file:
                with redirect_stdout(null_file):
                    reward = main_ngp(dataset=self.dataset)
            reward = (reward - 0.5) * 10 - 0.01 * action
        print("reward: ", reward)
        done = True
        info = {}

        return self.state, reward, done, info

class MatrixEnv(gym.Env):
    def __init__(self, dataset = "steak"):
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(26, 21), dtype=float  # U matrix with an extra row
        )
        self.action_space = spaces.Discrete(25)
        self.cameras_to_skip = [8, 12]
        self.dataset = dataset
        self.state = None
        self.set_dataset(dataset)
        self.num_images_to_transfer = 21
        self.task = 86
        self.step_counter = 0
        self.lpips_model = LPIPS(net='alex')

    def reset_task(self, task):
        self.task = task

    def set_dataset(self, dataset):
        self.dataset = dataset
        self.destination_folder = f"/home/socialai/instant-ngp-salmon/data/nerf/{self.dataset}/images/"
        self.parent_folder = f"/home/socialai/instant-ngp-salmon/data/nerf/{self.dataset}/frames/"
        self.stepper_folder = f"/home/socialai/instant-ngp-salmon/data/nerf/{self.dataset}/allframes/frames/"

    def count_images(self):
        folder_path = f"/home/socialai/instant-ngp-salmon/data/nerf/{self.dataset}/images"
        if not os.path.exists(folder_path):
            print(f"Error: Folder '{folder_path}' not found.")
            return 0
        files = os.listdir(folder_path)
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = [file for file in files if any(file.lower().endswith(ext) for ext in image_extensions)]
        num_images = len(image_files)
        
        return num_images

    def build_channel(self, 
                      lam: float = 0.5, 
                      miu: float = 0.5, 
                      l1: int = 30,
                      l2: int = 10,
                      l3: int = 30,
                      l4: int = 15):
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
        for i in range(100):
            send_frame = list()
            receive_frame = list()
            for ge in camera_list:
                g, d, s = ge.step()
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
        for i in range(len(receive_series)):
            for j in range(21):
                frames[int(receive_series[i, j]), j] = send_series[i, j]
        camera_frameshot = frames.tolist()
        channel_sim = dict()
        channel_sim["frames"] = camera_frameshot
        channel_sim["states"] = states
        channel_sim["para"] = [l1, l2, l3, l4]
        self.camera_frameshot = channel_sim

    def stepper(self) -> int:
        frameshots = self.camera_frameshot
        n = 25
        cameras_to_skip = self.cameras_to_skip
        destination_folder = self.parent_folder
        parent_folder = self.stepper_folder
        if os.path.exists(destination_folder):
            shutil.rmtree(destination_folder)
            os.makedirs(destination_folder)
        frame_counter = 0
        state_matrix = np.zeros((n, 21))
        while n >= 1:
            last_name = str(n+1).zfill(4)
            last_folder = destination_folder + last_name     
            destination_image_folder = os.path.join(destination_folder, f"{n:04d}")
            if os.path.exists(last_folder) and not os.path.exists(destination_image_folder):
                shutil.copytree(last_folder, destination_image_folder)  
            elif not os.path.exists(destination_image_folder):
                os.makedirs(destination_image_folder) 
            for i in range(21):
                n_to_transfer = int(frameshots["frames"][n + self.task][i])
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
                        if os.path.exists(source_image_path):
                            destination_image_path = os.path.join(destination_image_folder, f"{i:04d}.jpg")
                            if os.path.exists(destination_image_path):
                                shutil.os.remove(destination_image_path)
                            shutil.copy(source_image_path, destination_image_path)
                            state_matrix[n-1, i] = 1
            n -= 1
        return state_matrix

    def wait_for(self, action):
        source_path = os.path.join(self.parent_folder, str(action).zfill(4))
        if os.path.exists(self.destination_folder):
            shutil.rmtree(self.destination_folder)
        os.makedirs(self.destination_folder)
        frame_name = str(action).zfill(4)    
        source_path = os.path.join(self.parent_folder + frame_name) 
        if not os.path.exists(source_path):
            print("source_folder does not exist")
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

    def reset(self):
        self.build_channel(l1=10, l2=10, l3=20, l4=20)
        _U = self.stepper()
        obs = np.zeros((26, 21), dtype=float)
        obs[0, 0] = self.camera_frameshot["para"][0] / 1000.0
        obs[0, 1] = self.camera_frameshot["para"][1] / 1000.0
        obs[0, 2] = self.camera_frameshot["para"][2] / 1000.0
        obs[0, 3] = self.camera_frameshot["para"][3] / 1000.0
        obs[1:, :] = _U
        self.state = obs

        return self.state

    def step(self, action):
        self.wait_for(action)
        if self.count_images() == 0:
            reward = -1
        else:
            reward = 0  # Initialize reward
            with open('nul' if sys.platform == 'win32' else '/dev/null', 'w') as null_file:
                with redirect_stdout(null_file):
                    reward = main_ngp(dataset=self.dataset, lpips_model=self.lpips_model)
            reward = (reward - 0.5) * 10
        print("action: ", action)
        print("reward: ", reward)
        info = {}
        # self.task += 1
        # self.task %= 500

        return self.state, reward, True, info