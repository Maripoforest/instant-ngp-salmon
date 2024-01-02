import numpy as np
import random
import json

class Exponential:
    def __init__(self, lam: float) -> None:
        self.lam = lam
        # np.random.seed(5)

    def get_number(self) -> float:
        return int(np.random.exponential(scale=1/self.lam) * 100) + 1

class GE_Channel():
    def __init__(self, lam: float, miu: float, l1: int = 30, l2: int = 7, l3: int = 30, l4: int = 60) -> None:
        self.miu = miu
        self.lam = lam
        self.t = 1
        self.eps = 0.05

        self.gt_highrate = Exponential(l1) # Generation time
        self.gt_lowrate = Exponential(l2) # 300/30

        self.delay_highrate = Exponential(l3) # Time delay
        self.delay_lowrate = Exponential(l4) # 10/50   

        self.switch_to_highrate()

    def p_t(self, t: float) -> np.ndarray:
        pr_t = np.zeros((2, 2))
        pr_t[0][0] = (self.miu + self.lam * np.power(np.e, -t*self.eps*(self.miu + self.lam))) / (self.miu + self.lam)
        pr_t[0][1] = (self.lam - self.lam * np.power(np.e, -t*self.eps*(self.miu + self.lam))) / (self.miu + self.lam)
        pr_t[1][0] = (self.miu - self.miu * np.power(np.e, -t*self.eps*(self.miu + self.lam))) / (self.miu + self.lam)
        pr_t[1][1] = (self.lam + self.miu * np.power(np.e, -t*self.eps*(self.miu + self.lam))) / (self.miu + self.lam)
        # print(pr_t)
        return pr_t
    
    def switch_to_highrate(self) -> None:
        self.highrate = True
        self.gt = self.gt_highrate
        self.delay = self.delay_highrate
        # print("switch_to_highrate")
    
    def switch_to_lowrate(self) -> None:
        self.highrate = False
        self.gt = self.gt_lowrate
        self.delay = self.delay_lowrate
        # print("switch_to_lowrate")

    def switch(self) -> bool:
        pr_t = self.p_t(self.t)
        if not self.highrate:
            if random.random() > pr_t[0][0]:
                self.switch_to_highrate()
                self.t = 1
                return True
            else:
                self.t += 1
                return False

        if self.highrate:
            if random.random() > pr_t[1][1]:
                self.switch_to_lowrate()
                self.t = 1
                return False
            else:
                self.t += 1
                return True
     
    def step(self):
        generate_time = self.gt.get_number()
        delay = self.delay.get_number()
        self.highrate = self.switch()
        # print(generate_time)
        # print(delay)
        return generate_time, delay, self.highrate

    def sample(self):
        return self.gt.get_number()

if __name__ == "__main__":
    lam = 0.5
    miu = 0.5

    camera_list = list()
    for i in range(21):
        ge = GE_Channel(lam=lam, miu=miu)
        camera_list.append(ge)


    # MM1
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

    # print(send_series[:, -1])
    receive_series = receive_series.T
    send_series = send_series.T
    print(receive_series[1,5])

    # print(len(receive_series[:, -1]))
    l = int(np.max(receive_series)) + 1
    print(l)
    frames = np.zeros((l, 21))
    last_frame = np.zeros(21)
    for i in range(len(receive_series)):
        for j in range(21):
            frames[int(receive_series[i, j]), j] = send_series[i, j]

    camera_frameshot = frames.tolist()
    # int_list_2d = [[int(x) for x in inner_list] for inner_list in camera_frameshot]
    channel_sim = dict()
    channel_sim["frames"] = camera_frameshot
    channel_sim["states"] = states
    with open('./camera_frameshot.json', 'w') as f:
        json.dump(channel_sim, f)
        print("saved")

    # x = list()
    # ge = GE_Channel(lam=lam, miu=miu)
    # for i in range(1000):
    #     g, _ = ge.step()
    #     x.append(g)
    # import matplotlib.pyplot as plt
    # plt.plot(x)
    # plt.show()

    # ==================================== stepper
    import os
    import shutil

    frameshots = channel_sim
    n = 25
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
            n_to_transfer = int(frameshots["frames"][n+86][i])
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
                        if os.path.exists(destination_image_path):
                            shutil.os.remove(destination_image_path)
                        shutil.copy(source_image_path, destination_image_path)
        n -= 1