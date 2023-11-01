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
    def __init__(self, lam: float, miu: float) -> None:
        self.miu = miu
        self.lam = lam
        self.t = 1

        self.gt_highrate = Exponential(1000) # Generation time
        self.delay_highrate = Exponential(200) # Time delay

        self.gt_lowrate = Exponential(50)
        self.delay_lowrate = Exponential(50)

        self.switch_to_highrate()

    def p_t(self, t: float) -> np.ndarray:
        pr_t = np.zeros((2, 2))
        pr_t[0][0] = (self.miu + self.lam * np.power(np.e, -t*(miu + lam))) / (miu + lam)
        pr_t[0][1] = (self.lam - self.lam * np.power(np.e, -t*(miu + lam))) / (miu + lam)
        pr_t[1][0] = (self.miu - self.miu * np.power(np.e, -t*(miu + lam))) / (miu + lam)
        pr_t[1][1] = (self.lam + self.miu * np.power(np.e, -t*(miu + lam))) / (miu + lam)
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
                return True
            else:
                self.t += 1
                return False
        
     
    def step(self):
        generate_time = self.gt.get_number()
        delay = self.delay.get_number()
        self.switch()
        # print(generate_time)
        # print(delay)
        return generate_time, delay

    def generate(self):
        return self.gt.get_number()
    
lam = 0.01 # High latency to low latency
miu = 0.02 # Low latency to high latency

camera_list = list()
for i in range(19):
    ge = GE_Channel(lam=lam, miu=miu)
    camera_list.append(ge)


# MM1
receive_series = np.zeros((19, 1))
send_series = np.zeros((19, 1))
last_dt = 0
for i in range(30):
    send_frame = list()
    receive_frame = list()
    for ge in camera_list:
        g, d = ge.step()
        pdt = int(last_dt/g)
        while pdt >= 1:
            next_g = ge.generate()
            g += next_g
            pdt = int(last_dt/g)
        send_frame.append(g)
        receive_frame.append(g+d)
    send_frame = np.array(send_frame)
    receive_frame = np.array(receive_frame)
    send_frame = send_frame + send_series[:, -1].reshape(-1, 1).T
    receive_frame = receive_frame + send_series[:, -1].reshape(-1, 1).T
    send_series = np.hstack((send_series, send_frame.T))
    receive_series = np.hstack((receive_series, receive_frame.T))

print(send_series[:, -1])
print(receive_series[:, -1])



# print(len(receive_series[:, -1]))
# frames = np.zeros((19, int(max(receive_series[:, -1]))+1))

# camera_frameshot = send_series.T.tolist()
# int_list_2d = [[int(x) for x in inner_list] for inner_list in camera_frameshot]
# with open('./camera_frameshot.json', 'w') as f:
#     json.dump(int_list_2d, f)

# x = list()
# ge = GE_Channel(lam=lam, miu=miu)
# for i in range(1000):
#     g, _ = ge.step()
#     x.append(g)
# import matplotlib.pyplot as plt
# plt.plot(x)
# plt.show()
