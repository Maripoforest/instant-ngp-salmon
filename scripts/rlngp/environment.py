# import json
# import time


# while True:
#     try:
#         with open("temp/tmp.json", 'r') as f:
#             d = json.load(f)
#             break
#     except:
#         time.sleep(2)

# reward = d["lpips"] * 10 + d["time"] - action

import gym
from gym import spaces

class YourEnvironment(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=0, high=300, shape=(1,), dtype=int)
        self.state = self.reset()

    def reset(self):
        # Initialize the state (observation) when the environment is reset
        return self.observation_space.sample()

    def step(self, action):
        # Simulate one step in the environment given the action
        # Update the state and return the next observation, reward, done flag, and info
        next_state = self.observation_space.sample()
        reward = 0
        done = False
        info = {}
        return next_state, reward, done, info

# Instantiate the environment
env = YourEnvironment()

# Print the initial state
print("Initial State:", env.state)
