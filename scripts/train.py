from env import CountEnv, MatrixEnv
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# Wrap the environment in a DummyVecEnv to work with stable-baselines3
env = DummyVecEnv([lambda: CountEnv()])
model = PPO("MlpPolicy", env, verbose=1, n_steps=3, learning_rate=3e-4, tensorboard_log="./tensorboard/")
model.learn(total_timesteps=1000)
model.save("ppo_model")
