


import gym
import numpy as np
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import gym_conservation


env = Monitor(gym.make("conservation-v7", reps=100))
check_env(env)
model = A2C("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=20000)


done = False
obs = env.reset()
while not done:
    action, _state = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()

## Evaluate model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
print("v5 with PPO", "mean reward:", mean_reward, "std:", std_reward)
