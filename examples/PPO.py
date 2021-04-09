import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

import gym_conservation


env = gym.make("conservation-v2")
check_env(env)
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=200000)

## Simulate a run with the trained model, visualize result
df = env.simulate(model)
env.plot(df, "results/v2-PPO.png")

## Evaluate model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
print("v2 with PPO", "mean reward:", mean_reward, "std:", std_reward)
