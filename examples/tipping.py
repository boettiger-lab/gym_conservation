import gym
import numpy as np
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

import gym_conservation


env = gym.make("conservation-v5", sigma=0.01)
check_env(env)
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=80000)

## Simulate a run with the trained model, visualize result
df = env.simulate(model, reps = 10)
env.plot(df, "PPO-v5-test.png")

## Evaluate model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("algo:", "PPO", "env:", "conservation-v2", "mean reward:", mean_reward, "std:", std_reward)
