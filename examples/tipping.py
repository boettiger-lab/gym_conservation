import gym
import numpy as np
from stable_baselines3 import A2C, PPO, DDPG, TD3, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

import gym_conservation


#env = gym.make("conservation-v3", beta = 0.1, sigma=0.01, cost = .5, benefit = 0.5, a = .22)
env = gym.make("conservation-v5")
check_env(env)
model = SAC("MlpPolicy", env, verbose=1, policy_kwargs = dict(net_arch=[256, 256]))
model.learn(total_timesteps=300000)

## Simulate a run with the trained model, visualize result
df = env.simulate(model, reps = 1)
env.plot(df, "results/v5-SAC.png")

## Evaluate model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("v5 with SAC", "mean reward:", mean_reward, "std:", std_reward)
