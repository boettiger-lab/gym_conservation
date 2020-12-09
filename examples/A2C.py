import gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

import gym_conservation


env = gym.make("conservation-v2", init_state = 0.1)
check_env(env)
model = A2C("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=300000)

## Simulate a run with the trained model, visualize result
df = env.simulate(model)
env.plot(df, "A2C-low.png")

## Evaluate model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)



env = gym.make("conservation-v2", init_state = 0.8)
check_env(env)
model = A2C("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=300000)

## Simulate a run with the trained model, visualize result
df = env.simulate(model)
env.plot(df, "A2C-high.png")

## Evaluate model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
