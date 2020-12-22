import gym
import numpy as np
from gym_conservation.models.policies import fixed_action, target_state, user_action

from stable_baselines3 import A2C, PPO, DDPG, TD3, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

import gym_conservation

K = 1.5
alpha = 0.001
A = alpha * 100 * 2 * K 
env = gym.make("conservation-v5")


model = fixed_action(env, fixed_action = A )
df = env.simulate(model, reps = 1)
env.plot(df, "fixed-plot.png")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("steady-state", "mean reward:", mean_reward, "std:", std_reward)

model = fixed_action(env, fixed_action = .9 )
df = env.simulate(model, reps = 1)
env.plot(df, "high-plot.png")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("high", "mean reward:", mean_reward, "std:", std_reward)



model = fixed_action(env, fixed_action = A - 0.1 )
df = env.simulate(model, reps = 1)
env.plot(df, "low-plot.png")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("low", "mean reward:", mean_reward, "std:", std_reward)



model = fixed_action(env, fixed_action = 0 )
df = env.simulate(model, reps = 1)
env.plot(df, "noaction-plot.png")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("no-action", "mean reward:", mean_reward, "std:", std_reward)
