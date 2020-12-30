import gym
import numpy as np
from gym_conservation.models.policies import fixed_action, target_state, user_action, target_a

from stable_baselines3 import A2C, PPO, DDPG, TD3, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

import gym_conservation

K = 1.5
alpha = 0.001
A = alpha * 100 * 2 * K 
env = gym.make("conservation-v5", sigma=0.0, cost = 20, a = 0.18)


model = fixed_action(env, fixed_action = A )
df = env.simulate(model, reps = 5)
env.plot(df, "fixed-plot.png")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("steady-state", "mean reward:", mean_reward, "std:", std_reward)

model = fixed_action(env, fixed_action = .9 )
df = env.simulate(model, reps = 5)
env.plot(df, "high-plot.png")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("high", "mean reward:", mean_reward, "std:", std_reward)



model = fixed_action(env, fixed_action = A - 0.1 )
df = env.simulate(model, reps = 5)
env.plot(df, "low-plot.png")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("low", "mean reward:", mean_reward, "std:", std_reward)



model = fixed_action(env, fixed_action = 0 )
df = env.simulate(model, reps = 5)
env.plot(df, "noaction-plot.png")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("no-action", "mean reward:", mean_reward, "std:", std_reward)



## Target-state only applies to models where the action changes state (v1, v3)
model = target_a(env, 0.21)
df = env.simulate(model, reps = 5)
env.plot(df, "target_a0.21.png")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("target a = 0.21", "mean reward:", mean_reward, "std:", std_reward)


model = target_a(env, 0.15 )
df = env.simulate(model, reps = 5)
env.plot(df, "target_a_0.15.png")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("target a=0.15", "mean reward:", mean_reward, "std:", std_reward)


model = target_a(env, 0.05 )
df = env.simulate(model, reps = 5)
env.plot(df, "target_a_0.05.png")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("target a=0.05", "mean reward:", mean_reward, "std:", std_reward)


model = target_a(env, .22)
df = env.simulate(model, reps = 5)
env.plot(df, "target_a_0.22.png")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("target a=0.22 ", "mean reward:", mean_reward, "std:", std_reward)

