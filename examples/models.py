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

## Optimal solution is steady-state at whatever value a is set to, since larger recovery is proportionally more costly, while benefits saturate
## stochasticity gives incentive to improve conservation if tipping point is 'in range'

env = gym.make("conservation-v5")


model = fixed_action(env, fixed_action = A ) # .3, calibrated to steady-state
df = env.simulate(model, reps = 5)
env.plot(df, "fixed-plot.png")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("steady-state", "mean reward:", mean_reward, "std:", std_reward)

# over-conserve, slowly move away from tipping point
model = fixed_action(env, fixed_action = A + .1 )
df = env.simulate(model, reps = 5)
env.plot(df, "over-conserve.png")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("over-conserve", "mean reward:", mean_reward, "std:", std_reward)

# over-conserve, slowly move away from tipping point
model = fixed_action(env, fixed_action = A + .05 )
df = env.simulate(model, reps = 5)
env.plot(df, "over-conserve-slow.png")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("over-conserve-slow", "mean reward:", mean_reward, "std:", std_reward)



# under-conserve, slows decline only
model = fixed_action(env, fixed_action = A - .1 )
df = env.simulate(model, reps = 5)
env.plot(df, "under-conserve.png")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("under-conserve", "mean reward:", mean_reward, "std:", std_reward)



model = fixed_action(env, fixed_action = 0 )
df = env.simulate(model, reps = 5)
env.plot(df, "noaction-plot.png")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("no-action", "mean reward:", mean_reward, "std:", std_reward)



## Target-state only applies to models where the action changes state (v1, v3)
## For V5 we target an "a" value.  larger than ~ 0.215 is tipping point


model = target_a(env, 0.18 )
df = env.simulate(model, reps = 5)
env.plot(df, "target_a_0.18.png")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("target a=0.18", "mean reward:", mean_reward, "std:", std_reward)


model = target_a(env, 0.13 )
df = env.simulate(model, reps = 5)
env.plot(df, "target_a_0.13.png")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("target a=0.13", "mean reward:", mean_reward, "std:", std_reward)


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

