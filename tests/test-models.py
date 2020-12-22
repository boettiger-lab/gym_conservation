import gym
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

import gym_conservation
from gym_conservation.models.policies import fixed_action, target_state, user_action


def test_fixed_action():
    env = gym.make("conservation-v5")
    check_env(env)
    model = fixed_action(env)
    df = env.simulate(model, reps=2)
    env.plot(df, "fixed_action-test.png")

    ## Evaluate model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
