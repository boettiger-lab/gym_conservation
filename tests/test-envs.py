import gym
import numpy as np
from stable_baselines3.common.env_checker import check_env

import gym_conservation

np.random.seed(0)


def test_v0():
    env = gym.make("conservation-v0")
    check_env(env)
