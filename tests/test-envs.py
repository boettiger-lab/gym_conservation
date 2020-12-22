import gym
import numpy as np
from stable_baselines3.common.env_checker import check_env

import gym_conservation
from gym_conservation.models.policies import user_action

np.random.seed(42)

def test_v0():
    env = gym.make("conservation-v0")
    env.reset()
    check_env(env)


def test_v2():
    env = gym.make("conservation-v2")
    check_env(env)


def test_v3():
    env = gym.make("conservation-v3")
    check_env(env)


def test_v5():
    env = gym.make("conservation-v5")
    check_env(env)


def test_basics():
    env = gym.make("conservation-v2", init_state=0.7)
    assert env.unscaled_state == env.init_state
    x = env.get_unscaled_action(env.get_action(0.0))
    assert (x < 0.01) & (x > -0.01)
    x = env.get_unscaled_state(env.get_state(0.1))
    assert (x < 0.11) & (x > 0.1 - 0.01)
    env.reset()
    s = env.unscaled_state
    x = env.perform_action(0.3)
    assert (x < 0.31) & (x > 0.3 - 0.01)
    y = env.population_draw()
    assert y > s


def test_user():
    env = gym.make("conservation-v2", init_state=0.8)
    model = user_action(env)
    # df = env.simulate(model)
    # env.plot(df, "user-may-test.png")
