import gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import gym_conservation
from gym_conservation.models.policies import fixed_action, target_state


def test_fixed_action():
    env = gym.make("conservation-v5")
    check_env(env)
    model = fixed_action(env)
    df = env.simulate(model, reps=2)
    env.plot(df, "fixed_action-test.png")
    evaluate_policy(model, Monitor(env), n_eval_episodes=50)


def test_target_state():
    env = gym.make("conservation-v3")
    check_env(env)
    model = target_state(env, 0.8)
    df = env.simulate(model, reps=2)
    env.plot(df, "fixed_state-test.png")
    evaluate_policy(model, Monitor(env), n_eval_episodes=50)
