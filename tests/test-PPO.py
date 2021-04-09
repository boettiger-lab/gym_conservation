import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy


def test_ppo():
    env = gym.make("conservation-v2")
    check_env(env)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=200)
    df = env.simulate(model)
    env.plot(df, "PPO-test.png")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
    env.plot_policy(df, "PPO-policy.png")
