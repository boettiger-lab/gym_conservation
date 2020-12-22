import gym
import numpy as np
from gym import spaces

from gym_conservation.envs.shared_env import (
    csv_entry,
    estimate_policyfn,
    plot_mdp,
    plot_policyfn,
    simulate_mdp,
)


class BaseEcologyEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        params={
            "r": 0.3,
            "K": 1,
            "sigma": 0.0,
            "x0": 0.1,
            "cost": 2.0,
            "benefit": 1.0,
        },
        Tmax=100,
        file="render.csv",
    ):

        # parameters
        self.K = params["K"]
        self.r = params["r"]
        self.sigma = params["sigma"]
        self.cost = params["cost"]
        self.benefit = params["benefit"]
        self.init_state = params["x0"]
        self.params = params

        # Preserve these for reset
        self.unscaled_state = self.init_state
        self.reward = 0
        self.unscaled_action = 0
        self.years_passed = 0
        self.Tmax = Tmax
        self.file = file

        # for render() method only
        if file is not None:
            self.write_obj = open(file, "w+")

        # Initial state
        self.state = np.array([self.init_state / self.K - 1])

        # Best if cts actions / observations are normalized to a [-1, 1] domain
        self.action_space = spaces.Box(
            np.array([-1], dtype=np.float32),
            np.array([1], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            np.array([-1], dtype=np.float32),
            np.array([1], dtype=np.float32),
            dtype=np.float32,
        )

    def step(self, action):

        # Map from re-normalized model space to [0,2K] real space
        unscaled_action = self.get_unscaled_action(action)
        self.get_unscaled_state(self.state)

        # Apply unscaled_action and population growth
        self.unscaled_action = self.perform_action(unscaled_action)
        self.population_draw()

        # Map population back to system state (normalized space):
        self.get_state(self.unscaled_state)

        # should be the instanteous reward, not discounted
        self.reward = self.compute_reward()
        self.years_passed += 1
        done = bool(self.years_passed > self.Tmax)

        if self.unscaled_state <= 0.0:
            done = True

        return self.state, self.reward, done, {}

    def reset(self):
        self.state = np.array([self.init_state / self.K - 1])
        self.unscaled_state = self.init_state
        self.years_passed = 0

        # for tracking only
        self.reward = 0
        self.unscaled_action = 0
        return self.state

    def compute_reward(self):
        return (
            self.benefit * self.unscaled_state
            - self.unscaled_action * self.cost
        )

    def render(self, mode="human"):
        return csv_entry(self)

    def close(self):
        if self.file is not None:
            self.write_obj.close()

    def simulate(env, model, reps=1):
        return simulate_mdp(env, model, reps)

    def plot(self, df, output="results.png"):
        return plot_mdp(self, df, output)

    def policyfn(env, model, reps=1):
        return estimate_policyfn(env, model, reps)

    def plot_policy(self, df, output="results.png"):
        return plot_policyfn(self, df, output)

    def perform_action(self, unscaled_action):
        """
        increase population by 'unscaled_action'
        """
        self.unscaled_action = unscaled_action
        self.unscaled_state = self.unscaled_state + self.unscaled_action
        return self.unscaled_action

    def population_draw(self):
        self.unscaled_state = (
            self.unscaled_state
            + self.r
            * self.unscaled_state
            * (1.0 - self.unscaled_state / self.K)
            + self.unscaled_state * self.sigma * np.random.normal(0, 1)
        )
        self.unscaled_state = np.clip(self.unscaled_state, 0, 2 * self.K)
        return self.unscaled_state

    def get_unscaled_action(self, action):
        """
        Convert action into unscaled_action
        """
        if isinstance(self.action_space, gym.spaces.discrete.Discrete):
            unscaled_action = (action / self.n_actions) * self.K
        # Continuous Actions
        else:
            action = np.clip(
                action, self.action_space.low, self.action_space.high
            )[0]
            unscaled_action = (action + 1) * self.K
        return unscaled_action

    def get_action(self, unscaled_action):
        """
        Convert unscaled_action into action
        """
        if isinstance(self.action_space, gym.spaces.discrete.Discrete):
            return round(unscaled_action * self.n_actions / self.K)
        else:
            return unscaled_action / self.K - 1

    def get_unscaled_state(self, state):
        self.unscaled_state = (state[0] + 1) * self.K
        return self.unscaled_state

    def get_state(self, unscaled_state):
        self.state = np.array([unscaled_state / self.K - 1])
        return self.state
