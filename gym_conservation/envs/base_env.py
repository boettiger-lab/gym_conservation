import math

import gym
import numpy as np
from gym import error, logger, spaces, utils
from gym.utils import seeding

from gym_conservation.envs.shared_env import *


class BaseEcologyEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        params={"r": 0.3, "K": 1, "sigma": 0.01, "x0": 0.75},
        Tmax=100,
        file=None,
    ):

        ## parameters
        self.K = params["K"]
        self.r = params["r"]
        self.sigma = params["sigma"]
        self.init_state = params["x0"]
        self.params = params

        ## Preserve these for reset
        self.unscaled_state = self.init_state
        self.reward = 0
        self.harvest = 0
        self.years_passed = 0
        self.Tmax = Tmax
        self.file = file

        # for render() method only
        if file != None:
            self.write_obj = open(file, "w+")

        ## Initial state
        self.state = np.array([self.init_state / self.K - 1])

        ## Best if cts actions / observations are normalized to a [-1, 1] domain
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

        ## Map from re-normalized model space to [0,2K] real space
        quota = self.get_quota(action)
        self.get_unscaled_state(self.state)

        ## Apply harvest and population growth
        self.harvest = self.harvest_draw(quota)
        self.population_draw()

        ## Map population back to system state (normalized space):
        self.get_state(self.unscaled_state)

        ## should be the instanteous reward, not discounted
        self.reward = max(self.harvest, 0.0)
        self.years_passed += 1
        done = bool(self.years_passed > self.Tmax)

        if self.unscaled_state <= 0.0:
            done = True

        return self.state, self.reward, done, {}

    def reset(self):
        self.state = np.array([self.init_state / self.K - 1])
        self.unscaled_state = self.init_state
        self.years_passed = 0

        ## for tracking only
        self.reward = 0
        self.harvest = 0
        return self.state

    def render(self, mode="human"):
        return csv_entry(self)

    def close(self):
        if self.file != None:
            self.write_obj.close()

    def simulate(env, model, reps=1):
        return simulate_mdp(env, model, reps)

    def plot(self, df, output="results.png"):
        return plot_mdp(self, df, output)

    def policyfn(env, model, reps=1):
        return estimate_policyfn(env, model, reps)

    def plot_policy(self, df, output="results.png"):
        return plot_policyfn(self, df, output)

    def harvest_draw(self, quota):
        """
        Select a value to harvest at each time step.
        """

        self.harvest = min(self.unscaled_state, quota)
        self.unscaled_state = max(self.unscaled_state - self.harvest, 0.0)
        return self.harvest

    def population_draw(self):
        """
        Select a value for population to grow or decrease at each time step.
        """
        self.unscaled_state = np.maximum(
            self.unscaled_state
            + self.r * self.unscaled_state * (1.0 - self.unscaled_state / self.K)
            + self.unscaled_state * self.sigma * np.random.normal(0, 1),
            0.0,
        )
        return self.unscaled_state

    def get_quota(self, action):
        """
        Convert action into quota
        """
        if isinstance(self.action_space, gym.spaces.discrete.Discrete):
            quota = (action / self.n_actions) * self.K
        ## Continuous Actions
        else:
            action = np.clip(action, self.action_space.low, self.action_space.high)[0]
            quota = (action + 1) * self.K
        return quota

    def get_action(self, quota):
        """
        Convert quota into action
        """
        if isinstance(self.action_space, gym.spaces.discrete.Discrete):
            return round(quota * self.n_actions / self.K)
        else:
            return quota / self.K - 1

    def get_unscaled_state(self, state):
        self.unscaled_state = (state[0] + 1) * self.K
        return self.unscaled_state

    def get_state(self, unscaled_state):
        self.state = np.array([unscaled_state / self.K - 1])
        return self.state
