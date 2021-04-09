from csv import writer

import gym
import numpy as np
from gym import spaces
from gym.envs.registration import register

from gym_conservation.envs.growth_models import may


class VectorEcologyEnv(gym.Env):
    """
    State space is defined over an ensemble of replicates, allowing the
    agent to attempt to maximize expected value over the ensemble during
    training.

    Action space and state space are defined on a domain space of -1, 1
    to facilitate RL performance.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        params=dict(
            r=0.7,
            K=1.5,
            M=1.2,
            q=3,
            b=0.15,
            sigma=0.2,
            a=0.19,
            alpha=0.001,
            beta=1.0,
            x0=0.8,
            cost=2.0,
            benefit=1.0,
        ),
        reps=1,
        Tmax=500,
        file="render.csv",
    ):
        self.params = params
        self.init_a = params["a"]
        self.reps = reps
        self.Tmax = Tmax
        self.file = file
        # Initialize reward, action, years_passed, etc
        self.reset()

        # for render() method only
        if file is not None:
            self.write_obj = open(file, "w+")

        self.action_space = spaces.Box(
            np.array([-1], dtype=np.float32),
            np.array([1], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            np.full(reps, -1, dtype=np.float32),
            np.full(reps, 1, dtype=np.float32),
            dtype=np.float32,
        )

    def step(self, action):
        self.action = action  # for record keeping/render purposes
        self.state = self.take_action(self.state, action)
        self.reward = self.compute_reward(self.state, action)
        reward = float(np.mean(self.reward))
        self.years_passed += 1
        done = bool(self.years_passed > self.Tmax)
        return self.state, reward, done, {}

    def take_action(self, state, action):
        a = self.get_unscaled_action(action)
        s = self.get_unscaled_state(state)
        s = self.perform_action(s, a)
        s = self.population_draw(s)
        return self.get_state(s)

    def reset(self):
        init_state = np.full(self.reps, self.params["x0"], dtype=np.float32)
        self.state = self.get_state(init_state)
        self.params["a"] = self.init_a
        self.years_passed = 0
        self.reward = 0
        self.action = self.get_action(0.0)
        return self.state

    def compute_reward(self, state, action):
        a = self.get_unscaled_action(action)
        s = self.get_unscaled_state(state)
        reward = self.params["benefit"] * s / (1 + s) - np.power(
            a, self.params["cost"]
        )
        return reward

    def render(self, mode="human"):
        state = self.get_unscaled_state(self.state)
        action = self.get_unscaled_action(self.action)
        reward = self.reward
        csv_writer = writer(self.write_obj)
        for i in range(self.reps):
            row_contents = [self.years_passed, state[i], action, reward, i]
            csv_writer.writerow(row_contents)
        return None

    def close(self):
        if self.file is not None:
            self.write_obj.close()

    def perform_action(self, s, a):
        action = a
        self.params["a"] = np.maximum(
            0.0,
            self.params["a"] - action / (2 * self.params["K"] * 100.0),
        )
        return s

    def population_draw(self, s):
        self.params["a"] = self.params["a"] + self.params["alpha"]
        next_state = may(s, self.params)
        return np.clip(next_state, 0, 2 * self.params["K"])

    def get_unscaled_action(self, action):
        if isinstance(self.action_space, gym.spaces.discrete.Discrete):
            unscaled_action = (action / self.n_actions) * self.params["K"]
        else:
            action = np.clip(
                action, self.action_space.low, self.action_space.high
            )[0]
            unscaled_action = (action + 1) * self.params["K"]
        return unscaled_action

    def get_action(self, unscaled_action):
        if isinstance(self.action_space, gym.spaces.discrete.Discrete):
            return round(unscaled_action * self.n_actions / self.params["K"])
        else:
            return unscaled_action / self.params["K"] - 1

    def get_unscaled_state(self, state):
        self.unscaled_state = (state + 1) * self.params["K"]
        return self.unscaled_state

    def get_state(self, unscaled_state):
        return np.full(self.reps, unscaled_state / self.params["K"] - 1)


register(
    id="conservation-v7",
    entry_point="gym_conservation.envs:VectorEcologyEnv",
)
