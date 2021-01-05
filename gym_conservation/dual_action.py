import numpy as np
from gym import spaces
from gym.envs.registration import register

from gym_conservation.envs.base_env import BaseEcologyEnv
from gym_conservation.envs.growth_models import may


# Consider stochastic change in "a",
# Consider dual-control with actions on both state and parameter
class NonStationaryV7(BaseEcologyEnv):
    """"""

    def __init__(
        self,
        r=0.7,
        K=1.5,
        M=1.2,
        q=3,
        b=0.15,
        sigma=0.0,
        a=0.19,
        alpha=0.001,
        beta=1.0,
        init_state=0.8,
        cost=2.0,
        benefit=1.0,
        Tmax=500,
        file="render.csv",
    ):
        super().__init__(
            params={
                "r": r,
                "K": K,
                "sigma": sigma,
                "q": q,
                "b": b,
                "a": a,
                "M": M,
                "x0": init_state,
                "cost": cost,
                "benefit": benefit,
                "alpha": alpha,
                "beta": beta,
            },
            Tmax=Tmax,
            file=file,
        )
        self.init_a = a
        self.unscaled_action = [0, 0]
        self.action_space = spaces.Box(
            np.array([-1, -1], dtype=np.float32),
            np.array([1, 1], dtype=np.float32),
            dtype=np.float32,
        )

    def population_draw(self):
        self.params["a"] = self.params["a"] + self.params["alpha"]
        self.unscaled_state = may(self.unscaled_state, self.params)
        return self.unscaled_state

    def perform_action(self, unscaled_action):
        # Can move away from tipping point
        self.unscaled_action = unscaled_action
        self.params["a"] = np.maximum(
            0.0,
            self.params["a"]
            - unscaled_action[0] / (2 * self.params["K"] * 100.0),
        )
        self.unscaled_state = self.unscaled_state + unscaled_action[1]

        return unscaled_action

    def compute_reward(self):
        return (
            self.benefit * self.unscaled_state / (1 + self.unscaled_state)
            - np.power(self.unscaled_action[0], self.cost)
            + (
                self.benefit * self.unscaled_state
                - np.power(self.unscaled_action[1], self.cost)
            )
        )

    def reset(self):
        self.state = np.array([self.init_state / self.K - 1])
        self.unscaled_state = self.init_state
        self.years_passed = 0
        self.params["a"] = self.init_a
        # for tracking only
        self.reward = 0
        self.unscaled_action = [0.0, 0.0]
        return self.state

    # Dual action space
    def scale_action(self, unscaled_action):
        return unscaled_action / self.K - 1

    def unscale_action(self, action):
        action = np.clip(
            action, self.action_space.low, self.action_space.high
        )[0]
        unscaled_action = (action + 1) * self.K
        return unscaled_action

    def get_unscaled_action(self, action):
        return [self.unscale_action(action[0]), self.unscale_action(action[1])]

    def get_action(self, unscaled_action):
        return [
            self.scale_action(unscaled_action[0]),
            self.scale_action(unscaled_action[1]),
        ]


register(
    id="conservation-v7",
    entry_point="gym_conservation.envs:NonStationaryV7",
)
