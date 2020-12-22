import numpy as np
from gym.envs.registration import register

from gym_conservation.envs.base_env import BaseEcologyEnv
from gym_conservation.envs.growth_models import may


class NonStationaryV3(BaseEcologyEnv):
    def __init__(
        self,
        r=0.7,
        K=1.5,
        M=1.2,
        q=3,
        b=0.15,
        sigma=0.0,
        a=0.20,
        alpha=0.001,
        beta=1.0,
        init_state=0.8,
        cost=10.0,
        benefit=15.0,
        Tmax=100,
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

    def population_draw(self):
        self.params["a"] = self.params["a"] + self.params["alpha"]
        self.unscaled_state = may(self.unscaled_state, self.params)
        return self.unscaled_state

    def reset(self):
        self.state = np.array([self.init_state / self.K - 1])
        self.unscaled_state = self.init_state
        self.years_passed = 0
        self.params["a"] = self.init_a
        # for tracking only
        self.reward = 0
        self.unscaled_action = 0
        return self.state


class NonStationaryV5(BaseEcologyEnv):
    def __init__(
        self,
        r=0.7,
        K=1.5,
        M=1.2,
        q=3,
        b=0.15,
        sigma=0.0,
        a=0.20,
        alpha=0.001,
        beta=1.0,
        init_state=0.8,
        cost=10.0,
        benefit=15.0,
        Tmax=100,
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

    def population_draw(self):
        self.params["a"] = self.params["a"] + self.params["alpha"]
        self.unscaled_state = may(self.unscaled_state, self.params)
        return self.unscaled_state

    def perform_action(self, unscaled_action):
        self.unscaled_action = unscaled_action
        # Can move away from tipping point
        self.params["a"] = np.maximum(
            0.0,
            self.params["a"]
            - self.unscaled_action / (2 * self.params["K"] * 100.0),
        )
        return self.unscaled_action

    #    def compute_reward(self):
    #        return self.params["benefit"] * self.unscaled_state / (self.params["beta"] + self.unscaled_state) - np.power(
    #            self.unscaled_action, self.params["cost"]
    #        )
    def reset(self):
        self.state = np.array([self.init_state / self.K - 1])
        self.unscaled_state = self.init_state
        self.years_passed = 0
        self.params["a"] = self.init_a
        # for tracking only
        self.reward = 0
        self.unscaled_action = 0
        return self.state


class NonStationaryV4(BaseEcologyEnv):
    def __init__(
        self,
        r=0.7,
        K=1.5,
        M=1.2,
        q=3,
        b=0.15,
        sigma=0.0,
        a=0.20,
        alpha=0.001,
        beta=1.0,
        init_state=0.8,
        cost=10.0,
        benefit=15.0,
        Tmax=100,
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

    def population_draw(self):
        self.params["a"] = self.params["a"] + self.params["alpha"]
        self.unscaled_state = may(self.unscaled_state, self.params)
        return self.unscaled_state

    def perform_action(self, unscaled_action):
        self.unscaled_action = unscaled_action
        # Can move away from tipping point
        self.params["a"] = np.maximum(
            0.0,
            self.params["a"]
            - self.unscaled_action / (2 * self.params["K"] * 100.0),
        )
        return self.unscaled_action

    def compute_reward(self):
        return (
            self.params["benefit"] * float(self.unscaled_state > 0.3)
            - self.params["cost"] * self.unscaled_action
        )

    def reset(self):
        self.state = np.array([self.init_state / self.K - 1])
        self.unscaled_state = self.init_state
        self.years_passed = 0
        self.params["a"] = self.init_a
        # for tracking only
        self.reward = 0
        self.unscaled_action = 0
        return self.state


register(
    id="conservation-v3",
    entry_point="gym_conservation.envs:NonStationaryV3",
)

register(
    id="conservation-v4",
    entry_point="gym_conservation.envs:NonStationaryV4",
)

register(
    id="conservation-v5",
    entry_point="gym_conservation.envs:NonStationaryV5",
)
