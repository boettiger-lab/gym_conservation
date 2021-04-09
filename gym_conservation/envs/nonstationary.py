import numpy as np
from gym.envs.registration import register

from gym_conservation.envs.base_env import BaseEcologyEnv
from gym_conservation.envs.growth_models import may

# Consider stochastic change in "a",
# Consider dual-control with actions on both state and parameter


class NonStationaryV3(BaseEcologyEnv):
    """
    non-stationary parameter `a` slowly moves system to a tipping point.
    In V3, the agent cannot halt  this decline, but can only influence the state
    directly.

    As in all models, value is proportional to the state and actions are costly.
    """

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
    """
    non-stationary parameter `a` slowly moves system to a tipping point.
    In V5, the agent can act to slow or even reverse the decline, moving
    the system away from the tipping point.  Note: Action is scaled to alter a
    by a factor of "unscaled_action" / (2*K*100), where the unscaled_action
    is as always defined as 0 to 2K.

    As in all models, value is proportional to the state and actions are costly.
    """

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
        return self.benefit * self.unscaled_state / (
            1 + self.unscaled_state
        ) - np.power(self.unscaled_action, self.cost)

    def reset(self):
        self.state = np.array([self.init_state / self.K - 1])
        self.unscaled_state = self.init_state
        self.years_passed = 0
        self.params["a"] = self.init_a
        # for tracking only
        self.reward = 0
        self.unscaled_action = 0
        return self.state


class NonStationaryV6(NonStationaryV5):
    """
    Identical to V5 but with stochastic default settings.
    """

    def __init__(
        self,
        r=0.7,
        K=1.5,
        M=1.2,
        q=3,
        b=0.15,
        sigma=0.2,
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
            r=r,
            K=K,
            M=M,
            q=q,
            b=b,
            sigma=sigma,
            a=a,
            alpha=alpha,
            beta=beta,
            init_state=init_state,
            cost=cost,
            benefit=benefit,
            Tmax=Tmax,
            file=file,
        )
        self.init_a = a


register(
    id="conservation-v3",
    entry_point="gym_conservation.envs:NonStationaryV3",
)

register(
    id="conservation-v6",
    entry_point="gym_conservation.envs:NonStationaryV6",
)

register(
    id="conservation-v5",
    entry_point="gym_conservation.envs:NonStationaryV5",
)
