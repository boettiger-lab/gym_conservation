import numpy as np
from gym.envs.registration import register

from gym_conservation.envs.base_env import BaseEcologyEnv


class Allen(BaseEcologyEnv):
    def __init__(
        self,
        r=0.3,
        K=1,
        C=0.5,
        sigma=0.0,
        init_state=0.75,
        cost=1.0,
        benefit=5.0,
        Tmax=100,
        file="render.csv",
    ):
        super().__init__(
            params={
                "r": r,
                "K": K,
                "sigma": sigma,
                "C": C,
                "x0": init_state,
                "cost": cost,
                "benefit": benefit,
            },
            Tmax=Tmax,
            file=file,
        )

    def population_draw(self):
        self.unscaled_state = allen(self.unscaled_state, self.params)
        return self.unscaled_state


class BevertonHolt(BaseEcologyEnv):
    def __init__(
        self,
        r=0.3,
        K=1,
        sigma=0.0,
        init_state=0.75,
        cost=1.0,
        benefit=5.0,
        Tmax=100,
        file="render.csv",
    ):
        super().__init__(
            params={
                "r": r,
                "K": K,
                "sigma": sigma,
                "x0": init_state,
                "cost": cost,
                "benefit": benefit,
            },
            Tmax=Tmax,
            file=file,
        )

    def population_draw(self):
        self.unscaled_state = beverton_holt(self.unscaled_state, self.params)
        return self.unscaled_state


class Myers(BaseEcologyEnv):
    def __init__(
        self,
        r=1.0,
        K=1.0,
        M=1.0,
        theta=3.0,
        sigma=0.0,
        init_state=1.5,
        cost=1.0,
        benefit=5.0,
        Tmax=100,
        file="render.csv",
    ):
        super().__init__(
            params={
                "r": r,
                "K": K,
                "sigma": sigma,
                "theta": theta,
                "M": M,
                "x0": init_state,
                "cost": cost,
                "benefit": benefit,
            },
            Tmax=Tmax,
            file=file,
        )

    def population_draw(self):
        self.unscaled_state = myers(self.unscaled_state, self.params)
        return self.unscaled_state


# (r =.7, M = 1.2, q = 3, b = 0.15, a = 0.2) # lower-state peak is optimal
# (r =.7, M = 1.5, q = 3, b = 0.15, a = 0.2) # higher-state peak is optimal
class May(BaseEcologyEnv):
    def __init__(
        self,
        r=0.7,
        K=1.5,
        M=1.2,
        q=3,
        b=0.15,
        sigma=0.0,
        a=0.2,
        init_state=0.1,
        cost=1.0,
        benefit=5.0,
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
            },
            Tmax=Tmax,
            file=file,
        )

    def population_draw(self):
        self.unscaled_state = may(self.unscaled_state, self.params)
        return self.unscaled_state


class Ricker(BaseEcologyEnv):
    def __init__(
        self,
        r=0.3,
        K=1,
        sigma=0.0,
        init_state=0.75,
        cost=1.0,
        benefit=5.0,
        Tmax=100,
        file="render.csv",
    ):
        super().__init__(
            params={
                "r": r,
                "K": K,
                "sigma": sigma,
                "x0": init_state,
                "cost": cost,
                "benefit": benefit,
            },
            Tmax=Tmax,
            file=file,
        )

    def population_draw(self):
        self.unscaled_state = ricker(self.unscaled_state, self.params)
        return self.unscaled_state


class ModelUncertainty(BaseEcologyEnv):
    def __init__(
        self,
        models=["allen", "beverton_holt", "myers", "may", "ricker"],
        params={
            "allen": {
                "r": 0.3,
                "K": 1.0,
                "sigma": 0.0,
                "C": 0.5,
                "x0": 0.75,
                "cost": 1.0,
                "benefit": 5.0,
            },
            "beverton_holt": {
                "r": 0.3,
                "K": 1,
                "sigma": 0.0,
                "x0": 0.75,
                "cost": 1.0,
                "benefit": 5.0,
            },
            "myers": {
                "r": 1.0,
                "K": 1.0,
                "M": 1.0,
                "theta": 3.0,
                "sigma": 0.0,
                "x0": 1.5,
                "cost": 1.0,
                "benefit": 5.0,
            },
            "may": {
                "r": 0.7,
                "K": 1.5,
                "M": 1.5,
                "q": 3,
                "b": 0.15,
                "sigma": 0.0,
                "a": 0.2,
                "x0": 0.1,
                "cost": 1.0,
                "benefit": 5.0,
            },
            "ricker": {
                "r": 0.3,
                "K": 1,
                "sigma": 0.0,
                "x0": 0.75,
                "cost": 1.0,
                "benefit": 5.0,
            },
        },
        Tmax=100,
        file="render.csv",
    ):
        super().__init__(
            Tmax=Tmax,
            file=file,
        )
        self.model = np.random.choice(models)
        self.models = models
        self.params = params

    def population_draw(self):
        f = population_model[self.model]
        p = self.params[self.model]
        self.unscaled_state = f(self.unscaled_state, p)
        return self.unscaled_state

    def reset(self):
        self.state = np.array([self.init_state / self.K - 1])
        self.unscaled_state = self.init_state
        self.model = np.random.choice(self.models)
        self.years_passed = 0
        self.reward = 0
        self.unscaled_action = 0
        return self.state


# Growth Functions
def allen(x, params):
    with np.errstate(divide="ignore"):
        mu = (
            np.log(x)
            + params["r"]
            * (1 - x / params["K"])
            * (1 - params["C"])
            / params["K"]
        )
    return np.maximum(0, np.random.lognormal(mu, params["sigma"]))


def beverton_holt(x, params):
    x = np.clip(x, 0.0, np.inf)
    A = np.clip(params["r"], 0.0, np.inf) + 1
    with np.errstate(divide="ignore"):
        B = np.clip(params["K"], 0, np.inf) / np.clip(params["r"], 0.0, np.inf)
        mu = np.log(A) + np.log(x) - np.log(1 + x / B)
    return np.maximum(0, np.random.lognormal(mu, params["sigma"]))


def may(x, params):
    with np.errstate(divide="ignore"):
        r = params["r"]
        M = params["M"]
        a = params["a"]
        q = params["q"]
        b = params["b"]
        exp_mu = (
            x
            + x * r * (1 - x / M)
            - a * np.power(x, q) / (np.power(x, q) + np.power(b, q))
        )
        mu = np.log(np.clip(exp_mu, 0, np.inf))
    return np.maximum(0, np.random.lognormal(mu, params["sigma"]))


# be careful that K is chosen correctly (independent of M) to ensure
# state space is correct size.  Default parameters of Myers class should work
def myers(x, params):
    A = params["r"] + 1
    with np.errstate(divide="ignore"):
        mu = (
            np.log(A)
            + params["theta"] * np.log(x)
            - np.log(1 + np.power(x, params["theta"]) / params["M"])
        )
    return np.maximum(0, np.random.lognormal(mu, params["sigma"]))


def ricker(x, params):
    with np.errstate(divide="ignore"):
        mu = np.log(x) + params["r"] * (1 - x / params["K"])
    return np.maximum(0, np.random.lognormal(mu, params["sigma"]))


population_model = {
    "allen": allen,
    "beverton_holt": beverton_holt,
    "myers": myers,
    "may": may,
    "ricker": ricker,
}


register(
    id="conservation-v0",
    entry_point="gym_conservation.envs:Ricker",
)


register(
    id="conservation-v2",
    entry_point="gym_conservation.envs:May",
)
