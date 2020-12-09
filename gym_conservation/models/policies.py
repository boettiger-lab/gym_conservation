import numpy as np


class user_action:
    def __init__(self, env, **kwargs):
        self.env = env

    def predict(self, obs, **kwargs):
        state = self.env.get_unscaled_state(obs)
        prompt = "state: " + str(state) + ". Your action: "
        unscaled_action = input(prompt)
        action = self.env.get_action(float(unscaled_action))
        return action, obs
