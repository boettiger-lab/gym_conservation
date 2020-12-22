class user_action:
    def __init__(self, env, **kwargs):
        self.env = env

    def predict(self, obs, **kwargs):
        state = self.env.get_unscaled_state(obs)
        prompt = "state: " + str(state) + ". Your action: "
        unscaled_action = input(prompt)
        action = self.env.get_action(float(unscaled_action))
        return action, obs


class fixed_action:
    def __init__(self, env, fixed_action=0.0, **kwargs):
        self.env = env
        self.fixed_action = fixed_action

    def predict(self, obs, **kwargs):
        state = self.env.get_unscaled_state(obs)
        unscaled_action = self.fixed_action
        action = self.env.get_action(float(unscaled_action))
        return action, state


class target_state:
    def __init__(self, env, target_state=1.0, **kwargs):
        self.env = env
        self.target_state = target_state

    def predict(self, obs, **kwargs):
        state = self.env.get_unscaled_state(obs)
        unscaled_action = self.target_state - state
        action = self.env.get_action(float(unscaled_action))
        return action, obs
