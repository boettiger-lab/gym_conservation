from gym.envs.registration import register

register(
    id="conservation-v0",
    entry_point="gym_conservation.envs:Allen",
)

register(
    id="conservation-v1",
    entry_point="gym_conservation.envs:BevertonHolt",
)

register(
    id="conservation-v2",
    entry_point="gym_conservation.envs:May",
)

register(
    id="conservation-v3",
    entry_point="gym_conservation.envs:Myers",
)

register(
    id="conservation-v4",
    entry_point="gym_conservation.envs:Ricker",
)

register(
    id="conservation-v5",
    entry_point="gym_conservation.envs:NonStationary",
)

register(
    id="conservation-v6",
    entry_point="gym_conservation.envs:ModelUncertainty",
)

from gym_conservation import models
