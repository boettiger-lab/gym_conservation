from gym.envs.registration import register

register(
    id="conservation-v0",
    entry_point="gym_conservation.envs:Allen",
)


register(
    id="conservation-v2",
    entry_point="gym_conservation.envs:May",
)

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

from gym_conservation import models
