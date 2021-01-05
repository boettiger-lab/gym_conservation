from gym.envs.registration import register

from gym_conservation.envs.growth_models import May, Ricker
from gym_conservation.envs.nonstationary import (
    NonStationaryV3,
    NonStationaryV5,
    NonStationaryV6,
)

register(
    id="conservation-v0",
    entry_point="gym_conservation.envs:Ricker",
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
    id="conservation-v5",
    entry_point="gym_conservation.envs:NonStationaryV5",
)

register(
    id="conservation-v6",
    entry_point="gym_conservation.envs:NonStationaryV6",
)
