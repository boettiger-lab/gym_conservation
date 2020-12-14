library(reticulate)
library(tidyverse)

## Python dependencies
gc <- import("gym_conservation")
gym <- import ("gym")
sb3 <- import ("stable_baselines3")
## initialize the environment
env <- gym$make("conservation-v5", cost = 1, benefit = 0.00)
model = gc$models$user_action(env)

model = sb3$PPO("MlpPolicy", env, verbose=1)
model$learn(total_timesteps=100000)

## Simulate a run with the trained model, visualize result
df = env$simulate(model, reps = 10L)

df %>% pivot_longer(c(state, action, reward)) %>%
  ggplot(aes(time, value, group=rep)) + geom_line() +
  facet_wrap(~name, ncol=1)

