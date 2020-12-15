library(reticulate)
library(tidyverse)
library(patchwork)

## Python dependencies
gc <- import("gym_conservation")
gym <- import ("gym")
sb3 <- import ("stable_baselines3")
## initialize the environment
# see: cost 1, benefit ~ 5
env <- gym$make("conservation-v5", cost = 5, benefit = 150,
                sigma = 0.02, alpha = 0.005)
model = gc$models$user_action(env)

model = sb3$PPO("MlpPolicy", env, verbose=1)
model$learn(total_timesteps=500000)

## Simulate a run with the trained model, visualize result
df = env$simulate(model, reps = 10L)
policy = env$policyfn(model, reps = 10L)


p1 <- df %>% pivot_longer(c(state, action, reward)) %>%
  ggplot(aes(time, value, group=rep)) + geom_line() +
  facet_wrap(~name, ncol=1, scales="free_y")


## Simulate a run with the trained model, visualize result
p2 <- policy %>%
  ggplot(aes(state, action, color=rep)) + geom_point()

p1 / p2