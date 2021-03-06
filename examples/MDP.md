
Our conservation model represents an ecosystem with two alternative
stable states. For simplicity, we will consider higher ecological states
to be more desirable, i.e. our reward function will be linear in the
ecosystem state. We will consider actions that directly intervene to
change the ecosystem state, with large increases being proportionally
more costly than smaller ones (i.e. quadratic costs).

``` r
library(MDPtoolbox)
library(sarsop)
library(tidyverse) # for plotting
library(mdplearning) # remotes::install_github("boettiger-lab/mdplearning")
```

``` r
states <- seq(0,2, length=100)
actions <- seq(0,2, length.out = 100)
observations <- states
sigma_g <- 0.05
sigma_m <- 0.0

benefit <- 1
cost <- 2 # quadratic costs
reward_fn <- function(x,h) benefit * x - (h ^ cost)
discount <- 0.999

p <- list(r =.7, K = 1.2, q = 3, b = 0.15, a = 0.2) # lower-state peak is optimal
#p <- list(r =.7, K = 1.5, q = 3, b = 0.15, a = 0.2) # higher-state peak is optimal

may <- 
  function(x, h = 0){ # May
    y <- x +  h
    pmax(
      ## controlling h is controlling the bifurcation point directly...
      y + y * p$r * (1 - y / p$K)  - p$a * y ^ p$q / (y ^ p$q + p$b ^ p$q),  
      0)
  }

landscape <- tibble(x = states, f = may(x,0) - x) 
```

## Optimal management

``` r
m <- fisheries_matrices(states, actions, observations, reward_fn, 
                        may, sigma_g, sigma_m, noise = "lognormal")
```

``` r
soln <- mdp_value_iteration(m$transition, m$reward, discount)
```

``` r
df <- tibble(state = states,
             action = actions[soln$policy],
             value = soln$V) %>% 
  inner_join(landscape %>% rename(state = x), by = "state")
```

The optimal solution is an action sufficient to tip the system into the
higher state of attraction. Because small actions are relatively cheap
under the quadratic cost assumption, the optimal strategy retains a
small conservation investment, keeping the state a bit a higher than the
upper equilibrium.

A deep RL solution may struggle with exploration to learn to escape a
low initial state.

``` r
df %>% ggplot(aes(state, action)) + geom_point() +
  geom_line(aes(state, 70*f), col = "red")  + 
  coord_cartesian(xlim=c(0,1.3), ylim = c(-2, 3))
```

![](MDP_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
x0 = which.min(abs(states - 0.05))
sim <- mdp_planning(m$transition, m$reward, discount, model_prior = c(1), 
                   policy = soln$policy, x0 = x0, Tmax = 100)
sim %>% mutate(state = states[state], action = actions[action]) %>% 
  ggplot(aes(time, state)) + geom_point()+geom_path() + 
  geom_point(aes(time, action), col="blue")
```

![](MDP_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->
