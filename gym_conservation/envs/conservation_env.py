import math
from math import floor
import gym
from gym import spaces, logger, error, utils
from gym.utils import seeding
import numpy as np
from csv import writer, reader
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt

class AbstractConservationEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 params = {"r": 0.3, "K": 1, "sigma": 0.1},
                 inits = {"state": 0.75, "action": 0},
                 T_max = 100,
                 action_space = spaces.Discrete(100),
                 observation_space = spaces.Box(np.array([0]), 
                                            np.array([2 * params["K"]]), 
                                            dtype = np.float32),
                 file = None
                 ):

        self.action_space = action_space
        self.observation_space = observation_space
        self.params = params
        self.inits = inits
        
        self.state = np.array([inits["state"]])
        self.action = inits["action"]        
        self.reward = 0
        self.timestep = 0
        self.Tmax = Tmax
        if(file != None):
          self.write_obj = open(file, 'w+')
    
    def population_draw(self, h):
        x = self.state
        r = self.params["r"]
        K = self.params["K"]
        q = self.params["q"]
        b = self.params["b"]
        sigma = self.params["sigma"]
        
        xt = x + x * r * (1 - x / K) - 
             a * x ^ q / (x ^ q + b ^ q) - 
             0.1 * (1 - h) * x + 
             x * sigma * p.random.normal(0,1)

        self.state = max(xt, 0.0)
        return self.state

    
    def step(self, action):
      
        action = np.clip(action, int(0), int(self.n_actions))
        if self.n_actions > 3:
          self.harvest = ( action / self.n_actions ) * self.K
        
        ## Discrete actions: increase, decrease, stay the same
        else:
          if action == 0:
            self.harvest = self.harvest
          elif action == 1:
            self.harvest = 1.2 * self.harvest
          else:
            self.harvest = 0.8 * self.harvest
      
        self.harvest_draw(self.harvest)
        self.population_draw()
        reward = max(self.price * self.harvest, 0.0)
        
        ## recording purposes only
        self.action = action
        self.reward = reward
        self.years_passed += 1
        done = bool(self.years_passed >= self.Tmax)

        if self.state <= 0.0:
            done = True
            return self.state, reward, done, {}

        return self.state, reward, done, {}
        
    def reset(self):
        self.state = np.array([self.inits["state"]])
        self.action = self.inits["action"]
        self.years_passed = 0
        return self.state

    def render(self, mode='human'):
      return csv_entry(self)
  
    def close(self):
      if(self.write_obj != None):
        self.write_obj.close()

    def simulate(env, model, reps = 1):
      return simulate_mdp(env, model, reps)
      
    def plot(self, df, output = "results.png"):
      return plot_mdp(self, df, output)




## Shared methods
def csv_entry(self):
  row_contents = [self.years_passed, 
                  self.state[0],
                  self.action,
                  self.reward]
  csv_writer = writer(self.write_obj)
  csv_writer.writerow(row_contents)
  return row_contents
    
def simulate_mdp(env, model, reps = 1):
  row = []
  for rep in range(reps):
    obs = env.reset()
    for t in range(env.Tmax):
      action, _state = model.predict(obs)
      obs, reward, done, info = env.step(action)
      row.append([t, obs, action, reward, rep])
      if done:
        break
  df = DataFrame(row, columns=['time', 'state', 'action', 'reward', "rep"])
  return df

def plot_mdp(self, df, output = "results.png"):
  fig, axs = plt.subplots(3,1)
  for i in range(np.max(df.rep)):
    results = df[df.rep == i]
    episode_reward = np.cumsum(results.reward)                    
    axs[0].plot(results.time, results.state, color="blue", alpha=0.3)
    axs[1].plot(results.time, results.action, color="blue", alpha=0.3)
    axs[2].plot(results.time, episode_reward, color="blue", alpha=0.3)
  
  axs[0].set_ylabel('state')
  axs[1].set_ylabel('action')
  axs[2].set_ylabel('reward')
  fig.tight_layout()
  plt.savefig(output)
  plt.close("all")



class ConservationEnv(AbstractConservationEnv):
    def __init__(self, **kargs):
        super(ConservationEnv, self).__init__(**kargs)


  
