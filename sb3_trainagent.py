import gymnasium as gym
import os
import numpy as np
from stable_baselines3.common.env_checker import check_env

from env_vec import MAISREnvVec
from utility.data_logging import load_env_config
from agents import *



config = './config_files/rl_training_config.json'
env_config = load_env_config(config)

env = MAISREnvVec(env_config,
                  None,
                  render_mode='none',
                  reward_type='balanced-sparse',
                  obs_type='vector',
                  action_type='continuous',)

#check_env(env, warn=True)

# print("Observation space:", env.observation_space)
# print("Shape:", env.observation_space.shape)
# print("Action space:", env.action_space)
#
actions = []
obs, info = env.reset()
action = env.action_space.sample()

actions.append((env.aircraft_ids[0], action))

#print("Sampled action:", action)
#print(type(action))
obs, reward, terminated, truncated, info = env.step(actions)
#print(type(obs))