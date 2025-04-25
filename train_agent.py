from agents import *
import sys
import os
import ctypes

from env import MAISREnv
from gui import *
from utility.data_logging import GameLogger, load_env_config
from config import x, y, config_dict, run_order, surveys_enabled, times

def train_rl_agent(episodes=1000, time_scale=10.0):
    # Initialize environment in training mode
    env_config = load_env_config("./config_files/rl_training_config.json")
    env = MAISREnv(env_config, render_mode=False, agent_training=True, time_scale=time_scale)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Choose action (from your RL agent)
            action = env.action_space.sample()
            #action = agent.choose_action(state)  # Replace with your agent's action selection

            # Step the environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Train your agent
            #agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        print(f"Episode {episode}: Total Reward = {total_reward}")

if __name__ == '__main__':
    train_rl_agent()