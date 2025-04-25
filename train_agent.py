from agents import *
import sys
import os
import ctypes

from env import MAISREnv
from gui import *
from utility.data_logging import GameLogger, load_env_config
from config import x, y, config_dict, run_order, surveys_enabled, times

def train_rl_agent(episodes=1000):
    # Initialize environment in training mode
    env_config = load_env_config("./config_files/rl_training_config.json")
    env = MAISREnv(env_config, render_mode=False, agent_training=True)

    for episode in range(episodes):
        state = env.reset()
        ep_reward = 0
        done = False
        env.agents[env.human_idx].x, env.agents[env.human_idx].y = 2000, 2000 # TODO hack to put human out of harms way

        ep_len = 0
        while not done:
            ep_len += 1
            #action = {'waypoint': env.action_space.sample(), 'id_method': 0}
            action = env.action_space.sample()

            #print(f'Selected action {action}')
            #action = agent.choose_action(state)  # Replace with your agent's action selection

            # Step the environment
            next_state, reward, terminated, truncated, info = env.step([(env.agent_idx, action)])
            done = terminated or truncated

            # Train your agent
            #agent.learn(state, action, reward, next_state, done)

            state = next_state
            ep_reward += reward
            #if ep_len % 20 == 0:
                #print(f'Env time {round(env.display_time,1)}, time left: {round(env.time_limit-env.display_time/1000,1)}')

        print(f"Episode {episode}: {ep_len} steps, reward = {ep_reward} ({'died' if terminated else 'times up'})")

if __name__ == '__main__':
    train_rl_agent()