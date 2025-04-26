from agents import *
import sys
import os
import ctypes
from custom_ppo import PPOTrainer
from a2c_policy import MAISRActorCritic

from env import MAISREnv
from gui import *
from utility.data_logging import GameLogger, load_env_config
from config import x, y, config_dict, run_order, surveys_enabled, times

def train_rl_agent(episodes=1000):

    checkpoint_dir = 'checkpoints'

    # Initialize environment in training mode
    env_config = load_env_config("./config_files/rl_training_config.json")
    env = MAISREnv(env_config, render_mode=False, agent_training=True)
    model = MAISRActorCritic(444, 2, 1, env.config["gameboard size"])

    ppo = PPOTrainer(env, model, checkpoint_dir, lr=3e-4, gamma=0.99, clip_ratio=0.2, target_kl=0.01)

    print('starting PPO')
    ppo.train()

    # for episode in range(episodes):
    #     state = env.reset()
    #     ep_reward = 0
    #     done = False
    #     env.agents[env.human_idx].x, env.agents[env.human_idx].y = 2000, 2000 # TODO hack to put human out of harms way
    #
    #     ep_len = 0
    #     while not done:
    #         ep_len += 1
    #
    #         ai_action = env.action_space.sample() # Formatted as {'waypoint': , 'id_method': }
    #
    #         # Handle "human" agent (currently just holds in place)
    #         human_loc = (env.agents[env.human_idx].x, env.agents[env.human_idx].y)
    #         human_action = {'waypoint': human_loc, 'id_method': 0}
    #
    #         #print(f'Selected action {action}')
    #         actions = [(env.agent_idx, ai_action), (env.human_idx, human_action)]
    #
    #         # Step the environment
    #         next_state, reward, terminated, truncated, info = env.step(actions)
    #         done = terminated or truncated
    #
    #         # Train your agent
    #         #agent.learn(state, action, reward, next_state, done)
    #
    #         state = next_state
    #         ep_reward += reward
    #         #if ep_len % 20 == 0:
    #             #print(f'Env time {round(env.display_time,1)}, time left: {round(env.time_limit-env.display_time/1000,1)}')
    #
    #     print(f"Episode {episode}: {ep_len} steps, reward = {ep_reward} ({'died' if terminated else 'times up'})")

if __name__ == '__main__':
    train_rl_agent()