from agents import *
import sys
import os
import ctypes
from custom_ppo import PPOTrainer
from a2c_policy import MAISRActorCritic
import wandb

from env import MAISREnv
#from gui import *
from utility.data_logging import GameLogger, load_env_config
from config import x, y, config_dict, run_order, surveys_enabled, times

def train_rl_agent(episodes=1000):

    checkpoint_dir = 'checkpoints'
    #checkpoint_path = 'checkpoints/best_model.pt'

    # Initialize environment in training mode
    env_config = load_env_config("./config_files/rl_training_config.json")
    env = MAISREnv(env_config, render_mode=False, agent_training=True)
    model = MAISRActorCritic(444, 2, 1, env.config["gameboard size"])

    ppo = PPOTrainer(env, model, checkpoint_dir, lr=3e-4, gamma=0.99, clip_ratio=0.2, target_kl=0.01)
    print(f"Training on: {ppo.device}")

    print('starting PPO')
    ppo.train()

if __name__ == '__main__':
    train_rl_agent()