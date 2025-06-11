"""
CNN training pipeline for MAISR environment using 84x84 pixel observations.
Uses existing frame_skip functionality and maintains full compatibility with
curriculum learning and evaluation systems.
"""

import warnings

warnings.filterwarnings("ignore", message="Your system is avx2 capable but pygame was not built with support for it")

import gymnasium as gym
import os
import numpy as np
import multiprocessing
import socket
import torch
import torch.nn as nn

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from env_combined import MAISREnvVec
from utility.data_logging import load_env_config
from utility.config_management import load_env_config_with_sweeps, generate_sweep_run_name

# Import the enhanced wandb callback from your existing training file
from train_sb3 import EnhancedWandbCallback


def generate_run_name_cnn(config):
    """Generate a unique, descriptive name for CNN training runs."""

    components = [
        f"{config['n_envs']}envs",
        "cnn",
        f"fs{config.get('frame_skip', 1)}",
    ]

    # Add critical hyperparameters
    components.extend([
        f"lr{config.get('lr', 0.00025)}".replace(".", ""),
        f"bs{config.get('batch_size', 32)}",
    ])

    # Add curriculum info if enabled
    if config.get('use_curriculum', False):
        components.append("curriculum")

    # Add timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d_%H%M")

    run_name = f"{timestamp}_" + "_".join(components)
    return run_name


def make_env_cnn(env_config, rank, seed, run_name='no_name'):
    """
    Create MAISR environment with CNN pixel observations.
    Uses existing frame_skip functionality.
    """

    def _init():
        env = MAISREnvVec(
            config=env_config,
            render_mode='headless',
            run_name=run_name,
            tag=f'train_mp{rank}',
            seed=seed + rank,
        )
        env = Monitor(env)
        env.reset()
        return env

    return _init


def train_cnn(
        env_config,
        n_envs,
        project_name,
        save_dir="./trained_models/",
        load_path=None,
        log_dir="./logs/",
        machine_name='machine',
        save_model=True
):
    """
    CNN training pipeline for MAISR environment with 84x84 pixel observations.
    """

    # Generate run name
    run_name = generate_run_name_cnn(env_config)

    os.makedirs(f"{save_dir}/{run_name}", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f'./logs/action_histories/{run_name}', exist_ok=True)

    run = wandb.init(
        project=project_name,
        name=f'{machine_name}_{n_envs}envs_' + run_name,
        config=env_config,
        sync_tensorboard=True,
        monitor_gym=True,
    )
    run.log_code(".")

    ################################################ Initialize environments ################################################
    if n_envs > 1:
        print(f"Training with {n_envs} CNN environments in parallel")

        env_fns = [make_env_cnn(env_config, i, env_config['seed'] + i, run_name=run_name)
                   for i in range(n_envs)]
        env = SubprocVecEnv(env_fns)
        env = VecMonitor(env, filename=os.path.join(log_dir, 'vecmonitor'))

        # Transpose images for CNN (channels first: C x H x W)
        env = VecTransposeImage(env)

        print(f"CNN Training env observation space: {env.observation_space}")

    else:
        env = MAISREnvVec(
            env_config,
            None,
            render_mode='headless',
            tag='train_0',
            run_name=run_name,
        )
        env = Monitor(env)
        print(f"Single CNN Training env observation space: {env.observation_space}")

    # Evaluation environment
    eval_env = MAISREnvVec(
        env_config,
        None,
        render_mode='headless',
        tag='eval',
        run_name=run_name,
    )

    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecTransposeImage(eval_env)

    print('CNN environments created successfully')

    ################################################# Setup callbacks #################################################
    checkpoint_callback = CheckpointCallback(
        save_freq=env_config['save_freq'] // n_envs,
        save_path=f"{save_dir}/{run_name}",
        name_prefix=f"maisr_cnn_checkpoint_{run_name}",
        save_replay_buffer=True,
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=100,  # Less frequent for CNN training
        model_save_path=f"{save_dir}/wandb/{run.id}" if save_model else None,
        verbose=1
    )

    enhanced_wandb_callback = EnhancedWandbCallback(
        env_config,
        eval_env=eval_env,
        run=run,
        log_freq=50
    )

    print('Callbacks created')

    ################################################# Setup CNN model #################################################

    # policy_kwargs = dict(
    #     features_extractor_class=MAISRCNN,
    #     features_extractor_kwargs=dict(features_dim=env_config.get('cnn_features_dim', 256)),
    #     net_arch=dict(
    #         pi=[env_config.get('policy_network_size', 64)],
    #         vf=[env_config.get('value_network_size', 64)]
    #     ),
    #     activation_fn=torch.nn.ReLU,
    # )

    model = PPO(
        "CnnPolicy",  # Use CNN policy
        env,
        verbose=2,
        tensorboard_log=f"logs/runs/{run.id}",
        batch_size=env_config.get('batch_size', 32),
        n_steps=env_config.get('ppo_update_steps', 128),
        learning_rate=env_config.get('lr', 2.5e-4),
        seed=env_config['seed'],
        device='auto',  # Use GPU if available, CPU otherwise
        gamma=env_config.get('gamma', 0.99),
        ent_coef=env_config.get('entropy_regularization', 0.01),
        clip_range=env_config.get('clip_range', 0.2),
        vf_coef=env_config.get('vf_coef', 0.5),
        max_grad_norm=env_config.get('max_grad_norm', 0.5)
    )

    print('CNN model instantiated')
    print(f"Model policy: {model.policy}")
    print(f"CNN feature extractor: {model.policy.features_extractor}")

    ################################################# Load checkpoint ##################################################
    if load_path:
        print(f'LOADING CNN MODEL FROM {load_path}')
        model = model.__class__.load(load_path, env=env)
    else:
        print('Training new CNN model from scratch')

    print(
        '##################################### Beginning CNN agent training... #######################################\n')

    # Log initial difficulty and model info
    run.log({
        "curriculum/difficulty_level": 0,
        "model/total_parameters": sum(p.numel() for p in model.policy.parameters()),
        "model/trainable_parameters": sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    }, step=0)

    print(f'Starting CNN training with difficulty level 0')
    print(f'Model has {sum(p.numel() for p in model.policy.parameters())} total parameters')

    model.learn(
        total_timesteps=int(env_config['num_timesteps']),
        callback=[checkpoint_callback, wandb_callback, enhanced_wandb_callback],
        reset_num_timesteps=True if load_path else False
    )

    print(
        '########################################## CNN TRAINING COMPLETE ############################################\n')

    # Final evaluation
    print("Running final evaluation...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env,
        n_eval_episodes=env_config.get('n_eval_episodes', 10),
        deterministic=True
    )
    print(f"Final CNN evaluation: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # Save final model
    final_model_path = os.path.join(save_dir, f"cnn_maisr_final_{run_name}")
    model.save(final_model_path)
    print(f"CNN training completed! Final model saved to {final_model_path}")

    # Log final metrics
    run.log({
        "final/mean_reward": mean_reward,
        "final/std_reward": std_reward,
        "final/training_completed": True
    })

    # Cleanup
    env.close()
    eval_env.close()
    run.finish()

    return final_model_path


if __name__ == "__main__":

    ############## ---- CNN TRAINING SETTINGS ---- ##############
    load_path = None  # Path to load existing CNN model
    config_filename = '../config_files/cnn_test.json'
    ###############################################

    # Get machine name
    machine_name = 'home' if socket.gethostname() == 'DESKTOP-3Q1FTUP' else 'lab_pc' if socket.gethostname() == 'isye-ae-2023pc3' else 'pace'
    project_name = 'maisr-rl-cnn' if machine_name in ['home', 'lab_pc'] else 'maisr-rl-cnn-pace'
    print(f'Machine: {machine_name}, Project: {project_name}')

    print(f'\n############################ STARTING CNN TRAINING ############################')

    # Load configuration
    try:
        all_configs, param_names = load_env_config_with_sweeps(config_filename)
    except FileNotFoundError:
        print(f"Config file {config_filename} not found. Please create it first.")
        exit(1)

    for i, env_config in enumerate(all_configs):
        print(f'\n--- Starting CNN training run {i + 1}/{len(all_configs)} ---')

        # Set number of environments (limit for CNN training)
        env_config['n_envs'] = multiprocessing.cpu_count()
        env_config['config_filename'] = config_filename

        final_run_name = generate_run_name_cnn(
            env_config) + f'{"".join("_" + str(name) + "-" + str(env_config[name]) for name in param_names)}'
        print(f"Running CNN training: {final_run_name}")

        model_path = train_cnn(
            env_config,
            n_envs=env_config['n_envs'],
            load_path=load_path,
            machine_name=machine_name,
            project_name=project_name
        )

        print(f"✓ Completed CNN training run {i + 1}/{len(all_configs)}")
        print(f"Model saved to: {model_path}")