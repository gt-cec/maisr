import ctypes
import warnings
import pygame
import gymnasium as gym
import os
import numpy as np
import multiprocessing
import socket
import torch
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from env_multi_new import MAISREnvVec
from utility.data_logging import load_env_config

warnings.filterwarnings("ignore", message="Your system is avx2 capable but pygame was not built with support for it")


def generate_run_name(config, seed, num_agents):
    """Generate a unique, descriptive name for this training run."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d_%H%M")

    components = [
        f"seed{seed}",
        f"{num_agents}agents",
        f"{config['n_envs']}envs"
    ]

    run_name = f"teammate_{timestamp}_" + "_".join(components)
    return run_name


def make_env(env_config, rank, seed, run_name='no_name'):
    """
    Callable function that creates a MAISR environment for training teammates.
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


def train_single_teammate(
        env_config,
        training_seed,
        num_agents,
        n_envs,
        project_name,
        use_normalize=True,
        save_dir="./trained_models/teammates/",
        log_dir="./logs/teammates/",
        machine_name='machine'
):
    """
    Train a single teammate agent with specified configuration.
    """

    # Modify config for this specific training run
    config = env_config.copy()
    config['num_aircraft'] = num_agents
    config['seed'] = training_seed

    # Generate unique run name
    run_name = generate_run_name(config, training_seed, num_agents)

    print(f'\n=== Training teammate: {run_name} ===')
    print(f'Seed: {training_seed}, Num agents: {num_agents}')

    # Create directories
    os.makedirs(f"{save_dir}/{run_name}", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize WandB
    run = wandb.init(
        project=project_name,
        name=f'{machine_name}_{run_name}',
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
    )

    print(f"Training with {n_envs} environments in parallel")

    # Create vectorized training environments
    env_fns = [make_env(config, i, training_seed + i, run_name=run_name) for i in range(n_envs)]
    if n_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    # Apply SB3 wrappers
    env = VecMonitor(env, filename=os.path.join(log_dir, f'vecmonitor_{run_name}'))
    if use_normalize:
        env = VecNormalize(env)

    # Create evaluation environment
    base_eval_env = MAISREnvVec(
        config=config,
        render_mode='headless',
        tag='eval',
        run_name=run_name,
        seed=training_seed + 1000  # Different seed for eval
    )
    eval_env = Monitor(base_eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])

    if use_normalize:
        eval_env = VecNormalize(eval_env, norm_reward=False, training=False)
        eval_env.obs_rms = env.obs_rms
        eval_env.ret_rms = env.ret_rms

    print('Environments created')

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config['save_freq'] // n_envs,
        save_path=f"{save_dir}/{run_name}",
        name_prefix=f"teammate_checkpoint_{run_name}",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=50,
        verbose=1,
        model_save_path=f"{save_dir}/wandb/{run.id}"
    )

    print('Callbacks created')

    # Setup model
    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(
            pi=[config['policy_network_size'], config['policy_network_size']],
            vf=[config['value_network_size'], config['value_network_size']]
        )
    )

    model = PPO(
        "CnnPolicy" if config['obs_type'] == 'pixel' else "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=f"logs/tb_runs/{run.id}",
        batch_size=config['batch_size'],
        n_steps=config['ppo_update_steps'],
        learning_rate=config['lr'],
        seed=training_seed,
        device='cpu',
        gamma=config['gamma'],
        ent_coef=config['entropy_regularization'],
        clip_range=config['clip_range']
    )

    print('Model instantiated')
    print(f'Training for {config["num_timesteps"]} timesteps')

    # Train the model
    model.learn(
        total_timesteps=int(config['num_timesteps']),
        callback=[checkpoint_callback, wandb_callback],
        reset_num_timesteps=True
    )

    # Save normalization stats
    if use_normalize:
        stats = {
            'obs_mean': env.obs_rms.mean,
            'obs_var': env.obs_rms.var,
            'obs_count': env.obs_rms.count,
            'ret_mean': env.ret_rms.mean,
            'ret_var': env.ret_rms.var,
        }
        np.save(f"{save_dir}/{run_name}/norm_stats.npy", stats)
        env.save(f"{save_dir}/{run_name}/vecnormalize.pkl")

    # Save the final model
    final_model_path = os.path.join(save_dir, f"{run_name}_trained_model")
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")

    # Final evaluation
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=config['n_eval_episodes'])
    print(f"Final evaluation: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # Log final metrics
    run.log({
        "final/mean_reward": mean_reward,
        "final/std_reward": std_reward,
        "config/seed": training_seed,
        "config/num_agents": num_agents
    })

    # Cleanup
    env.close()
    eval_env.close()
    run.finish()

    print(f"✓ Completed training: {run_name}")
    return final_model_path


def train_rl_teammates(
        config_filename='configs/june23_poc1.json',
        num_agents_to_train=7,
        seed_list=None,
        n_envs=None,
        project_name='maisr-rl-teammates',
        use_normalize=True,
        save_dir="./trained_models/teammates/",
        log_dir="./logs/teammates/"
):
    """
    Main function to train a collection of RL teammate agents.

    Args:
        config_filename: Path to the environment configuration file
        num_agents_to_train: Number of agents to train for each configuration
        seed_list: List of seeds to use for training (must have at least num_agents_to_train seeds)
        n_envs: Number of parallel environments (defaults to CPU count - 2)
        project_name: WandB project name
        use_normalize: Whether to use observation normalization
        save_dir: Directory to save trained models
        log_dir: Directory to save logs
    """

    print(f'\n############################ STARTING RL TEAMMATES TRAINING ############################')

    # Default seed list if not provided
    if seed_list is None:
        seed_list = [42, 123, 456, 789, 1337, 2048, 9999]

    # Validate seed list length
    if len(seed_list) < num_agents_to_train:
        raise ValueError(f"seed_list must contain at least {num_agents_to_train} seeds, got {len(seed_list)}")

    # Set default n_envs
    if n_envs is None:
        n_envs = max(1, multiprocessing.cpu_count() - 2)

    # Load base configuration
    config = load_env_config(config_filename)
    config['n_envs'] = n_envs
    config['config_filename'] = config_filename

    # Determine machine name for WandB logging
    machine_name = (
        'home' if socket.gethostname() == 'DESKTOP-3Q1FTUP'
        else 'lab_pc' if socket.gethostname() == 'isye-ae-2023pc3'
        else 'pace'
    )

    print(f'Machine: {machine_name}')
    print(f'Using {n_envs} parallel environments')
    print(f'Training {num_agents_to_train} agents for 1-agent config')
    print(f'Training {num_agents_to_train} agents for 2-agent config')
    print(f'Seeds: {seed_list[:num_agents_to_train]}')

    trained_models = []

    # Train agents with 1 aircraft
    print(f'\n=== PHASE 1: Training {num_agents_to_train} agents with 1 aircraft ===')
    for i in range(num_agents_to_train):
        seed = seed_list[i]
        try:
            model_path = train_single_teammate(
                env_config=config,
                training_seed=seed,
                num_agents=1,
                n_envs=n_envs,
                project_name=project_name,
                use_normalize=use_normalize,
                save_dir=save_dir,
                log_dir=log_dir,
                machine_name=machine_name
            )
            trained_models.append({
                'path': model_path,
                'seed': seed,
                'num_agents': 1,
                'phase': 1
            })
        except Exception as e:
            print(f"❌ Failed to train agent {i + 1}/7 (1-agent, seed {seed}): {e}")
            continue

    # Train agents with 2 aircraft
    print(f'\n=== PHASE 2: Training {num_agents_to_train} agents with 2 aircraft ===')
    for i in range(num_agents_to_train):
        seed = seed_list[i]
        try:
            model_path = train_single_teammate(
                env_config=config,
                training_seed=seed,
                num_agents=2,
                n_envs=n_envs,
                project_name=project_name,
                use_normalize=use_normalize,
                save_dir=save_dir,
                log_dir=log_dir,
                machine_name=machine_name
            )
            trained_models.append({
                'path': model_path,
                'seed': seed,
                'num_agents': 2,
                'phase': 2
            })
        except Exception as e:
            print(f"❌ Failed to train agent {i + 1}/7 (2-agent, seed {seed}): {e}")
            continue

    print(f'\n############################ TRAINING COMPLETE ############################')
    print(f'Successfully trained {len(trained_models)} agents total:')

    # Summary of trained models
    phase1_count = sum(1 for m in trained_models if m['phase'] == 1)
    phase2_count = sum(1 for m in trained_models if m['phase'] == 2)

    print(f'  - Phase 1 (1 agent): {phase1_count}/{num_agents_to_train} agents')
    print(f'  - Phase 2 (2 agents): {phase2_count}/{num_agents_to_train} agents')

    print(f'\nTrained model paths:')
    for model_info in trained_models:
        print(f"  - {model_info['path']} (seed {model_info['seed']}, {model_info['num_agents']} agents)")

    return trained_models


if __name__ == "__main__":
    # Configuration parameters
    num_agents_to_train = 7
    seed_list = [42, 123, 456, 789, 1337, 2048, 9999]
    config_filename = 'configs/june23_poc1_2ship.json'

    # Train the teammates
    trained_models = train_rl_teammates(
        config_filename=config_filename,
        num_agents_to_train=num_agents_to_train,
        seed_list=seed_list,
        n_envs=multiprocessing.cpu_count() - 1,  # Leave 2 CPUs free
        project_name='maisr-rl-teammates',
        use_normalize=True,
        save_dir="./trained_models/teammates/",
        log_dir="./logs/teammates/"
    )

    print(f"\n✅ All training completed! Trained {len(trained_models)} total agents.")