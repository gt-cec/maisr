"""
Max-Entropy Population Training for MAISR

Trains a population of independent PPO agents with elevated entropy coefficient
to produce a diverse pool of behavioral policies. Each agent trains independently
with a different seed; checkpoints for all agents are saved at evenly spaced intervals.

Adapted from pbt_model_pool_entropy_parallel.py (TF/Overcooked) into SB3/MAISR,
using train_revised.py as the structural template.

Usage:
    python training/maxentropy/train_maxent.py --testing
    python training/maxentropy/train_maxent.py --total_timesteps 3e6 --population_size 6
"""

import copy
import os
import argparse
import multiprocessing
import socket
from datetime import datetime

import numpy as np
import torch
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from base_env import MaisrEnv
from utility.localsearch_training_wrapper import MaisrLocalSearchWrapper
from utility.league_management import TeammateManager, LocalSearch, GoToNearestThreat, ChangeRegions
from utility.config_management import load_env_config
from utility.callbacks import EnhancedWandbCallback


def create_agent_env(env_config, n_envs, agent_idx, run_name, seed):
    """
    Create training and eval environments for a single agent.

    Follows train_revised.py:225-320 patterns:
        make_wrapped_env closure: MaisrEnv -> MaisrLocalSearchWrapper -> Monitor
        Vectorize: SubprocVecEnv (n_envs>1) or DummyVecEnv (n_envs==1)
        Wrap: VecMonitor -> VecNormalize
        Eval env: single DummyVecEnv with shared obs_rms/ret_rms

    Returns:
        (env, eval_env, teammate_manager)
    """
    agent_seed = seed + agent_idx * 100
    agent_run_name = f"{run_name}/agent_{agent_idx}"

    # Setup teammate manager
    teammate_manager = TeammateManager(
        league_type=env_config['league_type'],
        balance_method=env_config['balance_method'],
        selfplay_checkpoint_dir=f"outputs/{agent_run_name}/checkpoints",
        pretrained_teammate_dir='trained_models/pretrained_teammates',
        overfit_test=None,
    )

    def make_wrapped_env(env_config, rank, seed, run_name='no_name', save_episode_plots=True):
        def _init():
            if rank != 0:
                import sys
                sys.stdout = open(os.devnull, 'w')

            base_env = MaisrEnv(
                config=env_config,
                render_mode='headless',
                run_name=run_name,
                tag=f'train_mp{rank}',
                seed=seed + rank,
                save_episode_plots=save_episode_plots
            )

            local_search_policy = LocalSearch()
            go_to_highvalue_policy = GoToNearestThreat(model_path=None)
            change_region_subpolicy = ChangeRegions(model_path=None)
            evade_policy = None

            wrapped_env = MaisrLocalSearchWrapper(
                base_env,
                env_config['obs_noise_std_localsearch'],
                local_search_policy,
                go_to_highvalue_policy,
                change_region_subpolicy,
                evade_policy,
                teammate_manager=teammate_manager
            )

            wrapped_env = Monitor(wrapped_env)
            wrapped_env.reset()
            return wrapped_env

        return _init

    # Instantiate main env
    env_fns = [
        make_wrapped_env(env_config, i, agent_seed + i, run_name=agent_run_name, save_episode_plots=True)
        for i in range(n_envs)
    ]
    if n_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    env = VecMonitor(env, filename=f'outputs/{agent_run_name}/logs/{run_name}_agent{agent_idx}_vecmonitor')
    env = VecNormalize(env)
    env.training = True
    env.norm_reward = True

    # Create eval env
    base_eval_env = MaisrEnv(
        env_config, None, render_mode='headless',
        tag='eval', run_name=agent_run_name, save_episode_plots=True
    )
    eval_env = MaisrLocalSearchWrapper(
        base_eval_env,
        env_config['obs_noise_std_localsearch'],
        LocalSearch(model_path=None),
        GoToNearestThreat(model_path=None),
        ChangeRegions(model_path=None),
        None,
        teammate_manager=teammate_manager
    )
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(eval_env, norm_reward=False, training=False)
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms

    return env, eval_env, teammate_manager


def create_agent_model(env_config, env, agent_idx, seed, tb_log_dir):
    """
    Create a PPO model for a single agent.

    Follows train_revised.py:365-388 PPO config with elevated entropy coefficient.

    Returns:
        PPO model
    """
    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(
            pi=[env_config['network_size']] * env_config['network_numlayers'],
            vf=[env_config['network_size']] * env_config['network_numlayers']
        )
    )

    model = PPO(
        "CnnPolicy" if env_config['obs_type'] == 'pixel' else "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=2,
        tensorboard_log=tb_log_dir,
        batch_size=env_config['batch_size'],
        n_steps=env_config['ppo_update_steps'],
        learning_rate=env_config['lr'],
        seed=seed + agent_idx,
        device='cpu',
        gamma=env_config['gamma'],
        ent_coef=env_config['entropy_regularization'],
        clip_range=env_config['clip_range']
    )

    return model


def setup_callbacks(env_config, eval_env, agent_idx, run_name, run, n_envs,
                    total_timesteps, num_checkpoints, teammate_manager):
    """
    Create training callbacks for a single agent.

    Returns:
        list of callbacks
    """
    checkpoint_interval = int(total_timesteps / num_checkpoints)
    save_freq = max(checkpoint_interval // n_envs, 1)
    agent_run_name = f"{run_name}/agent_{agent_idx}"

    wandb_callback = WandbCallback(gradient_save_freq=50, verbose=1, model_save_path=None)

    # Override config to disable entropy decay and curriculum for max-entropy training
    maxent_config = copy.deepcopy(env_config)
    maxent_config['use_entropy_decay_schedule'] = False
    maxent_config['use_curriculum'] = False
    maxent_config['run_human_eval'] = False

    enhanced_wandb_callback = EnhancedWandbCallback(
        maxent_config,
        eval_env=eval_env,
        human_eval_env=None,
        run=run,
        log_freq=75,
        teammate_manager=teammate_manager
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=f"outputs/{agent_run_name}/checkpoints",
        name_prefix=f"agent{agent_idx}_checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    callbacks = [wandb_callback, enhanced_wandb_callback, checkpoint_callback]
    return callbacks


def train_single_agent(agent_idx, env_config, args, run_name):
    """
    Train a single agent in the population.

    Orchestrates the full lifecycle:
        1. Init WandB run grouped with other agents
        2. Create environments
        3. Create PPO model
        4. Wire up teammate manager
        5. Setup callbacks
        6. Save initial checkpoint
        7. Train
        8. Save final model/stats
        9. Evaluate
        10. Cleanup
    """
    agent_run_name = f"{run_name}/agent_{agent_idx}"
    total_timesteps = int(args.total_timesteps)
    machine = socket.gethostname()

    print(f'\n{"=" * 80}')
    print(f'  TRAINING AGENT {agent_idx} / {args.population_size - 1}')
    print(f'  Run: {agent_run_name}')
    print(f'  Seed: {args.seed + agent_idx * 100}')
    print(f'  ent_coef: {env_config["entropy_regularization"]}')
    print(f'  Total timesteps: {total_timesteps}')
    print(f'{"=" * 80}\n')

    # Create output directories
    for subfolder in ['trained_models', 'checkpoints', 'vecnorm_stats', 'logs']:
        os.makedirs(f"outputs/{agent_run_name}/{subfolder}", exist_ok=True)

    # Init WandB
    run = wandb.init(
        project=args.project_name,
        name=f"agent_{agent_idx}_{machine}_{args.n_envs}envs",
        group=run_name,
        config=env_config,
        sync_tensorboard=True,
        monitor_gym=True,
        reinit=True,
    )
    run.log_code(".")

    # Create environments
    env, eval_env, teammate_manager = create_agent_env(
        env_config, args.n_envs, agent_idx, run_name, args.seed
    )
    print(f'  [Agent {agent_idx}] Environments created')

    # Create model
    tb_log_dir = f"outputs/logs/tb_runs/{run.id}"
    model = create_agent_model(env_config, env, agent_idx, args.seed, tb_log_dir)
    print(f'  [Agent {agent_idx}] Model instantiated')
    print(model.policy)

    # Wire up teammate manager
    teammate_manager.set_current_model(model)
    if hasattr(env, 'obs_rms'):
        teammate_manager.set_normalization_stats(env.obs_rms, env.ret_rms)
    teammate_manager._create_selfplay_teammate()
    teammate_manager.current_teammate.env = env

    # Setup callbacks
    callbacks = setup_callbacks(
        env_config, eval_env, agent_idx, run_name, run,
        args.n_envs, total_timesteps, args.num_checkpoints, teammate_manager
    )
    print(f'  [Agent {agent_idx}] Callbacks created')

    # Save initial checkpoint
    initial_path = f"outputs/{agent_run_name}/checkpoints/agent{agent_idx}_checkpoint_0_steps.zip"
    vecnorm_path = f"outputs/{agent_run_name}/checkpoints/agent{agent_idx}_checkpoint_vecnormalize_0_steps.pkl"
    model.save(initial_path)
    if isinstance(env, VecNormalize):
        env.save(vecnorm_path)
    print(f'  [Agent {agent_idx}] Initial checkpoint saved')

    run.log({"curriculum/difficulty_level": 0}, step=0)

    # Train
    print(f'\n  [Agent {agent_idx}] Starting model.learn({total_timesteps} timesteps)...\n')
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
    )

    # Save final model and normalization stats
    print(f'\n  [Agent {agent_idx}] Training complete. Saving final model...')
    try:
        stats = {
            'obs_mean': env.obs_rms.mean,
            'obs_var': env.obs_rms.var,
            'obs_count': env.obs_rms.count,
            'ret_mean': env.ret_rms.mean,
            'ret_var': env.ret_rms.var,
        }
        np.save(f"outputs/{agent_run_name}/trained_models/agent{agent_idx}_norm_stats.npy", stats)
        env.save(f"outputs/{agent_run_name}/vecnorm_stats/agent{agent_idx}_vecnormalize.pkl")

        final_model_path = f"outputs/{agent_run_name}/trained_models/agent{agent_idx}_model.zip"
        model.save(final_model_path)
        print(f'  [Agent {agent_idx}] Final model saved to {final_model_path}')
    except Exception as e:
        print(f'  [Agent {agent_idx}] Failed to save model/stats: {e}')

    # Evaluate
    print(f'  [Agent {agent_idx}] Running final evaluation...')
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=env_config['n_eval_episodes'])
    print(f'  [Agent {agent_idx}] Final eval: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}')
    run.log({"final/mean_reward": mean_reward, "final/std_reward": std_reward})

    # Cleanup
    env.close()
    eval_env.close()
    run.finish()

    print(f'  [Agent {agent_idx}] Done.\n')
    return final_model_path if 'final_model_path' in dir() else None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='MAISR Max-Entropy Population Training')
    parser.add_argument('--total_timesteps', type=float, default=3e6,
                        help='Total timesteps per agent')
    parser.add_argument('--num_checkpoints', type=int, default=6,
                        help='Number of evenly spaced checkpoints per agent')
    parser.add_argument('--population_size', type=int, default=6,
                        help='Number of agents in the population')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--testing', action='store_true',
                        help='Reduced timesteps/envs for debugging')
    parser.add_argument('--n_envs', type=int, default=None,
                        help='Parallel envs per agent (default: cpu_count)')
    parser.add_argument('--ent_coef', type=float, default=0.5,
                        help='Max-entropy coefficient')
    parser.add_argument('--config', type=str, default='configs/main_config.json',
                        help='Path to config JSON file')
    parser.add_argument('--project_name', type=str, default='maisr-maxent-population',
                        help='WandB project name')
    args = parser.parse_args()

    # Handle defaults
    if args.n_envs is None:
        args.n_envs = multiprocessing.cpu_count()

    # Testing overrides
    if args.testing:
        args.total_timesteps = 5e5
        args.num_checkpoints = 3
        args.n_envs = 2
        args.project_name = 'maisr-maxent-tests'

    # Load config
    env_config = load_env_config(args.config)
    env_config['seed'] = args.seed
    env_config['n_envs'] = args.n_envs
    env_config['entropy_regularization'] = args.ent_coef

    # Generate run name
    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_name = f"maxent_pop{args.population_size}_ent{args.ent_coef}_seed{args.seed}_{timestamp}"

    print(f'\n{"#" * 80}')
    print(f'  MAISR Max-Entropy Population Training')
    print(f'  Population size: {args.population_size}')
    print(f'  Total timesteps per agent: {int(args.total_timesteps)}')
    print(f'  Checkpoints per agent: {args.num_checkpoints}')
    print(f'  Entropy coefficient: {args.ent_coef}')
    print(f'  Envs per agent: {args.n_envs}')
    print(f'  Base seed: {args.seed}')
    print(f'  Run name: {run_name}')
    print(f'{"#" * 80}\n')

    # Create top-level output directory
    os.makedirs(f"outputs/{run_name}", exist_ok=True)

    # Train agents sequentially (each uses SubprocVecEnv which saturates CPUs)
    saved_models = []
    for agent_idx in range(args.population_size):
        model_path = train_single_agent(agent_idx, env_config, args, run_name)
        saved_models.append(model_path)

    # Summary
    print(f'\n{"#" * 80}')
    print(f'  ALL AGENTS TRAINED')
    print(f'{"#" * 80}')
    for idx, path in enumerate(saved_models):
        print(f'  Agent {idx}: {path}')
    print()
