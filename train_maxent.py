"""
Max-Entropy Population Training for MAISR - CORRECTED IMPLEMENTATION

This implements Phase 1 of the MEP paper (Zhao et al. 2022):
Training a population of agents with Population Entropy (PE) reward.

Key differences from original train_maxent.py:
1. Agents train in parallel (not sequentially) 
2. Implements actual PE reward: r = task_reward - α log(π̄(a|s))
   where π̄(a|s) = (1/n) Σ π^(i)(a|s) is the mean policy
3. All agents update using the centralized PE bonus

Usage:
    python train_maxent_corrected.py --testing
    python train_maxent_corrected.py --total_timesteps 3e6 --population_size 6
"""

import copy
import os
import argparse
import multiprocessing
import socket
from datetime import datetime
from typing import List
import numpy as np
import torch
import wandb
import gymnasium as gym
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize, VecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from base_env import MaisrEnv
from utility.localsearch_training_wrapper import MaisrLocalSearchWrapper
from utility.league_management import TeammateManager, LocalSearch, GoToNearestThreat, ChangeRegions
from utility.config_management import load_env_config
from utility.callbacks import EnhancedWandbCallback


class PopulationEntropyVecWrapper(gym.Wrapper):
    """
    Environment wrapper that adds Population Entropy bonus to rewards.
    
    Implements the PE reward from MEP paper:
        augmented_reward = task_reward - α log(π̄(a|s))
    where π̄(a|s) = (1/n) Σ π^(i)(a|s) is the mean action probability across population.
    """
    
    def __init__(self, env, population_models, entropy_weight, current_agent_idx):
        super().__init__(env)
        self.population_models = population_models  # List of PPO models
        self.entropy_weight = entropy_weight  # α parameter
        self.current_agent_idx = current_agent_idx
        self.last_obs = None
        
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_obs = obs
        return obs
    
    def step(self, action):
        obs, reward, truncated, terminated, info = self.env.step(action)
        
        # Compute PE bonus using observation from BEFORE action was taken
        if self.last_obs is not None:
            pe_bonus = self._compute_pe_bonus(self.last_obs, action)
            reward += pe_bonus
            info['pe_bonus'] = pe_bonus
        
        self.last_obs = obs
        return obs, reward, truncated, terminated, info
    
    def _compute_pe_bonus(self, obs, action):
        """Compute -α log(π̄(a|s))"""
        try:
            # Get action probabilities from all agents in population
            action_probs_list = []
            
            for model in self.population_models:
                if model is None:
                    continue
                    
                with torch.no_grad():
                    # Convert obs to tensor
                    obs_tensor = torch.as_tensor(obs).float().unsqueeze(0)
                    if hasattr(model.policy, 'device'):
                        obs_tensor = obs_tensor.to(model.policy.device)
                    
                    # Get action distribution from policy
                    distribution = model.policy.get_distribution(obs_tensor)
                    
                    # Get probabilities for discrete action space
                    if hasattr(distribution.distribution, 'probs'):
                        probs = distribution.distribution.probs.cpu().numpy()[0]
                    else:
                        # For continuous actions, would need different approach
                        return 0.0
                    
                    action_probs_list.append(probs)
            
            if len(action_probs_list) == 0:
                return 0.0
            
            # Compute mean policy π̄(a|s) = (1/n) Σ π^(i)(a|s)
            mean_policy = np.mean(action_probs_list, axis=0)
            
            # Get probability of taken action under mean policy
            mean_prob = mean_policy[action]
            
            # PE bonus: -α log(π̄(a|s))
            # Clip to avoid log(0)
            pe_bonus = -self.entropy_weight * np.log(np.clip(mean_prob, 1e-10, 1.0))
            
            return float(pe_bonus)
            
        except Exception as e:
            print(f"Error computing PE bonus: {e}")
            return 0.0


def wrap_env_with_pe(env, population_models, entropy_weight, agent_idx):
    """Wrap a single env with PE reward"""
    return PopulationEntropyVecWrapper(env, population_models, entropy_weight, agent_idx)


def create_agent_env(env_config, n_envs, agent_idx, run_name, seed, population_models, entropy_weight):
    """
    Create training and eval environments for a single agent.
    Wraps environment with PopulationEntropyVecWrapper to add PE reward.
    """
    agent_seed = seed + agent_idx * 100
    agent_run_name = f"{run_name}/agent_{agent_idx}"

    # Setup teammate manager
    teammate_manager = TeammateManager(
        league_type=env_config['league_type'],
        balance_method=env_config['balance_method'],
        selfplay_checkpoint_dir=f"outputs/maxent/{agent_run_name}/checkpoints",
        pretrained_teammate_dir='trained_models/pretrained_teammates',
        overfit_test=None,
    )

    def make_wrapped_env(rank, seed, run_name, save_episode_plots=True):
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

            # Add PE wrapper BEFORE Monitor
            wrapped_env = PopulationEntropyVecWrapper(
                wrapped_env, 
                population_models, 
                entropy_weight, 
                agent_idx
            )
            
            wrapped_env = Monitor(wrapped_env)
            wrapped_env.reset()
            return wrapped_env

        return _init

    # Create vectorized environment
    env_fns = [
        make_wrapped_env(i, agent_seed + i, agent_run_name, save_episode_plots=(i==0))
        for i in range(n_envs)
    ]
    
    if n_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    env = VecMonitor(env, filename=f'outputs/maxent/{agent_run_name}/logs/{run_name}_agent{agent_idx}_vecmonitor')
    env = VecNormalize(env)
    env.training = True
    env.norm_reward = True

    # Create eval env (without PE wrapper for clean evaluation)
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
    
    NOTE: We use standard PPO ent_coef for individual entropy.
    The population entropy comes from the PE reward wrapper.
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
        # Use lower ent_coef since PE reward handles diversity
        ent_coef=0.01,  
        clip_range=env_config['clip_range']
    )

    return model


class PopulationEntropyLogCallback(BaseCallback):
    """Callback to log population entropy metrics"""
    
    def __init__(self, population_models, agent_idx, verbose=0):
        super().__init__(verbose)
        self.population_models = population_models
        self.agent_idx = agent_idx
        
    def _on_step(self) -> bool:
        # Log PE bonus from info if available
        if len(self.locals.get('infos', [])) > 0:
            pe_bonuses = [info.get('pe_bonus', 0.0) for info in self.locals['infos']]
            if pe_bonuses:
                mean_pe = np.mean([pb for pb in pe_bonuses if pb != 0.0] or [0.0])
                self.logger.record(f'agent_{self.agent_idx}/mean_pe_bonus', mean_pe)
        
        return True


def setup_callbacks(env_config, eval_env, agent_idx, run_name, run, n_envs,
                    total_timesteps, num_checkpoints, teammate_manager, population_models):
    """Create training callbacks for a single agent."""
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
        save_path=f"outputs/maxent/{agent_run_name}/checkpoints",
        name_prefix=f"agent{agent_idx}_checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    pe_log_callback = PopulationEntropyLogCallback(population_models, agent_idx)

    callbacks = [wandb_callback, enhanced_wandb_callback, checkpoint_callback, pe_log_callback]
    return callbacks


def train_agent_iteration(agent_idx, model, env, eval_env, teammate_manager, 
                         callbacks, steps_per_iteration, iteration):
    """Train a single agent for one iteration"""
    print(f'  [Agent {agent_idx}] Iteration {iteration}: training {steps_per_iteration} steps')
    
    model.learn(
        total_timesteps=steps_per_iteration,
        callback=callbacks,
        reset_num_timesteps=False,
    )
    
    return model


def train_population(env_config, args, run_name):
    """
    Train population with Population Entropy reward.
    
    Algorithm:
    1. Initialize all agents
    2. For each iteration:
       - Each agent collects rollouts (PE reward computed using all current policies)
       - Each agent updates its policy
    3. Save checkpoints
    """
    total_timesteps = int(args.total_timesteps)
    population_size = args.population_size
    machine = socket.gethostname()
    
    print(f'\n{"=" * 80}')
    print(f'  MAXIMUM ENTROPY POPULATION TRAINING - CORRECTED')
    print(f'  Population size: {population_size}')
    print(f'  PE weight (α): {args.ent_coef}')
    print(f'  Total timesteps per agent: {total_timesteps}')
    print(f'  Run: {run_name}')
    print(f'{"=" * 80}\n')

    # Create output directories
    for agent_idx in range(population_size):
        agent_run_name = f"{run_name}/agent_{agent_idx}"
        for subfolder in ['trained_models', 'checkpoints', 'vecnorm_stats', 'logs']:
            os.makedirs(f"outputs/maxent/{agent_run_name}/{subfolder}", exist_ok=True)

    # Initialize all models (need to create them all upfront for PE computation)
    print('\n[Phase 1] Initializing all agents...')
    population_models = [None] * population_size
    environments = []
    eval_environments = []
    teammate_managers = []
    wandb_runs = []
    all_callbacks = []
    
    for agent_idx in range(population_size):
        print(f'  Creating agent {agent_idx}...')
        agent_run_name = f"{run_name}/agent_{agent_idx}"
        
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
        wandb_runs.append(run)
        
        # Create environment (will reference population_models)
        env, eval_env, teammate_manager = create_agent_env(
            env_config, args.n_envs, agent_idx, run_name, args.seed,
            population_models, args.ent_coef  # Pass PE parameters
        )
        environments.append(env)
        eval_environments.append(eval_env)
        teammate_managers.append(teammate_manager)
        
        # Create model
        tb_log_dir = f"outputs/maxent/logs/tb_runs/{run.id}"
        model = create_agent_model(env_config, env, agent_idx, args.seed, tb_log_dir)
        population_models[agent_idx] = model
        print(f'  Agent {agent_idx} model created')
        
        # Setup callbacks
        callbacks = setup_callbacks(
            env_config, eval_env, agent_idx, run_name, run,
            args.n_envs, total_timesteps, args.num_checkpoints, 
            teammate_manager, population_models
        )
        all_callbacks.append(callbacks)
        
        # Wire up teammate manager
        teammate_manager.set_current_model(model)
        if hasattr(env, 'obs_rms'):
            teammate_manager.set_normalization_stats(env.obs_rms, env.ret_rms)
        teammate_manager._create_selfplay_teammate()
        teammate_manager.current_teammate.env = env
        
        # Save initial checkpoint
        initial_path = f"outputs/maxent/{agent_run_name}/checkpoints/agent{agent_idx}_checkpoint_0_steps.zip"
        vecnorm_path = f"outputs/maxent/{agent_run_name}/checkpoints/agent{agent_idx}_checkpoint_vecnormalize_0_steps.pkl"
        model.save(initial_path)
        if isinstance(env, VecNormalize):
            env.save(vecnorm_path)
        
        run.log({"curriculum/difficulty_level": 0}, step=0)
    
    print(f'\n[Phase 2] Training population with PE reward...')
    print(f'  All {population_size} agents initialized')
    print(f'  PE wrapper will compute π̄(a|s) from all {population_size} policies\n')
    
    # Training loop: iterate through agents cyclically
    # Paper's Algorithm 1: "Sample agent from population"
    # We'll train each agent in round-robin fashion
    steps_per_iteration = args.n_envs * 2048  # ~1 PPO update
    num_iterations = total_timesteps // steps_per_iteration
    
    for iteration in range(num_iterations):
        print(f'\n--- Iteration {iteration + 1}/{num_iterations} ---')
        
        # Sample agent to train (uniform random)
        #agent_idx = iteration % population_size
        agent_idx = np.random.randint(0, population_size)  # Uniform random sampling
        
        print(f'Training agent {agent_idx}')
        model = population_models[agent_idx]
        env = environments[agent_idx]
        eval_env = eval_environments[agent_idx]
        teammate_manager = teammate_managers[agent_idx]
        callbacks = all_callbacks[agent_idx]
        
        # Train this agent for one iteration
        # The PE reward is automatically computed in the environment wrapper
        model = train_agent_iteration(
            agent_idx, model, env, eval_env, teammate_manager,
            callbacks, steps_per_iteration, iteration
        )
        
        # Update the population model reference
        population_models[agent_idx] = model
    
    # Save final models
    print(f'\n[Phase 3] Saving final models...')
    for agent_idx in range(population_size):
        agent_run_name = f"{run_name}/agent_{agent_idx}"
        model = population_models[agent_idx]
        env = environments[agent_idx]
        eval_env = eval_environments[agent_idx]
        run = wandb_runs[agent_idx]
        
        try:
            # Save normalization stats
            stats = {
                'obs_mean': env.obs_rms.mean,
                'obs_var': env.obs_rms.var,
                'obs_count': env.obs_rms.count,
                'ret_mean': env.ret_rms.mean,
                'ret_var': env.ret_rms.var,
            }
            np.save(f"outputs/maxent/{agent_run_name}/trained_models/agent{agent_idx}_norm_stats.npy", stats)
            env.save(f"outputs/maxent/{agent_run_name}/vecnorm_stats/agent{agent_idx}_vecnormalize.pkl")

            # Save model
            final_model_path = f"outputs/maxent/{agent_run_name}/trained_models/agent{agent_idx}_model.zip"
            model.save(final_model_path)
            print(f'  Agent {agent_idx} saved to {final_model_path}')
            
            # Evaluate
            mean_reward, std_reward = evaluate_policy(
                model, eval_env, n_eval_episodes=env_config['n_eval_episodes']
            )
            print(f'  Agent {agent_idx} final eval: {mean_reward:.2f} +/- {std_reward:.2f}')
            run.log({"final/mean_reward": mean_reward, "final/std_reward": std_reward})
            
        except Exception as e:
            print(f'  Agent {agent_idx} save failed: {e}')
        
        # Cleanup
        env.close()
        eval_env.close()
        run.finish()
    
    print(f'\n{"#" * 80}')
    print(f'  POPULATION TRAINING COMPLETE')
    print(f'  All agents trained with Population Entropy reward')
    print(f'{"#" * 80}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_timesteps', type=float, default=3e6, help='Total timesteps per agent')
    parser.add_argument('--num_checkpoints', type=int, default=6, help='Number of evenly spaced checkpoints per agent')
    parser.add_argument('--population_size', type=int, default=6, help='Number of agents in the population')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--testing', action='store_true', help='Reduced timesteps/envs for debugging')
    parser.add_argument('--n_envs', type=int, default=None, help='Parallel envs per agent (default: cpu_count)')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='Population entropy weight (α in paper)')
    parser.add_argument('--config', type=str, default='configs/main_config.json', help='Path to config JSON file')
    parser.add_argument('--project_name', type=str, default='maisr-mep-corrected', help='WandB project name')
    args = parser.parse_args()

    # Handle defaults
    if args.n_envs is None:
        args.n_envs = max(1, multiprocessing.cpu_count() // args.population_size)

    # Testing overrides
    if args.testing:
        args.total_timesteps = 1e5
        args.num_checkpoints = 3
        args.n_envs = 2
        args.population_size = 3
        args.project_name = 'maisr-mep-tests'

    # Load config
    env_config = load_env_config(args.config)
    env_config['seed'] = args.seed
    env_config['n_envs'] = args.n_envs

    # Generate run name
    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_name = f"mep_phase1_pop{args.population_size}_alpha{args.ent_coef}_seed{args.seed}_{timestamp}"

    print(f'\n{"#" * 80}')
    print(f'  MEP Phase 1: Maximum Entropy Population Training (CORRECTED)')
    print(f'  Population size: {args.population_size}')
    print(f'  Total timesteps per agent: {int(args.total_timesteps)}')
    print(f'  Population entropy weight (α): {args.ent_coef}')
    print(f'  Envs per agent: {args.n_envs}')
    print(f'  Base seed: {args.seed}')
    print(f'  Run name: {run_name}')
    print(f'  ')
    print(f'  KEY: This implementation adds PE reward -α log(π̄(a|s))')
    print(f'       where π̄ is the mean policy across all {args.population_size} agents')
    print(f'{"#" * 80}\n')

    # Create top-level output directory
    os.makedirs(f"outputs/maxent/{run_name}", exist_ok=True)

    # Train population
    train_population(env_config, args, run_name)
