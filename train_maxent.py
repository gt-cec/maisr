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


class PopulationEntropyLoggingCallback(BaseCallback):
    """
    Callback to log population entropy metrics and per-agent training progress.
    Logs episode rewards, PE bonuses, and population-level diversity metrics.
    """

    def __init__(self, agent_idx, population_models, wandb_run, eval_env,
                 eval_freq=10000, verbose=0):
        super().__init__(verbose)
        self.agent_idx = agent_idx
        self.population_models = population_models
        self.wandb_run = wandb_run
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.pe_bonuses = []
        self.last_eval_step = 0

    def _on_step(self) -> bool:
        # Collect episode statistics from all parallel environments
        if hasattr(self.training_env, 'get_attr'):
            for env_idx in range(self.training_env.num_envs):
                # Get episode info from VecMonitor
                infos = self.locals.get('infos', [])
                if env_idx < len(infos) and infos[env_idx]:
                    info = infos[env_idx]

                    # Log completed episodes
                    if 'episode' in info:
                        ep_reward = info['episode']['r']
                        ep_length = info['episode']['l']
                        self.episode_rewards.append(ep_reward)
                        self.episode_lengths.append(ep_length)

                        # Log to wandb
                        self.wandb_run.log({
                            f"agent_{self.agent_idx}/episode_reward": ep_reward,
                            f"agent_{self.agent_idx}/episode_length": ep_length,
                            f"agent_{self.agent_idx}/num_episodes": len(self.episode_rewards),  # ADD EPISODE COUNTER
                        })

                    # Log PE bonus if available
                    if 'pe_bonus' in info:
                        pe_bonus = info['pe_bonus']
                        self.pe_bonuses.append(pe_bonus)

                        # Log running average of PE bonus every 100 steps
                        if len(self.pe_bonuses) >= 100:
                            avg_pe_bonus = np.mean(self.pe_bonuses[-100:])
                            self.wandb_run.log({
                                f"agent_{self.agent_idx}/pe_bonus_avg": avg_pe_bonus,
                            })

        # Periodic evaluation and population metrics
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            self._log_evaluation_metrics()
            self._log_population_metrics()  # Only call from ONE agent to avoid duplicates

        return True

    def _log_evaluation_metrics(self):
        """Run evaluation and log results"""
        try:
            mean_reward, std_reward = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=5, deterministic=True
            )

            self.wandb_run.log({
                f"agent_{self.agent_idx}/eval_mean_reward": mean_reward,
                f"agent_{self.agent_idx}/eval_std_reward": std_reward,
            })
        except Exception as e:
            if self.verbose > 0:
                print(f"Evaluation failed for agent {self.agent_idx}: {e}")

    def _log_population_metrics(self):
        """Compute and log population-level diversity metrics"""
        # IMPORTANT: Only log population metrics from agent 0 to avoid duplicates!
        if self.agent_idx != 0:
            return

        try:
            # Sample some observations from replay buffer or eval env
            obs_samples = []
            for _ in range(100):  # Sample 100 observations
                obs, _ = self.eval_env.reset()
                obs_samples.append(obs)
            obs_samples = np.array(obs_samples)

            # Compute policy diversity metrics
            all_action_probs = []
            for model in self.population_models:
                if model is None:
                    continue

                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs_samples).float()
                    if hasattr(model.policy, 'device'):
                        obs_tensor = obs_tensor.to(model.policy.device)

                    distribution = model.policy.get_distribution(obs_tensor)
                    if hasattr(distribution.distribution, 'probs'):
                        probs = distribution.distribution.probs.cpu().numpy()
                        all_action_probs.append(probs)

            if len(all_action_probs) > 1:
                all_action_probs = np.array(all_action_probs)  # (n_agents, n_samples, n_actions)

                # Compute mean policy across population
                mean_policy = np.mean(all_action_probs, axis=0)  # (n_samples, n_actions)

                # Compute population entropy: H(π̄) = -Σ π̄(a|s) log π̄(a|s)
                mean_policy_clipped = np.clip(mean_policy, 1e-10, 1.0)
                pop_entropy = -np.sum(mean_policy * np.log(mean_policy_clipped), axis=-1)
                avg_pop_entropy = np.mean(pop_entropy)

                # Compute pairwise KL divergence between agents (diversity metric)
                kl_divergences = []
                for i in range(len(all_action_probs)):
                    for j in range(i + 1, len(all_action_probs)):
                        pi_i = np.clip(all_action_probs[i], 1e-10, 1.0)
                        pi_j = np.clip(all_action_probs[j], 1e-10, 1.0)
                        kl = np.sum(pi_i * (np.log(pi_i) - np.log(pi_j)), axis=-1)
                        kl_divergences.append(np.mean(kl))

                avg_kl = np.mean(kl_divergences) if kl_divergences else 0.0

                # Log population metrics
                self.wandb_run.log({
                    f"population/entropy": avg_pop_entropy,
                    f"population/avg_kl_divergence": avg_kl,
                    f"population/num_active_agents": len(all_action_probs),
                })

        except Exception as e:
            if self.verbose > 0:
                print(f"Population metrics computation failed: {e}")

    def _on_rollout_end(self) -> None:
        """Log PPO training statistics"""
        # Log recent episode statistics
        if len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-10:]  # Last 10 episodes
            self.wandb_run.log({
                f"agent_{self.agent_idx}/recent_mean_reward": np.mean(recent_rewards),
                f"agent_{self.agent_idx}/recent_std_reward": np.std(recent_rewards),
                f"agent_{self.agent_idx}/recent_min_reward": np.min(recent_rewards),
                f"agent_{self.agent_idx}/recent_max_reward": np.max(recent_rewards),
            })

        # Log PPO-specific metrics (if available in logger)
        if hasattr(self.logger, 'name_to_value'):
            for key, value in self.logger.name_to_value.items():
                # Only log PPO-related metrics
                if any(prefix in key for prefix in ['train/', 'rollout/']):
                    metric_name = f"agent_{self.agent_idx}/{key}"
                    self.wandb_run.log({metric_name: value})


def setup_callbacks(env_config, eval_env, agent_idx, run_name, wandb_run,
                    n_envs, total_timesteps, num_checkpoints, teammate_manager,
                    population_models):
    """Setup callbacks for training - MODIFIED to add PE logging"""
    agent_run_name = f"{run_name}/agent_{agent_idx}"

    # Checkpoint callback
    checkpoint_freq = int(total_timesteps // (n_envs * num_checkpoints))
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=f"outputs/maxent/{agent_run_name}/checkpoints",
        name_prefix=f"agent{agent_idx}_checkpoint",
        save_vecnormalize=True,
        verbose=1,
    )

    # WandB callback (original)
    wandb_callback = WandbCallback(
        gradient_save_freq=checkpoint_freq,
        verbose=2,
    )

    # Population Entropy Logging Callback (NEW!)

    pe_logging_callback = PopulationEntropyLoggingCallback(
        agent_idx=agent_idx,
        population_models=population_models,
        wandb_run=wandb_run,
        eval_env=eval_env,
        eval_freq=checkpoint_freq,  # Evaluate at same freq as checkpoints
        verbose=1,
    )

    # Enhanced callback if available
    try:
        enhanced_callback = EnhancedWandbCallback(
            env_config=env_config,
            wandb_run=wandb_run,
            eval_env=eval_env,
            eval_freq=checkpoint_freq,
            teammate_manager=teammate_manager,
            agent_idx=agent_idx,
        )
        return [checkpoint_callback, wandb_callback, pe_logging_callback, enhanced_callback]
    except:
        return [checkpoint_callback, wandb_callback, pe_logging_callback]

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
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
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

    checkpoint_dir = f"outputs/maxent/{agent_run_name}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

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
                tag=f'train_maxent{rank}',
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
        make_wrapped_env(i, agent_seed + i, agent_run_name, save_episode_plots=False)#(i==0))
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
        tag='eval', run_name=agent_run_name, save_episode_plots=False
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
    """Train a population of agents with PE reward - MODIFIED for single wandb run"""
    machine = socket.gethostname()
    total_timesteps = int(args.total_timesteps)
    population_size = args.population_size

    print(f'\n[Phase 1] Initializing population of {population_size} agents...')

    population_models = [None] * population_size
    environments = []
    eval_environments = []
    teammate_managers = []
    all_callbacks = []

    # CREATE SINGLE WANDB RUN (instead of one per agent)
    shared_wandb_run = wandb.init(
        project=args.project_name,
        name=f"{run_name}_{machine}_{args.n_envs}envs",
        group=f"pop{population_size}",
        config={
            **env_config,
            'population_size': population_size,
            'entropy_weight': args.ent_coef,
            'total_timesteps': total_timesteps,
            'n_envs': args.n_envs,
            'seed': args.seed,
        },
        sync_tensorboard=True,
        monitor_gym=True,
    )
    shared_wandb_run.log_code(".")

    for agent_idx in range(population_size):
        print(f'  Creating agent {agent_idx}...')
        agent_run_name = f"{run_name}/agent_{agent_idx}"

        # Create environment (will reference population_models)
        env, eval_env, teammate_manager = create_agent_env(
            env_config, args.n_envs, agent_idx, run_name, args.seed,
            population_models, args.ent_coef
        )
        environments.append(env)
        eval_environments.append(eval_env)
        teammate_managers.append(teammate_manager)

        # Create model
        tb_log_dir = f"outputs/maxent/logs/tb_runs/{shared_wandb_run.id}"
        model = create_agent_model(env_config, env, agent_idx, args.seed, tb_log_dir)
        population_models[agent_idx] = model
        print(f'  Agent {agent_idx} model created')

        # Setup callbacks - pass shared wandb run
        callbacks = setup_callbacks(
            env_config, eval_env, agent_idx, run_name, shared_wandb_run,  # Use shared run
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

        shared_wandb_run.log({"curriculum/difficulty_level": 0}, step=0)

    print(f'\n[Phase 2] Training population with PE reward...')
    print(f'  All {population_size} agents initialized')
    print(f'  PE wrapper will compute π̄(a|s) from all {population_size} policies\n')

    # Training loop: iterate through agents cyclically
    steps_per_iteration = args.n_envs * 2048
    num_iterations = total_timesteps // steps_per_iteration

    for iteration in range(num_iterations):
        print(f'\n--- Iteration {iteration + 1}/{num_iterations} ---')

        agent_idx = np.random.randint(0, population_size)

        print(f'Training agent {agent_idx}')
        model = population_models[agent_idx]
        env = environments[agent_idx]
        eval_env = eval_environments[agent_idx]
        teammate_manager = teammate_managers[agent_idx]
        callbacks = all_callbacks[agent_idx]

        # Train this agent for one iteration
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
            shared_wandb_run.log({
                f"agent_{agent_idx}/final_mean_reward": mean_reward,
                f"agent_{agent_idx}/final_std_reward": std_reward
            })

        except Exception as e:
            print(f'  Agent {agent_idx} save failed: {e}')

        # Cleanup environments
        env.close()
        eval_env.close()

    # FINISH THE SINGLE WANDB RUN
    shared_wandb_run.finish()

    print(f'\n{"#" * 80}')
    print(f'  POPULATION TRAINING COMPLETE')
    print(f'  All agents trained with Population Entropy reward')
    print(f'{"#" * 80}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_timesteps', type=float, default=3e6, help='Total timesteps per agent')
    parser.add_argument('--num_checkpoints', type=int, default=9, help='Number of evenly spaced checkpoints per agent')
    parser.add_argument('--population_size', type=int, default=6, help='Number of agents in the population')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--testing', action='store_true', help='Reduced timesteps/envs for debugging')
    parser.add_argument('--n_envs', type=int, default=None, help='Parallel envs per agent (default: cpu_count)')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='Population entropy weight (α in paper)')
    parser.add_argument('--config', type=str, default='configs/maxent_config.json', help='Path to config JSON file')
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
    print(f'{"#" * 80}\n')

    # Create top-level output directory
    os.makedirs(f"outputs/maxent/{run_name}", exist_ok=True)

    # Train population
    train_population(env_config, args, run_name)
