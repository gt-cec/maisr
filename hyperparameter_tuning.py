import warnings
warnings.filterwarnings("ignore", message="Your system is avx2 capable but pygame was not built with support for it")

import optuna
import os
import numpy as np
import multiprocessing
import json
from datetime import datetime
from typing import Dict, Any, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from env_combined import MAISREnvVec
from utility.data_logging import load_env_config


class OptunaPruningCallback(BaseCallback):
    """
    Callback used for pruning unpromising trials with Optuna.
    
    Optuna's pruning feature automatically stops unpromising trials at the early stages
    of the training. This is useful for speeding up the hyperparameter optimization process.
    """
    
    def __init__(self, trial: optuna.trial.Trial, eval_env, n_eval_episodes: int = 5,
                 eval_freq: int = 10000, deterministic: bool = True, verbose: int = 0):
        super(OptunaPruningCallback, self).__init__(verbose)
        self.trial = trial
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.is_pruned = False
        
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate the agent
            mean_reward, _ = evaluate_policy(
                self.model, 
                self.eval_env, 
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic
            )
            
            # Report intermediate value to Optuna
            self.trial.report(mean_reward, self.n_calls)
            
            # Prune trial if needed
            if self.trial.should_prune():
                self.is_pruned = True
                return False
                
        return True


def make_env(env_config: Dict[str, Any], rank: int, seed: int, run_name: str = 'optuna_trial'):
    """
    Create a MAISR environment for vectorized training.
    """
    def _init():
        env = MAISREnvVec(
            config=env_config,
            render_mode='headless',
            run_name=run_name,
            tag=f'optuna_mp{rank}',
            seed=seed + rank,
        )
        env = Monitor(env)
        env.reset()
        return env
    return _init


def create_model(trial: optuna.trial.Trial, env_config: Dict[str, Any], 
                env, eval_env, n_envs: int = 1) -> PPO:
    """
    Create a PPO model with hyperparameters suggested by Optuna.
    
    Args:
        trial: Optuna trial object
        env_config: Environment configuration
        env: Training environment
        eval_env: Evaluation environment
        n_envs: Number of parallel environments
        
    Returns:
        PPO model with suggested hyperparameters
    """
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-5, log=True)
    
    # Learning rate schedule options
    lr_schedule_choice = trial.suggest_categorical("lr_schedule", ["constant", "linear"])
    
    if lr_schedule_choice == "linear":
        # For linear schedule, we'll use a lambda that decays linearly
        def linear_schedule(initial_value: float):
            def func(progress_remaining: float) -> float:
                return progress_remaining * initial_value
            return func
        lr_schedule = linear_schedule(learning_rate)
    else:
        lr_schedule = learning_rate
    
    # Batch size (should be <= n_steps * n_envs)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    
    # Number of epochs (n_epochs in SB3)
    n_epochs = trial.suggest_int("n_epochs", 3, 20)
    
    # Value function loss coefficient
    vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
    
    # Discount factor (gamma)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
    
    # Additional hyperparameters that often matter for PPO
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    
    # # Ensure batch_size <= n_steps * n_envs
    # max_batch_size = n_steps * n_envs
    # if batch_size > max_batch_size:
    #     batch_size = max_batch_size
    #     trial.set_user_attr("batch_size_adjusted", True)
    
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 2.0)
    
    # Network architecture
    net_arch_choice = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    
    net_arch_dict = {
        "small": [64, 64],
        "medium": [128, 128], 
        "large": [256, 256]
    }
    
    policy_kwargs = {
        "net_arch": net_arch_dict[net_arch_choice],
        "activation_fn": trial.suggest_categorical("activation_fn", ["tanh", "relu"]),
    }
    
    # Convert activation function string to actual function
    if policy_kwargs["activation_fn"] == "tanh":
        import torch.nn as nn
        policy_kwargs["activation_fn"] = nn.Tanh
    else:
        import torch.nn as nn
        policy_kwargs["activation_fn"] = nn.ReLU
    
    # Create the model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        lr_schedule = # TODO
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=env_config.get('seed', 42),
        device='cpu',
    )
    
    return model


def objective(trial: optuna.trial.Trial, 
              env_config_filename: str,
              n_envs: int = 1,
              n_timesteps: int = 100000,
              n_eval_episodes: int = 10,
              eval_freq: int = 10000) -> float:
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
        env_config_filename: Path to environment configuration file
        n_envs: Number of parallel environments
        n_timesteps: Total timesteps for training
        n_eval_episodes: Number of episodes for evaluation
        eval_freq: Frequency of evaluation during training
        
    Returns:
        Mean reward achieved by the agent
    """
    
    try:
        # Load environment configuration
        env_config = load_env_config(env_config_filename)
        
        # Generate a unique run name for this trial
        run_name = f"optuna_trial_{trial.number}_{datetime.now().strftime('%H%M%S')}"
        
        # Create environments
        if n_envs > 1:
            env_fns = [make_env(env_config, i, env_config['seed'] + i + trial.number * 1000, run_name) 
                      for i in range(n_envs)]
            env = SubprocVecEnv(env_fns)
            env = VecMonitor(env)
        else:
            env = MAISREnvVec(
                env_config,
                render_mode='headless',
                tag=f'optuna_trial_{trial.number}',
                run_name=run_name,
                seed=env_config['seed'] + trial.number
            )
            env = Monitor(env)
        
        # Create evaluation environment
        eval_env = MAISREnvVec(
            env_config,
            render_mode='headless',
            tag=f'optuna_eval_{trial.number}',
            run_name=run_name,
            seed=env_config['seed'] + trial.number + 10000
        )
        eval_env = Monitor(eval_env)
        
        # Create model with suggested hyperparameters
        model = create_model(trial, env_config, env, eval_env, n_envs)
        
        # Create pruning callback
        pruning_callback = OptunaPruningCallback(
            trial, eval_env, n_eval_episodes=n_eval_episodes, eval_freq=eval_freq
        )
        
        # Train the model
        model.learn(
            total_timesteps=n_timesteps,
            callback=pruning_callback,
            reset_num_timesteps=True
        )
        
        # Check if trial was pruned
        if pruning_callback.is_pruned:
            raise optuna.TrialPruned()
        
        # Final evaluation
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=True
        )
        
        # Clean up environments
        env.close()
        eval_env.close()
        
        return mean_reward
        
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        # Return a very low reward for failed trials
        return -np.inf


def optimize_hyperparameters(
    env_config_filename: str,
    study_name: str = "maisr_ppo_optimization",
    n_trials: int = 100,
    n_jobs: int = 1,
    n_envs: int = 1,
    n_timesteps: int = 100000,
    n_eval_episodes: int = 10,
    eval_freq: int = 10000,
    storage: Optional[str] = None,
    sampler: str = "tpe",
    pruner: str = "median",
    timeout: Optional[int] = None
) -> optuna.Study:
    """
    Run hyperparameter optimization using Optuna.
    
    Args:
        env_config_filename: Path to environment configuration file
        study_name: Name of the Optuna study
        n_trials: Number of trials to run
        n_jobs: Number of parallel jobs (should be 1 for RL to avoid issues)
        n_envs: Number of parallel environments per trial
        n_timesteps: Total timesteps for training each trial
        n_eval_episodes: Number of episodes for evaluation
        eval_freq: Frequency of evaluation during training
        storage: Database URL for storing study results (e.g., "sqlite:///optuna.db")
        sampler: Sampling algorithm ("tpe", "random", "cmaes")
        pruner: Pruning algorithm ("median", "hyperband", "none")
        timeout: Maximum time in seconds for the study
        
    Returns:
        Completed Optuna study object
    """
    
    # Create sampler
    if sampler == "tpe":
        sampler_obj = optuna.samplers.TPESampler()
    elif sampler == "random":
        sampler_obj = optuna.samplers.RandomSampler()
    elif sampler == "cmaes":
        sampler_obj = optuna.samplers.CmaEsSampler()
    else:
        raise ValueError(f"Unknown sampler: {sampler}")
    
    # Create pruner
    if pruner == "median":
        pruner_obj = optuna.pruners.MedianPruner()
    elif pruner == "hyperband":
        pruner_obj = optuna.pruners.HyperbandPruner()
    elif pruner == "none":
        pruner_obj = optuna.pruners.NopPruner()
    else:
        raise ValueError(f"Unknown pruner: {pruner}")
    
    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler_obj,
        pruner=pruner_obj,
        storage=storage,
        load_if_exists=True
    )
    
    print(f"Starting hyperparameter optimization with {n_trials} trials")
    print(f"Study name: {study_name}")
    print(f"Sampler: {sampler}, Pruner: {pruner}")
    print(f"Using {n_envs} environments per trial")
    print(f"Training for {n_timesteps} timesteps per trial")
    
    # Define the objective function with fixed parameters
    def objective_wrapper(trial):
        return objective(
            trial=trial,
            env_config_filename=env_config_filename,
            n_envs=n_envs,
            n_timesteps=n_timesteps,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq
        )
    
    # Optimize
    study.optimize(
        objective_wrapper,
        n_trials=n_trials,
        n_jobs=n_jobs,  # Keep at 1 for RL to avoid multiprocessing issues
        timeout=timeout,
        show_progress_bar=True
    )
    
    return study


def print_study_results(study: optuna.Study):
    """Print optimization results."""
    
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    
    if study.best_trial is not None:
        print(f"\nBest trial:")
        print(f"  Value: {study.best_trial.value:.4f}")
        print(f"  Params:")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")
        
        # Print parameter importance if available
        try:
            importance = optuna.importance.get_param_importances(study)
            print(f"\nParameter importance:")
            for key, value in importance.items():
                print(f"    {key}: {value:.4f}")
        except Exception as e:
            print(f"Could not compute parameter importance: {e}")
    else:
        print("No successful trials completed.")


def save_best_config(study: optuna.Study, env_config_filename: str, output_filename: str):
    """Save the best hyperparameters to a new config file."""
    
    if study.best_trial is None:
        print("No best trial found, cannot save config.")
        return
    
    # Load original config
    with open(env_config_filename, 'r') as f:
        config = json.load(f)
    
    # Update with best parameters
    best_params = study.best_trial.params
    
    # Map Optuna parameters to config file parameters
    param_mapping = {
        'learning_rate': 'lr',
        'batch_size': 'batch_size',
        'n_epochs': 'ppo_epochs',  # Note: this might need to be added to your config
        'vf_coef': 'vf_coef',      # Note: this might need to be added to your config  
        'gamma': 'gamma',
        'n_steps': 'ppo_update_steps',
        'gae_lambda': 'gae_lambda',  # Note: this might need to be added to your config
        'clip_range': 'clip_range',  # Note: this might need to be added to your config
        'ent_coef': 'entropy_regularization',
        'max_grad_norm': 'max_grad_norm',  # Note: this might need to be added to your config
    }
    
    for optuna_param, config_param in param_mapping.items():
        if optuna_param in best_params:
            config[config_param] = best_params[optuna_param]
    
    # Add metadata
    config['optuna_study_name'] = study.study_name
    config['optuna_best_value'] = study.best_trial.value
    config['optuna_trial_number'] = study.best_trial.number
    config['optimization_date'] = datetime.now().isoformat()
    
    # Save optimized config
    with open(output_filename, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Best hyperparameters saved to: {output_filename}")


if __name__ == "__main__":
    
    # Configuration
    env_config_filename = 'config_files/rl_entreg0.01_bs64.json'
    
    # Optimization settings
    study_name = "maisr_ppo_hyperopt_v1"
    n_trials = 5  # Start with fewer trials for testing
    n_envs = min(4, multiprocessing.cpu_count())  # Use fewer envs for faster trials
    n_timesteps = 24e6  # Reduced timesteps for faster optimization
    n_eval_episodes = 8
    eval_freq = 20000
    
    # Database storage (optional - remove if you don't want persistence)
    storage = "sqlite:///optuna_maisr.db"
    
    # Create directories
    os.makedirs("logs/optuna", exist_ok=True)
    os.makedirs("optimized_configs", exist_ok=True)
    
    print("Starting MAISR PPO Hyperparameter Optimization")
    print(f"Environment config: {env_config_filename}")
    print(f"Trials: {n_trials}, Environments per trial: {n_envs}")
    print(f"Timesteps per trial: {n_timesteps}")
    
    # Run optimization
    study = optimize_hyperparameters(
        env_config_filename=env_config_filename,
        study_name=study_name,
        n_trials=n_trials,
        n_jobs=1,  # Keep at 1 to avoid multiprocessing issues with RL
        n_envs=n_envs,
        n_timesteps=n_timesteps,
        n_eval_episodes=n_eval_episodes,
        eval_freq=eval_freq,
        storage=storage,
        sampler="tpe",  # Tree-structured Parzen Estimator (recommended)
        pruner="median",  # Median pruner for early stopping
        timeout=None  # No timeout - let all trials complete
    )
    
    # Print results
    print_study_results(study)
    
    # Save best configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_config = f"optimized_configs/best_config_{timestamp}.json"
    save_best_config(study, env_config_filename, output_config)
    
    # Optional: Create visualization plots (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Create optimization history plot
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(f"logs/optuna/optimization_history_{timestamp}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create parameter importance plot
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.savefig(f"logs/optuna/param_importances_{timestamp}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create contour plot for top 2 parameters
        if len(study.best_trial.params) >= 2:
            param_names = list(study.best_trial.params.keys())[:2]
            fig = optuna.visualization.matplotlib.plot_contour(study, params=param_names)
            plt.savefig(f"logs/optuna/contour_{timestamp}.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Visualization plots saved to logs/optuna/")
        
    except ImportError:
        print("Install matplotlib for visualization: pip install matplotlib")
    except Exception as e:
        print(f"Could not create plots: {e}")
    
    print("\nOptimization complete!")