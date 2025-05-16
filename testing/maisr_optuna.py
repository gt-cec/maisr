import gymnasium as gym
import os
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import multiprocessing
from typing import Dict, List, Tuple, Any

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback

# Import your environment and utility functions
from env_combined import MAISREnvVec
from utility.data_logging import load_env_config

# Simple callback to report to Optuna
class OptunaCallback(BaseCallback):
    def __init__(self, trial, eval_env, eval_freq=10000, n_eval_episodes=8, verbose=0):
        super(OptunaCallback, self).__init__(verbose)
        self.trial = trial
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -float("inf")
        
    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0:
            # Evaluate the model
            mean_reward, _ = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes
            )
            
            print(f"Step: {self.num_timesteps}, Mean Reward: {mean_reward}")
            
            # Report to Optuna
            self.trial.report(mean_reward, self.num_timesteps)
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(f"./optuna_models/trial_{self.trial.number}_best")
            
            # Handle pruning (early stopping if the trial is unpromising)
            if self.trial.should_prune():
                print(f"Trial {self.trial.number} pruned at step {self.num_timesteps}")
                return False
                
        return True

def make_env(env_config, rank, seed, obs_type, action_type, frame_skip, use_curriculum, difficulty=0):
    def _init():
        env = MAISREnvVec(
            config=env_config,
            render_mode='headless',
            reward_type=env_config['reward type'],
            obs_type=obs_type,
            tag=f'train_mp{rank}',
            action_type=action_type,
            seed=seed + rank,
            difficulty=difficulty,
            use_curriculum=use_curriculum,
            frame_skip=frame_skip
        )
        env = Monitor(env)
        env.reset()
        return env

    return _init

def objective(trial):
    """Optuna objective function"""
    # Define hyperparameters to search
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    # TODO add reward variations
    
    # Create directories
    os.makedirs("./optuna_logs", exist_ok=True)
    os.makedirs("./optuna_models", exist_ok=True)
    
    # Load environment config
    env_config = load_env_config('./config_files/rl_cl_phase1.json')
    
    # Create vectorized environment
    if n_envs > 1:
        env_fns = [make_env(env_config, i, 42 + i, "relative", "continuous-normalized", 
                           frame_skip, True, difficulty=0)
                  for i in range(n_envs)]
        env = SubprocVecEnv(env_fns)
        env = VecMonitor(env, filename=f'./optuna_logs/trial_{trial.number}')
    else:
        env = MAISREnvVec(
            env_config,
            None,
            render_mode='headless',
            obs_type="relative",
            action_type="continuous-normalized",
            tag='train',
            seed=42,
            difficulty=0,
            frame_skip=frame_skip,
            use_curriculum=True,
        )
        env = Monitor(env)
    
    # Create evaluation environment (always single env)
    eval_env = MAISREnvVec(
        env_config,
        None,
        render_mode='headless',
        obs_type="relative",
        action_type="continuous-normalized",
        tag='eval',
        seed=42,
        difficulty=0,
        frame_skip=frame_skip,
        use_curriculum=True,
    )
    eval_env = Monitor(eval_env)
    
    # Create the PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        batch_size=batch_size * n_envs,
        n_steps=ppo_update_steps // n_envs,
        learning_rate=lr,
        seed=42,
        device='cpu'
    )
    
    # Checkpoint callback - save every 500k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=(500000 // n_envs),
        save_path=f"./optuna_models/trial_{trial.number}",
        name_prefix="checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    # Evaluation callback for Optuna
    optuna_callback = OptunaCallback(
        trial=trial,
        eval_env=eval_env,
        eval_freq=(50000 // n_envs),
        n_eval_episodes=8
    )
    
    try:
        # Train the model
        model.learn(
            total_timesteps=1000000,  # 1M steps for quicker evaluation
            callback=[checkpoint_callback, optuna_callback]
        )
        
        # Final evaluation
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        print(f"Trial {trial.number} finished with mean reward: {mean_reward} (±{std_reward})")
        
        # Save final model
        model.save(f"./optuna_models/trial_{trial.number}_final")
        
        return mean_reward
        
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return float('-inf')  # Return worst score on failure

def main():
    # Create study with persistence
    study = optuna.create_study(
        study_name="maisr_optimization",
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        storage="sqlite:///optuna_maisr.db",
        load_if_exists=True
    )
    
    # Run optimization
    study.optimize(objective, n_trials=20)  # Adjust number of trials as needed
    
    # Print results
    print("\n----- Optimization Results -----")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best mean reward: {study.best_trial.value}")
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save best parameters to JSON file
    import json
    with open("best_params.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=4)
    
    # Generate optimization visualization
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        import matplotlib.pyplot as plt
        
        # Plot optimization history
        fig1 = plot_optimization_history(study)
        fig1.write_image("optimization_history.png")
        
        # Plot parameter importances
        fig2 = plot_param_importances(study)
        fig2.write_image("param_importances.png")
        
        print("Visualization plots saved to disk.")
    except:
        print("Could not generate visualization plots. Make sure plotly is installed.")

if __name__ == "__main__":
    main()