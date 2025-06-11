import warnings

warnings.filterwarnings("ignore", message="Your system is avx2 capable but pygame was not built with support for it")

import os
import numpy as np
import socket
import json
from datetime import datetime
from typing import Dict, Any, Optional

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.env_context import EnvContext
from ray.rllib.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID

import wandb
import gymnasium as gym

from env_combined import MAISREnvVec
from utility.data_logging import load_env_config


def make_env(config: EnvContext) -> MAISREnvVec:
    """
    Environment factory function for RLlib.

    Args:
        config: RLlib's EnvContext containing environment configuration

    Returns:
        MAISREnvVec: The environment instance
    """
    # Extract our custom config from RLlib's config
    env_config = config["env_config"]

    env = MAISREnvVec(
        config=env_config,
        render_mode='headless',
        run_name='rllib-test',
        tag='train',
        seed=config["seed"],
    )
    return env


class MAISRCallbacks(DefaultCallbacks):
    """Custom RLlib callbacks for MAISR environment."""

    def __init__(self):
        super().__init__()
        self.wandb_run = None
        self.current_difficulty = 0
        self.above_threshold_counter = 0
        self.eval_counter = 0

    def on_algorithm_init(self, *, algorithm, **kwargs):
        """Called when the algorithm is initialized."""
        # Initialize wandb if configured
        if hasattr(algorithm.config, 'wandb_config') and algorithm.config.wandb_config:
            wandb_config = algorithm.config.wandb_config
            self.wandb_run = wandb.init(
                project=wandb_config['project'],
                name=wandb_config['name'],
                config=wandb_config['config'],
                sync_tensorboard=True,
                monitor_gym=True,
            )
            print(f"Initialized WandB run: {self.wandb_run.name}")

        # Set initial difficulty
        if self.wandb_run:
            self.wandb_run.log({"curriculum/difficulty_level": 0}, step=0)

    def on_episode_end(self, *, worker: RolloutWorker, base_env, policies: Dict[PolicyID, Policy],
                       episode: Episode, env_index: int, **kwargs):
        """Called when an episode ends."""
        # Get episode information
        info = episode.last_info_for()

        if info:
            # Log training metrics to WandB
            if self.wandb_run:
                log_data = {
                    "train/episode_reward": episode.total_reward,
                    "train/episode_length": episode.length,
                }

                # Add environment-specific metrics
                if "target_ids" in info:
                    log_data["train/target_ids"] = info["target_ids"]
                if "detections" in info:
                    log_data["train/detections"] = info["detections"]

                self.wandb_run.log(log_data, step=episode.episode_id)

        # Print episode summary
        if info:
            target_ids = info["target_ids"]
            detections = info["detections"]
            print(f'Episode complete: reward {episode.total_reward:.3f}, '
                  f'length {episode.length}, target_ids {target_ids}, detections {detections}')

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        """Called after each training iteration."""
        if self.wandb_run:
            # Log training metrics
            log_data = {
                "train/episode_reward_mean": result["episode_reward_mean"],
                "train/episode_len_mean": result["episode_len_mean"],
                "train/episodes_this_iter": result["episodes_this_iter"],
                "train/policy_gradient_loss": result["info"]["learner"]["default_policy"]["policy_loss"],
                "train/value_loss": result["info"]["learner"]["default_policy"]["vf_loss"],
                "train/entropy_loss": result["info"]["learner"]["default_policy"]["entropy"],
                "train/learning_rate": result["info"]["learner"]["default_policy"]["cur_lr"],
            }

            self.wandb_run.log(log_data, step=result["training_iteration"])

        # Handle curriculum learning and evaluation
        self._handle_evaluation_and_curriculum(algorithm, result)

    def _handle_evaluation_and_curriculum(self, algorithm, result: dict):
        """Handle periodic evaluation and curriculum progression."""
        # Check if it's time to evaluate (every N training iterations)
        eval_freq = getattr(algorithm.config, 'eval_freq', 10)  # Default to every 10 iterations

        if result["training_iteration"] % eval_freq == 0:
            self.eval_counter += 1
            print(
                f'\n#################################################\nEVALUATING (iteration: {result["training_iteration"]})')

            # Run evaluation episodes
            eval_results = self._run_evaluation(algorithm)

            if self.wandb_run and eval_results:
                self.wandb_run.log({
                    "eval/mean_reward": eval_results["mean_reward"],
                    "eval/std_reward": eval_results["std_reward"],
                    "eval/mean_target_ids": eval_results["mean_target_ids"],
                    "eval/mean_episode_length": eval_results["mean_episode_length"],
                    "curriculum/difficulty_level": self.current_difficulty
                }, step=result["training_iteration"])

            # Handle curriculum progression
            if hasattr(algorithm.config, 'use_curriculum') and algorithm.config.use_curriculum:
                self._update_curriculum(algorithm, eval_results)

            print('#################################################\n\nReturning to training...')

    def _run_evaluation(self, algorithm) -> Optional[Dict[str, float]]:
        """Run evaluation episodes."""
        try:
            # Get evaluation configuration
            n_eval_episodes = getattr(algorithm.config, 'n_eval_episodes', 5)

            # Create evaluation environment
            env_config = algorithm.config.env_config
            eval_env = MAISREnvVec(
                config=env_config,
                render_mode='headless',
                tag='eval',
                run_name='rllib-test-eval',
            )

            # Run evaluation episodes
            rewards = []
            target_ids_list = []
            episode_lengths = []

            for i in range(n_eval_episodes):
                obs, _ = eval_env.reset()
                terminated = False
                truncated = False
                ep_reward = 0

                while not (terminated or truncated):
                    action = algorithm.compute_single_action(obs, explore=False)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    ep_reward += reward

                rewards.append(ep_reward)
                if info and "target_ids" in info:
                    target_ids_list.append(info["target_ids"])
                if info and "episode" in info:
                    episode_lengths.append(info["episode"]["l"])

            eval_env.close()

            return {
                "mean_reward": np.mean(rewards),
                "std_reward": np.std(rewards),
                "mean_target_ids": np.mean(target_ids_list) if target_ids_list else 0,
                "mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0,
            }

        except Exception as e:
            print(f"Error during evaluation: {e}")
            return None

    def _update_curriculum(self, algorithm, eval_results: Optional[Dict[str, float]]):
        """Update curriculum difficulty based on evaluation results."""
        if not eval_results:
            return

        min_target_ids_to_advance = getattr(algorithm.config, 'min_target_ids_to_advance', 8)
        max_ep_len_to_advance = getattr(algorithm.config, 'max_ep_len_to_advance', 10000)

        avg_target_ids = eval_results["mean_target_ids"]
        avg_eval_len = eval_results["mean_episode_length"]

        print(f'CURRICULUM: Checking if we should increase difficulty (current: {self.current_difficulty})')

        # Check advancement criteria based on difficulty level
        should_advance = False
        if self.current_difficulty == 0:
            should_advance = avg_target_ids >= min_target_ids_to_advance
        else:
            should_advance = (avg_target_ids >= min_target_ids_to_advance and
                              avg_eval_len <= max_ep_len_to_advance)

        if should_advance:
            self.above_threshold_counter += 1
        else:
            self.above_threshold_counter = 0

        # Advance difficulty if consistently performing well
        if self.above_threshold_counter >= 5:
            self.above_threshold_counter = 0
            self.current_difficulty += 1
            print(f'CURRICULUM: Increasing difficulty to level {self.current_difficulty}')

            # Update difficulty in all workers
            def update_env_difficulty(worker):
                for env in worker.env.get_sub_environments():
                    env.set_difficulty(self.current_difficulty)

            algorithm.workers.foreach_worker(update_env_difficulty)

            if self.wandb_run:
                self.wandb_run.log({"curriculum/difficulty_level": self.current_difficulty})
        else:
            print(f'CURRICULUM: Maintaining difficulty at level {self.current_difficulty} '
                  f'(avg target_ids: {avg_target_ids} < threshold: {min_target_ids_to_advance})')


def generate_run_name(config):
    """Generate a unique, descriptive name for this training run."""
    components = [
        f"{config['num_workers']}workers",
        f"obs-{config['obs_type']}",
        f"act-{config['action_type']}",
    ]

    # Add critical hyperparameters
    components.extend([
        f"lr-{config['lr']}",
        f"bs-{config['train_batch_size']}",
        f"g-{config['gamma']}",
        f"fs-{config['frame_skip']}",
        f"curriculum-{config['use_curriculum']}",
        f"rew-wtn-{config['shaping_coeff_wtn']}",
        f"rew-prox-{config['shaping_coeff_prox']}",
        f"rew-timepenalty-{config['shaping_time_penalty']}",
    ])

    # Add timestamp
    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_name = f"rllib_{timestamp}_" + "_".join(components)

    return run_name


def train(
        env_config_filename: str,
        num_workers: int = 4,
        project_name: str = "maisr-rllib",
        save_dir: str = "./trained_models/",
        load_path: Optional[str] = None,
        machine_name: str = 'machine',
        total_timesteps: int = 1000000,
):
    """
    Main training pipeline using Ray RLlib.

    Args:
        env_config_filename: Path to environment configuration file
        num_workers: Number of parallel workers for training
        project_name: WandB project name
        save_dir: Directory to save trained models
        load_path: Path to checkpoint to load (optional)
        machine_name: Machine identifier for naming
        total_timesteps: Total training timesteps
    """

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # Load environment config
    env_config = load_env_config(env_config_filename)

    # Generate run name
    env_config['num_workers'] = num_workers
    env_config['config_filename'] = env_config_filename
    run_name = generate_run_name(env_config)
    env_config['run_name'] = run_name

    # Create save directory
    os.makedirs(f"{save_dir}/{run_name}", exist_ok=True)
    os.makedirs('./logs/action_histories/' + run_name, exist_ok=True)

    # Configure PPO algorithm
    config = (
        PPOConfig()
        .environment(
            env=make_env,
            env_config=env_config,
        )
        .rollouts(
            num_rollout_workers=num_workers,
            num_envs_per_worker=1,
        )
        .training(
            # PPO-specific parameters
            lr=env_config['lr'],
            gamma=env_config['gamma'],
            lambda_=env_config['lambda'],
            train_batch_size=env_config['batch_size'],
            sgd_minibatch_size=env_config['sgd_minibatch_size'],
            num_sgd_iter=env_config['num_sgd_iter'],
            entropy_coeff=env_config['entropy_regularization'],
            clip_param=env_config['clip_param'],
            vf_loss_coeff=env_config['vf_loss_coeff'],
            use_gae=True,
        )
        .framework("torch")
        .debugging(
            seed=env_config['seed'],
        )
        .callbacks(MAISRCallbacks)
        .resources(
            num_gpus=1,  # Set to 1 if you have GPU
        )
    )

    # Add custom configuration attributes for callbacks
    config.use_curriculum = env_config['use_curriculum']
    config.min_target_ids_to_advance = env_config['min_target_ids_to_advance']
    config.max_ep_len_to_advance = 10000
    config.eval_freq = 10  # Evaluate every 10 training iterations
    config.n_eval_episodes = env_config['n_eval_episodes']
    config.env_config = env_config

    # Configure WandB
    config.wandb_config = {
        'project': project_name,
        'name': f'{machine_name}_{num_workers}workers_{run_name}',
        'config': env_config,
    }

    # Build the algorithm
    algorithm = config.build()

    # Load checkpoint if provided
    if load_path:
        print(f'LOADING FROM {load_path}')
        algorithm.restore(load_path)
    else:
        print('Training new model')

    print('##################################### Beginning agent training... #######################################\n')

    # Training loop
    checkpoint_freq = env_config['save_freq'] // (num_workers * 1000)  # Approximate conversion
    iteration = 0
    total_timesteps_trained = 0

    try:
        while total_timesteps_trained < total_timesteps:
            # Train for one iteration
            result = algorithm.train()
            iteration += 1
            total_timesteps_trained = result["timesteps_total"]

            # Print progress
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, timesteps: {total_timesteps_trained}, "
                      f"reward: {result['episode_reward_mean']:.3f}")

            # Save checkpoint periodically
            if iteration % checkpoint_freq == 0:
                checkpoint_path = algorithm.save(f"{save_dir}/{run_name}")
                print(f"Checkpoint saved at: {checkpoint_path}")

    except KeyboardInterrupt:
        print("Training interrupted by user")

    finally:
        print(
            '########################################## TRAINING COMPLETE ############################################\n')

        # Save final model
        final_checkpoint_path = algorithm.save(f"{save_dir}/{run_name}")
        print(f"Final model saved at: {final_checkpoint_path}")

        # Clean up
        algorithm.stop()
        ray.shutdown()


if __name__ == "__main__":
    config_list = [
        'config_files/june3c/june3c_base.json',
        # Add more config files as needed
    ]

    # Specify a checkpoint to load here
    load_path = None

    # Get machine name
    print(f'machine is {socket.gethostname()}')
    machine_name = 'home' if socket.gethostname() == 'DESKTOP-3Q1FTUP' else 'pace'
    project_name = 'maisr-rllib' if machine_name == 'home' else 'maisr-rllib-pace'
    print(f'Setting machine_name to {machine_name}. Using project {project_name}')

    print('\n################################################################################')
    print('################################################################################')
    print(f'############################ STARTING RLLIB TRAINING RUN ##################')
    print('################################################################################')

    for config_filename in config_list:
        train(
            env_config_filename=config_filename,
            num_workers=4,  # Adjust based on your system
            load_path=load_path,
            machine_name=machine_name,
            project_name=project_name,
            total_timesteps=1000000,  # Adjust as needed
        )