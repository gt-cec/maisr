import ctypes
import warnings
import pygame
from training_wrappers.localsearch_training_wrapper import MaisrLocalSearchWrapper
warnings.filterwarnings("ignore", message="Your system is avx2 capable but pygame was not built with support for it")
import gymnasium as gym
import os
import numpy as np
import multiprocessing
import socket
import torch

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from env_multi_new import MAISREnvVec
from training_wrappers.modeselector_training_wrapper import MaisrModeSelectorWrapper
from policies.league_management import TeammateManager, GenericTeammatePolicy, SubPolicy, LocalSearch, ChangeRegions, GoToNearestThreat, TargetSearchLocalTSP
from utility.data_logging import load_env_config


"""Script to train the mode selector top-level policy"""

def generate_run_name(config):
    """Generate a unique, descriptive name for this training run. Will be shared across logs, WandB, and action
    history plots to make it easy to match them."""

    components = []#[f"{config['n_envs']}envs",]
    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_name = f"{timestamp}_" + "_".join(components)
    return run_name



class EnhancedWandbCallback_Monolith(BaseCallback):
    """Custom Callback that:
    1. Logs training metrics to WandB
    2. Evaluates the agent periodically (also logged to WandB)
    3. Determines if the agent should progress to the next stage of the training curriculum
    4. Logs additional PPO training metrics
    """

    def __init__(self, env_config, verbose=0, eval_env=None, run=None,
                 use_curriculum=False, min_target_ids_to_advance=8, run_name='no_name',
                 log_freq=2):
        super(EnhancedWandbCallback_Monolith, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = env_config['eval_freq']
        self.n_eval_episodes = env_config['n_eval_episodes']
        self.run = run
        self.log_freq = log_freq  # Log every N steps instead of every step

        self.use_curriculum = env_config['use_curriculum']
        self.min_target_ids_to_advance = env_config['min_target_ids_to_advance']
        self.max_ep_len_to_advance = 130
        self.max_difficulty = env_config['max_difficulty']
        self.cl_lr_decrease = env_config['cl_lr_decrease'] # Divide learning rate by this every time we increase difficulty

        self.current_difficulty = 0
        self.above_threshold_counter = 0

        self.switched_to_twoship = False
        self.twoship_switch_threshold = env_config['twoship_switch_threshold']

        # Entropy decay parameters
        self.use_entropy_decay_schedule = env_config['use_entropy_decay_schedule']
        self.entropy_decay_enabled = False
        self.entropy_decay_trigger_threshold = 0.32  # mean_target_ids_per_step threshold
        self.entropy_decay_threat_threshold = 0.4 # eval/mean_threat_ids threshold
        self.entropy_decay_steps = 1e6  # Decay over this many steps
        self.entropy_final_ratio = 0.5  # Final entropy = 50% of original
        self.entropy_decay_start_step = None
        self.original_entropy_coeff = None

        # Buffer for accumulating data between log events
        self.episode_buffer = {
            'rewards': [],
            'lengths': [],
            'target_ids': [],
            'threat_ids': [],
            'detections': []
        }

        # Early stopping based on performance degradation
        self.best_eval_performance = -np.inf
        self.performance_crash_counter = 0
        self.performance_crash_threshold = 20  # Number of consecutive poor evals before stopping
        self.performance_crash_ratio = 0.4  # Performance must drop below 50% of best
        self.should_stop_training = False

    def _on_step(self):
        # Only log on the specified frequency
        should_log_episode_data = self.num_timesteps % self.log_freq == 0

        # Always collect episode data when available (lightweight)
        if self.locals.get("infos") and len(self.locals["infos"]) > 0:
            for env_idx, info in enumerate(self.locals["infos"]):
                if "episode" in info:
                    # Always buffer the data
                    self.episode_buffer['rewards'].append(info["episode"]["r"])
                    self.episode_buffer['lengths'].append(info["episode"]["l"])

                    if "target_ids" in info:
                        self.episode_buffer['target_ids'].append(info["target_ids"])
                    elif "new_target_ids" in info:
                        self.episode_buffer['target_ids'].append(info["new_target_ids"])

                    if "new_threat_ids" in info:
                        self.episode_buffer['threat_ids'].append(info["new_threat_ids"])
                    if "detections" in info:
                        self.episode_buffer['detections'].append(info["detections"])


        # Only log episode data at the specified frequency
        if should_log_episode_data and any(len(v) > 0 for v in self.episode_buffer.values()):
            log_data = {}

            # Log aggregated data from buffer
            if self.episode_buffer['rewards']:
                log_data["train/mean_episode_reward"] = np.mean(self.episode_buffer['rewards'])
                log_data["train/mean_episode_length"] = np.mean(self.episode_buffer['lengths'])

            if self.episode_buffer['target_ids']:
                log_data["train/mean_target_ids"] = np.mean(self.episode_buffer['target_ids'])
            if self.episode_buffer['threat_ids']:
                log_data["train/mean_threat_ids"] = np.mean(self.episode_buffer['threat_ids'])
            if self.episode_buffer['detections']:
                log_data["train/mean_detections"] = np.mean(self.episode_buffer['detections'])

            if log_data: # Log the aggregated data
                self.run.log(log_data, step=self.num_timesteps // self.model.get_env().num_envs)

            # Clear the buffer after logging
            self.episode_buffer = {
                'rewards': [],
                'lengths': [],
                'target_ids': [],
                'threat_ids': [],
                'detections': []
            }

        # Log training metrics less frequently (e.g., every 10 steps)
        should_log_training_metrics = self.num_timesteps % (self.log_freq * 2) == 0

        if should_log_training_metrics:
            training_metrics = {}

            if hasattr(self.model, '_n_updates'): training_metrics["train/n_updates"] = self.model._n_updates

            if hasattr(self.logger, 'name_to_value'):
                logger_dict = self.logger.name_to_value

                # Log the core training metrics
                metrics_to_log = [
                    "train/approx_kl", "train/entropy_loss", "train/explained_variance",
                    "train/n_updates", "train/policy_gradient_loss", "train/value_loss",
                    "train/clip_fraction", "train/clip_range", "train/learning_rate"
                ]

                for metric in metrics_to_log:
                    if metric in logger_dict:
                        training_metrics[metric] = logger_dict[metric]

            # Log training metrics if any are available
            if training_metrics:
                self.run.log(training_metrics, step=self.num_timesteps // self.model.get_env().num_envs)

        # Evaluation logic remains the same (already infrequent)
        if self.eval_env is not None and self.num_timesteps % self.eval_freq == 0:
            print(f'\n#################################################\nEVALUATING (step: {self.num_timesteps})')

            if hasattr(self.model.get_env(), 'obs_rms'):
                self.eval_env.obs_rms = self.model.get_env().obs_rms
                self.eval_env.ret_rms = self.model.get_env().ret_rms

            target_ids_list = []
            threat_ids_list = []
            target_ids_per_step_list = []
            mean_reward, std_reward = 0, 0
            total_eval_reward = 0
            eval_lengths = []

            obs = self.eval_env.reset()
            for i in range(self.n_eval_episodes):
                done = False
                ep_reward, ep_target_ids, ep_threat_ids = 0, 0, 0

                while not done:
                    action, other = self.model.predict(obs, deterministic=True)
                    obses, rewards, dones, infos = self.eval_env.step([action])
                    obs = obses[0]
                    reward = rewards[0]
                    info = infos[0]
                    done = dones[0]

                    ep_reward += reward
                    # TODO make sure this works
                    ep_target_ids += info['new_target_ids']
                    ep_threat_ids += info['new_threat_ids']

                    final_info = info

                ep_length = final_info["episode"]["l"]
                target_ids_list.append(ep_target_ids)
                threat_ids_list.append(ep_threat_ids)

                eval_lengths.append(ep_length)
                target_ids_per_step_list.append(ep_target_ids / ep_length)

                total_eval_reward += ep_reward

            mean_reward = total_eval_reward / self.n_eval_episodes

            # Log evaluation results
            eval_metrics = {
                "eval/mean_reward": mean_reward,
                "eval/mean_target_ids": np.mean(target_ids_list) if target_ids_list else 0,
                "eval/mean_threat_ids": np.mean(threat_ids_list) if threat_ids_list else 0,
                "eval/mean_episode_length": np.mean(eval_lengths) if eval_lengths else 0,
                "eval/mean_target_ids_per_step": np.mean(target_ids_per_step_list) if target_ids_per_step_list else 0,
                "curriculum/difficulty_level": self.current_difficulty
            }

            ######################## Entropy decay #####################################################
            if self.use_entropy_decay_schedule:
                current_target_ids_per_step = np.mean(target_ids_per_step_list) if target_ids_per_step_list else 0
                current_threat_ids = np.mean(threat_ids_list) if threat_ids_list else 0

                if not self.entropy_decay_enabled and current_target_ids_per_step >= self.entropy_decay_trigger_threshold and current_threat_ids >= self.entropy_decay_threat_threshold:
                    print(f'\n{"=" * 60}')
                    print(f'ENTROPY DECAY TRIGGERED! (step {self.num_timesteps})')
                    print(f'Target IDs per step ({current_target_ids_per_step:.3f}) exceeded threshold ({self.entropy_decay_trigger_threshold})')
                    print(f'Starting entropy decay from {self.model.ent_coef} to {self.model.ent_coef * self.entropy_final_ratio} over {self.entropy_decay_steps} steps')
                    print(f'{"=" * 60}\n')

                    self.entropy_decay_enabled = True
                    self.entropy_decay_start_step = self.num_timesteps
                    self.original_entropy_coeff = self.model.ent_coef

                    # Log the trigger
                    eval_metrics["entropy_decay/triggered"] = True
                    eval_metrics["entropy_decay/trigger_step"] = self.num_timesteps
                    #eval_metrics["entropy_decay/original_coeff"] = self.original_entropy_coeff

                # Apply entropy decay if enabled
                if self.entropy_decay_enabled and self.entropy_decay_start_step is not None:
                    steps_since_trigger = self.num_timesteps - self.entropy_decay_start_step
                    decay_progress = min(steps_since_trigger / self.entropy_decay_steps, 1.0)

                    # Linear decay from original to final ratio
                    current_ratio = 1.0 - (decay_progress * (1.0 - self.entropy_final_ratio))
                    new_entropy_coeff = self.original_entropy_coeff * current_ratio

                    # Update the model's entropy coefficient
                    self.model.ent_coef = new_entropy_coeff

                    # Log entropy decay metrics
                    eval_metrics["entropy_decay/current_coeff"] = new_entropy_coeff
                    eval_metrics["entropy_decay/decay_progress"] = decay_progress
                    #eval_metrics["entropy_decay/steps_since_trigger"] = steps_since_trigger

                    if decay_progress >= 1.0:
                        eval_metrics["entropy_decay/completed"] = True

            # Always log current entropy coefficient
            eval_metrics["train/current_entropy_coeff"] = self.model.ent_coef


            #################################### Aircraft switching ####################################
            print(f'About to check for 2 ship switch: self.switched_to_twoship = {self.switched_to_twoship}, target_ids_list = {target_ids_list}')
            if (not self.switched_to_twoship) and target_ids_list:
                avg_target_ids = np.mean(target_ids_list)
                if avg_target_ids > self.twoship_switch_threshold:
                    print(f'\n{"=" * 80}')
                    print(f'AIRCRAFT SWITCHING TRIGGERED! (step {self.num_timesteps})')
                    print(f'Average target IDs ({avg_target_ids:.2f}) exceeded threshold ({self.twoship_switch_threshold})')
                    print(f'Switching from 1 aircraft to 2 aircraft...')
                    print(f'{"=" * 80}\n')

                    self.model.get_env().env_method("set_teammate_active", True)
                    self.eval_env.env_method("set_teammate_active", True)
                    self.switched_to_twoship = True

                    # Log the switch
                    eval_metrics["aircraft/num_aircraft"] = 2
                    eval_metrics["aircraft/switch_step"] = self.num_timesteps


                if not self.switched_to_twoship:
                    eval_metrics["aircraft/num_aircraft"] = 1 if not self.switched_to_twoship else 2

            self.run.log(eval_metrics, step=self.num_timesteps)

            print(f'\nEVAL LOGGED (mean reward {mean_reward}, std {round(std_reward, 2)}, 'f'mean target_ids: {np.mean(target_ids_list) if target_ids_list else 0}')


            #################################### Curriculum learning ####################################
            if self.use_curriculum:
                print('CURRICULUM: Checking if we should increase difficulty')
                avg_target_ids = np.mean(target_ids_list) if target_ids_list else 0
                avg_eval_len = np.mean(eval_lengths) if eval_lengths else 0


                if avg_target_ids >= self.min_target_ids_to_advance and avg_eval_len <= self.max_ep_len_to_advance:
                    self.above_threshold_counter += 1
                else:
                    self.above_threshold_counter = 0

                if self.above_threshold_counter >= 5 and self.current_difficulty < self.max_difficulty:
                    self.above_threshold_counter = 0
                    self.current_difficulty += 1
                    print(f'CURRICULUM: Increasing difficulty to level {self.current_difficulty}')

                    self.model.get_env().env_method("set_difficulty", self.current_difficulty)
                    try: self.eval_env.env_method("set_difficulty", self.current_difficulty)
                    except Exception as e: print(f"Failed to set difficulty on eval env: {e}")

                    self.run.log({"curriculum/difficulty_level": self.current_difficulty}, step=self.num_timesteps)

                    # Decrease LR
                    print(f'Model lr reduced from {self.model.policy.optimizer.param_groups[0]['lr']} to {self.model.policy.optimizer.param_groups[0]['lr']/self.cl_lr_decrease}')
                    self.model.policy.optimizer.param_groups[0]['lr'] = self.model.policy.optimizer.param_groups[0]['lr']/self.cl_lr_decrease

                else:
                    print(f'CURRICULUM: Maintaining difficulty at level {self.current_difficulty} '
                          f'(avg target_ids: {avg_target_ids} < threshold: {self.min_target_ids_to_advance})')

            print('#################################################\n\nReturning to training... \n')

        # if self.should_stop_training:
        #     return False
        return True



class EnhancedWandbCallback_MS(BaseCallback):
    """Custom Callback that:
    1. Logs training metrics to WandB
    2. Evaluates the agent periodically (also logged to WandB)
    3. Determines if the agent should progress to the next stage of the training curriculum
    4. Logs additional PPO training metrics
    """

    def __init__(self, env_config, verbose=0, eval_env=None, run=None,
                 use_curriculum=False, min_target_ids_to_advance=8, run_name='no_name',
                 log_freq=2):
        super(EnhancedWandbCallback_MS, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = env_config['eval_freq']
        self.n_eval_episodes = env_config['n_eval_episodes']
        self.run = run
        self.log_freq = log_freq

        self.use_curriculum = env_config['use_curriculum']
        self.min_target_ids_to_advance = env_config['min_target_ids_to_advance']
        self.max_ep_len_to_advance = 130
        self.max_difficulty = env_config['max_difficulty']
        self.cl_lr_decrease = env_config['cl_lr_decrease']

        self.current_difficulty = 0
        self.above_threshold_counter = 0

        # Entropy decay parameters
        self.entropy_decay_enabled = False
        self.entropy_decay_trigger_threshold = 0.3  # mean_target_ids_per_step threshold
        self.entropy_decay_steps = 200000  # Decay over 200k steps
        self.entropy_final_ratio = 0.25  # Final entropy = 25% of original
        self.entropy_decay_start_step = None
        self.original_entropy_coeff = None

        # Buffer for accumulating data between log events
        self.episode_buffer = {
            'rewards': [],
            'lengths': [],
            'target_ids': [],
            'detections': [],
            'threat_ids': [],  # New: track threat identifications
            'policy_switches': [],  # New: track mode selector switches
            'subpolicy_usage': [],  # New: track which subpolicies were used
            'action_choices': []  # Add this line
        }

        # Early stopping based on performance degradation
        self.best_eval_performance = -np.inf
        self.performance_crash_counter = 0
        self.performance_crash_threshold = 20
        self.performance_crash_ratio = 0.4
        self.should_stop_training = False

    def _on_step(self):
        should_log_episode_data = self.num_timesteps % self.log_freq == 0

        # Always collect episode data when available (lightweight)
        if self.locals.get("infos") and len(self.locals["infos"]) > 0:
            for env_idx, info in enumerate(self.locals["infos"]):
                if "episode" in info:
                    # Always buffer the data
                    self.episode_buffer['rewards'].append(info["episode"]["r"])
                    self.episode_buffer['lengths'].append(info["episode"]["l"])

                    if "target_ids" in info:
                        self.episode_buffer['target_ids'].append(info["target_ids"])
                    if "detections" in info:
                        self.episode_buffer['detections'].append(info["detections"])

                    ### CLAUDE CHANGED ###
                    # Log mode selector specific metrics
                    if "threat_ids" in info:
                        self.episode_buffer['threat_ids'].append(info["threat_ids"])
                    elif "new_threat_ids" in info:
                        self.episode_buffer['threat_ids'].append(info["new_threat_ids"])

                    # Track policy switches from the wrapper
                    # Note: You'll need to add this to the wrapper's info dict
                    if "policy_switches" in info:
                        self.episode_buffer['policy_switches'].append(info["policy_switches"])

                    # Track subpolicy usage
                    if "final_subpolicy" in info:
                        self.episode_buffer['subpolicy_usage'].append(info["final_subpolicy"])

                if "final_subpolicy" in info:
                    self.episode_buffer['subpolicy_usage'].append(info["final_subpolicy"])

                # Track action choices (mode selector decisions)
                if "action_choice" in info:
                    self.episode_buffer['action_choices'].append(info["action_choice"])

        # Only log episode data at the specified frequency
        if should_log_episode_data and any(len(v) > 0 for v in self.episode_buffer.values()):
            log_data = {}

            # Log aggregated data from buffer
            if self.episode_buffer['rewards']:
                log_data["train/mean_episode_reward"] = np.mean(self.episode_buffer['rewards'])
                log_data["train/mean_episode_length"] = np.mean(self.episode_buffer['lengths'])



            if self.episode_buffer['target_ids']:
                log_data["train/mean_target_ids"] = np.mean(self.episode_buffer['target_ids'])
            if self.episode_buffer['detections']:
                log_data["train/mean_detections"] = np.mean(self.episode_buffer['detections'])

            # Log mode selector specific metrics
            if self.episode_buffer['threat_ids']:
                log_data["train/mean_threat_ids"] = np.mean(self.episode_buffer['threat_ids'])

            if self.episode_buffer['policy_switches']:
                log_data["train/mean_policy_switches"] = np.mean(self.episode_buffer['policy_switches'])

            if self.episode_buffer['subpolicy_usage']: # Log distribution of subpolicy usage
                subpolicy_counts = np.bincount(self.episode_buffer['subpolicy_usage'], minlength=4)  # Changed to 4
                total = len(self.episode_buffer['subpolicy_usage'])
                if total > 0:
                    log_data["train/subpolicy_usage_localsearch"] = subpolicy_counts[0] / total  # Local search
                    log_data["train/subpolicy_usage_changeregion"] = (subpolicy_counts[1]+subpolicy_counts[4]+subpolicy_counts[5]+subpolicy_counts[6]) / total  # Change region
                    log_data["train/subpolicy_usage_gotothreat"] = subpolicy_counts[2] / total  # Go to threat
                    log_data["train/subpolicy_usage_hold"] = subpolicy_counts[3] / total  # Hold,

            # # Log action choice distribution
            # if self.episode_buffer['action_choices']:
            #     action_counts = np.bincount(self.episode_buffer['action_choices'], minlength=4)
            #     total_actions = len(self.episode_buffer['action_choices'])
            #     if total_actions > 0:
            #         log_data["train/action_0_frequency"] = action_counts[0] / total_actions
            #         log_data["train/action_1_frequency"] = action_counts[1] / total_actions
            #         log_data["train/action_2_frequency"] = action_counts[2] / total_actions
            #         log_data["train/action_3_frequency"] = action_counts[3] / total_actions

            if log_data:
                self.run.log(log_data, step=self.num_timesteps // self.model.get_env().num_envs)

            # Clear the buffer after logging
            self.episode_buffer = {
                'rewards': [], 'lengths': [], 'target_ids': [], 'detections': [],
                'threat_ids': [], 'policy_switches': [], 'subpolicy_usage': [],
                'action_choices': []
            }

        # Log training metrics less frequently (e.g., every 10 steps)
        should_log_training_metrics = self.num_timesteps % (self.log_freq * 2) == 0

        if should_log_training_metrics:
            training_metrics = {}

            if hasattr(self.model, '_n_updates'):
                training_metrics["train/n_updates"] = self.model._n_updates

            if hasattr(self.logger, 'name_to_value'):
                logger_dict = self.logger.name_to_value

                # Log the core training metrics
                metrics_to_log = [
                    "train/approx_kl", "train/entropy_loss", "train/explained_variance",
                    "train/n_updates", "train/policy_gradient_loss", "train/value_loss",
                    "train/clip_fraction", "train/clip_range", "train/learning_rate"
                ]

                for metric in metrics_to_log:
                    if metric in logger_dict:
                        training_metrics[metric] = logger_dict[metric]

            if training_metrics:
                self.run.log(training_metrics, step=self.num_timesteps // self.model.get_env().num_envs)

        # Evaluation logic
        if self.eval_env is not None and self.num_timesteps % self.eval_freq == 0:
            print(f'\n#################################################\nEVALUATING (step: {self.num_timesteps})')

            if hasattr(self.model.get_env(), 'obs_rms'):
                self.eval_env.obs_rms = self.model.get_env().obs_rms
                self.eval_env.ret_rms = self.model.get_env().ret_rms

            target_ids_list = []
            target_ids_per_step_list = []
            ### CLAUDE CHANGED ###
            threat_ids_list = []
            policy_switches_list = []
            ### CLAUDE CHANGED ###
            mean_reward, std_reward = 0, 0
            eval_lengths = []

            obs = self.eval_env.reset()
            for i in range(self.n_eval_episodes):
                done = False
                ep_reward = 0

                while not done:
                    action, other = self.model.predict(obs, deterministic=True)
                    obses, rewards, dones, infos = self.eval_env.step([action])
                    obs = obses[0]
                    reward = rewards[0]
                    info = infos[0]
                    done = dones[0]
                    ep_reward += reward

                if "target_ids" in info:
                    target_ids_list.append(info["target_ids"])

                ### CLAUDE CHANGED ###
                # Collect mode selector specific eval metrics
                if "threat_ids" in info:
                    threat_ids_list.append(info["threat_ids"])
                if "policy_switches" in info:
                    policy_switches_list.append(info["policy_switches"])
                ### CLAUDE CHANGED ###

                mean_reward += ep_reward / self.n_eval_episodes
                eval_lengths.append(info["episode"]["l"])
                target_ids_per_step_list.append(info["target_ids"] / info["episode"]["l"])

            std_reward = np.std(target_ids_list) if target_ids_list else 0

            # Log evaluation results
            eval_metrics = {
                "eval/mean_reward": mean_reward,
                "misc/std_reward": std_reward,
                "eval/mean_target_ids": np.mean(target_ids_list) if target_ids_list else 0,
                "eval/mean_episode_length": np.mean(eval_lengths) if eval_lengths else 0,
                "eval/mean_target_ids_per_step": np.mean(target_ids_per_step_list) if target_ids_per_step_list else 0,
                "curriculum/difficulty_level": self.current_difficulty,
                "eval/mean_threat_ids": np.mean(threat_ids_list) if threat_ids_list else 0,
                "eval/mean_policy_switches": np.mean(policy_switches_list) if policy_switches_list else 0,
            }

            ####################################### Entropy Decay ######################################################
            current_target_ids_per_step = np.mean(target_ids_per_step_list) if target_ids_per_step_list else 0

            if not self.entropy_decay_enabled and current_target_ids_per_step >= self.entropy_decay_trigger_threshold:
                print(f'\n{"=" * 60}')
                print(f'ENTROPY DECAY TRIGGERED! (step {self.num_timesteps})')
                print(
                    f'Target IDs per step ({current_target_ids_per_step:.3f}) exceeded threshold ({self.entropy_decay_trigger_threshold})')
                print(
                    f'Starting entropy decay from {self.model.ent_coef} to {self.model.ent_coef * self.entropy_final_ratio} over {self.entropy_decay_steps} steps')
                print(f'{"=" * 60}\n')

                self.entropy_decay_enabled = True
                self.entropy_decay_start_step = self.num_timesteps
                self.original_entropy_coeff = self.model.ent_coef

                # Log the trigger
                eval_metrics["entropy_decay/triggered"] = True
                eval_metrics["entropy_decay/trigger_step"] = self.num_timesteps
                eval_metrics["entropy_decay/original_coeff"] = self.original_entropy_coeff

            # Apply entropy decay if enabled
            if self.entropy_decay_enabled and self.entropy_decay_start_step is not None:
                steps_since_trigger = self.num_timesteps - self.entropy_decay_start_step
                decay_progress = min(steps_since_trigger / self.entropy_decay_steps, 1.0)

                # Linear decay from original to final ratio
                current_ratio = 1.0 - (decay_progress * (1.0 - self.entropy_final_ratio))
                new_entropy_coeff = self.original_entropy_coeff * current_ratio

                # Update the model's entropy coefficient
                self.model.ent_coef = new_entropy_coeff

                # Log entropy decay metrics
                eval_metrics["entropy_decay/current_coeff"] = new_entropy_coeff
                eval_metrics["entropy_decay/decay_progress"] = decay_progress
                eval_metrics["entropy_decay/steps_since_trigger"] = steps_since_trigger

                if decay_progress >= 1.0:
                    eval_metrics["entropy_decay/completed"] = True

            # Always log current entropy coefficient
            eval_metrics["train/current_entropy_coeff"] = self.model.ent_coef

            ############################################################################################################

            # Early stopping logic
            current_performance = np.mean(target_ids_list) if target_ids_list else 0
            if current_performance > self.best_eval_performance:
                self.best_eval_performance = current_performance
                self.performance_crash_counter = 0
                eval_metrics["early_stopping/best_performance"] = self.best_eval_performance
                print(f'NEW BEST PERFORMANCE: {self.best_eval_performance:.3f}')

            performance_threshold = self.best_eval_performance * self.performance_crash_ratio
            if current_performance < performance_threshold and self.best_eval_performance > 0:
                self.performance_crash_counter += 1
                print(f'PERFORMANCE CRASH WARNING: {current_performance:.3f} < {performance_threshold:.3f} '
                      f'({self.performance_crash_counter}/{self.performance_crash_threshold})')
            else:
                self.performance_crash_counter = 0

            eval_metrics["early_stopping/performance_crash_counter"] = self.performance_crash_counter
            eval_metrics["early_stopping/performance_threshold"] = performance_threshold

            if self.performance_crash_counter >= self.performance_crash_threshold:
                self.should_stop_training = True
                print(f'\n{"=" * 80}')
                print(f'EARLY STOPPING TRIGGERED!')
                print(
                    f'Performance has been below {self.performance_crash_ratio * 100}% of best for {self.performance_crash_counter} consecutive evaluations')
                print(f'Best performance: {self.best_eval_performance:.3f}')
                print(f'Current performance: {current_performance:.3f}')
                print(f'Threshold: {performance_threshold:.3f}')
                print(f'{"=" * 80}\n')

            self.run.log(eval_metrics, step=self.num_timesteps)

            ### CLAUDE CHANGED ###
            print(f'\nEVAL LOGGED (mean reward {mean_reward}, std {round(std_reward, 2)}, '
                  f'mean target_ids: {np.mean(target_ids_list) if target_ids_list else 0}, '
                  f'mean threat_ids: {np.mean(threat_ids_list) if threat_ids_list else 0}, '
                  f'mean switches: {np.mean(policy_switches_list) if policy_switches_list else 0})')
            ### CLAUDE CHANGED ###

            # Curriculum learning logic
            if self.use_curriculum:
                print('CURRICULUM: Checking if we should increase difficulty')
                avg_target_ids = np.mean(target_ids_list) if target_ids_list else 0
                avg_eval_len = np.mean(eval_lengths) if eval_lengths else 0

                if avg_target_ids >= self.min_target_ids_to_advance and avg_eval_len <= self.max_ep_len_to_advance:
                    self.above_threshold_counter += 1
                else:
                    self.above_threshold_counter = 0

                if self.above_threshold_counter >= 5 and self.current_difficulty < self.max_difficulty:
                    self.above_threshold_counter = 0
                    self.current_difficulty += 1
                    print(f'CURRICULUM: Increasing difficulty to level {self.current_difficulty}')

                    self.model.get_env().env_method("env", "set_difficulty", self.current_difficulty)
                    try:
                        self.eval_env.env_method("env", "set_difficulty", self.current_difficulty)
                    except Exception as e:
                        print(f"Failed to set difficulty on eval env: {e}")

                    print(f'CURRICULUM: Resetting performance tracking due to difficulty increase')
                    try:
                        self.best_eval_performance = current_performance
                    except:
                        self.best_eval_performance = -np.inf
                    self.performance_crash_counter = 0

                    self.run.log({"curriculum/difficulty_level": self.current_difficulty}, step=self.num_timesteps)

                    # Decrease LR
                    print(
                        f'Model lr reduced from {self.model.policy.optimizer.param_groups[0]["lr"]} to {self.model.policy.optimizer.param_groups[0]["lr"] / self.cl_lr_decrease}')
                    self.model.policy.optimizer.param_groups[0]['lr'] = self.model.policy.optimizer.param_groups[0][
                                                                            'lr'] / self.cl_lr_decrease

                else:
                    print(f'CURRICULUM: Maintaining difficulty at level {self.current_difficulty} '
                          f'(avg target_ids: {avg_target_ids} < threshold: {self.min_target_ids_to_advance})')

            print('#################################################\n\nReturning to training... \n')

        # if self.should_stop_training:
        #     return False
        return True

def make_env(env_config, rank, seed, run_name='no_name'):
    """
    Callable function that creates a MAISR environment. This function is passed to the vectorized environment
    instantiation in train()
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


def setup_teammate_pool(league_type, balance_method, selfplay_checkpoint_dir, pretrained_teammate_dir, overfit_test):
    """Setup teammate manager with specified league type"""

    # Create subpolicies for teammates to use
    subpolicies = {
        'local_search': LocalSearch(model_path=None),  # Using heuristic
        'change_region': ChangeRegions(model_path=None),  # Using heuristic
        'go_to_threat': GoToNearestThreat(model_path=None),  # Using heuristic
        'local_tsp_nocoord': TargetSearchLocalTSP(search_radius = 200),
        'global_tsp_nocoord': TargetSearchLocalTSP(search_radius = 1000),

        'local_tsp_yescoord': TargetSearchLocalTSP(search_radius=200, spatial_coord=True),
        'global_tsp_yescoord': TargetSearchLocalTSP(search_radius=1000, spatial_coord=True)

    }

    teammate_manager = TeammateManager(
        league_type,
        balance_method,
        subpolicies=subpolicies,
        selfplay_checkpoint_dir=selfplay_checkpoint_dir,
        pretrained_teammate_dir=pretrained_teammate_dir,
        overfit_test=overfit_test
    )

    print(f"Teammate manager setup with league_type: {league_type}")
    return teammate_manager

def train_generic(

        env_config,
        n_envs,
        project_name,
        use_normalize,
        use_teammate_manager,
        train_type, # "mode_selector" or "monolith"
        #selfplay_checkpoint_dir,
        run_name='norunname',
        save_dir="./trained_models/",
        load_path=None,
        render=False,
        log_dir="./logs/",
        machine_name='machine',
        save_model=True,
        save_checkpoints = False,
        overfit_test=None,

):
    """
    Main training pipeline. Does the following:
    1. Loads training and env config from env_config filename
    2. Sets up WandB for training logging
    3. Instantiates environments (multiprocessed vectorized environments for traning, and 1 env for eval)
    4. Instantiates training callbacks (WandB logging, checkpointing)
    5. Sets up Stable-Baselines3 PPO training
    6. Loads a prior checkpoint if provided
    7. Runs PPO training and saves checkpoints and the final model
    """

    print(f'Setting machine_name to {machine_name}. Using project {project_name}')

    if render:
        pygame.display.init()
        pygame.font.init()
        clock = pygame.time.Clock()
        ctypes.windll.user32.SetProcessDPIAware()
        window_width, window_height = config['window_size'][0], config['window_size'][1]
        config['tick_rate'] = 30
        window = pygame.display.set_mode((window_width, window_height), flags=pygame.NOFRAME)
        pygame.display.set_caption("MAISR Human Interface")

    os.makedirs(f"{save_dir}/{run_name}", exist_ok=True)
    #os.makedirs(f"./trained_models/{run_name}/", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f'./logs/action_histories/{run_name}', exist_ok=True)

    run = wandb.init(
        project=project_name,
        name=run_name+f'{machine_name}_{n_envs}envs',
        config=env_config,
        sync_tensorboard=True,
        monitor_gym=True,
    )
    run.log_code(".")

    ################################################ Initialize envs ################################################

    if env_config['num_aircraft'] > 1 and use_teammate_manager:
        teammate_manager = setup_teammate_pool(
            league_type=env_config['league_type'],
            balance_method = env_config['balance_method'],
            selfplay_checkpoint_dir=f"trained_models/{run_name}/checkpoints",
            pretrained_teammate_dir=f'trained_models/pretrained_teammates',
            overfit_test=overfit_test
        )
        print('Instantiated teammate manager')
    else:
        teammate_manager = None
        print('NOT USING a teammate manager')

    print(f"Training with {n_envs} environments in parallel")

    def make_wrapped_env(env_config, rank, seed, run_name='no_name', render=False):
        def _init():
            base_env = MAISREnvVec( # Create base environment
                config=env_config,
                render_mode='headless',
                run_name=run_name,
                tag=f'train_mp{rank}',
                seed=seed + rank,
            )

            #localsearch_model = PPO.load('trained_models/local_search_2000000.0timesteps_0.1threatpenalty_0615_1541_6envs_maisr_trained_model.zip')
            local_search_policy = LocalSearch()
            go_to_highvalue_policy = GoToNearestThreat(model_path=None)
            change_region_subpolicy = ChangeRegions(model_path=None)
            evade_policy = None

            if train_type == 'mode_selector':
                wrapped_env = MaisrModeSelectorWrapper(
                    base_env,
                    local_search_policy,
                    go_to_highvalue_policy,
                    change_region_subpolicy,
                    evade_policy,
                    teammate_manager = teammate_manager,
                    observation_noise_std = env_config['obs_noise_std_localsearch']
                )
            elif train_type == 'monolith':
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
    env_fns = [make_wrapped_env(env_config, i, env_config['seed'] + i, run_name=run_name) for i in range(n_envs)]
    if n_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    # SB3 wrappers for main env
    env = VecMonitor(env, filename=os.path.join(log_dir, 'vecmonitor'))
    if use_normalize: env = VecNormalize(env)


    # Create and wrap eval environment
    base_eval_env = MAISREnvVec(env_config,None,render_mode='headless',tag='eval',run_name=run_name,)
    if train_type == 'mode_selector':
        eval_env = MaisrModeSelectorWrapper(
                    base_eval_env,
                    LocalSearch(model_path=None),
                    GoToNearestThreat(model_path=None),
                    ChangeRegions(model_path=None),
                    None,
                    teammate_manager = teammate_manager
                )
    elif train_type == 'monolith':
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

    if use_normalize:
        eval_env = VecNormalize(eval_env, norm_reward=False, training=False)
        eval_env.obs_rms = env.obs_rms
        eval_env.ret_rms = env.ret_rms

    print('Envs created')

    ################################################# Setup callbacks #################################################
    checkpoint_callback = CheckpointCallback(
        save_freq=env_config['save_freq'] // n_envs,
        save_path=f"trained_models/{run_name}/checkpoints",
        name_prefix=f"maisr_checkpoint_{run_name}",
        save_replay_buffer=True, save_vecnormalize=True,
    )
    wandb_callback = WandbCallback(gradient_save_freq=50, verbose=1, model_save_path = None) #f"{save_dir}/{run_name}/wandb_modelsave" if save_model else None)
    if train_type == 'mode_selector':
        enhanced_wandb_callback = EnhancedWandbCallback_MS(env_config, eval_env=eval_env, run=run, log_freq=50)
    elif train_type == 'monolith':
        enhanced_wandb_callback = EnhancedWandbCallback_Monolith(env_config, eval_env=eval_env, run=run, log_freq=50)

    callbacks = [wandb_callback, enhanced_wandb_callback]
    if save_checkpoints:
        callbacks.append(checkpoint_callback)
    print('Callbacks created')

    ################################################# Setup model #################################################

    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(
            pi=[env_config['network_size'], env_config['network_size']],
            vf=[env_config['network_size'], env_config['network_size']]
        ))

    if env_config['algo'] == 'PPO':
        model = PPO(
            "CnnPolicy" if env_config['obs_type'] == 'pixel' else "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=2,
            tensorboard_log=f"logs/tb_runs/{run.id}",
            batch_size=env_config['batch_size'],
            n_steps=env_config['ppo_update_steps'],
            learning_rate=env_config['lr'],
            seed=env_config['seed'],
            device='cpu',
            gamma=env_config['gamma'],
            ent_coef=env_config['entropy_regularization'],
            clip_range=env_config['clip_range']
        )
    else:
        raise ValueError('Unsupported algo')

    print('Model instantiated')
    print(model.policy)

    print(f'Initial entropy coefficient: {model.ent_coef}')
    run.log({"entropy_decay/initial_coeff": model.ent_coef}, step=0)

    ################################################# Load checkpoint ##################################################
    if load_path:
        print(f'LOADING FROM {load_path}')
        model = model.__class__.load(load_path, env=env)
    else: print('No checkpoint provided, training new model')


    print('##################################### Beginning agent training... #######################################\n')

    # Log initial difficulty
    run.log({"curriculum/difficulty_level": 0}, step=0)
    print(f'Starting with difficulty level {0}')

    model.learn(
        total_timesteps=int(env_config['num_timesteps']),
        callback=callbacks,
        reset_num_timesteps=True if load_path else False  # TODO check this
    )

    # Save normalization stats for deployment
    stats = {
        'obs_mean': env.obs_rms.mean,
        'obs_var': env.obs_rms.var,
        'obs_count': env.obs_rms.count,
        'ret_mean': env.ret_rms.mean,
        'ret_var': env.ret_rms.var,
    }


    print("Training Normalization Stats:")
    print(f"Obs mean: {env.obs_rms.mean}")
    print(f"Obs std: {np.sqrt(env.obs_rms.var + 1e-8)}")
    print(f"Obs count: {env.obs_rms.count}")

    print('########################################## TRAINING COMPLETE ############################################\n')
    env.close()
    eval_env.close()

    # Save the final model
    if save_model:
        try:
            np.save(f"trained_models/{run_name}/{run_name}_norm_stats.npy", stats)
            env.save(f"trained_models/{run_name}/{run_name}local_search_vecnormalize.pkl")
            final_model_path = os.path.join(save_dir, f"{run_name}/{run_name}_maisr_trained_model")
            model.save(final_model_path)
            print(f"Training completed! Final model saved to {final_model_path}")
        except:
            print('Failed to save model and norm stats')

    # Run a final evaluation
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=env_config['n_eval_episodes'])
    print(f"Final evaluation: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # Log final metrics to wandb
    run.log({"final/mean_reward": mean_reward, "final/std_reward": std_reward, })
    run.finish()


if __name__ == "__main__":
    print(f'\n############################ STARTING TRAINING ############################')

    ############## ---- SETTINGS ---- ##############
    load_path = None
    config_filename = 'configs/Monolith_R7H_july9.json'
    num_envs = multiprocessing.cpu_count()
    train_type = 'monolith'
    project_name = 'maisr-rl-lab' if socket.gethostname() == 'DESKTOP-3Q1FTUP' else 'maisr-rl-pace' # 'isye-ae-2023pc3'
    machine = ('home' if socket.gethostname() == 'DESKTOP-3Q1FTUP' else 'lab' if socket.gethostname() == 'isye-ae-2023pc3' else 'pace')
    note = 'R7' + machine[0].upper()

    # Define hyperparameter sweep
    hyperparams = {
        #"network_size": [128, 196],
        #"num_observed_targets": [5],
        #"use_entropy_decay_schedule": [True, False],
        #"num_observed_threats":[1],
        #"use_stuck_detection": [False, True],
        'max_steps':[1500, 1800],
        'threat_reward_scaling':[1,1.5],
        'shaping_coeff_earlyfinish':[0.07]
        #"network_size":[128],
        #"lr": [0.001, 0.0015]
        #"team_spread_bonus_coeff": [0.0035], # 0.005,
        #"force_specific_level": [99],
        #"observe_teammate_direction":[True],
        #'entropy_regularization': [0.07],
        #"teammate_reward_scale": [0.5, 0.75],
        #"obs_noise": [0.01],
    }
    overfit_tests =  ["low_risk", "high_risk"] #["noisy_actions", "stable_actions", , "no_coord", "yes_coord"] #  "greedy_planning", "cluster_planning", ,

    param_shorthand = {
        'entropy_regularization': 'entreg',
        'teammate_reward_scale': 'trs',
        'team_spread_bonus_coeff': 'spreadbns',
        'num_observed_targets': 'obstgts',
        'num_observed_threats':'obstrts',
        'obs_noise': 'noise',
        'network_size': 'modelsize',
        "observe_teammate_direction":"obs-tmt-dir",
        "force_specific_level":"frclvl",
        "entropy_decay_schedule": "entdcy",
        "use_stuck_detection":"stuckdtct",
        "lr":"lr",
        "max_steps":'mxstps',
        'threat_reward_scaling':'thrtrwdscl',
        'shaping_coeff_earlyfinish':'erlyfnsh'
    }

    ################################################

    config = load_env_config(config_filename)
    config['n_envs'] = num_envs
    config['config_filename'] = config_filename

    import itertools
    param_names = list(hyperparams.keys())
    param_values = list(hyperparams.values())

    for param_combination in itertools.product(*param_values):
        current_params = dict(zip(param_names, param_combination))
        for param_name, param_value in current_params.items():
            config[param_name] = param_value

        param_strings = []
        for param_name, param_value in current_params.items():
            param_key = param_shorthand[param_name]
            param_strings.append(f'{param_key}-{param_value}')

        for overfit_test in overfit_tests:
            temp_identifier = 'overfit-' + overfit_test + '_' + '_'.join([s for s in param_strings if not s.startswith('overfittest-')])
            run_name = f'{note}-{train_type}_{temp_identifier}_' + generate_run_name(config)

            print(f'\n--- Starting training run with params: {current_params} ---')
            train_generic(
                config,
                run_name=run_name,
                use_normalize=True,
                use_teammate_manager=True,
                train_type = train_type,
                render=False,
                n_envs=num_envs,
                load_path=load_path,
                machine_name=('home' if socket.gethostname() == 'DESKTOP-3Q1FTUP' else 'lab' if socket.gethostname() == 'isye-ae-2023pc3' else 'pace'),
                project_name=project_name,
                save_model = True,
                save_checkpoints = True,
                overfit_test = overfit_test,
                save_dir=f"./trained_models/{train_type}/overfit_tests/" if overfit_test is not None else f'./trained_models/{train_type}',
            )
            print(f"✓ Completed training run")