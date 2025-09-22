import_complete = False
while not import_complete:
    try:
        import copy
        import itertools
        import ctypes
        import json
        import warnings
        import random
        import os, glob
        import pygame
        from PIL import Image
        from datetime import datetime

        from utility.localsearch_training_wrapper import MaisrLocalSearchWrapper
        import gymnasium as gym
        import numpy as np
        import multiprocessing
        import socket
        import torch
        import argparse
        import time

        import wandb
        from wandb.integration.sb3 import WandbCallback
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.callbacks import BaseCallback

        from base_env import MAISREnvVec
        from utility.league_management import TeammateManager, GenericTeammatePolicy, SubPolicy, LocalSearch, ChangeRegions, GoToNearestThreat, TargetSearchLocalTSP, RecordedTrajectoryTeammate
        from utility.config_management import load_env_config
        import_complete = True
    except:
        import_complete = False



def get_latest_checkpoint_and_vecnorm(seed: int, note_prefix: str = "pretrain") -> tuple[str, str]:
    """
    Automatically find the latest checkpoint .zip and VecNormalize .pkl for a given seed.

    Args:
        seed (int): The training seed used in the run folder name.
        note_prefix (str): The prefix used in the run folder name, e.g., "pretrainH".

    Returns:
        (load_path, vecnorm_path): Tuple of strings with the latest checkpoint and vecnormalize file paths.
    """
    # Pattern for run folder: <note>_MMDD_HHMM_seed<seed>/checkpoints
    pattern = f"outputs/{note_prefix}_*_seed{seed}/checkpoints"
    checkpoint_dirs = glob.glob(pattern)

    if not checkpoint_dirs:
        raise FileNotFoundError(f"[AutoLoad] No checkpoint directories found for seed {seed} using pattern {pattern}")

    # Use the most recently modified directory if multiple matches
    latest_dir = max(checkpoint_dirs, key=os.path.getmtime)

    # Get all checkpoint zips and vecnorm pkls
    checkpoint_zips = glob.glob(os.path.join(latest_dir, "*_checkpoint_*_steps.zip"))
    vecnorm_pkls = glob.glob(os.path.join(latest_dir, "*_checkpoint_vecnormalize_*_steps.pkl"))

    if not checkpoint_zips or not vecnorm_pkls:
        raise FileNotFoundError(f"[AutoLoad] No valid checkpoints or vecnormalize files found in {latest_dir}")

    # Helper to extract step number from file names
    def extract_step(path: str, vecnorm: bool = False) -> int:
        base = os.path.basename(path)
        if vecnorm:
            # <run>_checkpoint_vecnormalize_<steps>_steps.pkl
            step_str = base.split("_vecnormalize_")[-1].replace("_steps.pkl", "")
        else:
            # <run>_checkpoint_<steps>_steps.zip
            step_str = base.split("_checkpoint_")[-1].replace("_steps.zip", "")
        return int(step_str)

    # Pick the files with the highest step count
    latest_zip = max(checkpoint_zips, key=lambda p: extract_step(p, vecnorm=False))
    latest_pkl = max(vecnorm_pkls, key=lambda p: extract_step(p, vecnorm=True))

    print(f"[AutoLoad] Using latest checkpoint: {latest_zip}")
    print(f"[AutoLoad] Using latest vecnorm stats: {latest_pkl}")

    return latest_zip, latest_pkl


class LeagueTypeTransitionCallback(BaseCallback):
    """
    Custom callback that transitions the league type after a preset number of timesteps.
    This allows for curriculum-style league training where you start with one league type
    and transition to another (e.g., start with selfplay, then transition to mixed training).
    """

    def __init__(self,
                 transition_timesteps: float,
                 initial_league_type: str,
                 target_league_type: str,
                 eval_env=None,
                 run=None,
                 verbose: int = 1):
        """
        Args:
            transition_timesteps (int): Number of timesteps after which to transition
            initial_league_type (str): Starting league type (e.g., 'selfplay')
            target_league_type (str): League type to transition to (e.g., 'mixed50')
            eval_env: Evaluation environment (optional)
            run: WandB run object for logging (optional)
            verbose (int): Verbosity level
        """
        super(LeagueTypeTransitionCallback, self).__init__(verbose)

        self.transition_timesteps = transition_timesteps
        self.initial_league_type = initial_league_type
        self.target_league_type = target_league_type
        self.eval_env = eval_env
        self.run = run

        self.transition_completed = False
        self.transition_logged = False

        # Validate league types
        valid_league_types = ['selfplay', 'strategy_diverse', 'mixed25', 'mixed50', 'mixed75', 'fcp']
        if initial_league_type not in valid_league_types:
            raise ValueError(f"Invalid initial_league_type: {initial_league_type}. Must be one of {valid_league_types}")
        if target_league_type not in valid_league_types:
            raise ValueError(f"Invalid target_league_type: {target_league_type}. Must be one of {valid_league_types}")

        print(
            f"[League Transition] Initialized: {initial_league_type} → {target_league_type} at step {transition_timesteps}")

    # def _on_training_start(self) -> None:
    #     """Called at the start of training to ensure initial league type is set."""
    #     try:
    #         # Set initial league type for training environment
    #         self.model.get_env().env_method("set_league_type", self.initial_league_type)
    #
    #         # Set initial league type for eval environment if provided
    #         if self.eval_env is not None:
    #             self.eval_env.env_method("set_league_type", self.initial_league_type)
    #
    #         # Log initial state
    #         if self.run is not None:
    #             self.run.log({
    #                 "league_transition/current_league_type": self.initial_league_type,
    #                 "league_transition/transition_timesteps": self.transition_timesteps,
    #                 "league_transition/target_league_type": self.target_league_type
    #             }, step=0)
    #
    #         if self.verbose >= 1:
    #             print(f"[League Transition] Set initial league type to: {self.initial_league_type}")
    #
    #     except Exception as e:
    #         print(f"[League Transition] Warning: Failed to set initial league type: {e}")

    def _on_step(self) -> bool:
        """Called at each training step to check if transition should occur."""

        # Check if it's time to transition
        if not self.transition_completed and self.num_timesteps >= self.transition_timesteps:
            self._execute_transition()

        return True

    def _execute_transition(self):
        """Execute the league type transition."""
        try:
            print(f'\n{"=" * 80}')
            print(f'LEAGUE TYPE TRANSITION TRIGGERED! (step {self.num_timesteps})')
            print(f'Transitioning from {self.initial_league_type} to {self.target_league_type}')
            print(f'{"=" * 80}\n')

            # Transition training environment
            self.model.get_env().env_method("set_league_type", self.target_league_type)

            # Transition eval environment if provided
            if self.eval_env is not None:
                self.eval_env.env_method("set_league_type", self.target_league_type)

            # Mark transition as completed
            self.transition_completed = True

            # Log the transition
            if self.run is not None:
                self.run.log({
                    #"league_transition/transition_executed": True,
                    "league_transition/transition_step": self.num_timesteps,
                    #"league_transition/current_league_type": self.target_league_type,
                    #"league_transition/previous_league_type": self.initial_league_type
                }, step=self.num_timesteps // self.model.get_env().num_envs)

            if self.verbose >= 1:
                print(f"[League Transition] Successfully transitioned to: {self.target_league_type}")

        except Exception as e:
            print(f"[League Transition] Error during transition: {e}")
            raise ValueError
            # Don't mark as completed if there was an error, so it can retry

    def get_current_league_type(self) -> str:
        """Get the currently active league type."""
        if self.transition_completed:
            return self.target_league_type
        else:
            return self.initial_league_type

class PrintObsEvery50Steps(BaseCallback):
    """
    Custom callback to print the agent's environment observation every 50 steps.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if self.num_timesteps % 50 == 0:
            observations = self.locals.get('new_obs')
            if observations is None:
                observations = self.locals.get('obs')

            if observations is not None:
                # Print first environment's observation
                print(f"\n%%%%%%%%% [Step {self.num_timesteps}] Observation[0]: {observations[0][:3]}\n")
            else:
                print(f"[Step {self.num_timesteps}] No observations found in locals.")
        return True


class EnhancedWandbCallback_Monolith(BaseCallback):
    """Custom Callback that:
    1. Logs training metrics to WandB
    2. Evaluates the agent periodically (also logged to WandB)
    3. Determines if the agent should progress to the next stage of the training curriculum
    4. Logs additional PPO training metrics
    """

    def __init__(self, env_config, verbose=0, eval_env=None, human_eval_env=None,run=None,
                 use_curriculum=False, min_target_ids_to_advance=8, run_name='no_name',
                 log_freq=4, teammate_manager=None):
        super(EnhancedWandbCallback_Monolith, self).__init__(verbose)
        self.avg_mean_diffs = []
        self.avg_var_diffs = []
        self.config = env_config
        self.eval_env = eval_env
        self.human_eval_env = human_eval_env

        self.eval_freq = env_config['eval_freq']
        self.n_eval_episodes = env_config['n_eval_episodes']
        self.run = run
        self.run_name = run_name
        self.log_freq = log_freq  # Log every N steps instead of every step

        self.run_human_eval = env_config["run_human_eval"]
        # if self.run_human_eval:
        #     self.human_eval_env = copy.deepcopy(eval_env)
        #     self.human_eval_env.envs[0].env.env.tag = "human_eval0"

        self.use_curriculum = env_config['use_curriculum']
        self.min_target_ids_to_advance = env_config['min_target_ids_to_advance']
        self.max_ep_len_to_advance = 130
        self.max_difficulty = env_config['max_difficulty']

        self.current_difficulty = 0
        self.above_threshold_counter = 0

        self.switched_to_twoship = False
        self.twoship_switch_threshold = env_config['twoship_switch_threshold']
        #self.twoship_switch_reward_threshold = 23

        # Entropy decay parameters
        self.use_entropy_decay_schedule = env_config['use_entropy_decay_schedule']
        self.entropy_decay_enabled = False
        self.entropy_decay_trigger_threshold = env_config['entropy_decay_trigger_threshold']  # mean_target_ids_per_step threshold
        self.entropy_decay_threat_threshold = env_config['entropy_decay_threat_threshold'] # mean_threat_ids threshold
        self.entropy_decay_steps = env_config['entropy_decay_steps'] # Decay over this many steps
        self.entropy_final_ratio = env_config['entropy_final_ratio']  # Final entropy = 50% of original
        self.entropy_decay_start_step = None
        self.original_entropy_coeff = None

        # For mixed league training league updates
        self.ratio_schedule = env_config.get("ratio_schedule", {})  # dict: {reward_threshold: new_ratio}
        self.ratio_update_milestones = set()  # track which thresholds we’ve already applied

        # Buffer for accumulating data between log events
        self.episode_buffer = {
            'rewards': [],
            'lengths': [],
            'target_ids': [],
            'threat_ids': [],
            'detections': [],
            'teammate_names': []
        }

        # Early stopping based on performance degradation
        self.best_eval_performance = -np.inf
        self.performance_crash_counter = 0
        self.performance_crash_threshold = 20  # Number of consecutive poor evals before stopping
        self.performance_crash_ratio = 0.4  # Performance must drop below 50% of best
        self.should_stop_training = False

        self.teammate_manager = teammate_manager

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

                    if "teammate_name" in info:
                        self.episode_buffer['teammate_names'].append(info["teammate_name"])


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

            if self.config['league_type'] != "selfplay":
                if self.episode_buffer['teammate_names']:
                    from collections import Counter
                    teammate_counts = Counter(self.episode_buffer['teammate_names'])
                    total_episodes = len(self.episode_buffer['teammate_names'])

                    # Create a bar chart data structure for WandB
                    teammate_freq_data = []
                    for teammate_name, count in teammate_counts.items():
                        teammate_freq_data.append([teammate_name, count, count / total_episodes])

                    # Log as a table that WandB can convert to a bar chart
                    log_data["teammate_frequencies/teammate_frequency_table"] = wandb.Table(
                        data=teammate_freq_data,
                        columns=["teammate_name", "count", "frequency"]
                    )

                    # Also log individual frequencies for easier tracking
                    for teammate_name, count in teammate_counts.items():
                        # Clean the name for WandB (replace special characters)
                        clean_name = teammate_name.replace("/", "_").replace(" ", "_")
                        log_data[f"teammate_frequencies/teammate_freq_{clean_name}"] = count / total_episodes

            if log_data: # Log the aggregated data
                self.run.log(log_data, step=self.num_timesteps // self.model.get_env().num_envs)

            # Clear the buffer after logging
            self.episode_buffer = {
                'rewards': [],
                'lengths': [],
                'target_ids': [],
                'threat_ids': [],
                'detections': [],
                'teammate_names': []
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
            print(f'\n################################################# EVALUATING (step {self.num_timesteps}) #################################################')

            target_ids_list = []
            threat_ids_list = []
            target_ids_per_step_list = []
            mean_reward, std_reward = 0, 0
            total_eval_reward = 0
            eval_lengths = []
            teammate_names = []
            eval_episode_data_list = []

            level_metrics = {}

            obs = self.eval_env.reset()
            for i in range(self.n_eval_episodes):
                done = False
                ep_reward, ep_target_ids, ep_threat_ids = 0, 0, 0

                try:
                    #teammate_names.append(self.eval_env.get_wrapper_attr("current_teammate").name)
                    teammate_names.append(self.eval_env.envs[0].current_teammate.name)

                except Exception as e:
                    print('Error, failed to get teammate name using get_wrapper_attr')
                    print(e)
                    raise ValueError

                while not done:

                    action, other = self.model.predict(obs, deterministic=True)
                    #action = action[0]

                    # if isinstance(action, np.ndarray) and action.ndim > 0 and action.shape[0] > 1:
                    #     print(f"[Eval] Warning: Got vector action {action}, using first element for eval")
                    #     action = action[0]

                    obses, rewards, dones, infos = self.eval_env.step([action])
                    obs = obses[0]
                    reward = rewards[0]
                    info = infos[0]
                    done = dones[0]

                    ep_reward += reward
                    ep_target_ids += info['new_target_ids']
                    ep_threat_ids += info['new_threat_ids']

                    final_info = info

                ep_length = final_info["episode"]["l"]

                level_idx = self.eval_env.envs[0].env.env.level_idx
                #print(f'eval level idx is {level_idx}')
                #self.eval_env.envs[0].env.level_idx
                if level_idx not in level_metrics:
                    level_metrics[level_idx] = {
                        "rewards": [],
                        "target_ids": [],
                        "threat_ids": [],
                        "episode_lengths": [],
                        "target_ids_per_step": []
                    }

                level_metrics[level_idx]["rewards"].append(ep_reward)
                level_metrics[level_idx]["target_ids"].append(ep_target_ids)
                level_metrics[level_idx]["threat_ids"].append(ep_threat_ids)
                level_metrics[level_idx]["episode_lengths"].append(ep_length)
                level_metrics[level_idx]["target_ids_per_step"].append(ep_target_ids / ep_length)

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

            # Add per-level eval metrics
            for level, metrics in level_metrics.items():
                if len(metrics["rewards"]) == 0:
                    continue
                eval_metrics.update({
                    f"eval_levels/level{level}_reward": np.mean(metrics["rewards"]),
                    f"eval_levels/level{level}_target_ids": np.mean(metrics["target_ids"]),
                    f"eval_levels/level{level}_threat_ids": np.mean(metrics["threat_ids"]),
                    #f"eval/level{level}_episode_length": np.mean(metrics["episode_lengths"]),
                    #f"eval/level{level}_target_ids_per_step": np.mean(metrics["target_ids_per_step"]),
                })

            #main_tag = self.eval_env.envs[0].env.env.tag
            if self.run_human_eval:
                print("\n\n ++++++++++++++++ [Eval] Running additional evaluation with recorded human trajectory ++++++++++++++++ \n")

                base_human_env = self.human_eval_env.envs[0].env.env
                base_human_env.tag = 'human_eval0'

                recorded_teammate_indices = [0, 1]  # <-- set to your actual indices
                num_trajectories = len(recorded_teammate_indices)

                target_ids_list, threat_ids_list, target_ids_per_step_list = [], [], []
                mean_reward, std_reward, total_eval_reward = 0, 0, 0
                eval_lengths = []
                teammate_names = []

                timescale_correction = 10

                for level in range(7):

                    # Pick a random teammate idx
                    rand_idx = random.choice(recorded_teammate_indices)

                    # Grab all json trajectories for that subject and level
                    traj_pattern = f"./human_trajectories/subject_{rand_idx}/timesteps_A{level+1}_*.json"
                    candidate_files = glob.glob(traj_pattern)
                    if not candidate_files:
                        print(f"[Eval] No trajectories found for subject {rand_idx} level {level}")
                        continue

                    # Pick a random json trajectory file
                    trajectory_file = random.choice(candidate_files)

                    base_human_env.level_idx = level
                    base_human_env.config['force_specific_level'] = level


                    with open(trajectory_file, 'r') as f:
                        data = json.load(f)

                    if isinstance(data, dict) and "timesteps" in data:
                        timesteps = data["timesteps"]
                    elif isinstance(data, list):
                        timesteps = data

                    waypoints = [entry["human_custom_waypoint"] for entry in timesteps]
                    waypoints = waypoints[::10] # Timescale correction


                    current_pos = base_human_env.agents[base_human_env.aircraft_ids[1]].x, base_human_env.agents[base_human_env.aircraft_ids[1]].y
                    waypoints = [wp if wp is not None else current_pos for wp in waypoints]
                    #print(f"[Eval] Waypoints: {waypoints}")


                    print(f'Selected human trajectory {rand_idx}. Loaded trajectory from trajectory_file with timescale correction {timescale_correction}')
                    obs = self.human_eval_env.reset()
                    done = False
                    ep_reward, ep_target_ids, ep_threat_ids = 0, 0, 0
                    step_idx = 0

                    while not done:
                        if step_idx < len(waypoints):
                            base_human_env.agents[base_human_env.aircraft_ids[1]].waypoint_override = tuple(waypoints[step_idx])
                        else:
                            base_human_env.agents[base_human_env.aircraft_ids[1]].waypoint_override = tuple(waypoints[-1])  # hold last

                        action, other = self.model.predict(obs, deterministic=True)

                        obses, rewards, dones, infos = self.human_eval_env.step([action])

                        obs = obses[0]
                        reward = rewards[0]
                        info = infos[0]
                        done = dones[0]

                        ep_reward += reward
                        ep_target_ids += info['new_target_ids']
                        ep_threat_ids += info['new_threat_ids']

                        step_idx += 1

                        final_info = info

                    ep_length = final_info["episode"]["l"]
                    target_ids_list.append(ep_target_ids)
                    threat_ids_list.append(ep_threat_ids)

                    eval_lengths.append(ep_length)
                    target_ids_per_step_list.append(ep_target_ids / ep_length)

                    total_eval_reward += ep_reward

                mean_reward = total_eval_reward / self.n_eval_episodes

                # Log evaluation results
                eval_metrics.update({
                    "eval_with_human/mean_reward": mean_reward,
                    "eval_with_human/mean_target_ids": np.mean(target_ids_list) if target_ids_list else 0,
                    "eval_with_human/mean_threat_ids": np.mean(threat_ids_list) if threat_ids_list else 0,
                    "eval_with_human/mean_episode_length": np.mean(eval_lengths) if eval_lengths else 0,
                    "eval_with_human/mean_target_ids_per_step": np.mean(target_ids_per_step_list) if target_ids_per_step_list else 0,
                    "curriculum/difficulty_level": self.current_difficulty
                })

                self.run.log({"eval_with_human/mean_reward": mean_reward}, step=self.num_timesteps)
                #except Exception as e:
                    #print(f"[Eval] Failed to run recorded teammate eval: {e}")

            print("++++++++ [Human Eval] Human eval complete ++++++++\n")
            self.eval_env.envs[0].env.env.config['force_specific_level'] = 99
            #self.eval_env.envs[0].env.env.tag = main_tag


            ###################### === Dynamic League Ratio Update Based on Evaluation Reward === ######################
            for threshold, new_ratio in self.ratio_schedule.items():
                if mean_reward >= threshold and threshold not in self.ratio_update_milestones:
                    print(f'\n[League Ratio Update] Reward {mean_reward:.2f} exceeded threshold {threshold}, updating league ratio to {new_ratio}')
                    try:
                        self.model.get_env().env_method("change_league_ratio", new_ratio)
                        self.ratio_update_milestones.add(threshold)
                        self.run.log({
                            "league_ratio/update_triggered": True,
                            "league_ratio/new_ratio": new_ratio,
                            "league_ratio/trigger_step": self.num_timesteps,
                            "league_ratio/trigger_threshold": threshold
                        }, step=self.num_timesteps)
                    except Exception as e:
                        print(f"[League Ratio Update] Failed to update league ratio: {e}")

            ### Tracking teammate frequency
            try:
                counts = {
                    "Diverse": sum(1 for name in teammate_names if name.startswith("Diverse_")),
                    "SelfPlay": sum(1 for name in teammate_names if name.startswith("SelfPlay_")),
                    "Pretrained": sum(1 for name in teammate_names if name.startswith("Pretrained_")),
                    "Total": len(teammate_names)
                }
                ratios = {
                    "eval_teammate_ratios/teammate_ratio_diverse": counts["Diverse"] / counts["Total"] if counts["Total"] else 0,
                    "eval_teammate_ratios/teammate_ratio_selfplay": counts["SelfPlay"] / counts["Total"] if counts["Total"] else 0,
                    "eval_teammate_ratios/teammate_ratio_pretrained": counts["Pretrained"] / counts["Total"] if counts["Total"] else 0
                }
                self.run.log(ratios, step=self.num_timesteps)
            except Exception as e:
                print(f'Failed to log teammate frequency counts: {e}')

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
                    eval_metrics["monitoring/current_entropy_coeff"] = new_entropy_coeff
                    #eval_metrics["entropy_decay/decay_progress"] = decay_progress
                    #eval_metrics["entropy_decay/steps_since_trigger"] = steps_since_trigger

                    if decay_progress >= 1.0:
                        eval_metrics["entropy_decay/completed"] = True

            # Always log current entropy coefficient
            eval_metrics["train/current_entropy_coeff"] = self.model.ent_coef


            #################################### Aircraft switching ####################################
            #print(f'About to check for 2 ship switch: self.switched_to_twoship = {self.switched_to_twoship}, target_ids_list = {target_ids_list}')
            if (not self.switched_to_twoship) and target_ids_list:
                avg_target_ids = np.mean(target_ids_list)
                if avg_target_ids > self.twoship_switch_threshold:
                #if mean_reward > self.twoship_switch_reward_threshold:
                    print(f'\n{"=" * 80}')
                    print(f'AIRCRAFT SWITCHING TRIGGERED! (step {self.num_timesteps})')
                    #print(f'Average target IDs ({avg_target_ids:.2f}) exceeded threshold ({self.twoship_switch_threshold})')
                    print(f'Switching from 1 aircraft to 2 aircraft...')
                    print(f'{"=" * 80}\n')

                    self.model.get_env().env_method("set_teammate_active", True)
                    self.eval_env.env_method("set_teammate_active", True)
                    self.switched_to_twoship = True

                    # Log the switch
                    eval_metrics["monitoring/num_aircraft"] = 2
                    eval_metrics["monitoring/2aircraft_switch_step"] = self.num_timesteps


                #if not self.switched_to_twoship:
                eval_metrics["monitoring/num_aircraft"] = 1 if not self.switched_to_twoship else 2

            self.run.log(eval_metrics, step=self.num_timesteps)

            print(f'\n ########## EVAL LOGGED (mean reward {round(mean_reward,1)}, std {round(std_reward, 2)}, 'f'mean target_ids: {round(np.mean(target_ids_list),2) if target_ids_list else 0} ##########\n')


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


                else:
                    print(f'CURRICULUM: Maintaining difficulty at level {self.current_difficulty} '
                          f'(avg target_ids: {avg_target_ids} < threshold: {self.min_target_ids_to_advance})')
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


def setup_teammate_pool(league_type, balance_method, selfplay_checkpoint_dir, pretrained_teammate_dir, overfit_test, fcp_ratio=1.0):
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

    #print(f"        Teammate manager setup with league_type: {league_type}")
    return teammate_manager

def train_generic(
        env_config,
        n_envs,
        project_name,
        use_normalize,
        use_teammate_manager,
        train_type, # "mode_selector" or "monolith"
        run_name='norunname',
        load_path=None,
        vecnorm_load_path=None,
        render=False,
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

    if vecnorm_load_path is None and load_path is not None:
        raise ValueError('Provided model path without vecnorm stats')

    #paths = get_output_paths(run_name)
    print('\n[train_generic] Initializing...')

    print(f'        Setting machine_name = {machine_name} \n        WandB project = {project_name}')

    if render:
        pygame.display.init()
        pygame.font.init()
        clock = pygame.time.Clock()
        ctypes.windll.user32.SetProcessDPIAware()
        window_width, window_height = config['window_size'][0], config['window_size'][1]
        config['tick_rate'] = 30
        window = pygame.display.set_mode((window_width, window_height), flags=pygame.NOFRAME)
        pygame.display.set_caption("MAISR Human Interface")

    print('\nCreating output folders:')
    for subfolder in ['episode_plots','trained_models', 'checkpoints','vecnorm_stats','logs']:
        folder_name = f"outputs/{run_name}/{subfolder}"
        try:
            os.makedirs(folder_name, exist_ok=True)
        except:
            print(f'failed to create folder {subfolder}, retrying...')
            os.makedirs(folder_name, exist_ok=True)
        print(f'        {folder_name}')
    print('\n')

    init_successful = False
    while not init_successful:
        try:
            run = wandb.init(
                project=project_name,
                name=run_name+f'{machine_name}_{n_envs}envs',
                config=env_config,
                sync_tensorboard=True,
                monitor_gym=True,
            )
            init_successful = True
        except:
            print('         WandB init failed, retrying')
            init_successful = False
        if init_successful:
            print(f'        WandB init successful')
            break

    run.log_code(".")

    ################################################ Initialize envs ################################################

    if env_config['num_aircraft'] > 1 and use_teammate_manager:
        teammate_manager = setup_teammate_pool(
            league_type=env_config['league_type'],
            balance_method = env_config['balance_method'],
            selfplay_checkpoint_dir=f"outputs/{run_name}/checkpoints",
            pretrained_teammate_dir=f'trained_models/pretrained_teammates',
            overfit_test=overfit_test,
        )
        print('        Instantiated teammate manager')
    else:
        teammate_manager = None
        print('        Not using a teammate manager')

    print(f"Training with {n_envs} environments in parallel\n")

    def make_wrapped_env(env_config, rank, seed, run_name='no_name', render=False):
        def _init():

            if rank != 0:
                import sys
                import os
                sys.stdout = open(os.devnull, 'w')

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
    env = VecMonitor(env, filename=f'outputs/{run_name}/logs/{run_name}vecmonitor')

    if use_normalize:
        if vecnorm_load_path is not None:
            env = VecNormalize.load(vecnorm_load_path, venv=env)
            env.training = True
            env.norm_reward = True
        else:
            env = VecNormalize(env)
            env.training = True
            env.norm_reward = True


    # Create and wrap eval environment
    base_eval_env = MAISREnvVec(env_config,None,render_mode='headless',tag='eval',run_name=run_name)
    eval_env = MaisrLocalSearchWrapper(
        base_eval_env,
        env_config['obs_noise_std_localsearch'],
        LocalSearch(model_path=None),
        GoToNearestThreat(model_path=None),
        ChangeRegions(model_path=None),
        None,
        teammate_manager=teammate_manager)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])


    base_human_eval_env = MAISREnvVec(env_config, None, render_mode='headless', tag='human_eval0', run_name=run_name)
    human_eval_env = MaisrLocalSearchWrapper(
        base_human_eval_env,
        env_config['obs_noise_std_localsearch'],
        LocalSearch(model_path=None),
        GoToNearestThreat(model_path=None),
        ChangeRegions(model_path=None),
        None,
        teammate_manager=teammate_manager)
    human_eval_env = Monitor(human_eval_env)
    human_eval_env = DummyVecEnv([lambda: human_eval_env])


    if use_normalize:
        if vecnorm_load_path is not None:
            eval_env = VecNormalize.load(vecnorm_load_path, venv=eval_env)
            eval_env.norm_reward = False
            eval_env.training = False
        else:
            eval_env = VecNormalize(eval_env, norm_reward=False, training=False)
        eval_env.obs_rms = env.obs_rms
        eval_env.ret_rms = env.ret_rms

    print('        Envs created')

    ################################################# Setup callbacks #################################################

    wandb_callback = WandbCallback(gradient_save_freq=50, verbose=1, model_save_path = None) #f"{save_dir}/{run_name}/wandb_modelsave" if save_model else None)

    enhanced_wandb_callback = EnhancedWandbCallback_Monolith(
        env_config,
        eval_env=eval_env,
        human_eval_env=human_eval_env,
        run=run,
        log_freq=75,
        teammate_manager=teammate_manager
    )

    printcallback = PrintObsEvery50Steps(verbose=1)

    callbacks = [wandb_callback, enhanced_wandb_callback]  # printcallback

    if env_config['switch_leagues']:
        league_transition_callback = LeagueTypeTransitionCallback(
            transition_timesteps=2e6,  # Transition after this many steps
            initial_league_type='selfplay',
            target_league_type='strategy_diverse',
            eval_env=eval_env,
            run=run,
            verbose=1
        )

        callbacks.append(league_transition_callback)


    if save_checkpoints:
        checkpoint_callback = CheckpointCallback(
            save_freq=env_config['save_freq'] // n_envs,
            save_path=f"outputs/{run_name}/checkpoints",
            # save_path=paths["checkpoints"],
            name_prefix=f"{run_name}_checkpoint",
            save_replay_buffer=True, save_vecnormalize=True,
        )

        callbacks.append(checkpoint_callback)
    print('        Callbacks created')

    ################################################# Setup model #################################################

    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(
            pi=[env_config['network_size']] * env_config['network_numlayers'],
            vf=[env_config['network_size']] * env_config['network_numlayers']
        ))

    if env_config['algo'] == 'PPO':
        model = PPO(
            "CnnPolicy" if env_config['obs_type'] == 'pixel' else "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=2,
            tensorboard_log=f"outputs/logs/tb_runs/{run.id}",
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

    print('        Model instantiated\n')
    print(model.policy)

    if teammate_manager is not None:
        teammate_manager.set_current_model(model)
        if use_normalize and hasattr(env, 'obs_rms'):
            teammate_manager.set_normalization_stats(env.obs_rms, env.ret_rms)

    ################################################# Load checkpoint ##################################################
    if load_path:
        print(f'        Checkpoint: Loading from {load_path}')
        model = PPO.load(load_path, env=env)
    else: print('        Checkpoint: None provided, training new model')

    # Log initial difficulty
    run.log({"curriculum/difficulty_level": 0}, step=0)

    # === Save initial checkpoint immediately ===
    if save_checkpoints:
        initial_checkpoint_path = f"outputs/{run_name}/checkpoints/{run_name}_checkpoint_0_steps.zip"
        vecnormalize_path =  f"outputs/{run_name}/checkpoints/{run_name}_checkpoint_vecnormalize_0_steps.pkl"
        model.save(initial_checkpoint_path)
        if isinstance(env, VecNormalize):
            env.save(vecnormalize_path)
        print(f"[Startup] Initial checkpoint saved to {initial_checkpoint_path}")

    teammate_manager._create_selfplay_teammate()
    teammate_manager.current_teammate.env = env

    print('\n\n###### Running model.learn... ######\n')
    model.learn(
        total_timesteps=int(env_config['num_timesteps']),
        callback=callbacks,
        reset_num_timesteps=False if load_path else True
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

    print('\n#########################################################################################################')
    print('########################################## TRAINING COMPLETE ############################################\n')
    print('#########################################################################################################')
    env.close()
    eval_env.close()

    # Save the final model
    if save_model:
        try:
            np.save(f"outputs/{run_name}/trained_models/{run_name}_norm_stats.npy", stats)
            env.save(f"outputs/{run_name}/vecnorm_stats/{run_name}local_search_vecnormalize.pkl")
            final_model_path = f'outputs/{run_name}/trained_models/{run_name}_model.zip'  #os.path.join(save_dir, f"{run_name}/{run_name}_model.zip")

            model.save(final_model_path)
            print(f"Training completed!\nFinal model saved to {final_model_path}")
        except:
            print('Failed to save model and norm stats')

    # Run a final evaluation
    print('Running final eval:')
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=env_config['n_eval_episodes'])
    print(f"\nFinal evaluation: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # Log final metrics to wandb
    run.log({"final/mean_reward": mean_reward, "final/std_reward": std_reward, })
    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='MAISR RL training script')
    parser.add_argument('--version', required=True, help='Which training version to run. You can define multiple versions later in this script')
    parser.add_argument('--seed', required=True, help='Seed to run')
    parser.add_argument('--testing', action='store_true', help='Set to testing mode. Simplifies some aspects of training for faster debugging')

    args = parser.parse_args()
    version = args.version
    condition = args.condition

    print(f'\n############################ STARTING TRAINING ############################')

    ############## ---- SETTINGS ---- ##############
    config_filename = 'configs/Monolith_index_August.json'
    num_envs = 2 if args.testing else multiprocessing.cpu_count() # Use all CPU cores for multiprocessing, but only use 2 if args.testing (for faster init)
    train_type = 'monolith' # What type of agent to train. "monolith" for a single policy that chooses directional or target index control. "mode_selector" for a hybrid agent that chooses subpolicies (not currently implemented)
    project_name = 'maisr-rl-mixedtraining'#'insert_wandb_project_name'
    machine = socket.gethostname()

    config = load_env_config(config_filename)

    # Add parameters to the config so they're logged
    config['seed'] = int(args.seed)
    config['n_envs'] = num_envs
    config['config_filename'] = config_filename

    # An example of different training versions you can set up here. Specify using the --version arg.
    if version == 'main':
        run_prefix = '' + machine[0].upper()
        project_name = 'Add your project name here' # For WandB

        # If you want to sweep over multiple hyperparameter settings, you can define them here. These will override the values in the config.json
        hyperparams = {
            "network_size": [128, 196],
            "lr": [0.001, 0.0015],
            'entropy_regularization': [0.07, 0.08],
            "teammate_reward_scale": [0.5, 0.75],
            "teammate_active_at_start": True,
            "league_type": 'selfplay',
            "obs_noise": [0.00, 0.01],
        }

        load_path = None # You can specify a policy .zip file here if you want to continue training from a prior run
        vecnorm_load_path = None # Specify the path to the above policy's vecnormalize .pkl file here.
        config['load_path'] = load_path

    # Shorthand names for hyperparameters to reduce length of run names
    param_shorthand = {
        'entropy_regularization': 'entreg',
        'teammate_reward_scale': 'trs',
        'team_spread_bonus_coeff': 'spreadbns',
        'num_observed_targets': 'obstgts',
        'num_observed_threats': 'obstrts',
        'obs_noise': 'noise',
        'network_size': 'modelsize',
        "observe_teammate_direction": "obs-tmt-dir",
        "force_specific_level": "frclvl",
        "entropy_decay_schedule": "entdcy",
        "use_stuck_detection": "stuckdtct",
        "lr": "lr",
        "potential_ratio":"potratio",
        "max_steps": 'mxstps',
        'threat_reward_scaling': 'thrtrwdscl',
        'shaping_coeff_earlyfinish': 'erlyfnsh',
        'entropy_decay_steps': 'entdcystps',
        'seed': 'seed',
        "gamma":"gamma",
        'league_type':'lgtype',
        'quick_id_shaping_coeff':'quick_id_cf',
        'use_dynamic_potential':'dynpotential',
        "use_teammate_priority_shaping":"tmtprishaping",
        "switch_leagues":"switch_lgs"
    }

    if args.testing:
        config["eval_freq"] = 50
        config['num_eval_episodes'] = 5
        config['save_freq'] = 500
        config['num_timesteps'] = 5e5
        project_name = 'maisr-tests'

    ################################################

    param_names = list(hyperparams.keys())
    param_values = list(hyperparams.values())

    # Loop through the hyperparameters selected in the version block above
    for param_combination in itertools.product(*param_values):
        current_params = dict(zip(param_names, param_combination))
        for param_name, param_value in current_params.items():
            config[param_name] = param_value

        param_strings = []
        for param_name, param_value in current_params.items():
            try: param_key = param_shorthand[param_name]
            except: param_key = param_name
            param_strings.append(f'{param_key}-{param_value}')

        temp_identifier = '_'.join([s for s in param_strings if not s.startswith('overfittest-')])
        #
        run_name = f'{run_prefix}_' + datetime.now().strftime("%m%d_%H%M") + f'_seed{str(args.seed)}' + temp_identifier


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
            vecnorm_load_path=vecnorm_load_path,
            machine_name=('home' if socket.gethostname() == 'DESKTOP-3Q1FTUP' else 'lab' if socket.gethostname() == 'isye-ae-2023pc3' else 'pace'),
            project_name=project_name,
            save_model = True,
            save_checkpoints = True,
            #save_dir=f'./outputs/trained_models/',
        )
        print(f"✓ Completed training run")