import glob
import json

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import wandb
import random
import os

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

class PrintObsCallback(BaseCallback):
    """
    Custom callback to print the agent's environment observation every n steps.
    """
    def __init__(self, print_freq = 50, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            observations = self.locals.get('new_obs')
            if observations is None:
                observations = self.locals.get('obs')

            if observations is not None:
                # Print first environment's observation
                print(f"\n%%%%%%%%% [Step {self.num_timesteps}] Observation[0]: {observations[0][:3]}\n")
            else:
                print(f"[Step {self.num_timesteps}] No observations found in locals.")
        return True


class EnhancedWandbCallback(BaseCallback):
    """Custom Callback that:
    1. Logs training metrics to WandB
    2. Evaluates the agent periodically (also logged to WandB)
    3. Determines if the agent should progress to the next stage of the training curriculum
    4. Logs additional PPO training metrics

    # TODO: We plan to break this into multiple callbacks in the future.
    """

    def __init__(self, env_config, verbose=0, eval_env=None, human_eval_env=None,run=None,
                 use_curriculum=False, min_target_ids_to_advance=8, run_name='no_name',
                 log_freq=4, teammate_manager=None):
        super(EnhancedWandbCallback, self).__init__(verbose)
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

