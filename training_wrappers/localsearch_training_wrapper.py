import random

import gymnasium as gym
import numpy as np
from utility.league_management import TeammateManager, TeammatePolicy, RecordedTrajectoryTeammate


class MaisrLocalSearchWrapper(gym.Env):
    """Wrapper for training the mode selector
    Subpolicies are treated as part of the environment dynamics.
    """
    def __init__(self,
                 env,
                 obs_noise_std,
                 local_search_policy=None, # For teammate
                 go_to_highvalue_policy=None, # For teammate
                 change_region_subpolicy=None, # For teammate
                 evade_policy=None, # For teammate
                 teammate_manager: TeammateManager = None,
                 teammate_policy: TeammatePolicy = None
                 ):

        self.env = env

        # Define observation space
        self.obs_size = self.env.obs_size
        self.observation_space = gym.spaces.Box(
            low=-1, high=1,
            shape=(self.obs_size,),
            dtype=np.float32)

        # Action space: 3 possible sub-policies to choose from
        #self.action_space = gym.spaces.Discrete(16)
        if self.env.config['action_type'] == 'target_index':
            # Total selectable entities: targets + threats
            #total_entities = self.env.config['num_targets'] + self.env.config['num_threats']
            total_observed_entities = self.env.config['num_observed_targets'] + self.env.config['num_observed_threats']
            self.action_space = gym.spaces.Discrete(total_observed_entities)

        elif self.env.config['action_type'] == 'Discrete8':
            self.action_space = gym.spaces.Discrete(8)  # 8 directions
        elif self.env.config['action_type'] == 'Discrete16':
            self.action_space = gym.spaces.Discrete(16)  # 16 directions
        else:
            self.action_space = gym.spaces.Discrete(16)  # Default to 16 directions
        self.action_rate = 1

        self.render_mode = self.env.render_mode
        self.run_name = self.env.run_name  # For logging
        self.tag = self.env.tag

        self.obs_noise_std = obs_noise_std

        self.teammate_active = self.env.config['teammate_active_at_start']

        if teammate_policy is not None and teammate_manager is not None:
            raise ValueError('Cannot specify a teammate policy and a teammate manager at the same time')

        self.teammate_manager = teammate_manager
        self.teammate_policy = teammate_policy

        self.local_search_policy = local_search_policy
        self.go_to_highvalue_policy = go_to_highvalue_policy
        self.change_region_subpolicy = change_region_subpolicy
        self.evade_policy = evade_policy

        self.current_teammate = None
        self.teammate_subpolicy_choice = 0

        # For teammate decision speed
        self.teammate_decision_timer = 0
        self.teammate_cached_direction = None
        self.teammate_decision_interval = 5  # Change direction only every N steps

        self.periodic_pause_interval = 7  # how often to pause
        self.periodic_pause_duration = 2  # how long to pause
        self.periodic_step_counter = 0
        self.pause_toggle = False
        self.last_real_direction = None

        # For detecting stuck agent
        if self.env.config['use_stuck_detection']:
            self.recent_actions = []
            self.action_history_length = 10  # Can be tuned
            self.oscillation_threshold = 3.5  # Max average difference before declaring stuck

            self.position_history = []  # Track recent positions
            self.history_length = 10  # Number of positions to track
            self.stuck_threshold = 4  # Pixel distance threshold for being "stuck"
            self.override_active = False
            self.override_target_pos = None
            self.override_arrival_threshold = 150  # Distance to target before giving control back
            self.last_progress_step = 0
            self.no_progress_threshold = 20  # Steps without progress before override

            self.override_step_counter = 0  # Track override steps
            self.max_override_steps = 20  # Run override for 5 steps

        #print(f'Wrapped env created for local search training. Action space = {self.action_space}, obs space = {self.observation_space}')


    def reset(self, seed=None, options=None):
        raw_obs, _ = self.env.reset()
        self.num_switches = 0 # How many times the agent has switch policies in this round. Slight penalty to encourage consistency
        self.last_action = 0
        self.steps_since_last_selection = 0
        self.current_subpolicy = None

        self.env.final_wrapper_reward = 0

        # Reset stuck detection variables
        if self.env.config['use_stuck_detection']:

            self.recent_actions = []

            self.position_history = []
            self.override_active = False
            self.override_target_pos = None
            self.last_progress_step = 0
            self.last_targets_identified = 0

            self.override_step_counter = 0

        # Reset teammate selection for new episode

        #if self.env.tag == 'human_eval0':

        if self.teammate_manager:
            if hasattr(self.current_teammate, 'name'):
                if self.current_teammate.name == 'Recorded_Human_Teammate':
                    pass
                else:
                    # Update the teammate manager with the current model before selecting a teammate
                    #self.teammate_manager.set_current_model(self.env.model)

                    self.teammate_manager.reset_for_episode()
                    self.current_teammate = self.teammate_manager.select_random_teammate()
                    self.current_teammate.env = self.env
                    print(f'[localsearchwrapper] Teammate manager is true, current teammate set to {self.current_teammate}')
            else:
                self.teammate_manager.reset_for_episode()
                self.current_teammate = self.teammate_manager.select_random_teammate()
                self.current_teammate.env = self.env
                print(f'[localsearchwrapper] Teammate manager is true, current teammate set to {self.current_teammate}')

        elif self.teammate_policy:
            self.current_teammate = self.teammate_policy
            #print(f'[localsearchwrapper] teammate_policy is true, current teammate set to {self.current_teammate}')
        else:
            self.current_teammate = None
            print(f'[localsearchwrapper] current teammate is {self.current_teammate}')

        return raw_obs, _


    def step(self, action: np.int32):
        """ Apply the monolith's action (Directional movement))"""

        if self.env.config['action_type'] == 'target_index':
            # Action is already an index, pass it through
            processed_action = action
            #print(f'\n\n %%%%%%%% Agent took action {action} %%%%%%%%%% \n \n')

        # Get teammate action
        if self.env.config['num_aircraft'] == 2 and self.teammate_active:
            if self.env.tag == 'human_eval0':
                pass

            # if hasattr(self.current_teammate, 'name'):
            #     if self.current_teammate.name == 'Recorded_Human_Teammate':
            #         self.teammate_action = self.current_teammate.get_action()
            #         self.env.agents[self.env.aircraft_ids[1]].x, self.env.agents[self.env.aircraft_ids[1]].y = self.current_teammate.get_action()
            #     else:
            else:
                self.teammate_action = self.get_teammate_action()
                if isinstance(self.teammate_action, np.ndarray):
                    self.teammate_action = (self.teammate_action[0], self.teammate_action[1])
                self.env.agents[self.env.aircraft_ids[1]].waypoint_override = self.teammate_action

            # elif self.env.tag != 'human_eval0':
            #     self.teammate_action = self.get_teammate_action()
            #     #print(f'[Wrapper] Teammate action is {self.teammate_action} (type {type(self.teammate_action)}')
            #     if isinstance(self.teammate_action, np.ndarray):
            #         self.teammate_action = (self.teammate_action[0],self.teammate_action[1])
            #         #print(f'converted teammate action to tuple: {self.teammate_action}')
            #     self.env.agents[self.env.aircraft_ids[1]].waypoint_override = self.teammate_action
            # else:
            #     raise ValueError(f'Current teammate does not have .name but env tag is not human_eval0 (tag is {self.env.tag}')

        ############ Stuck detection ############

        if self.env.config['use_stuck_detection']:
            self.recent_actions.append(action)
            if len(self.recent_actions) > self.action_history_length:
                self.recent_actions.pop(0)

        if self.env.config['use_stuck_detection']:# and self.env.episode_counter >= 500:
            current_pos = np.array([self.env.agents[self.env.aircraft_ids[0]].x, self.env.agents[self.env.aircraft_ids[0]].y])
            self.position_history.append(current_pos.copy())

            if len(self.position_history) > self.history_length * 2:
                self.position_history = self.position_history[-self.history_length:]
            self.update_progress_tracking()

            # Check for stuck condition and activate override if needed
            #self.is_agent_stuck()
            if not self.override_active and self.is_agent_stuck():
                self.override_target_pos = self.get_nearest_unknown_target()
                if self.override_target_pos is not None:
                    self.override_active = True
                    #print(f"Agent stuck detected! Taking control - moving to target at {self.override_target_pos}")

            # Use override action if active
            if self.override_active:
                override_action = self.get_override_action()

                if override_action is not None:
                    action = override_action
                    self.override_step_counter += 1
                    if self.override_step_counter >= self.max_override_steps:
                        print("Override deactivated after 5 steps.")
                        self.override_active = False
                        self.override_target_pos = None
                        self.override_step_counter = 0

                #
                # if override_action is not None:
                #     action = override_action
                #     print(f"Override action: {action} (distance to target: {np.linalg.norm(current_pos - self.override_target_pos):.1f})")
                # else: print("Override deactivated")

        ####################################

        # Step the environment
        base_obs, base_reward, base_terminated, base_truncated, base_info = self.env.step(action)

        observation = self.env.get_observation_nearest_n()

        if self.obs_noise_std > 0:
            noise = np.random.normal(0, self.obs_noise_std, observation.shape)
            #observation = np.clip(observation + noise, -1, 1)  # Clip to valid range # TODO temp removed

        # Convert base_env elements to wrapper elements if needed
        reward = base_reward
        info = base_info
        terminated = base_terminated
        truncated = base_truncated

        if terminated or truncated:
            info["teammate_name"] = self.teammate_manager.current_teammate.name if (
                    self.teammate_manager and
                    self.teammate_manager.current_teammate and
                    hasattr(self.teammate_manager.current_teammate, 'name')
            ) else "No_Teammate"

            # Pass teammate info directly to environment for plotting
            if self.teammate_manager and self.teammate_manager.current_teammate:
                self.env.teammate_name = self.teammate_manager.current_teammate.name
            else:
                self.env.teammate_name = "No_Teammate"

        self.env.final_wrapper_reward += reward

        return observation, reward, terminated, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

########################################################################################################################
######################################    Observations and sub-observations     ########################################
########################################################################################################################

    def get_reward(self, info):
        """Generate reward for the mode selector."""
        pass

    def get_observation(self, agent_id):
        """
        Generates the observation for the mode selector using env attributes
        """
        # Initialize observation as float32 (not int32)
        obs = np.zeros(10, dtype=np.float32)

        # Calculate targets left
        targets_left = self.env.config['num_targets'] - self.env.targets_identified

        # obs[0]: Steps remaining (normalized 0-1, where 1 = all steps left, 0 = no steps left)
        max_steps_outer = self.env.max_steps / self.env.config['frame_skip']
        obs[0] = (max_steps_outer - self.env.step_count_outer) / self.env.max_steps

        # obs[1] = (self.env.max_detections - self.env.detections) / self.env.max_detections # Detections remaining before game over (normalized 0-1, where 1 = no detections, 0 = max detections)
        obs[1] = self.env.num_threats_identified / 2.0  # Num threats ID'd / max allowed (2)

        # obs[2]: Targets remaining (normalized 0-1, where 1 = all targets left, 0 = no targets left)
        obs[2] = targets_left / self.env.config['num_targets']

        # obs[3]: Ratio of remaining targets in current quadrant (0-1)
        if targets_left > 0:
            unknown_in_quad = self.unknown_targets_in_current_quadrant(agent_id)
            obs[3] = unknown_in_quad / targets_left
        else:
            obs[3] = 0.0

        # obs[4]: Distance to teammate (normalized 0-1, where 0 = same position, 1 = max distance)
        obs[4] = self.get_distance_to_teammate(agent_id) / self.env.config['gameboard_size']

        # obs[5]: Adaptation signal (placeholder)
        obs[5] = self.get_adaptation_signal()

        # obs[6] and obs[7]: dx, dy to nearest threat
        agent_pos = np.array(
            [self.env.agents[self.env.aircraft_ids[agent_id]].x, self.env.agents[self.env.aircraft_ids[agent_id]].y])
        if len(self.env.threats) > 0:

            # Find nearest threat
            threat_distances = []
            for threat_idx in range(len(self.env.threats)):
                threat_pos = np.array([self.env.threats[threat_idx, 0], self.env.threats[threat_idx, 1]])
                distance = np.sqrt(np.sum((threat_pos - agent_pos) ** 2))
                threat_distances.append((distance, threat_pos))

            # Get nearest threat position
            threat_distances.sort(key=lambda x: x[0])
            nearest_threat_pos = threat_distances[0][1]

            # Calculate dx, dy vector to nearest threat
            threat_vector = nearest_threat_pos - agent_pos
            obs[6] = threat_vector[0] / self.env.config['gameboard_size']  # dx to nearest threat
            obs[7] = threat_vector[1] / self.env.config['gameboard_size']  # dy to nearest threat
        else:
            obs[6] = 0.0  # No threats
            obs[7] = 0.0

        obs[8] = agent_pos[0]
        obs[9] = agent_pos[1]

        if self.obs_noise_std > 0:
            noise = np.random.normal(0, self.obs_noise_std, obs.shape)
            obs = obs + noise

        # if self.env.step_count_outer % 100 == 0:  # Print every 10 steps to avoid spam
        #     obs_labels = [
        #         "Steps remaining (0-1)",
        #         "Threats ID'd / 2",
        #         "Targets remaining (0-1)",
        #         "% targets in current quad",
        #         "Distance to teammate (norm)",
        #         "Adaptation signal",
        #         "dx to nearest threat",
        #         "dy to nearest threat"
        #     ]
        #
        #     print(f"\n=== Wrapper Observation (Step {self.env.step_count_outer}) ===")
        #     for i, (label, value) in enumerate(zip(obs_labels, obs)):
        #         print(f"obs[{i}]: {label:<25} = {value:.3f}")
        #     print("=" * 50)

        return obs

    def _unwrap_action(self, action):
        teammate_info = "Unavailable"
        # subpolicy_map = {
        #     0: "LocalSearch",
        #     1: "ChangeRegion",
        #     2: "GoToThreat",
        #     3: "HoldPosition",
        #     4: "ChangeRegion_NE",
        #     5: "ChangeRegion_SE",
        #     6: "ChangeRegion_SW"
        # }
        #
        # if self.current_teammate:
        #     name = getattr(self.current_teammate, 'name', 'Unknown')
        #     policy_type = type(self.current_teammate).__name__
        #     stability = getattr(self.current_teammate, 'action_stability', 'N/A')
        #     decision_speed = getattr(self.current_teammate, 'decision_speed', 'N/A')
        #     subpolicy_idx = getattr(self, 'teammate_subpolicy_choice', 'N/A')
        #     subpolicy_name = subpolicy_map.get(subpolicy_idx, f"Unknown({subpolicy_idx})")
        #
        #     teammate_info = (
        #         f"name={name}, type={policy_type}, stability={stability}, speed={decision_speed}, "
        #         f"subpolicy={subpolicy_name} (index={subpolicy_idx})"
        #     )
        if isinstance(action, tuple):
            # warnings.warn(
            #     f"WARNING: Teammate action is a tuple {action}. Unwrapping first element.\n"
            #     f"Teammate debug info: {teammate_info}"
            # )
            return action[0]
        #print(f"NOT A TUPLE - WARNING: Teammate action is NOT a tuple {action}. Unwrapping first element.\n", f"Teammate debug info: {teammate_info}\n")
        return action

    def get_teammate_action(self):
        #if self.env.tag == 'human_eval0':
        # if hasattr(self.current_teammate, 'name'):
        #     if self.current_teammate.name == 'Recorded_Human_Teammate':
        #         return self.env.agents[self.env.aircraft_ids[1]].x, self.env.agents[self.env.aircraft_ids[1]].y


        obs_agent1_raw = self.env.get_observation_nearest_n(1)
        obs_agent1_norm = self.current_teammate._normalize_observation(obs_agent1_raw)

        if self.env.config['action_type'] == 'target_index' and hasattr(self.current_teammate, 'model'):
            teammate_obs = self.current_teammate._normalize_observation(self.env.get_observation_nearest_n(1))
            teammate_target_index = self.current_teammate.model.predict(teammate_obs, deterministic=True)
            teammate_target_index = self._unwrap_action(teammate_target_index)
            return self.env._index_to_waypoint(int(teammate_target_index))

        #if self.env.step_count_outer % 50 == 0:
            #print(f"\n    &&&&& Step {self.env.step_count_outer} Teammate obs Raw:", obs_agent1_raw[:3])
            #print("    &&&&&          Teammate obs Norm:", obs_agent1_norm[:3])

        if isinstance(self.current_teammate, RecordedTrajectoryTeammate):
            return self.current_teammate.get_action()

        elif self.teammate_manager or self.teammate_policy:
            # if hasattr(self.current_teammate, 'env') and self.current_teammate.env is None: # TODO potential problem 1
            #     self.current_teammate.env = self.env

            # Get teammate observation (and normalize it)
            if self.env.config['league_type'] == 'selfplay':

                # If using a selfplay model
                if hasattr(self.current_teammate, 'model'):
                    teammate_obs = self.current_teammate._normalize_observation(self.env.get_observation_nearest_n(1)) # TODO temp
                    #teammate_obs = self.env.get_observation_nearest_n(1)
                    direction_to_move = self.current_teammate.model.predict(teammate_obs, deterministic=True)
                    direction_to_move = self._unwrap_action(direction_to_move)


                else:
                    teammate_subpolicy_observation = self.get_subpolicy_observation(self.teammate_subpolicy_choice, 1)
                    direction_to_move = self.current_teammate.local_search_policy.act(teammate_subpolicy_observation,env=self.env, agent_id=1)
                    direction_to_move = self._unwrap_action(direction_to_move)

                #print(f'Teammate action is {direction_to_move}')
                teammate_action = self.env._direction_to_waypoint(direction_to_move, 1)
                return teammate_action

            else:
                teammate_obs = self.current_teammate._normalize_observation(self.get_observation(1))

            self.teammate_subpolicy_choice = self.current_teammate.choose_subpolicy(teammate_obs,self.teammate_subpolicy_choice)
            teammate_subpolicy_observation = self.get_subpolicy_observation(self.teammate_subpolicy_choice, 1)

            if self.teammate_subpolicy_choice == 0:  # Local search
                direction_to_move = self.current_teammate.local_search_policy.act(teammate_subpolicy_observation,env=self.env, agent_id=1)
                direction_to_move = self._unwrap_action(direction_to_move)

                # Apply action noise
                if hasattr(self.current_teammate, 'action_stability'):
                    noise_options = {'noisy': [-3, -2, -1, 1, 2, 3], 'very_noisy': [-4, -3, -2, -1, 1, 2, 3, 4]}
                    noise_chance = {'noisy': 0.4, 'very_noisy': 0.7}

                    if self.current_teammate.action_stability == "periodic_stopping":
                        self.periodic_step_counter += 1

                        # Determine if we are in a pause window
                        if (self.periodic_step_counter % self.periodic_pause_interval) < self.periodic_pause_duration:
                            if self.last_real_direction is None:
                                self.last_real_direction = direction_to_move  # seed it the first time

                            #direction_to_move = (self.last_real_direction + 8) % 16 if self.pause_toggle else self.last_real_direction
                            if isinstance(self.last_real_direction, tuple):
                                self.last_real_direction = self.last_real_direction[0]
                                print('(subpol choice 0) fixed last_real_direction')

                            try:
                                if self.pause_toggle:
                                    direction_to_move = (self.last_real_direction + 4) % 16
                                else:
                                    direction_to_move = (self.last_real_direction - 4) % 16
                            except:
                                print(f'ERROR with direction_to_move = (self.last_real_direction + 4) % 16. last real direction is {self.last_real_direction}, type {type(self.last_real_direction)}')

                            self.pause_toggle = not self.pause_toggle

                        else:
                            # We're not pausing — take and store the real direction
                            self.last_real_direction = direction_to_move

                    elif self.current_teammate.action_stability in ['noisy', 'very_noisy'] and random.random() < noise_chance[self.current_teammate.action_stability]:
                        #old_direction_to_move = direction_to_move
                        noise = random.choice(noise_options[self.current_teammate.action_stability])
                        direction_to_move = int(self._unwrap_action(direction_to_move))
                        try:
                            direction_to_move = (direction_to_move + noise) % 16
                        except: print(f'ERROR: failed to add noise, teammate action is {direction_to_move}, type {type(direction_to_move)}')
                        #print(f'[DEBUG - LocalSearchWrapper.get_teammate_action] Applying noise to teammate action ({old_direction_to_move} + {noise} -> {direction_to_move})')

                #print(f'Teammate action is {direction_to_move}')
                teammate_action = self.env._direction_to_waypoint(direction_to_move, 1)

            elif self.teammate_subpolicy_choice == 1:  # Change region - NW
                waypoint_to_go = self._get_quadrant_waypoint(0)
                teammate_action = self.env._denormalize_waypoint(waypoint_to_go)

            elif self.teammate_subpolicy_choice == 2:  # go to high value target
                direction_to_move = self.go_to_highvalue_policy.act(teammate_subpolicy_observation)
                direction_to_move = self._unwrap_action(direction_to_move)

                # Apply action noise
                if hasattr(self.current_teammate, 'action_stability'):
                    noise_options = {'noisy': [-3, -2, -1, 1, 2, 3], 'very_noisy': [-4, -3, -2, -1, 1, 2, 3, 4]}
                    noise_chance = {'noisy': 0.4, 'very_noisy': 0.7}

                    # Check if it's time to enter or continue pause
                    if self.current_teammate.action_stability == "periodic_stopping":
                        self.periodic_step_counter += 1

                        # Determine if we are in a pause window
                        if (self.periodic_step_counter % self.periodic_pause_interval) < self.periodic_pause_duration:
                            if self.last_real_direction is None: # Oscillate using the last real direction
                                self.last_real_direction = direction_to_move  # seed it the first time
                            try:
                                if self.pause_toggle: # oscillate +90°
                                    direction_to_move = (int(self.last_real_direction) + 4) % 16
                                else: # original direction (or oscillate -90°)
                                    direction_to_move = (int(self.last_real_direction) - 4) % 16
                            except:
                                print( f'ERROR: failed to add noise, teammate action is {direction_to_move}, type {type(direction_to_move)}')

                            self.pause_toggle = not self.pause_toggle

                        else: # We're not pausing — take and store the real direction
                            self.last_real_direction = direction_to_move

                    elif self.current_teammate.action_stability in ['noisy', 'very_noisy'] and random.random() < noise_chance[self.current_teammate.action_stability]:
                        noise = random.choice(noise_options[self.current_teammate.action_stability])
                        direction_to_move = int(self._unwrap_action(direction_to_move))
                        try:
                            direction_to_move = (direction_to_move + noise) % 16
                        except:
                            print( f'ERROR: failed to add noise, teammate action is {direction_to_move}, type {type(direction_to_move)}')

                #print(f'Teammate action is {direction_to_move}')
                teammate_action = self.env._direction_to_waypoint(direction_to_move, 1)

            elif self.teammate_subpolicy_choice == 3:  # Hold at current location
                print(' %%%%%%%%%%%%%%%% \nteammate subpolicy choice is 3, overriding action')
                teammate_action = np.array([
                    self.env.agents[self.env.aircraft_ids[1]].x / self.env.config['gameboard_size'],
                    self.env.agents[self.env.aircraft_ids[1]].y / self.env.config['gameboard_size']])

            elif self.teammate_subpolicy_choice == 4:  # Change region - NE
                waypoint_to_go = self._get_quadrant_waypoint(1)
                teammate_action = self.env._denormalize_waypoint(waypoint_to_go)

            elif self.teammate_subpolicy_choice == 5:  # Change region - SE
                waypoint_to_go = self._get_quadrant_waypoint(3)
                teammate_action = self.env._denormalize_waypoint(waypoint_to_go)

            elif self.teammate_subpolicy_choice == 6:  # Change region - SW
                waypoint_to_go = self._get_quadrant_waypoint(2)
                teammate_action = self.env._denormalize_waypoint(waypoint_to_go)
            else:
                raise ValueError(f'ERROR: Got invalid subpolicy selection {self.teammate_subpolicy_choice} (type {type(self.teammate_subpolicy_choice)})')

        # Add subpolicy noise
        # if self.teammate_subpolicy_choice in [0, 2] and self.current_teammate.action_stability == 'noisy' and random.random() < 0.4:
        #     old_teammate_action = teammate_action
        #     noise = random.choice([-2, -1, 1, 2])
        #     print(f'Noise: {noise}')
        #     print(f'Teammate action: {teammate_action}')
        #     teammate_action = (teammate_action + noise) % 16
        #     print(f'[DEBUG - LocalSearchWrapper.get_teammate_action] Applying noise to teammate action ({old_teammate_action} + {noise} -> {teammate_action})')

        else: # Fallback greedy search
            # Access teammate location
            teammate_x = self.env.agents[self.env.aircraft_ids[1]].x
            teammate_y = self.env.agents[self.env.aircraft_ids[1]].y
            teammate_pos = np.array([teammate_x, teammate_y])

            # Get target positions and info levels
            target_positions = self.env.targets[:, 3:5]  # x,y coordinates
            target_info_levels = self.env.targets[:, 2]  # info levels

            # Create mask for unknown targets (info_level < 1.0)
            unknown_mask = target_info_levels < 1.0

            if np.any(unknown_mask):
                # Find nearest unknown target
                unknown_positions = target_positions[unknown_mask]
                distances = np.sqrt(np.sum((unknown_positions - teammate_pos) ** 2, axis=1))
                nearest_idx = np.argmin(distances)
                nearest_target_pos = unknown_positions[nearest_idx]

                # Return raw waypoint coordinates (NOT normalized)
                teammate_action = (float(nearest_target_pos[0]), float(nearest_target_pos[1]))
            else: # No unknown targets remaining, stay at current position
                teammate_action = (teammate_x, teammate_y)

        return teammate_action


########################################################################################################################
###############################################    Helper functions     ################################################
########################################################################################################################

    def get_config(self):
        """Return the environment configuration dictionary"""
        return self.env.config.copy()

    def set_teammate_active(self, should_activate):
        self.teammate_active = should_activate
        self.env.teammate_active = should_activate
        return

    def _get_quadrant_waypoint(self, quadrant_id):
        """Get waypoint for specific quadrant (0=NW, 1=NE, 2=SW, 3=SE)"""
        quadrant_centers = {
            0: np.array([-0.5, 0.5]),  # NW
            1: np.array([0.5, 0.5]),  # NE
            2: np.array([-0.5, -0.5]),  # SW
            3: np.array([0.5, -0.5])  # SE
        }
        return quadrant_centers.get(quadrant_id, np.array([0.0, 0.0]))

########################################################################################################################
############################################    Subpolicy Observations     #############################################
########################################################################################################################

    def get_subpolicy_observation(self, selected_subpolicy, agent_id):
        if selected_subpolicy == 0:  # Get obs for local search
            observation = self.get_observation_localsearch(agent_id)
            # observation = self.normalize_local_search_obs(observation)

        elif selected_subpolicy in [1, 4, 5, 6]:  # Change region
            observation = self.get_observation_changeregion(agent_id)

        elif selected_subpolicy == 2:  # Go to nearest
            observation = self.get_observation_nearest_threat(agent_id)

        elif selected_subpolicy == 3:  # Hold
            # observation = self.get_observation_evade(agent_id)
            observation = self.get_observation_localsearch(agent_id)

        elif selected_subpolicy == 7:  # Custom human waypoint
            observation = self.get_observation_localsearch(agent_id)

        return observation

    def get_observation_localsearch(self, agent_id):
        # return self.env.get_observation_nearest_n(agent_id)
        return self.env.get_observation_nearest_n(agent_id)


    def get_observation_changeregion(self, agent_id=0):
        """Get observation for the change_region policy.
        obs[0] = # Ratio of targets in quadrant NW
        obs[1] = # Agent distance to quadrant NW
        obs[2] = # Teammate distance to quadrant NW

        obs[3] =  # Ratio of targets in quadrant NW
        obs[4] =  # Agent distance to quadrant NW
        obs[5] =  # Teammate distance to quadrant NW

        obs[6] =  # Ratio of targets in quadrant NW
        obs[7] =  # Agent distance to quadrant NW
        obs[8] =  # Teammate distance to quadrant NW

        obs[9] =  # Ratio of targets in quadrant NW
        obs[10] =  # Agent distance to quadrant NW
        obs[11] =  # Teammate distance to quadrant NW
        """
        # TODO use agent_id to return observation relative to that agent
        obs = np.zeros(12, dtype=np.float32)

        # Get agent position
        agent_x = self.env.agents[self.env.aircraft_ids[agent_id]].x
        agent_y = self.env.agents[self.env.aircraft_ids[agent_id]].y

        # Get teammate position (if exists)
        if self.env.config['num_aircraft'] >= 2:
            teammate_x = self.env.agents[self.env.aircraft_ids[1 if agent_id == 0 else 0]].x
            teammate_y = self.env.agents[self.env.aircraft_ids[1 if agent_id == 0 else 0]].y
        else:
            teammate_x, teammate_y = 0, 0  # Default to center if no teammate

        # Get target info
        target_positions = self.env.targets[:self.env.config['num_targets'], 3:5]  # x,y coordinates
        target_info_levels = self.env.targets[:self.env.config['num_targets'], 2]  # info levels
        unknown_mask = target_info_levels < 1.0  # Unknown targets only

        # Calculate quadrant centers (for distance calculations)
        map_half_size = self.env.config['gameboard_size'] / 2
        quadrant_centers = {
            'NW': (-map_half_size / 2, map_half_size / 2),
            'NE': (map_half_size / 2, map_half_size / 2),
            'SW': (-map_half_size / 2, -map_half_size / 2),
            'SE': (map_half_size / 2, -map_half_size / 2)
        }

        quadrants = ['NW', 'NE', 'SW', 'SE']

        for i, quad in enumerate(quadrants):
            base_idx = i * 3
            center_x, center_y = quadrant_centers[quad]

            # Determine which targets are in this quadrant
            if quad == 'NW':
                quad_mask = (target_positions[:, 0] < 0) & (target_positions[:, 1] >= 0)
            elif quad == 'NE':
                quad_mask = (target_positions[:, 0] >= 0) & (target_positions[:, 1] >= 0)
            elif quad == 'SW':
                quad_mask = (target_positions[:, 0] < 0) & (target_positions[:, 1] < 0)
            elif quad == 'SE':
                quad_mask = (target_positions[:, 0] >= 0) & (target_positions[:, 1] < 0)

            # Count unknown targets in this quadrant
            unknown_targets_in_quad = np.sum(unknown_mask & quad_mask)
            total_unknown_targets = np.sum(unknown_mask)

            # Ratio of unknown targets in this quadrant
            obs[base_idx] = unknown_targets_in_quad / max(1, total_unknown_targets)

            # Agent distance to quadrant center (normalized)
            agent_dist = np.sqrt((agent_x - center_x) ** 2 + (agent_y - center_y) ** 2)
            obs[base_idx + 1] = agent_dist / self.env.config['gameboard_size']

            # Teammate distance to quadrant center (normalized)
            teammate_dist = np.sqrt((teammate_x - center_x) ** 2 + (teammate_y - center_y) ** 2)
            obs[base_idx + 2] = teammate_dist / self.env.config['gameboard_size']

        return obs

    def get_observation_nearest_threat(self, agent_id):
        """
        Get observation for go to nearest threat policy including identification status
        Now dynamically selects the 2 closest unidentified threats, falling back to closest identified if needed
        Returns: [dx_threat1, dy_threat1, identified1, dx_threat2, dy_threat2, identified2]
        """
        obs = np.zeros(6, dtype=np.float32)

        # Get agent position
        agent_pos = np.array([self.env.agents[self.env.aircraft_ids[agent_id]].x,
                              self.env.agents[self.env.aircraft_ids[agent_id]].y])

        # Get threat positions
        threat_positions = self.env.threats

        if len(threat_positions) == 0:
            return obs  # Return zeros if no threats

        # Calculate distances and create threat info list
        threat_info = []
        for i, threat_pos in enumerate(threat_positions):
            distance = np.linalg.norm(threat_pos - agent_pos)
            is_identified = self.env.threat_identified[i] if i < len(self.env.threat_identified) else False

            threat_info.append({
                'index': i,
                'position': threat_pos,
                'distance': distance,
                'identified': is_identified
            })

        # Sort by distance (closest first)
        threat_info.sort(key=lambda x: x['distance'])

        # Priority selection: prefer unidentified threats, but include closest overall
        selected_threats = []

        # First, try to get up to 2 unidentified threats
        unidentified_threats = [t for t in threat_info if not t['identified']]
        selected_threats.extend(unidentified_threats[:2])

        # If we need more threats and have identified ones, add closest identified threats
        if len(selected_threats) < 2:
            identified_threats = [t for t in threat_info if t['identified']]
            remaining_slots = 2 - len(selected_threats)
            selected_threats.extend(identified_threats[:remaining_slots])

        # Fill observation with selected threats
        for i, threat in enumerate(selected_threats):
            if i < 2:  # Ensure we don't exceed observation size
                base_idx = i * 3
                vector_to_threat = threat['position'] - agent_pos
                obs[base_idx] = vector_to_threat[0]  # dx
                obs[base_idx + 1] = vector_to_threat[1]  # dy
                obs[base_idx + 2] = float(threat['identified'])  # identification status

        return obs


    def get_distance_to_teammate(self, agent_id):
        """Returns pixel range between current location and teammate's location
        Note: Should NOT be normalized (should be in range [- gameboard_size, + gameboard_size])"""

        # Check if there are multiple aircraft (teammates exist)
        if self.env.config['num_aircraft'] < 2: # No teammate, return maximum distance as default
            return self.env.config['gameboard_size']

        # Get agent position (aircraft 0)
        agent_x = self.env.agents[self.env.aircraft_ids[agent_id]].x
        agent_y = self.env.agents[self.env.aircraft_ids[agent_id]].y

        teammate_id = 1 if agent_id == 0 else 0

        # Get teammate position (aircraft 1)
        teammate_x = self.env.agents[self.env.aircraft_ids[teammate_id]].x
        teammate_y = self.env.agents[self.env.aircraft_ids[teammate_id]].y

        # Calculate Euclidean distance
        distance = np.sqrt((agent_x - teammate_x) ** 2 + (agent_y - teammate_y) ** 2)

        return distance

    def get_adaptation_signal(self):
        """Gets adaptation signal, e.g. from external physiological measurement
        Currently placeholder as 0 until implemented"""

        return 0

    def unknown_targets_in_current_quadrant(self, agent_id):
        """Returns the number of unknown targets in the agent's quadrant"""

        agent_x = self.env.agents[self.env.aircraft_ids[agent_id]].x
        agent_y = self.env.agents[self.env.aircraft_ids[agent_id]].y

        target_positions = self.env.targets[:self.env.config['num_targets'], 3:5]  # x,y coordinates
        target_info_levels = self.env.targets[:self.env.config['num_targets'], 2]  # info levels

        unknown_mask = target_info_levels < 1.0 # Create mask for unknown targets (info_level < 1.0)

        # Determine agent's quadrant based on sign of coordinates
        agent_in_right = agent_x >= 0  # True if in right half (NE or SE)
        agent_in_top = agent_y >= 0  # True if in top half (NE or NW)

        # Create masks for targets in same quadrant as agent
        targets_in_right = target_positions[:, 0] >= 0  # x >= 0
        targets_in_top = target_positions[:, 1] >= 0  # y >= 0

        same_quadrant_mask = (targets_in_right == agent_in_right) & (targets_in_top == agent_in_top) # Targets are in same quadrant if they match agent's quadrant
        num_unknown_targets = np.sum(unknown_mask & same_quadrant_mask) # Count unknown targets in same quadrant

        return num_unknown_targets


    ############################ Functions for stuck detection ############################

    def is_agent_stuck(self, agent_id=0):
        if len(self.recent_actions) < self.action_history_length:
            return False

        # Compute pairwise action differences in circular space
        diffs = []
        for i in range(len(self.recent_actions) - 1):
            a1 = self.recent_actions[i]
            a2 = self.recent_actions[i + 1]
            # Circular distance: min steps around the 16-direction circle
            diff = min(abs(a1 - a2), 16 - abs(a1 - a2))
            diffs.append(diff)

        avg_diff = sum(diffs) / len(diffs)

        condition_1 = avg_diff > self.oscillation_threshold
        if condition_1: print('[STUCK DETECTION] Oscillating actions detected')

        # Condition 2: No forward progress
        current_pos = np.array([
            self.env.agents[self.env.aircraft_ids[agent_id]].x,
            self.env.agents[self.env.aircraft_ids[agent_id]].y
        ])

        # Check if agent hasn't moved much in recent history
        recent_positions = np.array(self.position_history[-self.history_length:])
        distances_from_current = np.linalg.norm(recent_positions - current_pos, axis=1)

        # If most recent positions are within stuck_threshold, agent is stuck
        stuck_positions = np.sum(distances_from_current < self.stuck_threshold)
        stuck_ratio = stuck_positions / len(distances_from_current)
        condition_3 = stuck_ratio > 0.8
        if condition_3: print('[STUCK DETECTION] Stuck ratio exceeded')

        # Also check for no progress towards targets
        # steps_since_progress = self.env.step_count_outer - self.last_progress_step
        # condition_2 = steps_since_progress > self.no_progress_threshold
        # if condition_2: print('[STUCK DETECTION] No progress detected')

        #print(f"[DEBUG] avg_diff={avg_diff:.2f}, stuck_ratio={stuck_ratio:.2f}, steps_since_progress=")

        return condition_1 or condition_3 # condition_2


    def get_nearest_unknown_target(self, agent_id=0):
        """Find the nearest unknown target position"""
        agent_pos = np.array([
            self.env.agents[self.env.aircraft_ids[agent_id]].x,
            self.env.agents[self.env.aircraft_ids[agent_id]].y
        ])

        # Get target positions and info levels
        target_positions = self.env.targets[:, 3:5]  # x,y coordinates
        target_info_levels = self.env.targets[:, 2]  # info levels
        unknown_mask = target_info_levels < 1.0

        if not np.any(unknown_mask):
            return None  # No unknown targets

        unknown_positions = target_positions[unknown_mask]
        distances = np.linalg.norm(unknown_positions - agent_pos, axis=1)
        nearest_idx = np.argmin(distances)

        return unknown_positions[nearest_idx]

    def get_override_action(self, agent_id=0):
        """Get action to move towards override target"""
        if self.override_target_pos is None:
            return None

        agent_pos = np.array([
            self.env.agents[self.env.aircraft_ids[agent_id]].x,
            self.env.agents[self.env.aircraft_ids[agent_id]].y
        ])

        # Calculate direction to target
        direction_vector = self.override_target_pos - agent_pos
        distance_to_target = np.linalg.norm(direction_vector)

        # Check if we've arrived at target
        # if distance_to_target < self.override_arrival_threshold:
        #     self.override_active = False
        #     self.override_target_pos = None
        #     print(f"Override complete - arrived at target (distance: {distance_to_target:.1f})")
        #     return None

        if distance_to_target > 0:
            # Normalize direction vector
            unit_direction = direction_vector / distance_to_target

            # Your environment's direction mapping:
            # 0: (0, 1)   # North
            # 1: (0.383, 0.924)  # NNE
            # 2: (0.707, 0.707)  # NE
            # 3: (0.924, 0.383)  # ENE
            # 4: (1, 0)   # East
            # etc.

            # Calculate angle from positive x-axis (standard math convention)
            angle = np.arctan2(unit_direction[1], unit_direction[0])

            # Convert to environment's convention where:
            # - Action 0 points North (0, 1) = 90° in standard math
            # - Action 4 points East (1, 0) = 0° in standard math
            # So we need to map: math_angle to env_action

            # Environment action 0 = 90° math angle
            # Environment action 4 = 0° math angle
            # Environment rotates clockwise from North

            # Convert math angle to environment angle
            # Rotate by 90° and reverse direction (clockwise vs counterclockwise)
            env_angle = (np.pi / 2 - angle) % (2 * np.pi)

            # Convert to discrete action (16 directions)
            action = int(env_angle / (2 * np.pi / 16)) % 16

            # print(f"Override debug:")
            # print(f"  Agent pos: {agent_pos}")
            # print(f"  Target pos: {self.override_target_pos}")
            # print(f"  Direction vector: {direction_vector}")
            # print(f"  Unit direction: {unit_direction}")
            # print(f"  Math angle: {angle:.3f} rad ({np.degrees(angle):.1f}°)")
            # print(f"  Env angle: {env_angle:.3f} rad ({np.degrees(env_angle):.1f}°)")
            # print(f"  Selected action: {action}")

            # Let's also verify what this action should do
            direction_map = {
                0: (0, 1), 1: (0.383, 0.924), 2: (0.707, 0.707), 3: (0.924, 0.383),
                4: (1, 0), 5: (0.924, -0.383), 6: (0.707, -0.707), 7: (0.383, -0.924),
                8: (0, -1), 9: (-0.383, -0.924), 10: (-0.707, -0.707), 11: (-0.924, -0.383),
                12: (-1, 0), 13: (-0.924, 0.383), 14: (-0.707, 0.707), 15: (-0.383, 0.924)
            }
            expected_direction = direction_map.get(action, (0, 0))
            #print(f"  Expected movement direction: {expected_direction}")

            return action

        return None

    # def get_override_action(self, agent_id=0):
    #     """Get action to move towards override target"""
    #     if self.override_target_pos is None:
    #         return None
    #
    #     agent_pos = np.array([
    #         self.env.agents[self.env.aircraft_ids[agent_id]].x,
    #         self.env.agents[self.env.aircraft_ids[agent_id]].y
    #     ])
    #
    #     # Calculate direction to target
    #     direction_vector = self.override_target_pos - agent_pos
    #     distance_to_target = np.linalg.norm(direction_vector)
    #
    #     # Check if we've arrived at target
    #     if distance_to_target < self.override_arrival_threshold:
    #         self.override_active = False
    #         self.override_target_pos = None
    #         print(f"Override complete - arrived at target (distance: {distance_to_target:.1f})")
    #         return None
    #
    #     # Normalize direction and convert to action
    #     if distance_to_target > 0:
    #         unit_direction = direction_vector / distance_to_target
    #
    #         # Convert to discrete action (find closest direction)
    #         if self.env.config['action_type'] in ['Discrete8', 'Discrete16']:
    #             angle = np.arctan2(unit_direction[1], unit_direction[0])
    #             # Convert to discrete action (8 or 16 directions)
    #             num_directions = 8 if self.env.config['action_type'] == 'Discrete8' else 16
    #             action = int(((angle + np.pi) / (2 * np.pi)) * num_directions) % num_directions
    #             return action
    #         else: # For continuous actions, return normalized direction
    #             return unit_direction
    #
    #     return None

    def update_progress_tracking(self):
        """Update progress tracking for stuck detection"""
        # Check if any new targets were identified this step
        current_identified = self.env.targets_identified
        if not hasattr(self, 'last_targets_identified'):
            self.last_targets_identified = current_identified

        if current_identified > self.last_targets_identified:
            self.last_progress_step = self.env.step_count_outer
            self.last_targets_identified = current_identified

    def change_league_ratio(self, new_ratio):
        if self.teammate_manager is not None:
            self.teammate_manager.fcp_ratio = new_ratio

    @property
    def episode_counter(self):
        return self.env.episode_counter