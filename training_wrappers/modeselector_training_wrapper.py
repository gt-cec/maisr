import warnings

import gymnasium as gym
import numpy as np
from sympy import trunc
from torch.ao.quantization.backend_config.onednn import observation_type

from policies.sub_policies import SubPolicy, GoToNearestThreat, LocalSearch, ChangeRegions
from utility.league_management import TeammateManager, TeammatePolicy


class MaisrModeSelectorWrapper(gym.Env):
    """Wrapper for training the mode selector
    Subpolicies are treated as part of the environment dynamics.
    """
    def __init__(self,
                 env,
                 local_search_policy: SubPolicy,
                 go_to_highvalue_policy: SubPolicy,
                 change_region_subpolicy: SubPolicy,
                 evade_policy: SubPolicy,
                 teammate_policy: TeammatePolicy=None,
                 teammate_manager: TeammateManager = None,
                 ):

        self.env = env

        # Load primary agent subpolicies and teammate policies
        self.local_search_policy = local_search_policy
        self.go_to_highvalue_policy = go_to_highvalue_policy
        self.change_region_subpolicy = change_region_subpolicy
        self.evade_policy = evade_policy

        if teammate_policy is not None and teammate_manager is not None:
            raise ValueError('Cannot specify a teammate policy and a teammate manager at the same time')
        self.teammate_policy = teammate_policy
        self.teammate_manager = teammate_manager


        # Define observation space (6 high-level elements about the current game state)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1,
            shape=(6,),
            dtype=np.float32)

        # Action space: 3 possible sub-policies to choose from
        self.action_space = gym.spaces.Discrete(3)
        self.action_rate = 20

        self.render_mode = self.env.render_mode
        self.run_name = self.env.run_name  # For logging
        self.tag = self.env.tag

        # Set rewards for mode selector
        self.reward_per_target_id = 1
        self.reward_per_threat_id = 5
        self.penalty_for_policy_switch = 0.02
        self.reward_per_step_early = 0.05
        self.penalty_per_detection = 0 # Currently none (but episode ends if we exceed max)

        self.mode_dict = {0:"local search", 1:'change_region', 2:'go_to_threat'}

        # Goal tracking for evade policy
        self.evade_goal = None
        self.evade_goal_threshold = 30.0  # Distance threshold to consider goal "reached"
        self.last_evade_step = -1  # Track when we last used evade to detect continuous usage

        self.circumnavigation_state = {
            'active': False,
            'threat_pos': None,
            'chosen_direction': None,  # 'clockwise' or 'counterclockwise'
            'last_angle': None,
            'start_angle': None,
            'safety_distance': None
        }

        # Load normalization stats
        # try:
        #     self.norm_stats = np.load(local_search_policy.norm_stats_filepath, allow_pickle=True).item()
        #     print("Loaded normalization stats for local search policy")
        # except FileNotFoundError:
        #     print("Warning: No normalization stats found, using raw observations")
        #     self.norm_stats = None


    def reset(self, seed=None, options=None):
        raw_obs, _ = self.env.reset()
        observation = self.get_observation(0)
        self.total_switches = 0
        self.last_action = 0
        self.steps_since_last_selection = 0
        self.subpolicy_choice = None
        self.teammate_subpolicy_choice = 0
        self.switched_policies = False

        self.episode_reward = 0

        self.subpolicy_history = []  # Track subpolicy choices over time

        self._reset_circumnavigation_state()

        # NEW: Track wrapper observations at key timesteps
        self.wrapper_observations = {}
        self.wrapper_observations[0] = observation.copy()  # Store initial observation

        # Instantiate teammate for this episode
        if self.teammate_manager:
            self.teammate_manager.reset_for_episode()
            # Choose selection strategy: 'random' or 'curriculum'
            self.current_teammate = self.teammate_manager.select_random_teammate()
            print(f"Selected teammate: {self.current_teammate.name if self.current_teammate else 'None'}")
        else: # No manager specified, use fixed teammate
            self.current_teammate = teammate_policy

        return observation, _

    def step(self, action: np.int32):
        """ Apply the mode selector's action (Index of selected subpolicy)"""

        ######################## Choose a subpolicy ########################
        # Check if we need to evade with direct tangential movement
        # TODO temp removed for training test
        # if self.near_threat() and not np.int32(action) == 2:
        #     # Compute direct tangential escape action
        #
        #     escape_action = self.compute_tangential_escape_action()
        #     #print(f'Escape action {escape_action} ({type(escape_action)})')
        #
        #     # Process teammate's action if needed
        #     if self.current_teammate and self.env.config['num_aircraft'] >= 2:
        #         teammate_obs = self.get_observation(1)
        #         self.teammate_subpolicy_choice = self.current_teammate.choose_subpolicy(teammate_obs, self.teammate_subpolicy_choice)
        #
        #         teammate_subpolicy_observation = self.get_subpolicy_observation(self.teammate_subpolicy_choice, 1)
        #         if self.teammate_subpolicy_choice == 0:  # Local search
        #             direction_to_move = self.current_teammate.local_search_policy.act(teammate_subpolicy_observation)
        #             teammate_subpolicy_action = direction_to_move
        #
        #         elif self.teammate_subpolicy_choice == 1:  # Change region
        #             waypoint_to_go = self.change_region_subpolicy.act(teammate_subpolicy_observation)
        #             teammate_subpolicy_action = waypoint_to_go
        #
        #         elif self.teammate_subpolicy_choice == 2:  # go to high value target
        #             waypoint_to_go = self.go_to_highvalue_policy.act(teammate_subpolicy_observation)
        #             teammate_subpolicy_action = waypoint_to_go
        #
        #         if isinstance(teammate_subpolicy_action, tuple):
        #             teammate_subpolicy_action = teammate_subpolicy_action[0]
        #
        #         # Apply teammate action to aircraft[1]
        #         teammate_waypoint = self.env.process_action(teammate_subpolicy_action, agent_id=1)
        #         self.env.agents[self.env.aircraft_ids[1]].waypoint_override = teammate_waypoint
        #
        #     # Apply escape action directly to environment
        #     base_obs, base_reward, base_terminated, base_truncated, base_info = self.env.step(escape_action)
        #
        #     # Convert base_env elements to wrapper elements
        #     observation = self.get_observation(0)
        #     reward = self.get_reward(base_info)
        #     info = base_info
        #     terminated = base_terminated
        #     truncated = base_truncated
        #
        #     self.steps_since_last_selection += 1
        #
        #     #print(f"EVADE: Taking tangential escape action {escape_action}")
        #     return observation, reward, terminated, truncated, info

        # Normal subpolicy selection logic when not evading
        if self.subpolicy_choice is None or self.steps_since_last_selection >= self.action_rate:
            self.steps_since_last_selection = 0

            # Track policy switching for penalty later
            self.switched_policies = False
            if self.last_action != action:
                self.total_switches += 1
                self.switched_policies = True

            # Check if we should auto-switch from change_region to local_search
            if action == 1 and self.has_reached_target_region(0):  # action 1 = change_region
                #print("Auto-switching from change_region to local_search - target region reached")
                self.subpolicy_choice = 0  # Switch to local search

                # Reset the change_region policy's target so it will select a new one next time
                if hasattr(self.change_region_subpolicy, 'target_region'):
                    self.change_region_subpolicy.target_region = None

            else:
                self.subpolicy_choice = action
        # Normal subpolicy selection logic when not evading
        if self.subpolicy_choice is None or self.steps_since_last_selection >= self.action_rate:
            self.steps_since_last_selection = 0

            # Track policy switching for penalty later
            self.switched_policies = False
            if self.last_action != action:
                self.total_switches += 1
                self.switched_policies = True

            self.subpolicy_choice = action

        ######################## Process subpolicy's action ########################

        subpolicy_observation = self.get_subpolicy_observation(self.subpolicy_choice, 0)
        if self.subpolicy_choice == 0:  # Local search
            subpolicy_action = self.local_search_policy.act(subpolicy_observation)

        elif self.subpolicy_choice == 1:  # Change region
            subpolicy_action = self.change_region_subpolicy.act(subpolicy_observation)

        elif self.subpolicy_choice == 2:  # go to high value target
            subpolicy_action = self.go_to_highvalue_policy.act(subpolicy_observation)

        elif self.subpolicy_choice == 3:  # Evade (shouldn't reach here with new logic)
            subpolicy_action = self.evade_policy.act(subpolicy_observation)

        else:
            raise ValueError(f'ERROR: Got invalid subpolicy selection {self.subpolicy_choice}')

        if isinstance(subpolicy_action, tuple):
            if subpolicy_action[1] == None:
                subpolicy_action = subpolicy_action[0]

        if isinstance(subpolicy_action, int):
            subpolicy_action = np.int32(subpolicy_action)

        if self.subpolicy_choice is not None:
            self.subpolicy_history.append(self.subpolicy_choice)
        else:
            # If no subpolicy chosen yet, use -1 or previous choice
            if len(self.subpolicy_history) > 0:
                self.subpolicy_history.append(self.subpolicy_history[-1])
            else:
                self.subpolicy_history.append(0)  # Default to local search

        ########################################## Process teammate's action ###########################################

        if self.current_teammate and self.env.config['num_aircraft'] >= 2:
            teammate_obs = self.get_observation(1)
            self.teammate_subpolicy_choice = self.current_teammate.choose_subpolicy(teammate_obs, self.teammate_subpolicy_choice)

            teammate_subpolicy_observation = self.get_subpolicy_observation(self.teammate_subpolicy_choice, 1)
            if self.teammate_subpolicy_choice == 0:  # Local search
                direction_to_move = self.current_teammate.local_search_policy.act(teammate_subpolicy_observation)
                teammate_subpolicy_action = direction_to_move

            elif self.teammate_subpolicy_choice == 1:  # Change region
                waypoint_to_go = self.change_region_subpolicy.act(teammate_subpolicy_observation)
                teammate_subpolicy_action = waypoint_to_go


            elif self.teammate_subpolicy_choice == 2:  # go to high value target
                waypoint_to_go = self.go_to_highvalue_policy.act(teammate_subpolicy_observation)
                teammate_subpolicy_action = waypoint_to_go

            if isinstance(teammate_subpolicy_action, tuple):
                teammate_subpolicy_action = teammate_subpolicy_action[0]

            # Apply teammate action to aircraft[1]
            teammate_waypoint = self.env.process_action(teammate_subpolicy_action, agent_id=1)
            self.env.agents[self.env.aircraft_ids[1]].waypoint_override = teammate_waypoint

        ############################################ Step the environment #############################################

        #print(f'applying action {subpolicy_action} {type(subpolicy_action)}')
        base_obs, base_reward, base_terminated, base_truncated, base_info = self.env.step(subpolicy_action)

        # Convert base_env elements to wrapper elements if needed
        observation = self.get_observation(0)
        reward = self.get_reward(base_info)
        self.episode_reward += reward
        info = base_info
        info['policy_switches'] = getattr(self, 'total_switches', 0)
        info['final_subpolicy'] = self.subpolicy_choice
        info['threat_ids'] = getattr(self.env, 'num_threats_identified', 0)
        terminated = base_terminated
        truncated = base_truncated

        current_step = self.env.step_count_outer
        if current_step == 100:
            self.wrapper_observations[100] = observation.copy()

        self.last_action = action
        self.steps_since_last_selection += 1

        if hasattr(self.env, 'set_subpolicy_history'):
            self.env.set_subpolicy_history(self.subpolicy_history)

        # Start logging final wrapper reward after step 100 (hack to make sure we don't terminate without it
        if self.env.step_count_outer >= 100 or terminated or truncated:
            self.env.final_wrapper_reward = self.episode_reward

        if terminated or truncated:
            self.print_episode_statistics()
            self.env.final_wrapper_reward = self.episode_reward

            # NEW: Pass wrapper observations to environment
            self.wrapper_observations[current_step] = observation.copy()
            if hasattr(self.env, 'set_wrapper_observations'):
                self.env.set_wrapper_observations(self.wrapper_observations)

            if hasattr(self.env, 'set_subpolicy_history'):
                self.env.set_subpolicy_history(self.subpolicy_history)

        return observation, reward, terminated, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

########################################################################################################################
######################################    Observations and sub-observations     ########################################
########################################################################################################################

    def get_observation(self, agent_id):
        """
        Generates the observation for the mode selector using env attributes
        """
        # Initialize observation as float32 (not int32)
        obs = np.zeros(6, dtype=np.float32)

        # Calculate targets left
        targets_left = self.env.config['num_targets'] - self.env.targets_identified

        # obs[0]: Steps remaining (normalized 0-1, where 1 = all steps left, 0 = no steps left)
        max_steps_outer = self.env.max_steps / self.env.config['frame_skip']
        obs[0] = (max_steps_outer - self.env.step_count_outer) / self.env.max_steps

        # obs[1]: Detections remaining before game over (normalized 0-1, where 1 = no detections, 0 = max detections)
        obs[1] = (self.env.max_detections - self.env.detections) / self.env.max_detections

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

        return obs

    def get_reward(self, info):
        """Generate reward for the mode selector.

        Reward components:
            (Reward)  Identifying a target
            (Reward)  Finishing early (Maybe?)
            (Penalty) Changing modes too frequently
            (Penalty) Getting detected by a high value target
        """
        target_reward = info['new_target_ids'] * self.reward_per_target_id
        threat_reward = info['new_threat_ids'] * self.reward_per_threat_id
        finish_reward = info['steps_left'] * self.reward_per_step_early \
            if info['done'] else 0
        switch_penalty = self.switched_policies * self.penalty_for_policy_switch  # Bool times penalty
        detect_penalty = info['new_detections'] * self.penalty_per_detection

        # 15 * 1 rew/tgt = 15
        # 2 * 5 r/threat = 10
        # ~600 early * 0.05 rew/step early = 30
        # 600 policy switches * 0.02 = -12 penalty

        reward = switch_penalty + detect_penalty + target_reward + + threat_reward + finish_reward
        return reward


########################################################################################################################
############################################    Subpolicy Observations     #############################################
########################################################################################################################

    def get_subpolicy_observation(self, selected_subpolicy, agent_id):
        #print(f'[get_subpolicy_observation] selected_subpolicy: {selected_subpolicy}')
        if selected_subpolicy == 0: # Get obs for local search
            observation = self.get_observation_localsearch(agent_id)
            #observation = self.normalize_local_search_obs(observation)

        elif selected_subpolicy == 1: # Change region
            observation = self.get_observation_changeregion(agent_id)

        elif selected_subpolicy == 2: # Go to nearest
            observation = self.get_observation_nearest_threat(agent_id)

        elif selected_subpolicy == 3: # Evade
            observation = self.get_observation_evade(agent_id)

        return observation

    def get_observation_localsearch(self, agent_id):
        #return self.env.get_observation_nearest_n(agent_id)
        return self.env.get_observation_nearest_n_safe(agent_id)

    def get_observation_evade(self, agent_id):
        """
        Get observation for evade policy including goal position
        Returns: [goal_dx, goal_dy, threat_dx, threat_dy]
        """
        obs = np.zeros(4, dtype=np.float32)

        agent_pos = np.array([self.env.agents[self.env.aircraft_ids[agent_id]].x,
                              self.env.agents[self.env.aircraft_ids[agent_id]].y])

        # Goal position (dx, dy)
        if self.evade_goal is not None:
            goal_vector = self.evade_goal - agent_pos
            obs[0] = goal_vector[0]  # dx to goal
            obs[1] = goal_vector[1]  # dy to goal

        # Nearest threat position (dx, dy)
        if len(self.env.threats) > 0:
            threat_distances = []
            for threat_idx in range(len(self.env.threats)):
                threat_pos = np.array([self.env.threats[threat_idx, 0], self.env.threats[threat_idx, 1]])
                distance = np.sqrt(np.sum((threat_pos - agent_pos) ** 2))
                threat_distances.append((distance, threat_pos))

            # Get nearest threat
            threat_distances.sort(key=lambda x: x[0])
            nearest_threat_pos = threat_distances[0][1]
            threat_vector = nearest_threat_pos - agent_pos
            obs[2] = threat_vector[0]  # dx to nearest threat
            obs[3] = threat_vector[1]  # dy to nearest threat

        return obs


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
            teammate_x = self.env.agents[self.env.aircraft_ids[1 if agent_id==0 else 0]].x
            teammate_y = self.env.agents[self.env.aircraft_ids[1 if agent_id==0 else 0]].y
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
            if quad == 'NW': quad_mask = (target_positions[:, 0] < 0) & (target_positions[:, 1] >= 0)
            elif quad == 'NE': quad_mask = (target_positions[:, 0] >= 0) & (target_positions[:, 1] >= 0)
            elif quad == 'SW': quad_mask = (target_positions[:, 0] < 0) & (target_positions[:, 1] < 0)
            elif quad == 'SE': quad_mask = (target_positions[:, 0] >= 0) & (target_positions[:, 1] < 0)

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

########################################################################################################################
###############################################    Teammate Methods     ################################################
########################################################################################################################


########################################################################################################################
###############################################    Helper functions     ################################################
########################################################################################################################

    def has_reached_target_region(self, agent_id=0):
        """
        Check if agent has reached the target region when using change_region subpolicy
        Returns True if agent is close enough to the target region center
        """
        if not hasattr(self.change_region_subpolicy,
                       'target_region') or self.change_region_subpolicy.target_region is None:
            return False

        # Get agent position
        agent_x = self.env.agents[self.env.aircraft_ids[agent_id]].x
        agent_y = self.env.agents[self.env.aircraft_ids[agent_id]].y
        agent_pos = np.array([agent_x, agent_y])

        # Get target region center in actual coordinates
        target_region_id = self.change_region_subpolicy.target_region
        region_centers = {
            0: np.array([-0.5, 0.5]),  # NW
            1: np.array([0.5, 0.5]),  # NE
            2: np.array([-0.5, -0.5]),  # SW
            3: np.array([0.5, -0.5])  # SE
        }

        # Convert normalized center to actual coordinates
        map_half_size = self.env.config['gameboard_size'] / 2
        region_center_norm = region_centers.get(target_region_id, np.array([0.0, 0.0]))
        region_center_actual = region_center_norm * map_half_size

        # Calculate distance to region center
        distance_to_region = np.linalg.norm(agent_pos - region_center_actual)

        # Define arrival threshold (about 25% of quadrant size)
        arrival_threshold = map_half_size * 0.25

        reached = distance_to_region <= arrival_threshold
        # if reached:
        #     print(f"Agent reached target region {target_region_id} (distance: {distance_to_region:.1f})")

        return reached

    def near_threat(self, agent_id = 0):
        """
        Check if the agent is near a threat and should automatically switch to evade mode.
        Returns True if agent is within threat radius or warning zone of any threat.
        """
        # Get agent position
        agent_pos = np.array([self.env.agents[self.env.aircraft_ids[agent_id]].x,
                              self.env.agents[self.env.aircraft_ids[agent_id]].y])

        # Check distance to all threats
        for threat_idx in range(len(self.env.threats)):
            threat_pos = np.array([self.env.threats[threat_idx, 0], self.env.threats[threat_idx, 1]])
            distance_to_threat = np.sqrt(np.sum((threat_pos - agent_pos) ** 2))

            threat_radius = self.env.config['threat_radius']
            warning_radius = threat_radius * 1.7  # 50% larger than threat radius for early warning

            # Trigger evade mode if within warning radius
            if distance_to_threat <= warning_radius:
                return True
        return False


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

    def set_evade_goal(self):
        """Set the evade goal to a safe position away from threats"""
        agent_pos = np.array([self.env.agents[self.env.aircraft_ids[0]].x,
                              self.env.agents[self.env.aircraft_ids[0]].y])

        # Find the nearest threat position
        nearest_threat_pos = None
        min_threat_distance = float('inf')

        for threat_idx in range(len(self.env.threats)):
            threat_pos = np.array([self.env.threats[threat_idx, 0], self.env.threats[threat_idx, 1]])
            distance = np.sqrt(np.sum((threat_pos - agent_pos) ** 2))
            if distance < min_threat_distance:
                min_threat_distance = distance
                nearest_threat_pos = threat_pos

        if nearest_threat_pos is not None:
            # Calculate direction away from nearest threat
            threat_to_agent = agent_pos - nearest_threat_pos
            threat_distance = np.linalg.norm(threat_to_agent)

            if threat_distance > 0:
                # Set goal to a safe distance away from threat in the same direction
                safe_distance = self.env.config['threat_radius'] * 2.5  # 2.5x threat radius
                direction_away = threat_to_agent / threat_distance
                self.evade_goal = nearest_threat_pos + direction_away * safe_distance

                # Clamp goal to map boundaries
                map_half_size = self.env.config['gameboard_size'] / 2
                self.evade_goal[0] = np.clip(self.evade_goal[0], -map_half_size, map_half_size)
                self.evade_goal[1] = np.clip(self.evade_goal[1], -map_half_size, map_half_size)

                print(f"Set evade goal to safe position: {self.evade_goal}")
            else:
                # Fallback: move to map center
                self.evade_goal = np.array([0.0, 0.0])
        else:
            # No threats found, move to center
            self.evade_goal = np.array([0.0, 0.0])

    def reached_evade_goal(self):
        """Check if agent has reached the evade goal"""
        if self.evade_goal is None:
            return True

        agent_pos = np.array([self.env.agents[self.env.aircraft_ids[0]].x,
                              self.env.agents[self.env.aircraft_ids[0]].y])

        distance_to_goal = np.sqrt(np.sum((self.evade_goal - agent_pos) ** 2))
        return distance_to_goal <= self.evade_goal_threshold

    def compute_tangential_escape_action(self, agent_id=0):
        """
        Compute a direct tangential escape action when near a threat.
        Uses state persistence to maintain consistent circumnavigation direction.
        """
        # Get agent position
        agent_pos = np.array([self.env.agents[self.env.aircraft_ids[agent_id]].x,
                              self.env.agents[self.env.aircraft_ids[agent_id]].y])

        # Find nearest threat
        nearest_threat_pos = None
        min_distance = float('inf')
        nearest_threat_idx = None

        for threat_idx in range(len(self.env.threats)):
            threat_pos = np.array([self.env.threats[threat_idx, 0], self.env.threats[threat_idx, 1]])
            distance = np.sqrt(np.sum((threat_pos - agent_pos) ** 2))
            if distance < min_distance:
                min_distance = distance
                nearest_threat_pos = threat_pos
                nearest_threat_idx = threat_idx

        if nearest_threat_pos is None:
            return 0  # Default action if no threats

        threat_radius = self.env.config['threat_radius']
        safety_margin = threat_radius * 1.8  # Increased safety margin

        # Check if we need to start or continue circumnavigation
        if min_distance <= safety_margin:
            return self._circumnavigate_threat(agent_pos, nearest_threat_pos, threat_radius, agent_id)
        else:
            # Far enough from threat, reset circumnavigation state
            self._reset_circumnavigation_state()
            return 0

    def _circumnavigate_threat(self, agent_pos, threat_pos, threat_radius, agent_id=0):
        """Handle circumnavigation around a threat with state persistence"""

        # Direction mapping (16 directions)
        directions = np.array([
            (0, 1),  # North (0°)
            (0.383, 0.924),  # NNE (22.5°)
            (0.707, 0.707),  # NE (45°)
            (0.924, 0.383),  # ENE (67.5°)
            (1, 0),  # East (90°)
            (0.924, -0.383),  # ESE (112.5°)
            (0.707, -0.707),  # SE (135°)
            (0.383, -0.924),  # SSE (157.5°)
            (0, -1),  # South (180°)
            (-0.383, -0.924),  # SSW (202.5°)
            (-0.707, -0.707),  # SW (225°)
            (-0.924, -0.383),  # WSW (247.5°)
            (-1, 0),  # West (270°)
            (-0.924, 0.383),  # WNW (292.5°)
            (-0.707, 0.707),  # NW (315°)
            (-0.383, 0.924),  # NNW (337.5°)
        ], dtype=np.float32)

        # Vector from threat to agent
        threat_to_agent = agent_pos - threat_pos
        distance_to_threat = np.linalg.norm(threat_to_agent)

        if distance_to_threat < 1e-6:
            return 0  # Default if at threat center

        # Calculate current angle around threat
        current_angle = np.arctan2(threat_to_agent[1], threat_to_agent[0])

        # Initialize or update circumnavigation state
        if not self.circumnavigation_state['active']:
            self._initialize_circumnavigation(threat_pos, current_angle, threat_radius)

        # Check if circumnavigation is complete
        if self._is_circumnavigation_complete(current_angle, threat_pos, agent_pos):
            self._reset_circumnavigation_state()
            # Move toward original target
            return self._get_action_toward_nearest_target(agent_pos, directions)

        # Continue circumnavigation
        return self._get_circumnavigation_action(current_angle, threat_to_agent, directions)

    def _initialize_circumnavigation(self, threat_pos, start_angle, threat_radius):
        """Initialize circumnavigation state"""
        self.circumnavigation_state['active'] = True
        self.circumnavigation_state['threat_pos'] = threat_pos.copy()
        self.circumnavigation_state['start_angle'] = start_angle
        self.circumnavigation_state['last_angle'] = start_angle
        self.circumnavigation_state['safety_distance'] = threat_radius * 1.5

        # Choose direction based on which way moves us more toward targets
        # For now, default to counterclockwise
        self.circumnavigation_state['chosen_direction'] = 'counterclockwise'

        print(f"Starting circumnavigation: direction={self.circumnavigation_state['chosen_direction']}")

    def _is_circumnavigation_complete(self, current_angle, threat_pos, agent_pos):
        """Check if we've gone far enough around the threat to have a clear path"""
        if not self.circumnavigation_state['active']:
            return False

        # Calculate how far we've traveled around the threat
        start_angle = self.circumnavigation_state['start_angle']
        angle_traveled = current_angle - start_angle

        # Normalize angle difference to [-π, π]
        while angle_traveled > np.pi:
            angle_traveled -= 2 * np.pi
        while angle_traveled < -np.pi:
            angle_traveled += 2 * np.pi

        # Check if we've gone at least 90 degrees around
        min_angle_traveled = np.pi / 2  # 90 degrees

        if self.circumnavigation_state['chosen_direction'] == 'counterclockwise':
            sufficient_travel = angle_traveled >= min_angle_traveled
        else:  # clockwise
            sufficient_travel = angle_traveled <= -min_angle_traveled

        if sufficient_travel:
            # Also check if we now have a clear line to targets
            return self._has_clear_path_to_targets(agent_pos, threat_pos)

        return False

    def _has_clear_path_to_targets(self, agent_pos, threat_pos):
        """Check if there's a clear path from current position to nearest unknown target"""
        # Get unknown target positions
        target_positions = self.env.targets[:self.env.config['num_targets'], 3:5]
        target_info_levels = self.env.targets[:self.env.config['num_targets'], 2]
        unknown_mask = target_info_levels < 1.0

        if not np.any(unknown_mask):
            return True  # No targets left, circumnavigation complete

        unknown_positions = target_positions[unknown_mask]
        distances = np.sqrt(np.sum((unknown_positions - agent_pos) ** 2, axis=1))
        nearest_target_pos = unknown_positions[np.argmin(distances)]

        # Check if path to nearest target intersects threat
        return self._path_clear_of_threat(agent_pos, nearest_target_pos, threat_pos)

    def _path_clear_of_threat(self, start_pos, end_pos, threat_pos):
        """Check if straight line path from start to end clears the threat"""
        threat_radius = self.env.config['threat_radius'] * 1.2  # Safety buffer

        # Vector from start to end
        path_vector = end_pos - start_pos
        path_length = np.linalg.norm(path_vector)

        if path_length < 1e-6:
            return True

        path_unit = path_vector / path_length

        # Vector from start to threat
        start_to_threat = threat_pos - start_pos

        # Project threat onto path
        projection_length = np.dot(start_to_threat, path_unit)

        # Only check collision if projection is within the path segment
        if 0 <= projection_length <= path_length:
            closest_point_on_path = start_pos + projection_length * path_unit
            distance_to_threat = np.linalg.norm(threat_pos - closest_point_on_path)
            return distance_to_threat > threat_radius

        return True  # Threat is not along the path

    def _get_circumnavigation_action(self, current_angle, threat_to_agent, directions):
        """Get action to continue circumnavigation"""
        distance_to_threat = np.linalg.norm(threat_to_agent)

        if distance_to_threat > 0:
            # Calculate tangent direction
            threat_unit = threat_to_agent / distance_to_threat

            if self.circumnavigation_state['chosen_direction'] == 'counterclockwise':
                tangent_direction = np.array([-threat_unit[1], threat_unit[0]])  # +90 degrees
            else:  # clockwise
                tangent_direction = np.array([threat_unit[1], -threat_unit[0]])  # -90 degrees

            # Add slight outward bias to maintain safe distance
            outward_direction = threat_unit  # Away from threat
            bias_strength = 0.2

            combined_direction = tangent_direction * (1 - bias_strength) + outward_direction * bias_strength
            combined_direction = combined_direction / np.linalg.norm(combined_direction)

            # Find best matching action
            dot_products = np.dot(directions, combined_direction)
            best_action = np.argmax(dot_products)

            return np.int32(best_action)

        return 0

    def _get_action_toward_nearest_target(self, agent_pos, directions):
        """Get action to move toward nearest unknown target after circumnavigation"""
        # Get unknown target positions
        target_positions = self.env.targets[:self.env.config['num_targets'], 3:5]
        target_info_levels = self.env.targets[:self.env.config['num_targets'], 2]
        unknown_mask = target_info_levels < 1.0

        if not np.any(unknown_mask):
            return 0  # No targets left

        unknown_positions = target_positions[unknown_mask]
        distances = np.sqrt(np.sum((unknown_positions - agent_pos) ** 2, axis=1))
        nearest_target_pos = unknown_positions[np.argmin(distances)]

        # Direction to nearest target
        target_vector = nearest_target_pos - agent_pos
        target_distance = np.linalg.norm(target_vector)

        if target_distance > 0:
            target_direction = target_vector / target_distance
            dot_products = np.dot(directions, target_direction)
            best_action = np.argmax(dot_products)
            return np.int32(best_action)

        return 0

    def _reset_circumnavigation_state(self):
        """Reset circumnavigation state"""
        self.circumnavigation_state = {
            'active': False,
            'threat_pos': None,
            'chosen_direction': None,
            'last_angle': None,
            'start_angle': None,
            'safety_distance': None
        }

    def print_episode_statistics(self):
        """Print statistics about subpolicy usage and switches for the episode"""
        if not self.subpolicy_history:
            print("No subpolicy history available")
            return

        total_steps = len(self.subpolicy_history)
        if total_steps == 0:
            return

        # Count steps for each subpolicy
        subpolicy_counts = {}
        for policy in self.subpolicy_history:
            policy_key = int(policy) if hasattr(policy, 'item') else int(policy)
            subpolicy_counts[policy_key] = subpolicy_counts.get(policy_key, 0) + 1

        # Get percentages with safe access using .get() method
        local_search_pct = (subpolicy_counts.get(0, 0) / total_steps) * 100
        change_region_pct = (subpolicy_counts.get(1, 0) / total_steps) * 100
        go_to_threat_pct = (subpolicy_counts.get(2, 0) / total_steps) * 100

        # Calculate percentages
        print(
            f"\n=== Episode {getattr(self.env, 'episode_counter', 'N/A')}: {self.total_switches} policy switches ({self.total_switches / total_steps:.2f}/step), Local search {local_search_pct:.1f}% / ChangeRegion {change_region_pct:.1f}% / GoToThreat {go_to_threat_pct:.1f}%")

    def get_current_subpolicy_info(self):
        """Return current subpolicy information for display"""
        if self.subpolicy_choice is None:
            return 0, "Local Search"  # Default

        mode_names = {
            0: "Local Search",
            1: "Change Region",
            2: "Go to Threat",
            3: "Evade"
        }
        return self.subpolicy_choice, mode_names.get(self.subpolicy_choice, "Unknown")

    def get_teammate_subpolicy_info(self):
        """Return teammate's current subpolicy information for display"""
        if not hasattr(self, 'teammate_subpolicy_choice'):
            return 0, "Local Search"  # Default

        mode_names = {
            0: "Local Search",
            1: "Change Region",
            2: "Go to Threat",
            3: "Evade"
        }
        return self.teammate_subpolicy_choice, mode_names.get(self.teammate_subpolicy_choice, "Unknown")