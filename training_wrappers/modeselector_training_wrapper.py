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

        if teammate_policy is not None:
            self.current_teammate = teammate_policy

        self.teammate_manager = teammate_manager

        if teammate_policy and teammate_manager:
            raise ValueError('Cannot specify a teammate policy and a teammate manager at the same time')


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
        self.reward_per_threat_id = 3
        self.penalty_for_policy_switch = 0.05
        self.penalty_per_detection = 0 # Currently none (but episode ends if we exceed max)

        self.mode_dict = {0:"local search", 1:'change_region', 2:'go_to_threat'}

        # Goal tracking for evade policy
        self.evade_goal = None
        self.evade_goal_threshold = 30.0  # Distance threshold to consider goal "reached"
        self.last_evade_step = -1  # Track when we last used evade to detect continuous usage

        # Load normalization stats
        try:
            self.norm_stats = np.load(local_search_policy.norm_stats_filepath, allow_pickle=True).item()
            print("Loaded normalization stats for local search policy")
        except FileNotFoundError:
            print("Warning: No normalization stats found, using raw observations")
            self.norm_stats = None


    def reset(self, seed=None, options=None):
        raw_obs, _ = self.env.reset()
        self.num_switches = 0
        self.last_action = 0
        self.steps_since_last_selection = 0
        self.subpolicy_choice = None
        self.teammate_subpolicy_choice = 0

        # Reset evade goal tracking
        #self.evade_goal = None
        #self.last_evade_step = -1

        # Instantiate teammate for this episode
        if self.teammate_manager:
            self.teammate_manager.reset_for_episode()
            # Choose selection strategy: 'random' or 'curriculum'
            self.current_teammate = self.teammate_manager.select_random_teammate()
            print(f"Selected teammate: {self.current_teammate.name if self.current_teammate else 'None'}")

        return raw_obs, _


    # def step(self, action: np.int32):
    #     """ Apply the mode selector's action (Index of selected subpolicy)"""
    #
    #     ######################## Choose a subpolicy ########################
    #     # TODO temp testing
    #     # Check if we need to evade
    #     if self.near_threat():
    #         # Compute direct tangential escape action
    #         escape_action = self.compute_tangential_escape_action()
    #
    #         # Apply escape action directly to environment
    #         base_obs, base_reward, base_terminated, base_truncated, base_info = self.env.step(escape_action)
    #
    #         # Convert base_env elements to wrapper elements if needed
    #         observation = self.get_observation(0)
    #         reward = self.get_reward(base_info)
    #         info = base_info
    #         terminated = base_terminated
    #         truncated = base_truncated
    #
    #         self.steps_since_last_selection += 1
    #
    #         print(f"EVADE: Taking tangential escape action {escape_action}")
    #         return observation, reward, terminated, truncated, info
    #     # if self.near_threat():
    #     #     # Set goal if this is the first step of evading
    #     #     if self.subpolicy_choice != 3 or self.last_evade_step != self.env.step_count_outer - 1:
    #     #         self.set_evade_goal()
    #     #
    #     #     self.subpolicy_choice = 3
    #     #     self.last_evade_step = self.env.step_count_outer
    #
    #
    #     elif self.subpolicy_choice == 3:  # We were evading but no longer near threat
    #
    #         # Clear evade goal immediately when no longer near threat
    #         # This prevents getting stuck trying to reach unreachable goals
    #         if not self.near_threat():
    #             print("No longer near threat, clearing evade goal and allowing new subpolicy selection")
    #             #self.evade_goal = None
    #             self.subpolicy_choice = action  # Allow new selection
    #
    #         else:
    #             # Continue evading
    #             print("Still evading toward goal")
    #
    #     elif self.subpolicy_choice is None or self.steps_since_last_selection >= self.action_rate:
    #         self.steps_since_last_selection = 0
    #         #print(f'SELECTOR TOOK ACTION {action} to switch to mode {self.mode_dict[int(action)]}')
    #
    #         # Track policy switching for penalty later
    #         self.switched_policies = False
    #         if self.last_action != action:
    #             self.num_switches += 1 # Not used yet
    #             self.switched_policies = True
    #
    #         self.subpolicy_choice = action # Activate selected subpolicy and generate its action
    #
    #
    #     ######################## Process subpolicy's action ########################
    #
    #     subpolicy_observation = self.get_subpolicy_observation(self.subpolicy_choice, 0)
    #     if self.subpolicy_choice == 0: # Local search
    #         subpolicy_action = self.local_search_policy.act(subpolicy_observation)
    #
    #     elif self.subpolicy_choice == 1: # Change region
    #         subpolicy_action = self.change_region_subpolicy.act(subpolicy_observation)
    #
    #     elif self.subpolicy_choice == 2: # go to high value target
    #         subpolicy_action = self.go_to_highvalue_policy.act(subpolicy_observation)
    #
    #     elif self.subpolicy_choice == 3: # Evade
    #         subpolicy_action = self.evade_policy.act(subpolicy_observation)
    #
    #     else:
    #         raise ValueError(f'ERROR: Got invalid subpolicy selection {self.subpolicy_choice}')
    #
    #     if isinstance(subpolicy_action, tuple):
    #         #warnings.warn(f"WARNING: Agent's action was a tuple {subpolicy_action}. Taking first element. This is likely a bug in how agent actions are being accessed.")
    #         subpolicy_action = subpolicy_action[0]
    #
    #
    #     ########################################## Process teammate's action ###########################################
    #
    #     # TODO rewrite to accept either mode-selector teammate, greedy, or human
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
    #     ############################################ Step the environment #############################################
    #
    #     base_obs, base_reward, base_terminated, base_truncated, base_info = self.env.step(subpolicy_action)
    #
    #     # Convert base_env elements to wrapper elements if needed
    #     observation = self.get_observation(0)
    #     reward = self.get_reward(base_info)
    #     info = base_info
    #     terminated = base_terminated
    #     truncated = base_truncated
    #
    #     self.last_action = action
    #
    #     self.steps_since_last_selection += 1
    #
    #     return observation, reward, terminated, truncated, info

    def step(self, action: np.int32):
        """ Apply the mode selector's action (Index of selected subpolicy)"""

        ######################## Choose a subpolicy ########################
        # Check if we need to evade with direct tangential movement
        if self.near_threat():
            # Compute direct tangential escape action

            escape_action = self.compute_tangential_escape_action()
            print(f'Escape action {escape_action} ({type(escape_action)})')

            # Process teammate's action if needed
            if self.current_teammate and self.env.config['num_aircraft'] >= 2:
                teammate_obs = self.get_observation(1)
                self.teammate_subpolicy_choice = self.current_teammate.choose_subpolicy(teammate_obs,
                                                                                        self.teammate_subpolicy_choice)

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

            # Apply escape action directly to environment
            base_obs, base_reward, base_terminated, base_truncated, base_info = self.env.step(escape_action)

            # Convert base_env elements to wrapper elements
            observation = self.get_observation(0)
            reward = self.get_reward(base_info)
            info = base_info
            terminated = base_terminated
            truncated = base_truncated

            self.steps_since_last_selection += 1

            print(f"EVADE: Taking tangential escape action {escape_action}")
            return observation, reward, terminated, truncated, info

        # Normal subpolicy selection logic when not evading
        elif self.subpolicy_choice is None or self.steps_since_last_selection >= self.action_rate:
            self.steps_since_last_selection = 0

            # Track policy switching for penalty later
            self.switched_policies = False
            if self.last_action != action:
                self.num_switches += 1
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
            subpolicy_action = subpolicy_action[0]

        ########################################## Process teammate's action ###########################################

        if self.current_teammate and self.env.config['num_aircraft'] >= 2:
            teammate_obs = self.get_observation(1)
            self.teammate_subpolicy_choice = self.current_teammate.choose_subpolicy(teammate_obs,
                                                                                    self.teammate_subpolicy_choice)

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

        print(f'applying action {subpolicy_action} {type(subpolicy_action)}')
        base_obs, base_reward, base_terminated, base_truncated, base_info = self.env.step(subpolicy_action)

        # Convert base_env elements to wrapper elements if needed
        observation = self.get_observation(0)
        reward = self.get_reward(base_info)
        info = base_info
        terminated = base_terminated
        truncated = base_truncated

        self.last_action = action
        self.steps_since_last_selection += 1

        return observation, reward, terminated, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

########################################################################################################################
######################################    Observations and sub-observations     ########################################
########################################################################################################################

    def get_observation(self, agent_id):
        """Generates the observation for the mode selector using env attributes"""
        # TODO use agent_id to return observation relative to that agent

        # Core state: observation vector
        targets_left = self.env.config['num_targets'] - self.env.targets_identified
        obs = np.zeros(6, dtype=np.int32)

        if targets_left:
            obs[0] = (self.env.max_steps - self.env.step_count_outer) / self.env.max_steps  # Steps remaining
            obs[1] = (self.env.max_detections - self.env.detections) / self.env.max_detections  # Progress toward max detections (game over)
            obs[2] = targets_left / self.env.config['num_targets']  # Ratio of targets ID'd
            obs[3] = self.unknown_targets_in_current_quadrant() / targets_left  # Ratio of all targets that are in current quadrant
            obs[4] = self.get_distance_to_teammate() / self.env.config['gameboard_size']  # Proximity to human (Helps decide whether to change regions
            obs[5] = self.get_adaptation_signal()  # Adaptation signal (placeholder as 0 for now)

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
        finish_reward = info['steps_left'] if info['done'] else 0
        switch_penalty = self.switched_policies * self.penalty_for_policy_switch  # Bool times penalty
        detect_penalty = info['new_detections'] * self.penalty_per_detection

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
        # TODO use agent_id to return observation relative to that agent
        return self.env.get_observation_nearest_n(agent_id)

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
        Contents:
            obs[0] - dx to nearest threat
            obs[1] - dy to nearest threat
            obs[2] - dx to 2nd nearest threat
            obs[3] - dy to 2nd nearest threat
        """
        # TODO use agent_id to return observation relative to that agent
        obs = np.zeros(4, dtype=np.float32)  # Changed from 2 to 4

        # Get agent position
        agent_pos = np.array([self.env.agents[self.env.aircraft_ids[0]].x,
                              self.env.agents[self.env.aircraft_ids[0]].y])

        # Get threat positions
        threat_positions = self.env.threats

        if len(threat_positions) == 0:
            return obs  # Return zeros if no threats

        # Calculate distances to all threats
        distances = np.linalg.norm(threat_positions - agent_pos, axis=1)
        sorted_indices = np.argsort(distances)

        # Get vector to nearest threat
        if len(sorted_indices) >= 1:
            nearest_pos = threat_positions[sorted_indices[0]]
            vector_to_nearest = nearest_pos - agent_pos
            obs[0] = vector_to_nearest[0]  # dx to nearest
            obs[1] = vector_to_nearest[1]  # dy to nearest

        # Get vector to second nearest threat
        if len(sorted_indices) >= 2:
            second_nearest_pos = threat_positions[sorted_indices[1]]
            vector_to_second = second_nearest_pos - agent_pos
            obs[2] = vector_to_second[0]  # dx to 2nd nearest
            obs[3] = vector_to_second[1]  # dy to 2nd nearest

        return obs

########################################################################################################################
###############################################    Teammate Methods     ################################################
########################################################################################################################


########################################################################################################################
###############################################    Helper functions     ################################################
########################################################################################################################

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
            warning_radius = threat_radius * 1.5  # 50% larger than threat radius for early warning

            # Trigger evade mode if within warning radius
            if distance_to_threat <= warning_radius:
                return True
        return False


    def unknown_targets_in_current_quadrant(self):
        """Returns the number of unknown targets in the agent's quadrant"""

        agent_x = self.env.agents[self.env.aircraft_ids[0]].x
        agent_y = self.env.agents[self.env.aircraft_ids[0]].y

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


    def get_distance_to_teammate(self):
        """Returns pixel range between current location and teammate's location
        Note: Should NOT be normalized (should be in range [- gameboard_size, + gameboard_size])"""

        # Check if there are multiple aircraft (teammates exist)
        if self.env.config['num_aircraft'] < 2: # No teammate, return maximum distance as default
            return self.env.config['gameboard_size']

        # Get agent position (aircraft 0)
        agent_x = self.env.agents[self.env.aircraft_ids[0]].x
        agent_y = self.env.agents[self.env.aircraft_ids[0]].y

        # Get teammate position (aircraft 1)
        teammate_x = self.env.agents[self.env.aircraft_ids[1]].x
        teammate_y = self.env.agents[self.env.aircraft_ids[1]].y

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
        Returns the action index (0-15) that moves tangentially around the nearest threat
        with a slight bias toward continuing around it.
        """
        # Get agent position
        agent_pos = np.array([self.env.agents[self.env.aircraft_ids[agent_id]].x,
                              self.env.agents[self.env.aircraft_ids[agent_id]].y])

        # Find nearest threat
        nearest_threat_pos = None
        min_distance = float('inf')

        for threat_idx in range(len(self.env.threats)):
            threat_pos = np.array([self.env.threats[threat_idx, 0], self.env.threats[threat_idx, 1]])
            distance = np.sqrt(np.sum((threat_pos - agent_pos) ** 2))
            if distance < min_distance:
                min_distance = distance
                nearest_threat_pos = threat_pos

        if nearest_threat_pos is None:
            return 0  # Default action if no threats

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

        # Vector from agent to threat
        threat_vector = nearest_threat_pos - agent_pos
        threat_distance = np.linalg.norm(threat_vector)

        if threat_distance < 1e-6:
            return 0  # Default if at threat center

        threat_unit = threat_vector / threat_distance

        # Calculate two tangent directions (perpendicular to threat vector)
        tangent1 = np.array([-threat_unit[1], threat_unit[0]])  # +90 degrees
        tangent2 = np.array([threat_unit[1], -threat_unit[0]])  # -90 degrees

        # Add slight bias away from threat (mix tangent with escape direction)
        escape_direction = -threat_unit  # Direction directly away from threat
        bias_strength = 0.3  # How much to bias toward escape (0.0 = pure tangent, 1.0 = pure escape)

        # Create biased tangent directions
        biased_tangent1 = tangent1 * (1 - bias_strength) + escape_direction * bias_strength
        biased_tangent2 = tangent2 * (1 - bias_strength) + escape_direction * bias_strength

        # Normalize the biased directions
        biased_tangent1 = biased_tangent1 / np.linalg.norm(biased_tangent1)
        biased_tangent2 = biased_tangent2 / np.linalg.norm(biased_tangent2)

        # Choose direction that leads toward more open space or mission objectives
        # For now, use a simple heuristic: prefer direction that moves toward map center
        map_center = np.array([0.0, 0.0])
        center_direction = map_center - agent_pos
        center_distance = np.linalg.norm(center_direction)

        if center_distance > 1e-6:
            center_unit = center_direction / center_distance

            # Choose tangent that better aligns with moving toward center
            dot1 = np.dot(biased_tangent1, center_unit)
            dot2 = np.dot(biased_tangent2, center_unit)

            chosen_direction = biased_tangent1 if dot1 > dot2 else biased_tangent2
        else:
            # Default to first tangent if at center
            chosen_direction = biased_tangent1

        # Find best matching action from available directions
        dot_products = np.dot(directions, chosen_direction)
        best_action = np.argmax(dot_products)

        return np.int32(best_action)