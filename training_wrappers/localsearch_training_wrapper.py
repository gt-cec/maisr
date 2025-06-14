import gymnasium as gym
import gym.spaces
import numpy as np
from sympy import trunc
from torch.ao.quantization.backend_config.onednn import observation_type

from policies.sub_policies import SubPolicy, GoToNearestThreat, LocalSearch, ChangeRegions



class MaisrLocalSearchWrapper(gym.Env):
    """Wrapper for training the mode selector
    Subpolicies are treated as part of the environment dynamics.
    """
    def __init__(self, env,):

        self.env = env

        # Define observation space
        self.obs_size = 2 * self.env.config['num_observed_targets'] + 2 * self.env.config['num_observed_threats']  # x,y components of unit vector
        self.observation_space = gym.spaces.Box(
            low=-1, high=1,
            shape=(self.obs_size,),
            dtype=np.float32)

        # Action space: 3 possible sub-policies to choose from
        self.action_space = gym.spaces.Discrete(16)
        self.action_rate = 1

        # Set rewards for mode selector
        self.reward_per_target_id = 1
        self.reward_per_threat_id = 3
        self.penalty_for_policy_switch = 0.05
        self.penalty_per_detection = 0 # Currently none (but episode ends if we exceed max)

        self.mode_dict = {0:"local search", 1:'change_region', 2:'go_to_threat'}

        self.render_mode = self.env.render_mode
        self.run_name = self.env.run_name  # For logging
        self.tag = self.env.tag

        print(f'Wrapped env created for local search training. Action space = {self.action_space}, obs space = {self.observation_space}')


    def reset(self):
        raw_obs, _ = self.env.reset()
        self.num_switches = 0 # How many times the agent has switch policies in this round. Slight penalty to encourage consistency
        self.last_action = 0
        self.steps_since_last_selection = 0
        self.current_subpolicy = None

        return raw_obs, _


    def step(self, action: np.int32):
        """ Apply the mode selector's action (Index of selected subpolicy)"""

        # Step the environment
        base_obs, base_reward, base_terminated, base_truncated, base_info = self.env.step(action)

        observation = self.env.get_observation_nearest_n()
        reward = base_reward

        # Convert base_env elements to wrapper elements if needed
        info = base_info
        terminated = base_terminated
        truncated = base_truncated

        return observation, reward, terminated, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

########################################################################################################################
######################################    Observations and sub-observations     ########################################
########################################################################################################################

    def get_observation(self):
        """Generates the observation for the mode selector using env attributes"""

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

    def get_subpolicy_observation(self, selected_subpolicy):
        if selected_subpolicy == 0: # Get obs for local search
            observation = self.env.get_observation_nearest_n()

        elif selected_subpolicy == 1: # Change region
            observation = self.get_observation_changeregion()

        elif selected_subpolicy == 2: # Go to nearest
            observation = self.get_observation_nearest_threat()

        return observation


    def get_observation_changeregion(self):
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
        obs = np.zeros(12, dtype=np.float32)

        # Get agent position
        agent_x = self.env.agents[self.env.aircraft_ids[0]].x
        agent_y = self.env.agents[self.env.aircraft_ids[0]].y

        # Get teammate position (if exists)
        if self.env.config['num_aircraft'] >= 2:
            teammate_x = self.env.agents[self.env.aircraft_ids[1]].x
            teammate_y = self.env.agents[self.env.aircraft_ids[1]].y
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

    def get_observation_nearest_threat(self):
        """
        Contents:
            obs[0] - dx to nearest threat
            obs[1] - dy to nearest threat
            obs[2] - dx to 2nd nearest threat
            obs[3] - dy to 2nd nearest threat
        """
        obs = np.zeros(2, dtype=np.float32)

        # Get agent position
        agent_pos = np.array([self.env.agents[self.env.aircraft_ids[0]].x,
                              self.env.agents[self.env.aircraft_ids[0]].y])

        # Get threat
        threat_position = self.env.threat

        # Get vector to nearest high-value target
        vector_to_threat = threat_position - agent_pos
        obs = vector_to_threat
        # if len(sorted_indices) >= 1:
        #     nearest_pos = highvalue_positions[sorted_indices[0]]
        #     vector_to_nearest = nearest_pos - agent_pos
        #     obs[0] = vector_to_nearest[0]  # dx to nearest
        #     obs[1] = vector_to_nearest[1]  # dy to nearest
        #
        # # Get vector to second nearest high-value target
        # if len(sorted_indices) >= 2:
        #     second_nearest_pos = highvalue_positions[sorted_indices[1]]
        #     vector_to_second = second_nearest_pos - agent_pos
        #     obs[2] = vector_to_second[0]  # dx to 2nd nearest
        #     obs[3] = vector_to_second[1]  # dy to 2nd nearest

        return obs



########################################################################################################################
###############################################    Helper functions     ################################################
########################################################################################################################

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