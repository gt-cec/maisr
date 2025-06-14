import gymnasium as gym
import gym.spaces
import numpy as np
from sympy import trunc
from policies.sub_policies import SubPolicy, GoToNearestHighValueTarget, LocalSearch, ChangeRegions



class MaisrModeSelectorWrapper(gym.Env):
    """Wrapper for training the mode selector
    Subpolicies are treated as part of the environment dynamics.
    """
    def __init__(self,
                 env,
                 local_search_policy: SubPolicy,
                 go_to_highvalue_policy: SubPolicy,
                 change_region_subpolicy: SubPolicy
                 ):

        self.env = env
        self.local_search_policy = local_search_policy
        self.go_to_highvalue_policy = go_to_highvalue_policy
        self.change_region_subpolicy = change_region_subpolicy

        # Define observation space (6 high-level elements about the current game state)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1,
            shape=(6,),
            dtype=np.float32)

        # Action space: 3 possible sub-policies to choose from
        self.action_space = gym.spaces.Discrete(3)

        # Set rewards for mode selector
        self.reward_per_target_id = 1
        self.reward_per_threat_id = 3
        self.penalty_for_policy_switch = 0.05
        self.penalty_per_detection = 0 # Currently none (but episode ends if we exceed max)


    def reset(self):
        raw_obs, _ = self.env.reset()
        self.num_switches = 0 # How many times the agent has switch policies in this round. Slight penalty to encourage consistency
        self.last_action = 0

        return raw_obs, _


    def step(self, action: int):
        """ Apply the mode selector's action (Index of selected subpolicy)"""

        print(f'SELECTOR TOOK ACTION {action}')

        # Track policy switching for penalty later
        self.switched_policies = False
        if self.last_action != action:
            self.num_switches += 1 # Not used yet
            self.switched_policies = True


        # Activate selected subpolicy and generate its action
        selected_subpolicy = action
        subpolicy_observation = self.get_subpolicy_observation(selected_subpolicy)

        if selected_subpolicy == 0: # Local search
            direction_to_move = self.local_search_policy.act(subpolicy_observation)
            subpolicy_action = direction_to_move

        elif selected_subpolicy == 1: # Change region
            waypoint_to_go = self.change_region_subpolicy.act(subpolicy_observation)
            subpolicy_action = waypoint_to_go

        elif selected_subpolicy == 2: # go to high value target
            waypoint_to_go = self.go_to_highvalue_policy.act(subpolicy_observation)
            subpolicy_action = waypoint_to_go

        else:
            raise ValueError(f'ERROR: Got invalid subpolicy selection {selected_subpolicy}')

        # Step the environment
        base_obs, base_reward, base_terminated, base_truncated, base_info = self.env.step(subpolicy_action)

        observation = self.get_observation()
        reward = self.get_reward(base_info)

        # Convert base_env elements to wrapper elements if needed
        info = base_info
        terminated = base_terminated
        truncated = base_truncated

        self.last_action = action

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
            print('Observation for local search:')
            print(observation)

        elif selected_subpolicy == 1: # Change region
            observation = self.get_observation_changeregion()

        elif selected_subpolicy == 2: # Go to nearest
            observation = self.get_observation_nearest_highvalue()

        return observation

    def get_observation_changeregion(self):
        pass

    def get_observation_nearest_highvalue(self):
        pass


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