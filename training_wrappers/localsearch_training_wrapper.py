import gymnasium as gym
import numpy as np


class MaisrLocalSearchWrapper(gym.Env):
    """Wrapper for training the mode selector
    Subpolicies are treated as part of the environment dynamics.
    """
    def __init__(self, env, obs_noise_std):

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

        self.render_mode = self.env.render_mode
        self.run_name = self.env.run_name  # For logging
        self.tag = self.env.tag

        self.obs_noise_std = obs_noise_std

        self.teammate_active = self.env.config['teammate_active_at_start']

        print(f'Wrapped env created for local search training. Action space = {self.action_space}, obs space = {self.observation_space}')


    def reset(self, seed=None, options=None):
        raw_obs, _ = self.env.reset()
        self.num_switches = 0 # How many times the agent has switch policies in this round. Slight penalty to encourage consistency
        self.last_action = 0
        self.steps_since_last_selection = 0
        self.current_subpolicy = None

        return raw_obs, _


    def step(self, action: np.int32):
        """ Apply the mode selector's action (Index of selected subpolicy)"""

        # Get teammate action
        if self.env.config['num_aircraft'] == 2 and self.teammate_active:
            teammate_action = self.get_teammate_action()
            self.env.agents[self.env.aircraft_ids[1]].waypoint_override = teammate_action

        # Step the environment
        base_obs, base_reward, base_terminated, base_truncated, base_info = self.env.step(action)

        observation = self.env.get_observation_nearest_n()

        if self.obs_noise_std > 0:
            noise = np.random.normal(0, self.obs_noise_std, observation.shape)
            observation = np.clip(observation + noise, -1, 1)  # Clip to valid range



        # Convert base_env elements to wrapper elements if needed
        reward = base_reward
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
        pass

    def get_reward(self, info):
        """Generate reward for the mode selector."""
        pass

    def get_teammate_action(self):
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
        else:
            # No unknown targets remaining, stay at current position
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
        return