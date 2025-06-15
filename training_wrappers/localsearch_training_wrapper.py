import gymnasium as gym
import numpy as np
from sympy import trunc
from torch.ao.quantization.backend_config.onednn import observation_type

from policies.sub_policies import SubPolicy, GoToNearestThreat, LocalSearch, ChangeRegions



class MaisrLocalSearchWrapper(gym.Env):
    """Wrapper for training the mode selector
    Subpolicies are treated as part of the environment dynamics.
    """
    def __init__(self, env):

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




    def reset(self, seed=None, options=None):
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
        pass

    def get_reward(self, info):
        """Generate reward for the mode selector."""
        pass


########################################################################################################################
###############################################    Helper functions     ################################################
########################################################################################################################