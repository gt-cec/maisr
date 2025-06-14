"""
Hierarchical MAISR Environment Wrapper
Implements a two-level hierarchical RL system where:
- High-level policy selects from sub-policies at 2 Hz
- Sub-policies execute until changed by high-level policy
- Each sub-policy has specialized observations, actions, and rewards
"""

import gymnasium as gym
import numpy as np
import math
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
from env_combined import MAISREnvVec


class SubPolicy(ABC):
    """Abstract base class for all sub-policies"""
    def __init__(self, name: str):
        self.name = name
        self._action_space = None
        self._observation_space = None

    def act(self, observation):
        pass

    @abstractmethod
    def get_action_space(self) -> gym.Space:
        """Return the action space for this sub-policy"""
        pass

    @abstractmethod
    def get_observation_space(self) -> gym.Space:
        """Return the observation space for this sub-policy"""
        pass

    @abstractmethod
    def get_observation(self, base_obs: np.ndarray, env_state: Dict[str, Any]) -> np.ndarray:
        """Transform base observation into sub-policy specific observation"""
        pass

    @abstractmethod
    def get_reward(self, base_reward: float, env_state: Dict[str, Any],
                   info: Dict[str, Any], progress: Dict[str, Any]) -> float:
        """Calculate sub-policy specific reward"""
        pass

    def get_action(self, obs: np.ndarray, model=None) -> np.ndarray:
        """Get action from sub-policy (can be overridden for trained models)"""
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            return action
        else:
            # Default random action
            return self.get_action_space().sample()

    def is_terminated(self, env_state: Dict[str, Any]) -> bool:
        """Check if sub-policy wants to terminate early"""
        return False

    def should_interrupt(self, env_state: Dict[str, Any]) -> bool:
        """Check if this sub-policy should interrupt current execution"""
        return False


class GoToNearestHighValueTarget(SubPolicy):
    """Sub-policy that navigates to the nearest high-value target"""

    def __init__(self):
        super().__init__("go_to_nearest_high_val_target")


    def act(self, observation):
        # TODO: Action = discrete direction toward nearest high val target
        action =
        return action

    def get_action_space(self) -> gym.Space:
        return gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def get_observation_space(self) -> gym.Space:
        # Agent pos (2) + nearest high-val target pos (2) + distance (1) + angle (1)
        return gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

    def get_observation(self, base_obs: np.ndarray, env_state: Dict[str, Any]) -> np.ndarray:
        agent_pos = env_state['agent_position']
        high_val_targets = env_state['high_value_targets']
        map_half_size = env_state['map_half_size']

        # Normalize agent position
        agent_pos_norm = agent_pos / map_half_size

        if len(high_val_targets) > 0:
            # Find nearest high-value target
            distances = np.linalg.norm(high_val_targets[:, 3:5] - agent_pos, axis=1)
            nearest_idx = np.argmin(distances)
            nearest_target = high_val_targets[nearest_idx, 3:5]

            # Normalize target position
            target_pos_norm = nearest_target / map_half_size

            # Calculate normalized distance and angle
            vector_to_target = nearest_target - agent_pos
            distance_norm = np.linalg.norm(vector_to_target) / (map_half_size * 2)
            angle = np.arctan2(vector_to_target[1], vector_to_target[0]) / np.pi

            obs = np.array([
                agent_pos_norm[0], agent_pos_norm[1],
                target_pos_norm[0], target_pos_norm[1],
                distance_norm, angle
            ], dtype=np.float32)
        else:
            # No high-value targets, return zeros for target info
            obs = np.array([
                agent_pos_norm[0], agent_pos_norm[1],
                0.0, 0.0, 0.0, 0.0
            ], dtype=np.float32)

        return obs

    def get_reward(self, base_reward: float, env_state: Dict[str, Any],
                   info: Dict[str, Any], progress: Dict[str, Any]) -> float:
        # Reward for getting closer to high-value targets
        high_val_targets = env_state['high_value_targets']
        if len(high_val_targets) == 0:
            return base_reward

        agent_pos = env_state['agent_position']
        distances = np.linalg.norm(high_val_targets[:, 3:5] - agent_pos, axis=1)
        min_distance = np.min(distances)

        # Normalize distance reward
        map_half_size = env_state['map_half_size']
        distance_reward = -min_distance / (map_half_size * 2)

        # Bonus for identifying high-value targets
        high_val_bonus = info.get('reward_components', {}).get('high val target id', 0) * 2.0

        return base_reward + 0.1 * distance_reward + high_val_bonus

    def is_terminated(self, env_state: Dict[str, Any]) -> bool:
        # Terminate if no more high-value targets
        return len(env_state['high_value_targets']) == 0


class EvadeDetection(SubPolicy):
    """Sub-policy that avoids threats and minimizes detection risk"""

    def __init__(self):
        super().__init__("evade_detection")

    def get_action_space(self) -> gym.Space:
        return gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def get_observation_space(self) -> gym.Space:
        # Agent pos (2) + threat pos (2) + threat distance (1) + detection risk (1) + escape vector (2)
        return gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)

    def get_observation(self, base_obs: np.ndarray, env_state: Dict[str, Any]) -> np.ndarray:
        agent_pos = env_state['agent_position']
        threat_pos = env_state['threat_position']
        map_half_size = env_state['map_half_size']

        # Normalize positions
        agent_pos_norm = agent_pos / map_half_size
        threat_pos_norm = threat_pos / map_half_size

        # Calculate threat distance and detection risk
        threat_distance = np.linalg.norm(threat_pos - agent_pos)
        threat_radius = env_state.get('threat_radius', 50.0)
        detection_risk = max(0, 1 - threat_distance / threat_radius)

        # Calculate escape vector (away from threat)
        if threat_distance > 0:
            escape_vector = (agent_pos - threat_pos) / threat_distance
        else:
            escape_vector = np.array([1.0, 0.0])  # Default escape direction

        obs = np.array([
            agent_pos_norm[0], agent_pos_norm[1],
            threat_pos_norm[0], threat_pos_norm[1],
            threat_distance / (map_half_size * 2),
            detection_risk,
            escape_vector[0], escape_vector[1]
        ], dtype=np.float32)

        return obs

    def get_reward(self, base_reward: float, env_state: Dict[str, Any],
                   info: Dict[str, Any], progress: Dict[str, Any]) -> float:
        # Strong penalty for detections
        detection_penalty = -20.0 * progress.get('new_detections', 0)

        # Reward for maintaining safe distance from threats
        threat_distance = np.linalg.norm(env_state['threat_position'] - env_state['agent_position'])
        threat_radius = env_state.get('threat_radius', 50.0)
        safety_bonus = min(1.0, threat_distance / (threat_radius * 1.5)) * 0.1

        return base_reward + detection_penalty + safety_bonus

    def should_interrupt(self, env_state: Dict[str, Any]) -> bool:
        # Interrupt if detection risk is high
        return env_state['detection_risk'] > 0.7


class LocalSearch(SubPolicy):
    # TODO replace with trained
    """Sub-policy that searches locally for unknown targets"""

    def __init__(self, model):
        super().__init__("local_search")
        self.search_radius = 300.0  # Search within this radius
        self.model = model

    def act(self, observation):
        action = self.model.predict(observation)
        return action

    def get_action_space(self) -> gym.Space:
        return gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def get_observation_space(self) -> gym.Space:
        # Agent pos (2) + local unknown targets (up to 5 targets * 2 coords = 10) + search progress (1)
        return gym.spaces.Box(low=-1, high=1, shape=(13,), dtype=np.float32)

    def get_observation(self, base_obs: np.ndarray, env_state: Dict[str, Any]) -> np.ndarray:
        # TODO replace with env.get_observation_nearest_n

        agent_pos = env_state['agent_position']
        unknown_targets = env_state['unknown_targets']
        map_half_size = env_state['map_half_size']

        # Normalize agent position
        agent_pos_norm = agent_pos / map_half_size

        # Find local unknown targets within search radius
        if len(unknown_targets) > 0:
            distances = np.linalg.norm(unknown_targets[:, 3:5] - agent_pos, axis=1)
            local_mask = distances <= self.search_radius
            local_targets = unknown_targets[local_mask]
        else:
            local_targets = np.array([]).reshape(0, 5)

        # Include up to 5 closest local targets
        target_obs = np.zeros(10)  # 5 targets * 2 coordinates
        if len(local_targets) > 0:
            # Sort by distance and take closest 5
            distances = np.linalg.norm(local_targets[:, 3:5] - agent_pos, axis=1)
            sorted_indices = np.argsort(distances)[:5]

            for i, idx in enumerate(sorted_indices):
                target_pos_norm = local_targets[idx, 3:5] / map_half_size
                target_obs[i * 2:(i + 1) * 2] = target_pos_norm

        # Search progress (fraction of total targets identified)
        search_progress = env_state['targets_identified'] / max(1, env_state['total_targets'])

        obs = np.concatenate([
            agent_pos_norm,
            target_obs,
            [search_progress]
        ]).astype(np.float32)

        return obs

    def get_reward(self, base_reward: float, env_state: Dict[str, Any],
                   info: Dict[str, Any], progress: Dict[str, Any]) -> float:
        # TODO rewrite

        # Reward for identifying any targets
        identification_bonus = info.get('reward_components', {}).get('regular val target id', 0) * 1.5

        # Small reward for staying in areas with unknown targets
        agent_pos = env_state['agent_position']
        unknown_targets = env_state['unknown_targets']

        if len(unknown_targets) > 0:
            distances = np.linalg.norm(unknown_targets[:, 3:5] - agent_pos, axis=1)
            local_targets = np.sum(distances <= self.search_radius)
            exploration_bonus = min(0.05, local_targets * 0.01)
        else:
            exploration_bonus = 0

        return base_reward + identification_bonus + exploration_bonus


class ChangeRegions(SubPolicy):
    """Sub-policy that moves to a specific region of the map"""

    def __init__(self, model):
        super().__init__(f"change__region")
        self.model = model

    def act(self, observation):
        # TODO figure out how this should be handled. I don't want agent
        """
        1. Select the region of the map (9-tile grid) to fly to
        2. Fly to the edge of the selected region
        """
        if not hasattr(self, 'new_region'):
            new_region = self.model.predict(observation)

        action = # TODO Discrete action to fly to center of new_region
        return action

    def get_action_space(self) -> gym.Space:
        return gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def get_observation_space(self) -> gym.Space:
        # Agent pos (2) + target region center (2) + distance to target (1) + current region (1)
        return gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

    def get_observation(self, base_obs: np.ndarray, env_state: Dict[str, Any]) -> np.ndarray:
        # TODO:
        #  Densities of all regions
        #  Distances to each region
        #  Location of human (in relation to agent and all regions)

        return obs

    def get_reward(self):
        # TODO
        return reward

    def is_terminated(self, env_state: Dict[str, Any]) -> bool:
        # Terminate when arrived at target region
        agent_pos = env_state['agent_position']
        map_half_size = env_state['map_half_size']
        region_center = self._get_region_center(self.target_region, map_half_size)
        distance_to_region = np.linalg.norm(region_center - agent_pos)
        return distance_to_region <= self.arrival_threshold

    def _get_region_center(self, region_id: int, map_half_size: float) -> np.ndarray:
        """Get the center coordinates of a region (0=NW, 1=NE, 2=SW, 3=SE)"""
        quarter_size = map_half_size * 0.5
        centers = {
            0: np.array([-quarter_size, quarter_size]),  # NW
            1: np.array([quarter_size, quarter_size]),  # NE
            2: np.array([-quarter_size, -quarter_size]),  # SW
            3: np.array([quarter_size, -quarter_size])  # SE
        }
        return centers.get(region_id, np.array([0.0, 0.0]))


class HierarchicalMAISREnv(gym.Env):
    """
    Hierarchical wrapper for MAISR environment that implements two-level RL:
    - High-level policy selects sub-policies at 2 Hz
    - Sub-policies execute until changed by high-level policy
    """

    def __init__(self, base_env_config: Dict[str, Any], hierarchy_config: Dict[str, Any] = None):
        super().__init__()

        # Initialize base environment
        self.base_env = MAISREnvVec(base_env_config, render_mode='headless')
        self.base_env_config = base_env_config

        # Hierarchy configuration
        if hierarchy_config is None:
            hierarchy_config = {}

        self.high_level_freq = hierarchy_config.get('high_level_freq', 2)  # 2 Hz
        self.low_level_freq = hierarchy_config.get('low_level_freq', 60)  # Base env frequency
        self.steps_per_high_action = self.low_level_freq // self.high_level_freq

        # Current state
        self.current_option = 0
        self.option_remaining_steps = 0
        self.high_level_step_count = 0
        self.total_steps = 0

        # Initialize sub-policies
        self.sub_policies = self._initialize_sub_policies()
        self.sub_policy_models = {}  # Will store trained models for each sub-policy

        # Define spaces
        self.action_space = gym.spaces.Discrete(len(self.sub_policies))
        self.observation_space = self._get_high_level_obs_space()

        # Tracking
        self.episode_rewards = {'high_level': 0, 'sub_policies': {name: 0 for name in self.sub_policies.keys()}}
        self.option_history = []

    def _initialize_sub_policies(self) -> Dict[str, SubPolicy]:
        """Initialize all available sub-policies"""
        sub_policies = {
            'go_to_high_val': GoToNearestHighValueTarget(),
            'evade_detection': EvadeDetection(),
            'local_search': LocalSearch(),
            'change_to_nw': ChangeRegions(0),
            'change_to_ne': ChangeRegions(1),
            'change_to_sw': ChangeRegions(2),
            'change_to_se': ChangeRegions(3)
        }
        return sub_policies

    def _get_high_level_obs_space(self) -> gym.Space:
        """Define observation space for high-level policy"""
        # High-level state includes: agent pos (2), targets info (4), threat info (3), time (1), region (1)
        return gym.spaces.Box(low=-1, high=1, shape=(11,), dtype=np.float32)

    def _get_high_level_observation(self) -> np.ndarray:
        """Get observation for high-level policy"""
        env_state = self.base_env.get_high_level_state()
        map_half_size = self.base_env.config["gameboard_size"] / 2

        # Agent position (normalized)
        agent_pos_norm = env_state['agent_position'] / map_half_size

        # Target information
        unknown_targets = env_state['unknown_targets']
        high_val_targets = env_state['high_value_targets']

        num_unknown = len(unknown_targets)
        num_high_val = len(high_val_targets)
        targets_identified_ratio = env_state['targets_identified'] / max(1, env_state['total_targets'])

        # Find nearest unknown target distance
        if num_unknown > 0:
            distances = np.linalg.norm(unknown_targets[:, 3:5] - env_state['agent_position'], axis=1)
            nearest_unknown_dist = np.min(distances) / (map_half_size * 2)
        else:
            nearest_unknown_dist = 1.0

        # Threat information
        threat_distance = np.linalg.norm(env_state['threat_position'] - env_state['agent_position'])
        threat_distance_norm = threat_distance / (map_half_size * 2)
        detection_risk = env_state['detection_risk']

        # Time and region
        time_remaining_norm = env_state['time_remaining'] / self.base_env.max_steps
        current_region_norm = env_state['current_region'] / 3.0

        obs = np.array([
            agent_pos_norm[0], agent_pos_norm[1],  # Agent position
            num_unknown / 30.0, num_high_val / 30.0,  # Target counts (normalized)
            targets_identified_ratio, nearest_unknown_dist,  # Target progress
            threat_distance_norm, detection_risk,  # Threat info
            time_remaining_norm,  # Time
            current_region_norm,  # Region
            self.current_option / (len(self.sub_policies) - 1)  # Current option (normalized)
        ], dtype=np.float32)

        return obs

    def reset(self, **kwargs):
        """Reset the hierarchical environment"""
        base_obs, info = self.base_env.reset(**kwargs)

        # Reset hierarchical state
        self.current_option = 0
        self.option_remaining_steps = self.steps_per_high_action
        self.high_level_step_count = 0
        self.total_steps = 0

        # Reset tracking
        self.episode_rewards = {'high_level': 0, 'sub_policies': {name: 0 for name in self.sub_policies.keys()}}
        self.option_history = []

        # Get initial high-level observation
        high_level_obs = self._get_high_level_observation()

        info.update({
            'current_option': self.current_option,
            'option_remaining_steps': self.option_remaining_steps,
            'sub_policy_name': list(self.sub_policies.keys())[self.current_option]
        })

        return high_level_obs, info

    def step(self, high_level_action: Optional[int] = None):
        """
        Execute one step in the hierarchical environment

        Args:
            high_level_action: Optional high-level action to change sub-policy
        """
        # High-level decision making
        option_changed = False
        if high_level_action is not None and high_level_action != self.current_option:
            self.current_option = high_level_action
            self.option_remaining_steps = self.steps_per_high_action
            self.high_level_step_count += 1
            option_changed = True
        elif self.option_remaining_steps <= 0:
            # Continue with current option
            self.option_remaining_steps = self.steps_per_high_action
            self.high_level_step_count += 1

        # Track option usage
        self.option_history.append(self.current_option)

        # Get current sub-policy
        sub_policy_name = list(self.sub_policies.keys())[self.current_option]
        sub_policy = self.sub_policies[sub_policy_name]

        # Get sub-policy specific observation
        base_obs = self.base_env.get_observation()
        env_state = self.base_env.get_high_level_state()
        sub_obs = sub_policy.get_observation(base_obs, env_state)

        # Get action from sub-policy
        sub_policy_model = self.sub_policy_models.get(sub_policy_name, None)
        sub_action = sub_policy.get_action(sub_obs, sub_policy_model)

        # Execute action in base environment
        base_obs, base_reward, terminated, truncated, info = self.base_env.step(sub_action)

        # Get updated environment state for reward calculation
        env_state_post = self.base_env.get_high_level_state()

        # Calculate progress info for sub-policy rewards
        progress = {
            'new_detections': info.get('detections', 0) - env_state.get('detections', 0),
            'new_identifications': len(info.get('new_identifications', []))
        }

        # Calculate sub-policy specific reward
        sub_reward = sub_policy.get_reward(base_reward, env_state_post, info, progress)

        # Calculate high-level reward (based on overall progress)
        high_level_reward = self._calculate_high_level_reward(info, env_state_post, option_changed)

        # Update counters
        self.option_remaining_steps -= 1
        self.total_steps += 1

        # Check if sub-policy wants to terminate early
        if sub_policy.is_terminated(env_state_post):
            self.option_remaining_steps = 0

        # Check for interruptions from other sub-policies
        for other_policy in self.sub_policies.values():
            if other_policy != sub_policy and other_policy.should_interrupt(env_state_post):
                self.option_remaining_steps = 0
                break

        # Update reward tracking
        self.episode_rewards['high_level'] += high_level_reward
        self.episode_rewards['sub_policies'][sub_policy_name] += sub_reward

        # Get high-level observation
        high_level_obs = self._get_high_level_observation()

        # Enhanced info
        info.update({
            'current_option': self.current_option,
            'sub_policy_name': sub_policy_name,
            'option_remaining_steps': self.option_remaining_steps,
            'high_level_reward': high_level_reward,
            'sub_policy_reward': sub_reward,
            'base_reward': base_reward,
            'option_changed': option_changed,
            'high_level_step': self.high_level_step_count,
            'total_steps': self.total_steps,
            'episode_rewards': self.episode_rewards.copy(),
            'sub_policy_obs': sub_obs,
            'sub_action': sub_action
        })

        return high_level_obs, high_level_reward, terminated, truncated, info

    def _calculate_high_level_reward(self, info: Dict[str, Any], env_state: Dict[str, Any],
                                     option_changed: bool) -> float:
        """Calculate reward for high-level policy"""
        # Base reward from environment
        base_reward = info.get('episode', {}).get('r', 0) - self.episode_rewards['high_level']

        # Efficiency bonus (completing objectives quickly)
        if env_state['targets_identified'] == env_state['total_targets']:
            efficiency_bonus = 2.0
        else:
            efficiency_bonus = 0.0

        # Small penalty for frequent option changes (encourage stability)
        option_change_penalty = -0.1 if option_changed else 0.0

        # Progress reward (identifying targets)
        identification_bonus = len(info.get('new_identifications', [])) * 0.5

        return base_reward + efficiency_bonus + option_change_penalty + identification_bonus

    def load_sub_policy_model(self, sub_policy_name: str, model):
        """Load a trained model for a specific sub-policy"""
        if sub_policy_name in self.sub_policies:
            self.sub_policy_models[sub_policy_name] = model
            print(f"Loaded model for sub-policy: {sub_policy_name}")
        else:
            raise ValueError(f"Unknown sub-policy: {sub_policy_name}")

    def get_sub_policy_env(self, sub_policy_name: str):
        """Get an environment wrapper for training a specific sub-policy"""
        if sub_policy_name not in self.sub_policies:
            raise ValueError(f"Unknown sub-policy: {sub_policy_name}")

        return SubPolicyEnvWrapper(self.base_env, self.sub_policies[sub_policy_name])

    def render(self, mode='human'):
        """Render the environment"""
        return self.base_env.render()

    def close(self):
        """Close the environment"""
        self.base_env.close()


class SubPolicyEnvWrapper(gym.Env):
    """Wrapper for training individual sub-policies"""

    def __init__(self, base_env: MAISREnvVec, sub_policy: SubPolicy):
        super().__init__()
        self.base_env = base_env
        self.sub_policy = sub_policy

        # Use sub-policy's action and observation spaces
        self.action_space = sub_policy.get_action_space()
        self.observation_space = sub_policy.get_observation_space()

    def reset(self, **kwargs):
        base_obs, info = self.base_env.reset(**kwargs)
        env_state = self.base_env.get_high_level_state()
        sub_obs = self.sub_policy.get_observation(base_obs, env_state)
        return sub_obs, info

    def step(self, action):
        # Execute action in base environment
        base_obs, base_reward, terminated, truncated, info = self.base_env.step(action)

        # Get sub-policy specific observation and reward
        env_state = self.base_env.get_high_level_state()
        sub_obs = self.sub_policy.get_observation(base_obs, env_state)

        progress = {
            'new_detections': 0,  # Would need to track this properly
            'new_identifications': len(info.get('new_identifications', []))
        }

        sub_reward = self.sub_policy.get_reward(base_reward, env_state, info, progress)

        # Check if sub-policy is terminated
        if self.sub_policy.is_terminated(env_state):
            terminated = True

        return sub_obs, sub_reward, terminated, truncated, info

    def render(self, mode='human'):
        return self.base_env.render()

    def close(self):
        pass  # Don't close the base env as it might be shared


# Add missing methods to MAISREnvVec class
def add_hierarchical_methods_to_base_env():
    """
    Add these methods to the MAISREnvVec class in env_combined.py
    These methods provide the hierarchical environment with necessary state information
    """

    def get_high_level_state(self):
        """Return state information for high-level policy decision making"""
        map_half_size = self.config["gameboard_size"] / 2

        # Get unknown targets (info_level < 1.0)
        unknown_mask = self.targets[:, 2] < 1.0
        unknown_targets = self.targets[unknown_mask] if np.any(unknown_mask) else np.array([]).reshape(0, 5)

        # Get high-value targets
        high_val_mask = self.targets[:, 1] == 1.0
        high_value_targets = self.targets[high_val_mask] if np.any(high_val_mask) else np.array([]).reshape(0, 5)

        return {
            'agent_position': np.array([self.agents[self.aircraft_ids[0]].x, self.agents[self.aircraft_ids[0]].y]),
            'unknown_targets': unknown_targets,
            'high_value_targets': high_value_targets,
            'threat_position': self.threat,
            'threat_radius': self.config.get('threat_radius', 50.0),
            'targets_identified': self.targets_identified,
            'total_targets': self.num_targets,
            'detection_risk': self._calculate_detection_risk(),
            'current_region': self._get_current_region(),
            'time_remaining': self.max_steps - self.step_count_inner,
            'map_half_size': map_half_size,
            'detections': self.detections
        }

    def _calculate_detection_risk(self):
        """Calculate current detection risk based on proximity to threats"""
        agent_pos = np.array([self.agents[self.aircraft_ids[0]].x, self.agents[self.aircraft_ids[0]].y])
        threat_pos = np.array([self.threat[0], self.threat[1]])
        distance = np.linalg.norm(agent_pos - threat_pos)
        threat_radius = self.config.get('threat_radius', 50.0)
        return max(0, 1 - distance / (threat_radius * 1.5))

    def _get_current_region(self):
        """Determine which quadrant/region the agent is currently in"""
        agent_x, agent_y = self.agents[self.aircraft_ids[0]].x, self.agents[self.aircraft_ids[0]].y
        # Return region ID (0=NW, 1=NE, 2=SW, 3=SE)
        return int(agent_x >= 0) + 2 * int(agent_y < 0)


# Example usage and training integration
class HierarchicalTrainingManager:
    """
    Manager class for training hierarchical policies
    Handles both high-level and sub-policy training
    """

    def __init__(self, base_env_config: Dict[str, Any], hierarchy_config: Dict[str, Any] = None):
        self.base_env_config = base_env_config
        self.hierarchy_config = hierarchy_config or {}
        self.hier_env = HierarchicalMAISREnv(base_env_config, hierarchy_config)

    def train_sub_policies(self, sub_policy_configs: Dict[str, Dict[str, Any]]):
        """
        Train individual sub-policies

        Args:
            sub_policy_configs: Dict mapping sub-policy names to their training configs
        """
        from stable_baselines3 import PPO

        trained_models = {}

        for sub_policy_name in self.hier_env.sub_policies.keys():
            if sub_policy_name not in sub_policy_configs:
                print(f"Skipping training for {sub_policy_name} (no config provided)")
                continue

            print(f"\nTraining sub-policy: {sub_policy_name}")

            # Get sub-policy environment
            sub_env = self.hier_env.get_sub_policy_env(sub_policy_name)

            # Get training config for this sub-policy
            config = sub_policy_configs[sub_policy_name]

            # Create and train model
            model = PPO(
                "MlpPolicy",
                sub_env,
                learning_rate=config.get('learning_rate', 3e-4),
                n_steps=config.get('n_steps', 2048),
                batch_size=config.get('batch_size', 64),
                gamma=config.get('gamma', 0.99),
                verbose=1
            )

            model.learn(
                total_timesteps=config.get('total_timesteps', 100000),
                progress_bar=True
            )

            # Save and load model into hierarchical environment
            model_path = f"./trained_models/sub_policy_{sub_policy_name}"
            model.save(model_path)
            self.hier_env.load_sub_policy_model(sub_policy_name, model)
            trained_models[sub_policy_name] = model

            print(f"Completed training for {sub_policy_name}")

        return trained_models

    def train_high_level_policy(self, high_level_config: Dict[str, Any]):
        """
        Train the high-level policy that selects sub-policies

        Args:
            high_level_config: Training configuration for high-level policy
        """
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        print("\nTraining high-level policy...")

        # Wrap in vectorized environment
        env = DummyVecEnv([lambda: self.hier_env])

        # Create high-level model
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=high_level_config.get('learning_rate', 3e-4),
            n_steps=high_level_config.get('n_steps', 2048),
            batch_size=high_level_config.get('batch_size', 64),
            gamma=high_level_config.get('gamma', 0.99),
            verbose=1
        )

        # Train model
        model.learn(
            total_timesteps=high_level_config.get('total_timesteps', 500000),
            progress_bar=True
        )

        # Save model
        model_path = "./trained_models/high_level_policy"
        model.save(model_path)

        print("Completed high-level policy training")
        return model

    def evaluate_hierarchical_policy(self, high_level_model, n_episodes: int = 10):
        """
        Evaluate the complete hierarchical policy

        Args:
            high_level_model: Trained high-level policy
            n_episodes: Number of episodes to evaluate
        """
        episode_rewards = []
        episode_lengths = []
        target_ids_per_episode = []
        option_usage = {i: 0 for i in range(len(self.hier_env.sub_policies))}

        for episode in range(n_episodes):
            obs, _ = self.hier_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                # Get action from high-level policy
                action, _ = high_level_model.predict(obs, deterministic=True)

                # Execute action
                obs, reward, terminated, truncated, info = self.hier_env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

                # Track option usage
                option_usage[info['current_option']] += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            target_ids_per_episode.append(info.get('target_ids', 0))

            print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
                  f"Length={episode_length}, Targets={info.get('target_ids', 0)}")

        # Print evaluation summary
        print(f"\nEvaluation Summary ({n_episodes} episodes):")
        print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Average Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
        print(f"Average Targets ID'd: {np.mean(target_ids_per_episode):.2f} ± {np.std(target_ids_per_episode):.2f}")

        print("\nOption Usage:")
        sub_policy_names = list(self.hier_env.sub_policies.keys())
        for option_id, usage_count in option_usage.items():
            usage_pct = (usage_count / sum(option_usage.values())) * 100
            print(f"  {sub_policy_names[option_id]}: {usage_pct:.1f}% ({usage_count} steps)")

        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'target_ids_per_episode': target_ids_per_episode,
            'option_usage': option_usage
        }


def train_highlevel_policy():
    """At 2Hz, policy chooses a subpolicy option"""

def train_lowlevel_policy():
    """"""

def train_hierarchical_maisr():
    """
    Example script showing how to train the hierarchical MAISR system
    """

    # Load base environment config
    from utility.data_logging import load_env_config
    base_env_config = load_env_config('')

    # Hierarchy configuration
    hierarchy_config = {
        'high_level_freq': 2,  # 2 Hz for high-level decisions
        'low_level_freq': 60  # 60 Hz for low-level execution
    }

    # Create training manager
    trainer = HierarchicalTrainingManager(base_env_config, hierarchy_config)

    # Sub-policy training configurations
    sub_policy_configs = {
        'go_to_high_val': {
            'learning_rate': 3e-4,
            'total_timesteps': 200000,
            'n_steps': 2048,
            'batch_size': 64
        },
        'evade_detection': {
            'learning_rate': 5e-4,
            'total_timesteps': 150000,
            'n_steps': 1024,
            'batch_size': 32
        },
        'local_search': {
            'learning_rate': 3e-4,
            'total_timesteps': 300000,
            'n_steps': 2048,
            'batch_size': 64
        },
        'change_to_nw': {
            'learning_rate': 5e-4,
            'total_timesteps': 100000,
            'n_steps': 1024,
            'batch_size': 32
        },
        'change_to_ne': {
            'learning_rate': 5e-4,
            'total_timesteps': 100000,
            'n_steps': 1024,
            'batch_size': 32
        },
        'change_to_sw': {
            'learning_rate': 5e-4,
            'total_timesteps': 100000,
            'n_steps': 1024,
            'batch_size': 32
        },
        'change_to_se': {
            'learning_rate': 5e-4,
            'total_timesteps': 100000,
            'n_steps': 1024,
            'batch_size': 32
        }
    }

    # Train sub-policies
    print("Starting sub-policy training...")
    sub_policy_models = trainer.train_sub_policies(sub_policy_configs)

    # High-level policy training configuration
    high_level_config = {
        'learning_rate': 1e-4,  # Lower learning rate for stability
        'total_timesteps': 1000000,
        'n_steps': 4096,  # Longer episodes for strategic learning
        'batch_size': 128,
        'gamma': 0.995  # Higher discount factor for long-term planning
    }

    # Train high-level policy
    print("Starting high-level policy training...")
    high_level_model = trainer.train_high_level_policy(high_level_config)

    # Evaluate the complete system
    print("Evaluating hierarchical policy...")
    evaluation_results = trainer.evaluate_hierarchical_policy(high_level_model, n_episodes=20)

    print("Training complete!")
    return trainer, high_level_model, sub_policy_models, evaluation_results


if __name__ == "__main__":
    # Run the training
    trainer, high_level_model, sub_policy_models, results = train_hierarchical_maisr()