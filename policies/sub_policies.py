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
from env_multi_new import MAISREnvVec


class SubPolicy(ABC):
    """Abstract base class for all sub-policies"""
    def __init__(self, name: str):
        self.name = name
        self._action_space = None
        self._observation_space = None

    def act(self, observation):
        pass

    # @abstractmethod
    # def get_action_space(self) -> gym.Space:
    #     """Return the action space for this sub-policy"""
    #     pass
    #
    # @abstractmethod
    # def get_observation_space(self) -> gym.Space:
    #     """Return the observation space for this sub-policy"""
    #     pass

    # @abstractmethod
    # def get_observation(self, base_obs: np.ndarray, env_state: Dict[str, Any]) -> np.ndarray:
    #     """Transform base observation into sub-policy specific observation"""
    #     pass

    # @abstractmethod
    # def get_reward(self, base_reward: float, env_state: Dict[str, Any],
    #                info: Dict[str, Any], progress: Dict[str, Any]) -> float:
    #     """Calculate sub-policy specific reward"""
    #     pass
    #
    # def get_action(self, obs: np.ndarray, model=None) -> np.ndarray:
    #     """Get action from sub-policy (can be overridden for trained models)"""
    #     if model is not None:
    #         action, _ = model.predict(obs, deterministic=True)
    #         return action
    #     else:
    #         # Default random action
    #         return self.get_action_space().sample()
    #
    # def is_terminated(self, env_state: Dict[str, Any]) -> bool:
    #     """Check if sub-policy wants to terminate early"""
    #     return False
    #
    # def should_interrupt(self, env_state: Dict[str, Any]) -> bool:
    #     """Check if this sub-policy should interrupt current execution"""
    #     return False


class GoToNearestHighValueTarget(SubPolicy):
    """Sub-policy that navigates to the nearest high-value target"""

    def __init__(self, heuristic = None, model = None):
        super().__init__()
        self.model = model
        self.heuristic = heuristic
        if heuristic is None and model is None:
            raise ValueError('ERROR (GoToNearestHighValueTarget): Neither heuristic nor model provided')

        if heuristic is not None and model is not None:
            raise ValueError('ERROR (GoToNearestHighValueTarget): Both heuristic and model provided')


    def act(self, observation):
        # TODO: Action = discrete direction toward nearest high val target
        action = self.heuristic(observation)
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

    def __init__(self, heuristic=None, model=None):
        super().__init__("local_search")
        self.search_radius = 300.0  # Search within this radius
        self.model = model
        self.heuristic = heuristic
        if heuristic is None and model is None:
            raise ValueError('ERROR (LocalSearch): Neither heuristic nor model provided')

        if heuristic is not None and model is not None:
            raise ValueError('ERROR (LocalSearch): Both heuristic and model provided')

    def act(self, observation):
        if self.model:
            action = self.model.predict(observation)
        elif self.heuristic:
            action = self.heuristic(observation)

        return action

    def destroy(self):
        # TODO
        pass

    # def get_action_space(self) -> gym.Space:
    #     return gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    #
    # def get_observation_space(self) -> gym.Space:
    #     # Agent pos (2) + local unknown targets (up to 5 targets * 2 coords = 10) + search progress (1)
    #     return gym.spaces.Box(low=-1, high=1, shape=(13,), dtype=np.float32)

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

    def __init__(self, model=None, heuristic=None):
        super().__init__(f"change_region")
        self.model = model
        self.heuristic = heuristic
        if heuristic is None and model is None:
            raise ValueError('ERROR (LocalSearch): Neither heuristic nor model provided')

        if heuristic is not None and model is not None:
            raise ValueError('ERROR (LocalSearch): Both heuristic and model provided')

    def act(self, observation):
        # TODO figure out how this should be handled. I don't want agent
        """
        1. Select the region of the map (9-tile grid) to fly to
        2. Fly to the edge of the selected region
        """
        if self.heuristic:
            new_region = self.heuristic(observation)
        elif self.model:
            new_region = self.model.predict(observation)

        # TODO set waypoint directly to center of new region

        action = new_region
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