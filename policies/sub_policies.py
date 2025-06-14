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


class GoToNearestThreat(SubPolicy):
    """Sub-policy that navigates to the nearest high-value target"""

    def __init__(self, model=None):
        super().__init__("go_to_nearest_threat")

        self.model = model
        if model: print('[GoToNearestThreat]: Using provided model for inference')
        else: print('[GoToNearestThreat]: No model provided, using internal heuristic')

        # Internal state of heuristic
        self._current_target_id = None
        self._current_target_pos = None
        self._last_action = None
        self._action_repeat_count = 0
        self._max_repeat_count = 3  # Minimum steps to take in same direction
        self._target_switch_threshold = 20.0  # Distance threshold to consider switching targets

    def act(self, observation):
        # TODO: Action = discrete direction toward nearest high val target
        if self.target_region is None or self.steps_since_update >= self.update_rate:
            if self.model:
                self.target_region = self.model.predict(observation)
            else:
                self.target_region = self.heuristic(observation)

    def heuristic(self, observation):
        """
        Input: Observation vector of the dx, dy vector to the nearest two threats (4 elements total)
        Output:
        Pick the """

        obs = np.array(observation)

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
        ], dtype=float)

        # Extract nearest target vector (first two components)
        target_vector_x = obs[0]
        target_vector_y = obs[1]

        # Check if there's a valid target (non-zero vector)
        if target_vector_x == 0.0 and target_vector_y == 0.0:
            # No targets or at target location
            self.reset_heuristic_state()
            return 0

        # The observation already gives us the vector to the nearest target
        direction_to_target = np.array([target_vector_x, target_vector_y])

        # Normalize direction vectors
        direction_norms = np.linalg.norm(directions, axis=1)
        normalized_directions = directions / direction_norms[:, np.newaxis]

        # Normalize target direction
        target_norm = np.linalg.norm(direction_to_target)
        if target_norm > 0:
            direction_to_target_norm = direction_to_target / target_norm
        else:
            return self._last_action if self._last_action is not None else 0

        # Calculate dot products
        dot_products = np.dot(normalized_directions, direction_to_target_norm)

        # Find best action
        best_action = np.argmax(dot_products)

        # Anti-oscillation: if we just took an action, continue for minimum steps
        if (self._last_action is not None and
                self._action_repeat_count < self._max_repeat_count and
                self._last_action != best_action):

            # Check if last action is still reasonable (dot product > 0.5)
            last_dot_product = dot_products[self._last_action]
            if last_dot_product > 0.5:  # Still pointing roughly toward target
                best_action = self._last_action
                self._action_repeat_count += 1
            else:
                _action_repeat_count = 0  # Reset if direction is too far off
        else:
            _action_repeat_count = 0

        # Additional anti-oscillation: prevent direct opposite actions
        if (self._last_action is not None and abs(self._last_action - best_action) == 4):  # Opposite directions
            # Choose a compromise direction
            adjacent_actions = [(self._last_action + 1) % 8, (self._last_action - 1) % 8]
            adjacent_dots = [dot_products[a] for a in adjacent_actions]
            best_adjacent_idx = np.argmax(adjacent_dots)
            best_action = adjacent_actions[best_adjacent_idx]

        _last_action = best_action
        return np.int32(best_action)

    def reset_heuristic_state(self):
        """Reset the global state for the heuristic policy."""
        #global _current_target_id, _current_target_pos, _last_action, _action_repeat_count
        self._current_target_id = None
        self._current_target_pos = None
        self._last_action = None
        self._action_repeat_count = 0

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

    def __init__(self, model=None):
        super().__init__("local_search")
        self.search_radius = 300.0  # Search within this radius
        self.model = model
        if model:
            print('Using provided model for inference')
        else:
            print('No model provided, using internal heuristic')

        self._current_target_id = None
        self._current_target_pos = None
        self._last_action = None
        self._action_repeat_count = 0
        self._max_repeat_count = 3  # Minimum steps to take in same direction
        self._target_switch_threshold = 20.0  # Distance threshold to consider switching targets

    def act(self, observation):
        if self.model:
            action = self.model.predict(observation)
        else:
            action = self.heuristic(observation)
        return action

    def destroy(self):
        # TODO
        pass

    def heuristic(self, observation):
        """Simple heuristic to fly to nearest unknown target. Can be used if RL model is not provided"""

        obs = np.array(observation)

        # Direction mapping
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
        ], dtype=float)

        # Extract nearest target vector (first two components)
        target_vector_x = obs[0]
        target_vector_y = obs[1]

        # Check if there's a valid target (non-zero vector)
        if target_vector_x == 0.0 and target_vector_y == 0.0:
            # No targets or at target location
            self.reset_heuristic_state()
            return 0

        # The observation already gives us the vector to the nearest target
        direction_to_target = np.array([target_vector_x, target_vector_y])

        # Normalize direction vectors
        direction_norms = np.linalg.norm(directions, axis=1)
        normalized_directions = directions / direction_norms[:, np.newaxis]

        # Normalize target direction
        target_norm = np.linalg.norm(direction_to_target)
        if target_norm > 0:
            direction_to_target_norm = direction_to_target / target_norm
        else:
            return self._last_action if self._last_action is not None else 0

        # Calculate dot products
        dot_products = np.dot(normalized_directions, direction_to_target_norm)

        # Find best action
        best_action = np.argmax(dot_products)

        # Anti-oscillation: if we just took an action, continue for minimum steps
        if (self._last_action is not None and
                self._action_repeat_count < self._max_repeat_count and
                self._last_action != best_action):

            # Check if last action is still reasonable (dot product > 0.5)
            last_dot_product = dot_products[self._last_action]
            if last_dot_product > 0.5:  # Still pointing roughly toward target
                best_action = self._last_action
                self._action_repeat_count += 1
            else:
                _action_repeat_count = 0  # Reset if direction is too far off
        else:
            _action_repeat_count = 0

        # Additional anti-oscillation: prevent direct opposite actions
        if (self._last_action is not None and abs(self._last_action - best_action) == 4):  # Opposite directions
            # Choose a compromise direction
            adjacent_actions = [(self._last_action + 1) % 8, (self._last_action - 1) % 8]
            adjacent_dots = [dot_products[a] for a in adjacent_actions]
            best_adjacent_idx = np.argmax(adjacent_dots)
            best_action = adjacent_actions[best_adjacent_idx]

        _last_action = best_action
        return np.int32(best_action)

    def reset_heuristic_state(self):
        """Reset the global state for the heuristic policy."""
        #global _current_target_id, _current_target_pos, _last_action, _action_repeat_count
        self._current_target_id = None
        self._current_target_pos = None
        self._last_action = None
        self._action_repeat_count = 0


class ChangeRegions(SubPolicy):
    """Sub-policy that moves to a specific region of the map"""

    def __init__(self, model=None):
        super().__init__(f"change_region")
        self.model = model

        self.update_rate = 10 # Recalculate every 10 steps to reduce computation cost
        self.steps_since_update = 0

        self.target_region = None
        self.arrival_threshold = 0.05

    def act(self, observation):
        # TODO need to add action masking
        """
        1. Select the region of the map (4-tile grid) to fly to
        2. Fly to the edge of the selected region
        """
        if self.target_region is None or self.steps_since_update >= self.update_rate:
            if self.model:
                self.target_region = self.model.predict(observation)
            else:
                self.target_region = self.heuristic(observation)

        # Set waypoint directly to center of new region
        action = self._get_region_center(self.target_region)

        return action

    def heuristic(self, observation):
        """Simple heuristic to choose a region. Can be used if model is not provided"""

        obs = np.array(observation)

        # Each region has 3 values: [target_ratio, agent_distance, teammate_distance]
        regions_info = []

        for region_id in range(4):  # 4 regions: NW, NE, SW, SE
            base_idx = region_id * 3
            target_ratio = obs[base_idx]  # Ratio of unknown targets in this region
            agent_distance = obs[base_idx + 1]  # Agent distance to region center
            teammate_distance = obs[base_idx + 2]  # Teammate distance to region center

            regions_info.append({
                'region_id': region_id,
                'target_ratio': target_ratio,
                'agent_distance': agent_distance,
                'teammate_distance': teammate_distance
            })

        # Determine which region teammate is likely in
        # Teammate is probably in the region they're closest to
        teammate_distances = [info['teammate_distance'] for info in regions_info]
        teammate_region = np.argmin(teammate_distances)

        # Sort regions by target density (highest ratio first)
        regions_info.sort(key=lambda x: x['target_ratio'], reverse=True)

        # Choose the highest density region that doesn't have teammate
        target_region = None
        for region_info in regions_info:
            region_id = region_info['region_id']

            # Skip if teammate is in this region (with small tolerance for distance comparison)
            if region_id == teammate_region and region_info[
                'teammate_distance'] < 0.3:  # 0.3 is normalized distance threshold
                continue

            target_region = region_id
            break

        # Fallback: if all regions have teammate or no targets anywhere, choose the region with highest density
        if target_region is None:
            target_region = regions_info[0]['region_id']

        return target_region


    def is_terminated(self, env_state: Dict[str, Any]) -> bool:
        """Terminate when arrived at target region"""
        # TODO check this
        agent_pos = env_state['agent_position']
        region_center = self._get_region_center(self.target_region)
        distance_to_region = np.linalg.norm(region_center - agent_pos)
        return distance_to_region <= self.arrival_threshold

    def _get_region_center(self, region_id: int) -> np.ndarray:
        """Get the center coordinates of a region (0=NW, 1=NE, 2=SW, 3=SE)"""
        centers = {
            0: np.array([-0.25, 0.25]),  # NW
            1: np.array([0.25, 0.25]),  # NE
            2: np.array([-0.25, -0.25]),  # SW
            3: np.array([0.25, -0.25])  # SE
        }
        return centers.get(region_id, np.array([0.0, 0.0]))