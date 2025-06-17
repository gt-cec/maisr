import random
import numpy as np
from abc import ABC, abstractmethod
import pygame
from stable_baselines3 import PPO
#from policies.sub_policies import LocalSearch, ChangeRegions, GoToNearestThreat
import gymnasium as gym
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

    def is_terminated(self, observation):
        pass

class TeammatePolicy(ABC):
    """Abstract base class for teammate policies"""

    @abstractmethod
    def choose_subpolicy(self, observation):
        pass

    # @abstractmethod
    # def reset(self):
    #     pass
    #
    # @property
    # @abstractmethod
    # def name(self):
    #     pass

class GenericTeammatePolicy(TeammatePolicy):
    def __init__(self,
                 env,
                 local_search_policy: SubPolicy,
                 go_to_highvalue_policy: SubPolicy,
                 change_region_subpolicy: SubPolicy,
                 mode_selector: str, # RL, heuristic, human, or none
                 use_collision_avoidance: bool, # If true, approaching threat while in local search will trigger collision avoidance
                 ):

        self.env = env

        if mode_selector == "RLSelector":
            # TODO load RL policy into self.mode_selector
            pass
        self.mode_selector = mode_selector
        self.use_collision_avoidance = use_collision_avoidance

        self.local_search_policy = local_search_policy
        self.go_to_highvalue_policy = go_to_highvalue_policy
        self.change_region_subpolicy = change_region_subpolicy
        self.key_to_action = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2}

    def choose_subpolicy(self, observation, current_subpolicy):
        if self.mode_selector == 'human':
            action = None
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key in self.key_to_action:
                        action = self.key_to_action[event.key]
                        print(f"[choose subpolicy] Chose subpolicy {action} via keypress")
                        return action
            if action is None:
                action = current_subpolicy
            return action

        elif self.mode_selector == 'RlSelector':
            pass

        elif self.mode_selector == 'HeuristicSelectorGreedySafe': # Always do local search unless we need to evade
            if self.near_a_threat(): # True if we are approaching threat and need to call Evade
                pass
            else:
                return 0 # Local search

        elif self.mode_selector == 'HeuristicSelectorGreedyRisky': # Always do local search
            return 0 # Local search

        elif self.mode_selector == 'HeuristicSelectorConservative': # Always go to nearest threat
            pass # Implement heuristic policy that chooses local search or get region but never nearest threat

        elif self.mode_selector == 'HeuristicSelectorAggressive': # Always go to nearest threat
            return 2

    def near_a_threat(self):
        """Return true if near threat and need to call evade"""
        pass

# class GenericTeammatePolicy(TeammatePolicy):
#     """Heuristic-based teammate policy"""
#
#     def __init__(self, strategy_type, config=None):
#         self.strategy_type = strategy_type
#
#         self.config = config or {}
#         self._name = f"Heuristic_{strategy_type}"
#         self.current_mode = 0  # 0: local_search, 1: change_region, 2: go_to_threat
#         self.steps_since_mode_change = 0
#         self.mode_duration = self.config.get('mode_duration', 50)
#
#     def choose_subpolicy(self, observation, env_state):
#         if self.strategy_type == "aggressive":
#             return self._aggressive_strategy(observation, env_state)
#         elif self.strategy_type == "conservative":
#             return self._conservative_strategy(observation, env_state)
#         elif self.strategy_type == "adaptive":
#             return self._adaptive_strategy(observation, env_state)
#         else:
#             return self._random_strategy(observation, env_state)
#
#     def _aggressive_strategy(self, observation, env_state):
#         """Always prioritize going to threats"""
#         return 2  # go_to_threat mode
#
#     def _conservative_strategy(self, observation, env_state):
#         """Prefer local search, avoid threats"""
#         threats_nearby = self._check_threats_nearby(env_state)
#         if threats_nearby:
#             return 1  # change_region to escape
#         return 0  # local_search
#
#     def _adaptive_strategy(self, observation, env_state):
#         """Switch modes based on game state"""
#         targets_remaining = env_state.get('targets_remaining', 0)
#         detection_risk = env_state.get('detection_risk', 0)
#
#         if detection_risk > 0.7:
#             return 1  # change_region
#         elif targets_remaining > 5:
#             return 0  # local_search
#         else:
#             return 2  # go_to_threat
#
#     def _random_strategy(self, observation, env_state):
#         """Random mode switching with some persistence"""
#         self.steps_since_mode_change += 1
#
#         if self.steps_since_mode_change >= self.mode_duration:
#             self.current_mode = random.randint(0, 2)
#             self.steps_since_mode_change = 0
#
#         return self.current_mode
#
#     def _check_threats_nearby(self, env_state):
#         """Check if threats are within danger zone"""
#         # Implement based on your environment's threat detection logic
#         return env_state.get('threat_distance', float('inf')) < 100
#
#     def reset(self):
#         self.current_mode = 0
#         self.steps_since_mode_change = 0
#
#     @property
#     def name(self):
#         return self._name


class TeammateManager:
    """Manages pool of teammate policies and selection"""

    def __init__(self):
        self.teammate_pool = []
        self.current_teammate = None
        self.episode_count = 0

    def add_rl_teammate(self, model_path, policy_name):
        """Add an RL-trained teammate to the pool"""
        teammate = RLTeammatePolicy(model_path, policy_name)
        self.teammate_pool.append(teammate)

    def add_heuristic_teammate(self, strategy_type, config=None):
        """Add a heuristic teammate to the pool"""
        teammate = HeuristicTeammatePolicy(strategy_type, config)
        self.teammate_pool.append(teammate)

    def select_random_teammate(self):
        """Randomly select a teammate from the pool"""
        if not self.teammate_pool:
            return None
        self.current_teammate = random.choice(self.teammate_pool)
        return self.current_teammate

    def select_teammate_by_curriculum(self):
        """Select teammate based on training curriculum"""
        if not self.teammate_pool:
            return None

        # Example curriculum: start with heuristic, gradually add RL teammates
        if self.episode_count < 1000:
            # Early training: only heuristic teammates
            heuristic_teammates = [t for t in self.teammate_pool if isinstance(t, HeuristicTeammatePolicy)]
            self.current_teammate = random.choice(heuristic_teammates) if heuristic_teammates else None
        else:
            # Later training: mix of heuristic and RL
            self.current_teammate = random.choice(self.teammate_pool)

        return self.current_teammate

    def reset_for_episode(self):
        """Reset teammate for new episode"""
        self.episode_count += 1
        if self.current_teammate:
            self.current_teammate.reset()


########################################################################################################################
##################################################### SUB POLICIES #####################################################
########################################################################################################################

class GoToNearestThreat(SubPolicy):
    """Sub-policy that navigates to the nearest high-value target"""

    def __init__(self, model_path=None):
        super().__init__("go_to_nearest_threat")

        if model_path is not None:
            self.model = PPO.load(model_path)
            print('[GoToNearestThreat]: Using provided model for inference')
        else:
            self.model = None
            print('[GoToNearestThreat]: No model provided, using internal heuristic')

        # Internal state of heuristic
        self._current_target_id = None
        self._current_target_pos = None
        self._last_action = None
        self._action_repeat_count = 0
        self._max_repeat_count = 3  # Minimum steps to take in same direction
        self._target_switch_threshold = 20.0  # Distance threshold to consider switching targets

        self.is_terminated = False

    def act(self, observation):
        if not self.has_unidentified_threats_remaining(observation):
            self.is_terminated = True
            return 0  # Default action when no unidentified threats remain

        if self.model:
            action = self.model.predict(observation)
        else:
            action = self.heuristic(observation)
        return action

    def heuristic(self, observation) -> np.int32:
        """
        Input: Observation vector with dx, dy, identified status for nearest two threats (6 elements total)
        Output: Direction to move toward nearest unidentified threat
        """

        # Check if any unidentified threats remain
        if not self.has_unidentified_threats_remaining(observation):
            self.reset_heuristic_state()
            self.is_terminated = True
            return np.int32(0)

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

        # Find the nearest unidentified threat
        target_vector = None

        # Check first threat (nearest by distance)
        threat1_identified = obs[2] if len(obs) > 2 else 1.0
        if threat1_identified < 0.5:  # Not identified
            target_vector_x = obs[0]
            target_vector_y = obs[1]
            target_vector = np.array([target_vector_x, target_vector_y])

        # Check second threat if first is identified
        elif len(obs) >= 6:
            threat2_identified = obs[5]
            if threat2_identified < 0.5:  # Not identified
                target_vector_x = obs[3]
                target_vector_y = obs[4]
                target_vector = np.array([target_vector_x, target_vector_y])

        # No unidentified threats found
        if target_vector is None or (target_vector[0] == 0.0 and target_vector[1] == 0.0):
            self.reset_heuristic_state()
            self.is_terminated = True
            return np.int32(0)

        # Normalize direction vectors
        direction_norms = np.linalg.norm(directions, axis=1)
        normalized_directions = directions / direction_norms[:, np.newaxis]

        # Normalize target direction
        target_norm = np.linalg.norm(target_vector)
        if target_norm > 0:
            direction_to_target_norm = target_vector / target_norm
        else:
            return self._last_action if self._last_action is not None else 0

        # Calculate dot products
        dot_products = np.dot(normalized_directions, direction_to_target_norm)

        # Find best action
        best_action = np.argmax(dot_products)

        # Anti-oscillation logic (same as before)
        if (self._last_action is not None and
                self._action_repeat_count < self._max_repeat_count and
                self._last_action != best_action):

            last_dot_product = dot_products[self._last_action]
            if last_dot_product > 0.5:
                best_action = self._last_action
                self._action_repeat_count += 1
            else:
                self._action_repeat_count = 0
        else:
            self._action_repeat_count = 0

        # Prevent direct opposite actions
        if (self._last_action is not None and abs(self._last_action - best_action) == 8):
            adjacent_actions = [(self._last_action + 1) % 16, (self._last_action - 1) % 16]
            adjacent_dots = [dot_products[a] for a in adjacent_actions]
            best_adjacent_idx = np.argmax(adjacent_dots)
            best_action = adjacent_actions[best_adjacent_idx]

        self._last_action = best_action
        print(f'Heuristic chose action {best_action} targeting unidentified threat')
        return np.int32(best_action)

    def has_unidentified_threats_remaining(self, observation) -> bool:
        """
        Check if there are any unidentified threats remaining to pursue
        Args:
            observation: The observation vector containing dx/dy/identified for threats
        Returns:
            bool: True if unidentified threats remain, False if all are identified
        """
        obs = np.array(observation)

        if len(obs) < 3:
            return False

        # Check first threat
        threat1_identified = obs[2] if len(obs) > 2 else 1.0
        threat1_exists = not (obs[0] == 0.0 and obs[1] == 0.0)

        if threat1_exists and threat1_identified < 0.5:
            return True

        # Check second threat if observation is long enough
        if len(obs) >= 6:
            threat2_identified = obs[5]
            threat2_exists = not (obs[3] == 0.0 and obs[4] == 0.0)

            if threat2_exists and threat2_identified < 0.5:
                return True

        return False

    def reset_heuristic_state(self):
        """Reset the global state for the heuristic policy."""
        #global _current_target_id, _current_target_pos, _last_action, _action_repeat_count
        self._current_target_id = None
        self._current_target_pos = None
        self._last_action = None
        self._action_repeat_count = 0



class EvadeDetection(SubPolicy):
    """Sub-policy that avoids threats and minimizes detection risk"""

    def __init__(self, model_path: str=None, norm_statistics_path=None):
        super().__init__("evade_detection")
        if model_path is not None:
            self.model = PPO.load(model_path)
            print('[EvadeDetection]: Using provided model for inference')
        else:
            self.model = None
            print('[EvadeDetection]: No model provided, using internal heuristic')

    def load_norm_statistics(self, norm_statistics_path):
        # TODO
        pass

    def act(self, observation):
        print(f'[EvadeDetection] EVADE TRIGGERED')

        if self.model:
            action = self.model.predict(observation)
        else:
            action = self.heuristic(observation)

        return action

    def heuristic(self, observation) -> np.int32:
        """
        Given dx, dy vector to goal position and dx,dy vector to the centerpoint of the threat to avoid, pick a direction to move around the threat (assuming the threat has a radius of 50 pixels
        Observation:
            [0] = dx to the goal position
            [1] = dy to the goal position
            [2] = dx to the center of the threat (danger zone begins 50 pixels from the centerpoint
            [3] = dy to the center of the threat (danger zone begins 50 pixels from the centerpoint
        Notes:
            * If sqrt(dx+dy)^2 <= 50, we are inside the danger zone and need to move directly away from it
            * Otherwise, we are outside the danger zone and need to pick one of 16 directions such that we move tangentially to the danger zone, following around the edge of the danger zone until we have a clear shot to the goal location
        """

        obs = np.array(observation)

        # Extract vectors
        goal_dx, goal_dy = obs[0], obs[1]
        threat_dx, threat_dy = obs[2], obs[3]

        # Calculate distance to threat center
        threat_distance = np.sqrt(threat_dx ** 2 + threat_dy ** 2)
        threat_radius = 50.0
        buffer_radius = threat_radius * 1.5

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

        # Case 1: Inside danger zone - move directly away from threat
        print(f'threat_distance {threat_distance} <= threat_radius {buffer_radius} = {threat_distance <= buffer_radius})')
        if threat_distance <= buffer_radius:
            print('Inside danger zone - evading directly away from threat')

            # Handle edge case where agent is exactly at threat center
            if threat_distance < 1e-6:  # Very small number to avoid division by zero
                # Move toward goal if available, otherwise move east
                if np.sqrt(goal_dx ** 2 + goal_dy ** 2) > 1e-6:
                    escape_direction = np.array([goal_dx, goal_dy])
                    escape_direction = escape_direction / np.linalg.norm(escape_direction)
                else:
                    escape_direction = np.array([1.0, 0.0])  # Default east
            else:
                # Move directly away from threat center
                escape_direction = np.array([-threat_dx, -threat_dy]) / threat_distance

            # Find best matching direction
            dot_products = np.dot(directions, escape_direction)
            action = np.argmax(dot_products)

        # Case 2: Outside danger zone - navigate around threat toward goal
        else:
            print('FALSE')
            # Calculate safe buffer distance
            safe_distance = threat_radius * 1.2  # 20% buffer

            # Check if we have a clear shot to goal (path doesn't intersect threat)
            goal_distance = np.sqrt(goal_dx ** 2 + goal_dy ** 2)
            if goal_distance > 0:
                goal_direction = np.array([goal_dx, goal_dy]) / goal_distance

                # Check if direct path to goal intersects with threat zone
                # Project threat center onto line from agent to goal
                threat_to_agent = np.array([-threat_dx, -threat_dy])
                projection_length = np.dot(threat_to_agent, goal_direction)

                # Only consider projection if it's between agent and goal
                if 0 <= projection_length <= goal_distance:
                    # Calculate closest point on path to threat center
                    closest_point_on_path = projection_length * goal_direction
                    distance_to_path = np.linalg.norm(threat_to_agent - closest_point_on_path)

                    # If path is clear, go directly toward goal
                    if distance_to_path > safe_distance:
                        dot_products = np.dot(directions, goal_direction)
                        action = np.argmax(dot_products)
                    else:
                        # Path blocked - need to go around threat
                        action = self._calculate_tangent_direction(threat_dx, threat_dy, goal_dx, goal_dy, threat_radius, directions)
                else:
                    # Direct path doesn't pass near threat
                    dot_products = np.dot(directions, goal_direction)
                    action = np.argmax(dot_products)
            else:
                # No goal or at goal - default behavior
                action = 0

        return np.int32(action)

    def _calculate_tangent_direction(self, threat_dx, threat_dy, goal_dx, goal_dy, threat_radius, directions):
        """Calculate direction to move tangentially around threat toward goal"""

        # Vector from agent to threat center
        threat_vector = np.array([threat_dx, threat_dy])
        threat_distance = np.linalg.norm(threat_vector)

        if threat_distance == 0:
            return 0

        threat_unit = threat_vector / threat_distance # Unit vector toward threat

        # Calculate two tangent directions (perpendicular to radius)
        # Rotate threat vector by +90 and -90 degrees
        tangent1 = np.array([-threat_unit[1], threat_unit[0]])  # +90 degrees
        tangent2 = np.array([threat_unit[1], -threat_unit[0]])  # -90 degrees

        # Choose tangent direction that brings us closer to goal
        goal_vector = np.array([goal_dx, goal_dy])
        goal_distance = np.linalg.norm(goal_vector)

        if goal_distance > 0: # Choose tangent that has better dot product with goal direction
            goal_unit = goal_vector / goal_distance
            dot1 = np.dot(tangent1, goal_unit)
            dot2 = np.dot(tangent2, goal_unit)
            chosen_tangent = tangent1 if dot1 > dot2 else tangent2
        else: # Default to first tangent if no goal
            chosen_tangent = tangent1

        # Find best matching direction from available actions
        dot_products = np.dot(directions, chosen_tangent)
        return np.argmax(dot_products)


class LocalSearch(SubPolicy):
    """Sub-policy that searches locally for unknown targets"""

    def __init__(self, model_path: str = None, norm_stats_filepath: str = None):
        super().__init__("local_search")
        self.search_radius = 300.0  # Search within this radius

        if model_path:
            self.model = PPO.load(model_path)
            print('[LocalSearch] Using provided model for inference')
        else:
            self.model = None
            print('[LocalSearch] No model provided, using internal heuristic')

        self.norm_statistics_path = norm_stats_filepath
        if norm_stats_filepath:
            self.norm_stats_filepath = norm_stats_filepath
            print(f'Loaded training normalization stats from {norm_stats_filepath}')
        else:
            self.norm_stats_filepath = None

        self._current_target_id = None
        self._current_target_pos = None
        self._last_action = None
        self._action_repeat_count = 0
        self._max_repeat_count = 3  # Minimum steps to take in same direction
        self._target_switch_threshold = 20.0  # Distance threshold to consider switching targets


    def act(self, observation):
        if self.model:
            action, _ = self.model.predict(observation)
            action = np.int32(action)
            #print(f'Model output is {action}')
        else:
            action, _ = self.heuristic(observation)
        #print(f'local search act: {action} {type(action)}')
        return action, None #np.int32(action)

    def heuristic(self, observation):
        """Simple heuristic to fly to nearest unknown target. Can be used if RL model is not provided"""

        # Handle both vectorized and non-vectorized observations
        obs = np.array(observation)

        # If observation is from vectorized environment, extract the first element
        if obs.ndim > 1:
            obs = obs[0]  # Extract first environment's observation

        # Ensure obs is at least 1D
        obs = np.atleast_1d(obs)

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
        if len(obs) < 2:
            print(f"Warning: observation too short, got {len(obs)} elements, expected at least 2")
            return np.int32(0)

        target_vector_x = obs[0]
        target_vector_y = obs[1]

        # Check if there's a valid target (non-zero vector)
        if target_vector_x == 0.0 and target_vector_y == 0.0:
            # No targets or at target location
            self.reset_heuristic_state()
            return np.int32(0)

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
            return np.int32(self._last_action if self._last_action is not None else 0)

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
                self._action_repeat_count = 0  # Reset if direction is too far off
        else:
            self._action_repeat_count = 0

        # Additional anti-oscillation: prevent direct opposite actions
        if (self._last_action is not None and abs(
                self._last_action - best_action) == 8):  # Opposite directions for 16-direction case
            # Choose a compromise direction
            adjacent_actions = [(self._last_action + 1) % 16, (self._last_action - 1) % 16]
            adjacent_dots = [dot_products[a] for a in adjacent_actions]
            best_adjacent_idx = np.argmax(adjacent_dots)
            best_action = adjacent_actions[best_adjacent_idx]

        self._last_action = best_action
        #print(f'Local search heuristic: action is {best_action} ({type(best_action)}')
        return np.int32(best_action), None

    def reset_heuristic_state(self):
        """Reset the global state for the heuristic policy."""
        #global _current_target_id, _current_target_pos, _last_action, _action_repeat_count
        self._current_target_id = None
        self._current_target_pos = None
        self._last_action = None
        self._action_repeat_count = 0


class ChangeRegions(SubPolicy):
    """Sub-policy that moves to a specific region of the map"""

    def __init__(self, model_path=None):
        super().__init__(f"change_region")
        if model_path is not None:
            self.model = PPO.load(model_path)
            print('[ChangeRegions]: Using provided model for inference')
        else:
            self.model = None
            print('[ChangeRegions]: No model provided, using internal heuristic')

        self.update_rate = 10 # Recalculate every 10 steps to reduce computation cost
        self.steps_since_update = 0

        self.target_region = None
        self.arrival_threshold = 0.05


    def act(self, observation):
        # Check if we've reached the current target region
        if self.target_region is not None and self._has_reached_target(observation):
            print(f'Reached target region {self.target_region}, selecting new region')
            self.target_region = None  # Force new selection
            self.steps_since_update = self.update_rate  # Force immediate update

        # Select new target region if needed
        if self.target_region is None or self.steps_since_update >= self.update_rate:
            self.steps_since_update = 0
            if self.model:
                self.target_region = self.model.predict(observation)
            else:
                self.target_region = self.heuristic(observation)
            print(f'Selected new target region: {self.target_region}')

        # Set waypoint directly to center of new region
        action = self._get_region_center(self.target_region)
        self.steps_since_update += 1
        print(f'Changeregion choice action {action}')
        return action

    def _has_reached_target(self, observation):
        """Check if agent has reached the current target region"""
        if self.target_region is None:
            return False

        # Extract agent distance to the target region from observation
        # Each region has 3 values: [target_ratio, agent_distance, teammate_distance]
        target_region_info_idx = self.target_region * 3 + 1  # +1 to get the agent distance
        agent_distance_to_target = observation[target_region_info_idx]

        # Check if agent is close enough to the target region
        # The distance is normalized, so we use a small threshold
        distance_threshold = 0.15  # Adjust this value as needed (normalized distance)

        return agent_distance_to_target <= distance_threshold

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
        if self.target_region is None:
            return False

        # Get agent position from env_state
        agent_pos = np.array([env_state['agent_x'], env_state['agent_y']])
        region_center = self._get_region_center(self.target_region)

        # Convert region center from normalized coordinates to actual coordinates
        map_half_size = 500
        region_center_actual = region_center * map_half_size

        distance_to_region = np.linalg.norm(region_center_actual - agent_pos)
        print(f'Distance to region: {distance_to_region}')
        arrival_threshold = self.arrival_threshold  # Convert normalized threshold to actual distance

        terminated = distance_to_region <= arrival_threshold
        #print(terminated)
        return terminated

    def _get_region_center(self, region_id: int) -> np.ndarray:
        """Get the center coordinates of a region (0=NW, 1=NE, 2=SW, 3=SE)"""
        # centers = {
        #     0: np.array([-0.5, 0.5]),  # NW
        #     1: np.array([0.5, 0.5]),  # NE
        #     2: np.array([-0.5, -0.5]),  # SW
        #     3: np.array([0.5, -0.5])  # SE
        # }
        centers = {
            0: np.array([-75, 75]),  # NW
            1: np.array([75, 75]),  # NE
            2: np.array([-75, -75]),  # SW
            3: np.array([75, -75])  # SE
        }
        # centers = {
        #     0: np.array([0.25, 0.25]),  # NW
        #     1: np.array([0.75, 0.25]),  # NE
        #     2: np.array([0.25, 0.75]),  # SW
        #     3: np.array([0.75, 0.75])  # SE
        # }
        return centers.get(region_id, np.array([0.0, 0.0]))