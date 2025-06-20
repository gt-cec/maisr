import gymnasium as gym
import numpy as np
import math
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List

from stable_baselines3 import PPO

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

        self.is_terminated = False


    def act(self, observation):
        # if not self.has_threats_remaining(observation):
        #     self.is_terminated = True
        #     return 0  # Default action when no threats remain

        if self.model:
            action = self.model.predict(observation)
        else:
            action = self.heuristic(observation)
        print(f'GOTOTHREAT: Obs {observation} -> action {action}')
        return action

    def heuristic(self, observation) -> np.int32:
        """
        Input: Observation vector with dx, dy, identified status for up to 2 threats
        Output: Direction to move toward the best available threat
        """
        # Check if any threats remain
        if not self.has_threats_remaining(observation):
            self.reset_heuristic_state()
            self.is_terminated = True
            return np.int32(0)

        obs = np.array(observation)

        # Extract threat information
        threats = []
        for i in range(2):  # Check up to 2 threats
            base_idx = i * 3
            if base_idx + 2 < len(obs):
                x, y = obs[base_idx], obs[base_idx + 1]
                identified = obs[base_idx + 2]

                # Only consider threats that exist and are unidentified
                if not (x == 0.0 and y == 0.0) and identified < 1.0:
                    distance = np.sqrt(x * x + y * y)
                    threats.append({
                        'vector': np.array([x, y]),
                        'distance': distance,
                        'identified': identified
                    })

        if not threats:
            self.reset_heuristic_state()
            return np.int32(0)

        # Choose the closest unidentified threat
        target_threat = min(threats, key=lambda t: t['distance'])
        direction_to_target = target_threat['vector']

        # Rest of the direction calculation logic remains the same...
        directions = np.array([
            (0, 1), (0.383, 0.924), (0.707, 0.707), (0.924, 0.383),
            (1, 0), (0.924, -0.383), (0.707, -0.707), (0.383, -0.924),
            (0, -1), (-0.383, -0.924), (-0.707, -0.707), (-0.924, -0.383),
            (-1, 0), (-0.924, 0.383), (-0.707, 0.707), (-0.383, 0.924),
        ], dtype=float)

        # Normalize and find best direction
        direction_norms = np.linalg.norm(directions, axis=1)
        normalized_directions = directions / direction_norms[:, np.newaxis]

        target_norm = np.linalg.norm(direction_to_target)
        if target_norm > 0:
            direction_to_target_norm = direction_to_target / target_norm
            dot_products = np.dot(normalized_directions, direction_to_target_norm)
            best_action = np.argmax(dot_products)
        else:
            best_action = 0

        # Apply anti-oscillation logic as before...
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

        self._last_action = best_action
        return np.int32(best_action)

    def reset_heuristic_state(self):
        """Reset the global state for the heuristic policy."""
        #global _current_target_id, _current_target_pos, _last_action, _action_repeat_count
        self._current_target_id = None
        self._current_target_pos = None
        self._last_action = None
        self._action_repeat_count = 0

    def has_threats_remaining(self, observation) -> bool:
        """
        Check if there are any high-value threats remaining to pursue
        Now checks both threat slots in the observation
        """
        obs = np.array(observation)

        # Check first threat
        threat1_x, threat1_y = obs[0], obs[1]
        threat1_identified = obs[2] if len(obs) > 2 else 0

        # Check second threat if available
        threat2_x, threat2_y = 0, 0
        threat2_identified = 1  # Default to identified if not present
        if len(obs) >= 6:
            threat2_x, threat2_y = obs[3], obs[4]
            threat2_identified = obs[5]

        # Return True if any unidentified threat exists
        threat1_exists = not (threat1_x == 0.0 and threat1_y == 0.0)
        threat2_exists = not (threat2_x == 0.0 and threat2_y == 0.0)

        threat1_unidentified = threat1_exists and threat1_identified < 1.0
        threat2_unidentified = threat2_exists and threat2_identified < 1.0

        return threat1_unidentified or threat2_unidentified



class EvadeDetection(SubPolicy):
    """Sub-policy that avoids threats and minimizes detection risk"""

    def __init__(self, model_path: str=None, norm_statistics_path=None):
        super().__init__("evade_detection")
        self.model = PPO.load(model_path)
        self.norm_statistics_path = norm_statistics_path
        if self.model:
            print(f"[Evade] Loading model {self.model}")

    def load_norm_statistics(self, norm_statistics_path):
        # TODO
        pass

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
        if threat_distance <= threat_radius:
            # Move directly away from threat center
            if threat_distance > 0:
                escape_direction = np.array([-threat_dx, -threat_dy]) / threat_distance
            else:
                # If exactly at threat center, move toward goal
                goal_distance = np.sqrt(goal_dx ** 2 + goal_dy ** 2)
                if goal_distance > 0:
                    escape_direction = np.array([goal_dx, goal_dy]) / goal_distance
                else:
                    escape_direction = np.array([1, 0])  # Default east

            # Find best matching direction
            dot_products = np.dot(directions, escape_direction)
            action = np.argmax(dot_products)

        # Case 2: Outside danger zone - navigate around threat toward goal
        else:
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


# class LocalSearch(SubPolicy):
#     """Sub-policy that searches locally for unknown targets with integrated evade logic"""
#
#     def __init__(self, model_path: str = None, norm_stats_filepath: str = None):
#         super().__init__("local_search")
#         self.search_radius = 300.0  # Search within this radius
#
#         if model_path:
#             self.model = PPO.load(model_path)
#             print('[LocalSearch] Using provided model for inference')
#         else:
#             self.model = None
#             print('[LocalSearch] No model provided, using internal heuristic')
#
#         self.norm_statistics_path = norm_stats_filepath
#         if norm_stats_filepath:
#             self.norm_stats_filepath = norm_stats_filepath
#             print(f'Loaded training normalization stats from {norm_stats_filepath}')
#         else:
#             self.norm_stats_filepath = None
#
#         # Heuristic state tracking
#         self._current_target_id = None
#         self._current_target_pos = None
#         self._last_action = None
#         self._action_repeat_count = 0
#         self._max_repeat_count = 3  # Minimum steps to take in same direction
#         self._target_switch_threshold = 20.0  # Distance threshold to consider switching targets
#
#         # Evade logic state (moved from wrapper)
#         self.evade_goal = None
#         self.evade_goal_threshold = 30.0  # Distance threshold to consider goal "reached"
#         self.last_evade_step = -1  # Track when we last used evade to detect continuous usage
#
#         self.circumnavigation_state = {
#             'active': False,
#             'threat_pos': None,
#             'chosen_direction': None,  # 'clockwise' or 'counterclockwise'
#             'last_angle': None,
#             'start_angle': None,
#             'safety_distance': None
#         }
#
#     def act(self, observation, env=None, agent_id=0):
#         """
#         Enhanced act method that includes evade logic for threats
#
#         Args:
#             observation: The observation vector for local search
#             env: Environment instance (needed for threat detection)
#             agent_id: Agent ID (default 0)
#         """
#         # Check if we need to evade threats first
#         if env is not None and self.near_threat(env, agent_id):
#             print(f"[LocalSearch] Threat detected, switching to evade mode")
#             evade_action = self.compute_tangential_escape_action(env, agent_id)
#             return evade_action, None
#
#         # Normal local search behavior
#         if self.model:
#             action, _ = self.model.predict(observation)
#             action = np.int32(action)
#         else:
#             try:
#                 action, _ = self.heuristic(observation)
#             except:
#                 action = self.heuristic(observation)
#
#         return action, None
#
#     def near_threat(self, env, agent_id=0):
#         """
#         Check if the agent is near a threat and should automatically switch to evade mode.
#         Returns True if agent is within threat radius or warning zone of any threat.
#         """
#         # Get agent position
#         agent_pos = np.array([env.agents[env.aircraft_ids[agent_id]].x,
#                               env.agents[env.aircraft_ids[agent_id]].y])
#
#         # Check distance to all threats
#         for threat_idx in range(len(env.threats)):
#             threat_pos = np.array([env.threats[threat_idx, 0], env.threats[threat_idx, 1]])
#             distance_to_threat = np.sqrt(np.sum((threat_pos - agent_pos) ** 2))
#
#             threat_radius = env.config['threat_radius']
#             warning_radius = threat_radius * 1.7  # 70% larger than threat radius for early warning
#
#             # Trigger evade mode if within warning radius
#             if distance_to_threat <= warning_radius:
#                 return True
#         return False
#
#     def compute_tangential_escape_action(self, env, agent_id=0):
#         """
#         Compute a direct tangential escape action when near a threat.
#         Uses state persistence to maintain consistent circumnavigation direction.
#         """
#         # Get agent position
#         agent_pos = np.array([env.agents[env.aircraft_ids[agent_id]].x,
#                               env.agents[env.aircraft_ids[agent_id]].y])
#
#         # Find nearest threat
#         nearest_threat_pos = None
#         min_distance = float('inf')
#         nearest_threat_idx = None
#
#         for threat_idx in range(len(env.threats)):
#             threat_pos = np.array([env.threats[threat_idx, 0], env.threats[threat_idx, 1]])
#             distance = np.sqrt(np.sum((threat_pos - agent_pos) ** 2))
#             if distance < min_distance:
#                 min_distance = distance
#                 nearest_threat_pos = threat_pos
#                 nearest_threat_idx = threat_idx
#
#         if nearest_threat_pos is None:
#             return 0  # Default action if no threats
#
#         threat_radius = env.config['threat_radius']
#         safety_margin = threat_radius * 1.8  # Increased safety margin
#
#         # Check if we need to start or continue circumnavigation
#         if min_distance <= safety_margin:
#             return self._circumnavigate_threat(agent_pos, nearest_threat_pos, threat_radius, env, agent_id)
#         else:
#             # Far enough from threat, reset circumnavigation state
#             self._reset_circumnavigation_state()
#             return 0
#
#     def _circumnavigate_threat(self, agent_pos, threat_pos, threat_radius, env, agent_id=0):
#         """Handle circumnavigation around a threat with state persistence"""
#
#         # Direction mapping (16 directions)
#         directions = np.array([
#             (0, 1),  # North (0°)
#             (0.383, 0.924),  # NNE (22.5°)
#             (0.707, 0.707),  # NE (45°)
#             (0.924, 0.383),  # ENE (67.5°)
#             (1, 0),  # East (90°)
#             (0.924, -0.383),  # ESE (112.5°)
#             (0.707, -0.707),  # SE (135°)
#             (0.383, -0.924),  # SSE (157.5°)
#             (0, -1),  # South (180°)
#             (-0.383, -0.924),  # SSW (202.5°)
#             (-0.707, -0.707),  # SW (225°)
#             (-0.924, -0.383),  # WSW (247.5°)
#             (-1, 0),  # West (270°)
#             (-0.924, 0.383),  # WNW (292.5°)
#             (-0.707, 0.707),  # NW (315°)
#             (-0.383, 0.924),  # NNW (337.5°)
#         ], dtype=np.float32)
#
#         # Vector from threat to agent
#         threat_to_agent = agent_pos - threat_pos
#         distance_to_threat = np.linalg.norm(threat_to_agent)
#
#         if distance_to_threat < 1e-6:
#             return 0  # Default if at threat center
#
#         # Calculate current angle around threat
#         current_angle = np.arctan2(threat_to_agent[1], threat_to_agent[0])
#
#         # Initialize or update circumnavigation state
#         if not self.circumnavigation_state['active']:
#             self._initialize_circumnavigation(threat_pos, current_angle, threat_radius)
#
#         # Check if circumnavigation is complete
#         if self._is_circumnavigation_complete(current_angle, threat_pos, agent_pos, env):
#             self._reset_circumnavigation_state()
#             # Move toward original target
#             return self._get_action_toward_nearest_target(agent_pos, directions, env)
#
#         # Continue circumnavigation
#         return self._get_circumnavigation_action(current_angle, threat_to_agent, directions)
#
#     def _initialize_circumnavigation(self, threat_pos, start_angle, threat_radius):
#         """Initialize circumnavigation state"""
#         self.circumnavigation_state['active'] = True
#         self.circumnavigation_state['threat_pos'] = threat_pos.copy()
#         self.circumnavigation_state['start_angle'] = start_angle
#         self.circumnavigation_state['last_angle'] = start_angle
#         self.circumnavigation_state['safety_distance'] = threat_radius * 1.5
#
#         # Choose direction based on which way moves us more toward targets
#         # For now, default to counterclockwise
#         self.circumnavigation_state['chosen_direction'] = 'counterclockwise'
#
#         print(f"[LocalSearch] Starting circumnavigation: direction={self.circumnavigation_state['chosen_direction']}")
#
#     def _is_circumnavigation_complete(self, current_angle, threat_pos, agent_pos, env):
#         """Check if we've gone far enough around the threat to have a clear path"""
#         if not self.circumnavigation_state['active']:
#             return False
#
#         # Calculate how far we've traveled around the threat
#         start_angle = self.circumnavigation_state['start_angle']
#         angle_traveled = current_angle - start_angle
#
#         # Normalize angle difference to [-π, π]
#         while angle_traveled > np.pi:
#             angle_traveled -= 2 * np.pi
#         while angle_traveled < -np.pi:
#             angle_traveled += 2 * np.pi
#
#         # Check if we've gone at least 90 degrees around
#         min_angle_traveled = np.pi / 2  # 90 degrees
#
#         if self.circumnavigation_state['chosen_direction'] == 'counterclockwise':
#             sufficient_travel = angle_traveled >= min_angle_traveled
#         else:  # clockwise
#             sufficient_travel = angle_traveled <= -min_angle_traveled
#
#         if sufficient_travel:
#             # Also check if we now have a clear line to targets
#             return self._has_clear_path_to_targets(agent_pos, threat_pos, env)
#
#         return False
#
#     def _has_clear_path_to_targets(self, agent_pos, threat_pos, env):
#         """Check if there's a clear path from current position to nearest unknown target"""
#         # Get unknown target positions
#         target_positions = env.targets[:env.config['num_targets'], 3:5]
#         target_info_levels = env.targets[:env.config['num_targets'], 2]
#         unknown_mask = target_info_levels < 1.0
#
#         if not np.any(unknown_mask):
#             return True  # No targets left, circumnavigation complete
#
#         unknown_positions = target_positions[unknown_mask]
#         distances = np.sqrt(np.sum((unknown_positions - agent_pos) ** 2, axis=1))
#         nearest_target_pos = unknown_positions[np.argmin(distances)]
#
#         # Check if path to nearest target intersects threat
#         return self._path_clear_of_threat(agent_pos, nearest_target_pos, threat_pos, env)
#
#     def _path_clear_of_threat(self, start_pos, end_pos, threat_pos, env):
#         """Check if straight line path from start to end clears the threat"""
#         threat_radius = env.config['threat_radius'] * 1.2  # Safety buffer
#
#         # Vector from start to end
#         path_vector = end_pos - start_pos
#         path_length = np.linalg.norm(path_vector)
#
#         if path_length < 1e-6:
#             return True
#
#         path_unit = path_vector / path_length
#
#         # Vector from start to threat
#         start_to_threat = threat_pos - start_pos
#
#         # Project threat onto path
#         projection_length = np.dot(start_to_threat, path_unit)
#
#         # Only check collision if projection is within the path segment
#         if 0 <= projection_length <= path_length:
#             closest_point_on_path = start_pos + projection_length * path_unit
#             distance_to_threat = np.linalg.norm(threat_pos - closest_point_on_path)
#             return distance_to_threat > threat_radius
#
#         return True  # Threat is not along the path
#
#     def _get_circumnavigation_action(self, current_angle, threat_to_agent, directions):
#         """Get action to continue circumnavigation"""
#         distance_to_threat = np.linalg.norm(threat_to_agent)
#
#         if distance_to_threat > 0:
#             # Calculate tangent direction
#             threat_unit = threat_to_agent / distance_to_threat
#
#             if self.circumnavigation_state['chosen_direction'] == 'counterclockwise':
#                 tangent_direction = np.array([-threat_unit[1], threat_unit[0]])  # +90 degrees
#             else:  # clockwise
#                 tangent_direction = np.array([threat_unit[1], -threat_unit[0]])  # -90 degrees
#
#             # Add slight outward bias to maintain safe distance
#             outward_direction = threat_unit  # Away from threat
#             bias_strength = 0.2
#
#             combined_direction = tangent_direction * (1 - bias_strength) + outward_direction * bias_strength
#             combined_direction = combined_direction / np.linalg.norm(combined_direction)
#
#             # Find best matching action
#             dot_products = np.dot(directions, combined_direction)
#             best_action = np.argmax(dot_products)
#
#             return np.int32(best_action)
#
#         return 0
#
#     def _get_action_toward_nearest_target(self, agent_pos, directions, env):
#         """Get action to move toward nearest unknown target after circumnavigation"""
#         # Get unknown target positions
#         target_positions = env.targets[:env.config['num_targets'], 3:5]
#         target_info_levels = env.targets[:env.config['num_targets'], 2]
#         unknown_mask = target_info_levels < 1.0
#
#         if not np.any(unknown_mask):
#             return 0  # No targets left
#
#         unknown_positions = target_positions[unknown_mask]
#         distances = np.sqrt(np.sum((unknown_positions - agent_pos) ** 2, axis=1))
#         nearest_target_pos = unknown_positions[np.argmin(distances)]
#
#         # Direction to nearest target
#         target_vector = nearest_target_pos - agent_pos
#         target_distance = np.linalg.norm(target_vector)
#
#         if target_distance > 0:
#             target_direction = target_vector / target_distance
#             dot_products = np.dot(directions, target_direction)
#             best_action = np.argmax(dot_products)
#             return np.int32(best_action)
#
#         return 0
#
#     def _reset_circumnavigation_state(self):
#         """Reset circumnavigation state"""
#         self.circumnavigation_state = {
#             'active': False,
#             'threat_pos': None,
#             'chosen_direction': None,
#             'last_angle': None,
#             'start_angle': None,
#             'safety_distance': None
#         }
#
#     def heuristic(self, observation):
#         """Simple heuristic to fly to nearest unknown target. Can be used if RL model is not provided"""
#
#         # Handle both vectorized and non-vectorized observations
#         obs = np.array(observation)
#
#         # If observation is from vectorized environment, extract the first element
#         if obs.ndim > 1:
#             obs = obs[0]  # Extract first environment's observation
#
#         # Ensure obs is at least 1D
#         obs = np.atleast_1d(obs)
#
#         # Direction mapping
#         directions = np.array([
#             (0, 1),  # North (0°)
#             (0.383, 0.924),  # NNE (22.5°)
#             (0.707, 0.707),  # NE (45°)
#             (0.924, 0.383),  # ENE (67.5°)
#             (1, 0),  # East (90°)
#             (0.924, -0.383),  # ESE (112.5°)
#             (0.707, -0.707),  # SE (135°)
#             (0.383, -0.924),  # SSE (157.5°)
#             (0, -1),  # South (180°)
#             (-0.383, -0.924),  # SSW (202.5°)
#             (-0.707, -0.707),  # SW (225°)
#             (-0.924, -0.383),  # WSW (247.5°)
#             (-1, 0),  # West (270°)
#             (-0.924, 0.383),  # WNW (292.5°)
#             (-0.707, 0.707),  # NW (315°)
#             (-0.383, 0.924),  # NNW (337.5°)
#         ], dtype=float)
#
#         # Extract nearest target vector (first two components)
#         if len(obs) < 2:
#             print(f"Warning: observation too short, got {len(obs)} elements, expected at least 2")
#             return np.int32(0), None
#
#         target_vector_x = obs[0]
#         target_vector_y = obs[1]
#
#         # Check if there's a valid target (non-zero vector)
#         if target_vector_x == 0.0 and target_vector_y == 0.0:
#             # No targets or at target location
#             self.reset_heuristic_state()
#             return np.int32(0), None
#
#         # The observation already gives us the vector to the nearest target
#         direction_to_target = np.array([target_vector_x, target_vector_y])
#
#         # Normalize direction vectors
#         direction_norms = np.linalg.norm(directions, axis=1)
#         normalized_directions = directions / direction_norms[:, np.newaxis]
#
#         # Normalize target direction
#         target_norm = np.linalg.norm(direction_to_target)
#         if target_norm > 0:
#             direction_to_target_norm = direction_to_target / target_norm
#         else:
#             return np.int32(self._last_action if self._last_action is not None else 0), None
#
#         # Calculate dot products
#         dot_products = np.dot(normalized_directions, direction_to_target_norm)
#
#         # Find best action
#         best_action = np.argmax(dot_products)
#
#         # Anti-oscillation: if we just took an action, continue for minimum steps
#         if (self._last_action is not None and
#                 self._action_repeat_count < self._max_repeat_count and
#                 self._last_action != best_action):
#
#             # Check if last action is still reasonable (dot product > 0.5)
#             last_dot_product = dot_products[self._last_action]
#             if last_dot_product > 0.5:  # Still pointing roughly toward target
#                 best_action = self._last_action
#                 self._action_repeat_count += 1
#             else:
#                 self._action_repeat_count = 0  # Reset if direction is too far off
#         else:
#             self._action_repeat_count = 0
#
#         # Additional anti-oscillation: prevent direct opposite actions
#         if (self._last_action is not None and abs(
#                 self._last_action - best_action) == 8):  # Opposite directions for 16-direction case
#             # Choose a compromise direction
#             adjacent_actions = [(self._last_action + 1) % 16, (self._last_action - 1) % 16]
#             adjacent_dots = [dot_products[a] for a in adjacent_actions]
#             best_adjacent_idx = np.argmax(adjacent_dots)
#             best_action = adjacent_actions[best_adjacent_idx]
#
#         self._last_action = best_action
#         return np.int32(best_action), None
#
#     def reset_heuristic_state(self):
#         """Reset the global state for the heuristic policy."""
#         self._current_target_id = None
#         self._current_target_pos = None
#         self._last_action = None
#         self._action_repeat_count = 0
#
#     def reset_evade_state(self):
#         """Reset evade-related state"""
#         self.evade_goal = None
#         self.last_evade_step = -1
#         self._reset_circumnavigation_state()







# TODO: This is the old (working) localsearch that doesn't have evade built in
# class LocalSearch(SubPolicy):
#     """Sub-policy that searches locally for unknown targets"""
#
#     def __init__(self, model_path: str = None, norm_stats_filepath: str = None):
#         super().__init__("local_search")
#         self.search_radius = 300.0  # Search within this radius
#
#
#         if model_path:
#             self.model = PPO.load(model_path)
#             print('[LocalSearch] Using provided model for inference')
#         else:
#             self.model = None
#             print('[LocalSearch] No model provided, using internal heuristic')
#
#         self.norm_statistics_path = norm_stats_filepath
#         if norm_stats_filepath:
#             self.norm_stats_filepath = norm_stats_filepath
#             print(f'Loaded training normalization stats from {norm_stats_filepath}')
#         else:
#             self.norm_stats_filepath = None
#
#         self._current_target_id = None
#         self._current_target_pos = None
#         self._last_action = None
#         self._action_repeat_count = 0
#         self._max_repeat_count = 3  # Minimum steps to take in same direction
#         self._target_switch_threshold = 20.0  # Distance threshold to consider switching targets
#
#
#     def act(self, observation):
#         if self.model:
#             action = self.model.predict(observation)
#         else:
#             action = self.heuristic(observation)
#         return action
#
#     def heuristic(self, observation):
#         """Simple heuristic to fly to nearest unknown target. Can be used if RL model is not provided"""
#
#         # Handle both vectorized and non-vectorized observations
#         obs = np.array(observation)
#
#         # If observation is from vectorized environment, extract the first element
#         if obs.ndim > 1:
#             obs = obs[0]  # Extract first environment's observation
#
#         # Ensure obs is at least 1D
#         obs = np.atleast_1d(obs)
#
#         print(f'[heuristic]obs: {obs}')
#
#         # Direction mapping
#         directions = np.array([
#             (0, 1),  # North (0°)
#             (0.383, 0.924),  # NNE (22.5°)
#             (0.707, 0.707),  # NE (45°)
#             (0.924, 0.383),  # ENE (67.5°)
#             (1, 0),  # East (90°)
#             (0.924, -0.383),  # ESE (112.5°)
#             (0.707, -0.707),  # SE (135°)
#             (0.383, -0.924),  # SSE (157.5°)
#             (0, -1),  # South (180°)
#             (-0.383, -0.924),  # SSW (202.5°)
#             (-0.707, -0.707),  # SW (225°)
#             (-0.924, -0.383),  # WSW (247.5°)
#             (-1, 0),  # West (270°)
#             (-0.924, 0.383),  # WNW (292.5°)
#             (-0.707, 0.707),  # NW (315°)
#             (-0.383, 0.924),  # NNW (337.5°)
#         ], dtype=float)
#
#         # Extract nearest target vector (first two components)
#         if len(obs) < 2:
#             print(f"Warning: observation too short, got {len(obs)} elements, expected at least 2")
#             return np.int32(0)
#
#         target_vector_x = obs[0]
#         target_vector_y = obs[1]
#
#         # Check if there's a valid target (non-zero vector)
#         if target_vector_x == 0.0 and target_vector_y == 0.0:
#             # No targets or at target location
#             self.reset_heuristic_state()
#             return np.int32(0)
#
#         # The observation already gives us the vector to the nearest target
#         direction_to_target = np.array([target_vector_x, target_vector_y])
#
#         # Normalize direction vectors
#         direction_norms = np.linalg.norm(directions, axis=1)
#         normalized_directions = directions / direction_norms[:, np.newaxis]
#
#         # Normalize target direction
#         target_norm = np.linalg.norm(direction_to_target)
#         if target_norm > 0:
#             direction_to_target_norm = direction_to_target / target_norm
#         else:
#             return np.int32(self._last_action if self._last_action is not None else 0)
#
#         # Calculate dot products
#         dot_products = np.dot(normalized_directions, direction_to_target_norm)
#
#         # Find best action
#         best_action = np.argmax(dot_products)
#
#         # Anti-oscillation: if we just took an action, continue for minimum steps
#         if (self._last_action is not None and
#                 self._action_repeat_count < self._max_repeat_count and
#                 self._last_action != best_action):
#
#             # Check if last action is still reasonable (dot product > 0.5)
#             last_dot_product = dot_products[self._last_action]
#             if last_dot_product > 0.5:  # Still pointing roughly toward target
#                 best_action = self._last_action
#                 self._action_repeat_count += 1
#             else:
#                 self._action_repeat_count = 0  # Reset if direction is too far off
#         else:
#             self._action_repeat_count = 0
#
#         # Additional anti-oscillation: prevent direct opposite actions
#         if (self._last_action is not None and abs(
#                 self._last_action - best_action) == 8):  # Opposite directions for 16-direction case
#             # Choose a compromise direction
#             adjacent_actions = [(self._last_action + 1) % 16, (self._last_action - 1) % 16]
#             adjacent_dots = [dot_products[a] for a in adjacent_actions]
#             best_adjacent_idx = np.argmax(adjacent_dots)
#             best_action = adjacent_actions[best_adjacent_idx]
#
#         self._last_action = best_action
#         return np.int32(best_action), None
#
#     def reset_heuristic_state(self):
#         """Reset the global state for the heuristic policy."""
#         #global _current_target_id, _current_target_pos, _last_action, _action_repeat_count
#         self._current_target_id = None
#         self._current_target_pos = None
#         self._last_action = None
#         self._action_repeat_count = 0




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
        centers = {
            0: np.array([-0.5, 0.5]),  # NW
            1: np.array([0.5, 0.5]),  # NE
            2: np.array([-0.5, -0.5]),  # SW
            3: np.array([0.5, -0.5])  # SE
        }
        return centers.get(region_id, np.array([0.0, 0.0]))