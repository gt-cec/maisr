import random
import numpy as np
from abc import ABC, abstractmethod
import pygame
from stable_baselines3 import PPO
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


class TeammateManager:
    """Manages pool of teammate policies and selection based on league type"""

    def __init__(self,league_type="baseline", subpolicies=None):
        """
        Initialize teammate manager with specified league type.

        Args:
            league_type (str): "baseline", "vanilla", or "strategy_diverse"
            subpolicies (dict): Dictionary containing subpolicy instances
                Expected keys: 'local_search', 'change_region', 'go_to_threat'
        """
        self.league_type = league_type
        self.subpolicies = subpolicies or {}
        self.current_teammate = None
        self.episode_count = 0

        # Validate league type
        valid_league_types = ["baseline", "vanilla", "strategy_diverse"]
        if league_type not in valid_league_types:
            raise ValueError(f"league_type must be one of {valid_league_types}")

        print(f"\nTeammateManager initialized with league_type: {league_type}")

    def select_random_teammate(self):
        """Randomly select a teammate based on league type configuration"""
        if self.league_type == "baseline":
            return self._create_baseline_teammate()
        elif self.league_type == "vanilla":
            return self._create_vanilla_teammate()
        elif self.league_type == "strategy_diverse":
            return self._create_strategy_diverse_teammate()
        else:
            raise ValueError(f"Unknown league_type: {self.league_type}")

    def _create_baseline_teammate(self):
        """Create baseline teammate: always heuristic with conservative settings"""
        heuristic_agent = HeuristicAgent(
            mode_selector="none",
            risk_tolerance="low",
            spatial_coord="none"
        )

        teammate = GenericTeammatePolicy(
            env=None,  # Will be set later if needed
            local_search_policy=self.subpolicies.get('local_search'),
            go_to_highvalue_policy=self.subpolicies.get('go_to_threat'),
            change_region_subpolicy=self.subpolicies.get('change_region'),
            mode_selector_agent=heuristic_agent,
            use_collision_avoidance=False
        )

        teammate.name = "Baseline_Greedy_noMS_lowrisk_nospatialcoord"
        self.current_teammate = teammate
        return teammate

    def _create_vanilla_teammate(self):
        """Create vanilla teammate: varied mode_selector, conservative spatial/risk settings"""
        # Randomly sample mode_selector
        mode_selector = random.choice(["none", "heuristic"]) # Removed "trained" for now

        if mode_selector == "trained":
            # TODO: Load trained model when available
            raise NotImplementedError

        heuristic_agent = HeuristicAgent(
            mode_selector=mode_selector,
            risk_tolerance="low",  # Always low for vanilla
            spatial_coord="none"  # Always none for vanilla
        )

        teammate = GenericTeammatePolicy(
            env=None,
            local_search_policy=self.subpolicies.get('local_search'),
            go_to_highvalue_policy=self.subpolicies.get('go_to_threat'),
            change_region_subpolicy=self.subpolicies.get('change_region'),
            mode_selector_agent=heuristic_agent,
            use_collision_avoidance=False
        )

        teammate.name = f"Vanilla_{mode_selector}MS_norisk_nospatialcoord"
        self.current_teammate = teammate
        return teammate

    def _create_strategy_diverse_teammate(self):
        """Create strategy diverse teammate: all parameters randomly sampled"""
        # Randomly sample all parameters
        mode_selector = random.choice(["none", "heuristic"]) # Removed "trained"
        risk_tolerance = random.choice(["low", "medium", "high", "extreme"])
        spatial_coord = random.choice(["none", "some", "high"])

        if mode_selector == "trained":
            # TODO: Load trained model when available
            raise NotImplementedError

        heuristic_agent = HeuristicAgent(
            mode_selector=mode_selector,
            risk_tolerance=risk_tolerance,
            spatial_coord=spatial_coord
        )

        teammate = GenericTeammatePolicy(
            env=None,
            local_search_policy=self.subpolicies.get('local_search'),
            go_to_highvalue_policy=self.subpolicies.get('go_to_threat'),
            change_region_subpolicy=self.subpolicies.get('change_region'),
            mode_selector_agent=heuristic_agent,
            use_collision_avoidance=False
        )

        teammate.name = f"Diverse_{mode_selector}MS_{risk_tolerance}risk_{spatial_coord}spatialcoord"
        self.current_teammate = teammate
        return teammate

    def reset_for_episode(self):
        """Reset teammate for new episode and select new random teammate"""
        self.episode_count += 1
        # Select a new random teammate for each episode
        self.select_random_teammate()

        if self.current_teammate and hasattr(self.current_teammate, 'reset'):
            self.current_teammate.reset()

    # Legacy methods for backward compatibility
    def add_rl_teammate(self, model_path, policy_name):
        """Legacy method - not used with new league system"""
        print("Warning: add_rl_teammate is deprecated with league-based teammate management")
        pass

    def add_heuristic_teammate(self, strategy_type, config=None):
        """Legacy method - not used with new league system"""
        print("Warning: add_heuristic_teammate is deprecated with league-based teammate management")
        pass

    def select_teammate_by_curriculum(self):
        """Legacy method - use select_random_teammate instead"""
        print("Warning: select_teammate_by_curriculum is deprecated, using select_random_teammate")
        return self.select_random_teammate()


class HeuristicAgent:
    """
    Heuristic agent that chooses subpolicies based on risk tolerance and spatial coordination settings.

    Subpolicies:
    0 = localsearch
    1 = changeregion
    2 = gotothreat
    """

    def __init__(self, mode_selector="heuristic", risk_tolerance="medium", spatial_coord="some"):
        """
        Initialize the heuristic agent.

        Args:
            mode_selector (str): "none" to always choose localsearch, or "heuristic" for decision logic
            risk_tolerance (str): "low", "medium", "high", or "extreme" - controls gotothreat usage
            spatial_coord (str): "none", "some", or "high" - controls localsearch vs changeregion choice
        """
        self.mode_selector = mode_selector
        self.risk_tolerance = risk_tolerance
        self.spatial_coord = spatial_coord

        # Anti-oscillation state tracking
        self.last_subpolicy = None
        self.subpolicy_commit_steps = 0
        self.min_commit_duration = 15  # Minimum steps to stick with a subpolicy
        self.changeregion_cooldown = 0  # Steps remaining before can choose changeregion again
        self.changeregion_cooldown_duration = 10  # Steps to wait after switching away from changeregion

        # Target-rich detection with hysteresis
        self.target_rich_threshold_high = 0.45  # Threshold to START considering quadrant target-rich
        self.target_rich_threshold_low = 0.15  # Threshold to STOP considering quadrant target-rich
        self.currently_consider_target_rich = False  # Current state with hysteresis

        # Validate configuration
        valid_risk_levels = ["low", "medium", "high", "extreme"]
        valid_spatial_levels = ["none", "some", "high"]
        valid_mode_selectors = ["none", "heuristic"]

        if risk_tolerance not in valid_risk_levels:
            raise ValueError(f"risk_tolerance must be one of {valid_risk_levels}")
        if spatial_coord not in valid_spatial_levels:
            raise ValueError(f"spatial_coord must be one of {valid_spatial_levels}")
        if mode_selector not in valid_mode_selectors:
            raise ValueError(f"mode_selector must be one of {valid_mode_selectors}")

    def choose_subpolicy(self, env, agent_id=0):
        """
        Choose a subpolicy based on the agent's configuration and current environment state.

        Args:
            env: The environment instance (MAISREnvVec)
            agent_id (int): ID of the agent making the decision (default 0)

        Returns:
            int: Subpolicy choice (0=localsearch, 1=changeregion, 2=gotothreat)
        """
        # Update cooldowns
        if self.changeregion_cooldown > 0:
            self.changeregion_cooldown -= 1

        # If mode_selector is "none", always choose localsearch
        if self.mode_selector == "none":
            self._update_tracking(0)
            #print(f"[HeuristicAgent] mode_selector=none -> localsearch(0)")
            return 0

        # Check if we should choose gotothreat based on risk tolerance and detections
        detections = env.num_threats_identified
        should_go_to_threat = self._should_go_to_threat(env)

        if should_go_to_threat:
            self._update_tracking(2)
            #print(f"[HeuristicAgent] detections={detections}, risk={self.risk_tolerance} -> gotothreat(2)")
            return 2  # gotothreat

        # Check if we should stick with current subpolicy to avoid oscillation
        if (self.last_subpolicy is not None and
                self.subpolicy_commit_steps < self.min_commit_duration):

            # Don't stick with changeregion if we're in cooldown
            if self.last_subpolicy == 1 and self.changeregion_cooldown > 0:
                choice = 0  # Switch to localsearch
                reason = "changeregion_cooldown"
            else:
                choice = self.last_subpolicy
                reason = f"committed_for_{self.subpolicy_commit_steps}_steps"

            self._update_tracking(choice)
            action_name = ["localsearch", "changeregion", "gotothreat"][choice]
            #print(f"[HeuristicAgent] {reason} -> {action_name}({choice})")
            return choice

        # Choose between localsearch (0) and changeregion (1) based on spatial coordination
        choice = self._choose_search_strategy(env, agent_id)

        # Apply changeregion cooldown
        if choice == 1 and self.changeregion_cooldown > 0:
            choice = 0  # Force localsearch if changeregion is in cooldown
            reason = "changeregion_in_cooldown"
        else:
            # Add debugging for the search strategy choice
            if self.spatial_coord == "none":
                reason = "spatial_coord=none"
            elif self.spatial_coord == "some":
                reason = f"spatial_coord=some, target_rich_hysteresis={self.currently_consider_target_rich}"
            elif self.spatial_coord == "high":
                same_quadrant = self._agents_in_same_quadrant(env, agent_id)
                reason = f"spatial_coord=high, same_quadrant={same_quadrant}"
            else:
                reason = "default"

        # Start cooldown if switching away from changeregion
        if self.last_subpolicy == 1 and choice != 1:
            self.changeregion_cooldown = self.changeregion_cooldown_duration

        self._update_tracking(choice)
        action_name = ["localsearch", "changeregion", "gotothreat"][choice]
        #print(f"[HeuristicAgent] detections={detections}, risk={self.risk_tolerance}, {reason} -> {action_name}({choice})")
        return choice

    def _update_tracking(self, chosen_subpolicy):
        """Update internal tracking for anti-oscillation"""
        if self.last_subpolicy == chosen_subpolicy:
            self.subpolicy_commit_steps += 1
        else:
            self.subpolicy_commit_steps = 1
        self.last_subpolicy = chosen_subpolicy

    def _should_go_to_threat(self, env):
        """
        Determine if the agent should choose gotothreat based on risk tolerance and current detections.

        Args:
            env: The environment instance

        Returns:
            bool: True if should choose gotothreat, False otherwise
        """
        detections = env.num_threats_identified

        if self.risk_tolerance == "low":
            return False  # Never choose gotothreat
        elif self.risk_tolerance == "medium":
            return detections == 0
        elif self.risk_tolerance == "high":
            return detections <= 1
        elif self.risk_tolerance == "extreme":
            return detections <= 2

        return False

    def _choose_search_strategy(self, env, agent_id):
        """
        Choose between localsearch and changeregion based on spatial coordination setting.

        Args:
            env: The environment instance
            agent_id (int): ID of the agent making the decision

        Returns:
            int: 0 for localsearch, 1 for changeregion
        """
        if self.spatial_coord == "none":
            return 0  # Always choose localsearch

        elif self.spatial_coord == "some":
            # Choose changeregion if there's a quadrant with lots of targets
            # and neither agent nor teammate is in that quadrant
            return self._check_target_rich_quadrant_with_hysteresis(env, agent_id)

        elif self.spatial_coord == "high":
            # Always choose changeregion if both agents are in the same quadrant
            if self._agents_in_same_quadrant(env, agent_id):
                return 1  # changeregion
            else:
                return 0  # localsearch

        return 0  # Default to localsearch

    def _check_target_rich_quadrant_with_hysteresis(self, env, agent_id):
        """
        Check if there's a target-rich quadrant with hysteresis to prevent oscillation.

        Args:
            env: The environment instance
            agent_id (int): ID of the agent making the decision

        Returns:
            int: 1 for changeregion if target-rich quadrant found, 0 for localsearch otherwise
        """
        if env.config['num_aircraft'] < 2:
            return 0  # No teammate, just do localsearch

        # Get agent and teammate positions
        agent_pos = self._get_agent_quadrant(env, agent_id)
        teammate_id = 1 if agent_id == 0 else 0
        teammate_pos = self._get_agent_quadrant(env, teammate_id)

        # Get target counts per quadrant (only unknown targets)
        quadrant_target_counts = self._get_unknown_targets_per_quadrant(env)
        total_unknown_targets = sum(quadrant_target_counts.values())

        if total_unknown_targets == 0:
            self.currently_consider_target_rich = False
            return 0  # No unknown targets, do localsearch

        # Find quadrant with highest target density
        max_targets = max(quadrant_target_counts.values())
        target_rich_quadrants = [q for q, count in quadrant_target_counts.items() if count == max_targets]

        # Calculate the ratio of targets in the richest quadrant
        max_target_ratio = max_targets / total_unknown_targets

        # Apply hysteresis thresholds
        if self.currently_consider_target_rich:
            # Currently considering target-rich - use lower threshold to stay
            threshold = self.target_rich_threshold_low
        else:
            # Not currently considering target-rich - use higher threshold to switch
            threshold = self.target_rich_threshold_high

        # Update hysteresis state
        if max_target_ratio >= threshold:
            # Check if the target-rich quadrant(s) are unoccupied
            for quadrant in target_rich_quadrants:
                if quadrant != agent_pos and quadrant != teammate_pos:
                    self.currently_consider_target_rich = True
                    return 1  # changeregion to target-rich quadrant

            # Target-rich quadrant is occupied
            self.currently_consider_target_rich = False
        else:
            # Below threshold
            self.currently_consider_target_rich = False

        return 0  # No suitable target-rich quadrant, do localsearch

    def _agents_in_same_quadrant(self, env, agent_id):
        """
        Check if the agent and teammate are in the same quadrant.

        Args:
            env: The environment instance
            agent_id (int): ID of the agent making the decision

        Returns:
            bool: True if both agents are in the same quadrant
        """
        if env.config['num_aircraft'] < 2:
            return False  # No teammate

        agent_quad = self._get_agent_quadrant(env, agent_id)
        teammate_id = 1 if agent_id == 0 else 0
        teammate_quad = self._get_agent_quadrant(env, teammate_id)

        return agent_quad == teammate_quad

    def _get_agent_quadrant(self, env, agent_id):
        """
        Get the quadrant that an agent is currently in.

        Args:
            env: The environment instance
            agent_id (int): ID of the agent

        Returns:
            str: Quadrant name ("NW", "NE", "SW", "SE")
        """
        agent_x = env.agents[env.aircraft_ids[agent_id]].x
        agent_y = env.agents[env.aircraft_ids[agent_id]].y

        # Determine quadrant based on sign of coordinates
        if agent_x >= 0 and agent_y >= 0:
            return "NE"
        elif agent_x < 0 and agent_y >= 0:
            return "NW"
        elif agent_x < 0 and agent_y < 0:
            return "SW"
        else:  # agent_x >= 0 and agent_y < 0
            return "SE"

    def _get_unknown_targets_per_quadrant(self, env):
        """
        Count unknown targets in each quadrant.

        Args:
            env: The environment instance

        Returns:
            dict: Mapping of quadrant names to target counts
        """
        target_positions = env.targets[:env.config['num_targets'], 3:5]  # x,y coordinates
        target_info_levels = env.targets[:env.config['num_targets'], 2]  # info levels

        # Only count unknown targets (info_level < 1.0)
        unknown_mask = target_info_levels < 1.0
        unknown_positions = target_positions[unknown_mask]

        quadrant_counts = {"NW": 0, "NE": 0, "SW": 0, "SE": 0}

        for pos in unknown_positions:
            x, y = pos[0], pos[1]

            if x >= 0 and y >= 0:
                quadrant_counts["NE"] += 1
            elif x < 0 and y >= 0:
                quadrant_counts["NW"] += 1
            elif x < 0 and y < 0:
                quadrant_counts["SW"] += 1
            else:  # x >= 0 and y < 0
                quadrant_counts["SE"] += 1

        return quadrant_counts





# class HeuristicAgent:
#     """
#     Heuristic agent that chooses subpolicies based on risk tolerance and spatial coordination settings.
#
#     Subpolicies:
#     0 = localsearch
#     1 = changeregion
#     2 = gotothreat
#     """
#
#     def __init__(self, mode_selector="heuristic", risk_tolerance="medium", spatial_coord="some"):
#         """
#         Initialize the heuristic agent.
#
#         Args:
#             mode_selector (str): "none" to always choose localsearch, or "heuristic" for decision logic
#             risk_tolerance (str): "low", "medium", "high", or "extreme" - controls gotothreat usage
#             spatial_coord (str): "none", "some", or "high" - controls localsearch vs changeregion choice
#         """
#         self.mode_selector = mode_selector
#         self.risk_tolerance = risk_tolerance
#         self.spatial_coord = spatial_coord
#
#         # Validate configuration
#         valid_risk_levels = ["low", "medium", "high", "extreme"]
#         valid_spatial_levels = ["none", "some", "high"]
#         valid_mode_selectors = ["none", "heuristic"]
#
#         if risk_tolerance not in valid_risk_levels:
#             raise ValueError(f"risk_tolerance must be one of {valid_risk_levels}")
#         if spatial_coord not in valid_spatial_levels:
#             raise ValueError(f"spatial_coord must be one of {valid_spatial_levels}")
#         if mode_selector not in valid_mode_selectors:
#             raise ValueError(f"mode_selector must be one of {valid_mode_selectors}")
#
#
#     def choose_subpolicy(self, env, agent_id=0):
#         """
#         Choose a subpolicy based on the agent's configuration and current environment state.
#
#         Args:
#             env: The environment instance (MAISREnvVec)
#             agent_id (int): ID of the agent making the decision (default 0)
#
#         Returns:
#             int: Subpolicy choice (0=localsearch, 1=changeregion, 2=gotothreat)
#         """
#         # If mode_selector is "none", always choose localsearch
#         if self.mode_selector == "none":
#             print(f"[HeuristicAgent] mode_selector=none -> localsearch(0)")
#             return 0
#
#         # Check if we should choose gotothreat based on risk tolerance and detections
#         detections = env.num_threats_identified
#         should_go_to_threat = self._should_go_to_threat(env)
#
#         if should_go_to_threat:
#             print(f"[HeuristicAgent] detections={detections}, risk={self.risk_tolerance} -> gotothreat(2)")
#             return 2  # gotothreat
#         else:
#             # Choose between localsearch (0) and changeregion (1) based on spatial coordination
#             choice = self._choose_search_strategy(env, agent_id)
#
#             # Add debugging for the search strategy choice
#             if self.spatial_coord == "none":
#                 reason = "spatial_coord=none"
#             elif self.spatial_coord == "some":
#                 target_rich_found = choice == 1
#                 reason = f"spatial_coord=some, target_rich_quadrant={target_rich_found}"
#             elif self.spatial_coord == "high":
#                 same_quadrant = self._agents_in_same_quadrant(env, agent_id)
#                 reason = f"spatial_coord=high, same_quadrant={same_quadrant}"
#             else:
#                 reason = "default"
#
#             action_name = "localsearch" if choice == 0 else "changeregion"
#             print(
#                 f"[HeuristicAgent] detections={detections}, risk={self.risk_tolerance}, {reason} -> {action_name}({choice})")
#
#             return choice
#
#     def _should_go_to_threat(self, env):
#         """
#         Determine if the agent should choose gotothreat based on risk tolerance and current detections.
#
#         Args:
#             env: The environment instance
#
#         Returns:
#             bool: True if should choose gotothreat, False otherwise
#         """
#         detections = env.num_threats_identified
#
#         if self.risk_tolerance == "low":
#             return False  # Never choose gotothreat
#         elif self.risk_tolerance == "medium":
#             return detections == 0
#         elif self.risk_tolerance == "high":
#             return detections <= 1
#         elif self.risk_tolerance == "extreme":
#             return detections <= 2
#
#         return False
#
#     def _choose_search_strategy(self, env, agent_id):
#         """
#         Choose between localsearch and changeregion based on spatial coordination setting.
#
#         Args:
#             env: The environment instance
#             agent_id (int): ID of the agent making the decision
#
#         Returns:
#             int: 0 for localsearch, 1 for changeregion
#         """
#         if self.spatial_coord == "none":
#             return 0  # Always choose localsearch
#
#         elif self.spatial_coord == "some":
#             # Choose changeregion if there's a quadrant with lots of targets
#             # and neither agent nor teammate is in that quadrant
#             return self._check_target_rich_quadrant(env, agent_id)
#
#         elif self.spatial_coord == "high":
#             # Always choose changeregion if both agents are in the same quadrant
#             if self._agents_in_same_quadrant(env, agent_id):
#                 return 1  # changeregion
#             else:
#                 return 0  # localsearch
#
#         return 0  # Default to localsearch
#
#     def _check_target_rich_quadrant(self, env, agent_id):
#         """
#         Check if there's a target-rich quadrant that neither agent nor teammate occupies.
#
#         Args:
#             env: The environment instance
#             agent_id (int): ID of the agent making the decision
#
#         Returns:
#             int: 1 for changeregion if target-rich quadrant found, 0 for localsearch otherwise
#         """
#         if env.config['num_aircraft'] < 2:
#             return 0  # No teammate, just do localsearch
#
#         # Get agent and teammate positions
#         agent_pos = self._get_agent_quadrant(env, agent_id)
#         teammate_id = 1 if agent_id == 0 else 0
#         teammate_pos = self._get_agent_quadrant(env, teammate_id)
#
#         # Get target counts per quadrant (only unknown targets)
#         quadrant_target_counts = self._get_unknown_targets_per_quadrant(env)
#         total_unknown_targets = sum(quadrant_target_counts.values())
#
#         if total_unknown_targets == 0:
#             return 0  # No unknown targets, do localsearch
#
#         # Find quadrant with highest target density
#         max_targets = max(quadrant_target_counts.values())
#         target_rich_quadrants = [q for q, count in quadrant_target_counts.items() if count == max_targets]
#
#         # Check if the target-rich quadrant(s) have significantly more targets (>40% of total)
#         threshold = 0.4 * total_unknown_targets
#
#         for quadrant in target_rich_quadrants:
#             if (quadrant_target_counts[quadrant] >= threshold and
#                     quadrant != agent_pos and quadrant != teammate_pos):
#                 #print(f'[Teammate] Target-rich quadrant found')
#                 return 1  # changeregion to target-rich quadrant
#
#         return 0  # No suitable target-rich quadrant, do localsearch
#
#     def _agents_in_same_quadrant(self, env, agent_id):
#         """
#         Check if the agent and teammate are in the same quadrant.
#
#         Args:
#             env: The environment instance
#             agent_id (int): ID of the agent making the decision
#
#         Returns:
#             bool: True if both agents are in the same quadrant
#         """
#         if env.config['num_aircraft'] < 2:
#             return False  # No teammate
#
#         agent_quad = self._get_agent_quadrant(env, agent_id)
#         teammate_id = 1 if agent_id == 0 else 0
#         teammate_quad = self._get_agent_quadrant(env, teammate_id)
#
#         return agent_quad == teammate_quad
#
#     def _get_agent_quadrant(self, env, agent_id):
#         """
#         Get the quadrant that an agent is currently in.
#
#         Args:
#             env: The environment instance
#             agent_id (int): ID of the agent
#
#         Returns:
#             str: Quadrant name ("NW", "NE", "SW", "SE")
#         """
#         agent_x = env.agents[env.aircraft_ids[agent_id]].x
#         agent_y = env.agents[env.aircraft_ids[agent_id]].y
#
#         # Determine quadrant based on sign of coordinates
#         if agent_x >= 0 and agent_y >= 0:
#             return "NE"
#         elif agent_x < 0 and agent_y >= 0:
#             return "NW"
#         elif agent_x < 0 and agent_y < 0:
#             return "SW"
#         else:  # agent_x >= 0 and agent_y < 0
#             return "SE"
#
#     def _get_unknown_targets_per_quadrant(self, env):
#         """
#         Count unknown targets in each quadrant.
#
#         Args:
#             env: The environment instance
#
#         Returns:
#             dict: Mapping of quadrant names to target counts
#         """
#         target_positions = env.targets[:env.config['num_targets'], 3:5]  # x,y coordinates
#         target_info_levels = env.targets[:env.config['num_targets'], 2]  # info levels
#
#         # Only count unknown targets (info_level < 1.0)
#         unknown_mask = target_info_levels < 1.0
#         unknown_positions = target_positions[unknown_mask]
#
#         quadrant_counts = {"NW": 0, "NE": 0, "SW": 0, "SE": 0}
#
#         for pos in unknown_positions:
#             x, y = pos[0], pos[1]
#
#             if x >= 0 and y >= 0:
#                 quadrant_counts["NE"] += 1
#             elif x < 0 and y >= 0:
#                 quadrant_counts["NW"] += 1
#             elif x < 0 and y < 0:
#                 quadrant_counts["SW"] += 1
#             else:  # x >= 0 and y < 0
#                 quadrant_counts["SE"] += 1
#
#         return quadrant_counts

class GenericTeammatePolicy(TeammatePolicy):
    def __init__(self,
                 env,
                 local_search_policy: SubPolicy,
                 go_to_highvalue_policy: SubPolicy,
                 change_region_subpolicy: SubPolicy,
                 mode_selector_agent: HeuristicAgent = None,
                 use_collision_avoidance: bool = False,
                 ):

        self.env = env
        self.mode_selector_agent = mode_selector_agent
        self.use_collision_avoidance = use_collision_avoidance

        self.local_search_policy = local_search_policy
        self.go_to_highvalue_policy = go_to_highvalue_policy
        self.change_region_subpolicy = change_region_subpolicy

        # For human control (if needed)
        self.key_to_action = {1: 0, 2: 1, 3: 2}  # Updated to use numbers instead of pygame keys

        # Default name
        self.name = "Generic_Teammate"

    def choose_subpolicy(self, observation, current_subpolicy):
        """Choose subpolicy using the embedded HeuristicAgent"""
        if self.mode_selector_agent is None:
            # Fallback to local search if no agent provided
            return 0

        # Use the heuristic agent to make the decision
        # We need to get the environment from the wrapper context
        # For now, we'll pass None and add error handling in HeuristicAgent
        if self.env is not None:
            return self.mode_selector_agent.choose_subpolicy(self.env, agent_id=1)  # Assuming teammate is agent 1
        else:
            # If no environment available, fallback to local search
            print("[GenericTeammatePolicy] Warning: No environment available, defaulting to localsearch")
            return 0

    def reset(self):
        """Reset any internal state"""
        pass

    def near_a_threat(self):
        """Return true if near threat and need to call evade"""
        if self.env is None:
            return False

        # Implementation would depend on environment structure
        # For now, return False as placeholder
        return False


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
        #print(f'[GoToThreat] Heuristic chose action {best_action} targeting unidentified threat')
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
        #print(f'[EvadeDetection] EVADE TRIGGERED')

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
        #print(f'[EvadeDetection] threat_distance {threat_distance} <= threat_radius {buffer_radius} = {threat_distance <= buffer_radius})')
        if threat_distance <= buffer_radius:
            #print('[EvadeDetection] Inside danger zone - evading directly away from threat')

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
    """Sub-policy that searches locally for unknown targets with integrated evade logic"""

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

        # Heuristic state tracking
        self._current_target_id = None
        self._current_target_pos = None
        self._last_action = None
        self._action_repeat_count = 0
        self._max_repeat_count = 3  # Minimum steps to take in same direction
        self._target_switch_threshold = 20.0  # Distance threshold to consider switching targets

        # Evade logic state (moved from wrapper)
        self.evade_goal = None
        self.evade_goal_threshold = 30.0  # Distance threshold to consider goal "reached"
        self.last_evade_step = -1  # Track when we last used evade to detect continuous usage

        self.circumnavigation_state = {
            'active': False,
            'threat_pos': None,
            'chosen_direction': None,  # 'clockwise' or 'counterclockwise'
            'last_angle': None,
            'start_angle': None,
            'safety_distance': None
        }

    def act(self, observation, env=None, agent_id=0):
        """
        Enhanced act method that includes evade logic for threats

        Args:
            observation: The observation vector for local search
            env: Environment instance (needed for threat detection)
            agent_id: Agent ID (default 0)
        """
        # Check if we need to evade threats first
        if env is not None and self.near_threat(env, agent_id):
            #print(f"[LocalSearch] Threat detected, switching to evade mode")
            evade_action = self.compute_tangential_escape_action(env, agent_id)
            return evade_action, None

        # Normal local search behavior
        if self.model:
            action, _ = self.model.predict(observation)
            action = np.int32(action)
        else:
            try:
                action, _ = self.heuristic(observation)
            except:
                action = self.heuristic(observation)

        return action, None

    def near_threat(self, env, agent_id=0):
        """
        Check if the agent is near a threat and should automatically switch to evade mode.
        Returns True if agent is within threat radius or warning zone of any threat.
        """
        # Get agent position
        agent_pos = np.array([env.agents[env.aircraft_ids[agent_id]].x,
                              env.agents[env.aircraft_ids[agent_id]].y])

        # Check distance to all threats
        for threat_idx in range(len(env.threats)):
            threat_pos = np.array([env.threats[threat_idx, 0], env.threats[threat_idx, 1]])
            distance_to_threat = np.sqrt(np.sum((threat_pos - agent_pos) ** 2))

            threat_radius = env.config['threat_radius']
            warning_radius = threat_radius * 1.7  # 70% larger than threat radius for early warning

            # Trigger evade mode if within warning radius
            if distance_to_threat <= warning_radius:
                return True
        return False

    def compute_tangential_escape_action(self, env, agent_id=0):
        """
        Compute a direct tangential escape action when near a threat.
        Uses state persistence to maintain consistent circumnavigation direction.
        """
        # Get agent position
        agent_pos = np.array([env.agents[env.aircraft_ids[agent_id]].x,
                              env.agents[env.aircraft_ids[agent_id]].y])

        # Find nearest threat
        nearest_threat_pos = None
        min_distance = float('inf')
        nearest_threat_idx = None

        for threat_idx in range(len(env.threats)):
            threat_pos = np.array([env.threats[threat_idx, 0], env.threats[threat_idx, 1]])
            distance = np.sqrt(np.sum((threat_pos - agent_pos) ** 2))
            if distance < min_distance:
                min_distance = distance
                nearest_threat_pos = threat_pos
                nearest_threat_idx = threat_idx

        if nearest_threat_pos is None:
            return 0  # Default action if no threats

        threat_radius = env.config['threat_radius']
        safety_margin = threat_radius * 1.8  # Increased safety margin

        # Check if we need to start or continue circumnavigation
        if min_distance <= safety_margin:
            return self._circumnavigate_threat(agent_pos, nearest_threat_pos, threat_radius, env, agent_id)
        else:
            # Far enough from threat, reset circumnavigation state
            self._reset_circumnavigation_state()
            return 0

    def _circumnavigate_threat(self, agent_pos, threat_pos, threat_radius, env, agent_id=0):
        """Handle circumnavigation around a threat with state persistence"""

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

        # Vector from threat to agent
        threat_to_agent = agent_pos - threat_pos
        distance_to_threat = np.linalg.norm(threat_to_agent)

        if distance_to_threat < 1e-6:
            return 0  # Default if at threat center

        # Calculate current angle around threat
        current_angle = np.arctan2(threat_to_agent[1], threat_to_agent[0])

        # Initialize or update circumnavigation state
        if not self.circumnavigation_state['active']:
            self._initialize_circumnavigation(threat_pos, current_angle, threat_radius)

        # Check if circumnavigation is complete
        if self._is_circumnavigation_complete(current_angle, threat_pos, agent_pos, env):
            self._reset_circumnavigation_state()
            # Move toward original target
            return self._get_action_toward_nearest_target(agent_pos, directions, env)

        # Continue circumnavigation
        return self._get_circumnavigation_action(current_angle, threat_to_agent, directions)

    def _initialize_circumnavigation(self, threat_pos, start_angle, threat_radius):
        """Initialize circumnavigation state"""
        self.circumnavigation_state['active'] = True
        self.circumnavigation_state['threat_pos'] = threat_pos.copy()
        self.circumnavigation_state['start_angle'] = start_angle
        self.circumnavigation_state['last_angle'] = start_angle
        self.circumnavigation_state['safety_distance'] = threat_radius * 1.5

        # Choose direction based on which way moves us more toward targets
        # For now, default to counterclockwise
        self.circumnavigation_state['chosen_direction'] = 'counterclockwise'

        #print(f"[LocalSearch] Starting circumnavigation: direction={self.circumnavigation_state['chosen_direction']}")

    def _is_circumnavigation_complete(self, current_angle, threat_pos, agent_pos, env):
        """Check if we've gone far enough around the threat to have a clear path"""
        if not self.circumnavigation_state['active']:
            return False

        # Calculate how far we've traveled around the threat
        start_angle = self.circumnavigation_state['start_angle']
        angle_traveled = current_angle - start_angle

        # Normalize angle difference to [-π, π]
        while angle_traveled > np.pi:
            angle_traveled -= 2 * np.pi
        while angle_traveled < -np.pi:
            angle_traveled += 2 * np.pi

        # Check if we've gone at least 90 degrees around
        min_angle_traveled = np.pi / 2  # 90 degrees

        if self.circumnavigation_state['chosen_direction'] == 'counterclockwise':
            sufficient_travel = angle_traveled >= min_angle_traveled
        else:  # clockwise
            sufficient_travel = angle_traveled <= -min_angle_traveled

        if sufficient_travel:
            # Also check if we now have a clear line to targets
            return self._has_clear_path_to_targets(agent_pos, threat_pos, env)

        return False

    def _has_clear_path_to_targets(self, agent_pos, threat_pos, env):
        """Check if there's a clear path from current position to nearest unknown target"""
        # Get unknown target positions
        target_positions = env.targets[:env.config['num_targets'], 3:5]
        target_info_levels = env.targets[:env.config['num_targets'], 2]
        unknown_mask = target_info_levels < 1.0

        if not np.any(unknown_mask):
            return True  # No targets left, circumnavigation complete

        unknown_positions = target_positions[unknown_mask]
        distances = np.sqrt(np.sum((unknown_positions - agent_pos) ** 2, axis=1))
        nearest_target_pos = unknown_positions[np.argmin(distances)]

        # Check if path to nearest target intersects threat
        return self._path_clear_of_threat(agent_pos, nearest_target_pos, threat_pos, env)

    def _path_clear_of_threat(self, start_pos, end_pos, threat_pos, env):
        """Check if straight line path from start to end clears the threat"""
        threat_radius = env.config['threat_radius'] * 1.2  # Safety buffer

        # Vector from start to end
        path_vector = end_pos - start_pos
        path_length = np.linalg.norm(path_vector)

        if path_length < 1e-6:
            return True

        path_unit = path_vector / path_length

        # Vector from start to threat
        start_to_threat = threat_pos - start_pos

        # Project threat onto path
        projection_length = np.dot(start_to_threat, path_unit)

        # Only check collision if projection is within the path segment
        if 0 <= projection_length <= path_length:
            closest_point_on_path = start_pos + projection_length * path_unit
            distance_to_threat = np.linalg.norm(threat_pos - closest_point_on_path)
            return distance_to_threat > threat_radius

        return True  # Threat is not along the path

    def _get_circumnavigation_action(self, current_angle, threat_to_agent, directions):
        """Get action to continue circumnavigation"""
        distance_to_threat = np.linalg.norm(threat_to_agent)

        if distance_to_threat > 0:
            # Calculate tangent direction
            threat_unit = threat_to_agent / distance_to_threat

            if self.circumnavigation_state['chosen_direction'] == 'counterclockwise':
                tangent_direction = np.array([-threat_unit[1], threat_unit[0]])  # +90 degrees
            else:  # clockwise
                tangent_direction = np.array([threat_unit[1], -threat_unit[0]])  # -90 degrees

            # Add slight outward bias to maintain safe distance
            outward_direction = threat_unit  # Away from threat
            bias_strength = 0.2

            combined_direction = tangent_direction * (1 - bias_strength) + outward_direction * bias_strength
            combined_direction = combined_direction / np.linalg.norm(combined_direction)

            # Find best matching action
            dot_products = np.dot(directions, combined_direction)
            best_action = np.argmax(dot_products)

            return np.int32(best_action)

        return 0

    def _get_action_toward_nearest_target(self, agent_pos, directions, env):
        """Get action to move toward nearest unknown target after circumnavigation"""
        # Get unknown target positions
        target_positions = env.targets[:env.config['num_targets'], 3:5]
        target_info_levels = env.targets[:env.config['num_targets'], 2]
        unknown_mask = target_info_levels < 1.0

        if not np.any(unknown_mask):
            return 0  # No targets left

        unknown_positions = target_positions[unknown_mask]
        distances = np.sqrt(np.sum((unknown_positions - agent_pos) ** 2, axis=1))
        nearest_target_pos = unknown_positions[np.argmin(distances)]

        # Direction to nearest target
        target_vector = nearest_target_pos - agent_pos
        target_distance = np.linalg.norm(target_vector)

        if target_distance > 0:
            target_direction = target_vector / target_distance
            dot_products = np.dot(directions, target_direction)
            best_action = np.argmax(dot_products)
            return np.int32(best_action)

        return 0

    def _reset_circumnavigation_state(self):
        """Reset circumnavigation state"""
        self.circumnavigation_state = {
            'active': False,
            'threat_pos': None,
            'chosen_direction': None,
            'last_angle': None,
            'start_angle': None,
            'safety_distance': None
        }

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
            return np.int32(0), None

        target_vector_x = obs[0]
        target_vector_y = obs[1]

        # Check if there's a valid target (non-zero vector)
        if target_vector_x == 0.0 and target_vector_y == 0.0:
            # No targets or at target location
            self.reset_heuristic_state()
            return np.int32(0), None

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
            return np.int32(self._last_action if self._last_action is not None else 0), None

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
        return np.int32(best_action), None

    def reset_heuristic_state(self):
        """Reset the global state for the heuristic policy."""
        self._current_target_id = None
        self._current_target_pos = None
        self._last_action = None
        self._action_repeat_count = 0

    def reset_evade_state(self):
        """Reset evade-related state"""
        self.evade_goal = None
        self.last_evade_step = -1
        self._reset_circumnavigation_state()


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
            #print(f'[ChangeRegions] Reached target region {self.target_region}, selecting new region')
            self.target_region = None  # Force new selection
            self.steps_since_update = self.update_rate  # Force immediate update

        # Select new target region if needed
        if self.target_region is None or self.steps_since_update >= self.update_rate:
            self.steps_since_update = 0
            if self.model:
                self.target_region = self.model.predict(observation)
            else:
                self.target_region = self.heuristic(observation)
            #print(f'[ChangeRegions] Selected new target region: {self.target_region}')

        # Set waypoint directly to center of new region
        action = self._get_region_center(self.target_region)
        self.steps_since_update += 1
        #print(f'[ChangeRegion.act] Chose action {action}')
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
        """
        Improved heuristic to choose a region with most targets that doesn't contain teammate.
        If we already have a target region, stick with it until we've searched it thoroughly.
        """
        obs = np.array(observation)

        # Each region has 3 values: [target_ratio, agent_distance, teammate_distance]
        regions_info = []

        for region_id in range(4):  # 4 regions: NW, NE, SW, SE
            base_idx = region_id * 3
            target_ratio = obs[base_idx]  # Ratio of unknown targets in this region
            agent_distance = obs[base_idx + 1]  # Agent distance to region center (normalized)
            teammate_distance = obs[base_idx + 2]  # Teammate distance to region center (normalized)

            regions_info.append({
                'region_id': region_id,
                'target_ratio': target_ratio,
                'agent_distance': agent_distance,
                'teammate_distance': teammate_distance
            })

        # If we already have a target region and haven't finished searching it, keep it
        if hasattr(self, 'target_region') and self.target_region is not None:
            current_region_info = regions_info[self.target_region]

            # Only switch if current region has no targets left OR teammate entered our region
            teammate_in_current_region = current_region_info['teammate_distance'] < 0.25
            no_targets_in_current = current_region_info['target_ratio'] < 0.1  # Less than 10% of targets

            if not (teammate_in_current_region or no_targets_in_current):
                #print(f"Continuing with current region {self.target_region}")
                return self.target_region

        # Need to select a new region
        # Define threshold for "teammate being in a region" (normalized distance)
        region_threshold = 0.25

        # Filter out regions where teammate is currently located
        # Note: We don't exclude where agent is, since agent needs to be able to enter regions
        available_regions = []
        for region_info in regions_info:
            teammate_in_region = region_info['teammate_distance'] < region_threshold

            # Only exclude regions where teammate is present
            if not teammate_in_region:
                available_regions.append(region_info)

        # If no regions are available (teammate coverage is too broad), fall back to all regions
        if not available_regions:
            print("[ChangeRegions] Warning: Teammate covers all regions, considering all regions")
            available_regions = regions_info

        # Sort available regions by target density (highest ratio first)
        available_regions.sort(key=lambda x: x['target_ratio'], reverse=True)

        # Choose the region with highest target density
        target_region = available_regions[0]['region_id']

        #print(f"[ChangeRegions] Selected NEW region {target_region} with target ratio {available_regions[0]['target_ratio']:.2f}")

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
            3: np.array([0.5, -0.5])}  # SE
        return centers.get(region_id, np.array([0.0, 0.0]))