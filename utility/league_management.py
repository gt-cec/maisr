import glob
import os
import random
from multiprocessing.managers import Value

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

    def __init__(self, league_type, balance_method, selfplay_checkpoint_dir, pretrained_teammate_dir,
                 subpolicies=None, overfit_test = None, current_model = None, fcp_ratio=1.0):
        """
        Initialize teammate manager with specified league type and balance method.

        Args:
            league_type (str): "baseline", "vanilla", or "strategy_diverse"
            subpolicies (dict): Dictionary containing subpolicy instances
                Expected keys: 'local_search', 'change_region', 'go_to_threat'
            balance_method (str): "uniform" for current random sampling, "complex" for advanced balancing
        """
        self.league_type = league_type
        self.subpolicies = subpolicies or {}
        self.balance_method = balance_method
        self.current_teammate = None
        self.current_model = current_model

        self.episode_count = 0
        self.overfit_test = overfit_test
        self.selfplay_checkpoint_dir = selfplay_checkpoint_dir
        self.pretrained_teammate_dir = pretrained_teammate_dir
        self.fcp_ratio = fcp_ratio

        # Validate league type
        valid_league_types = ["baseline", "vanilla", "strategy_diverse", "selfplay", 'fcp','mixed50','mixed25','mixed75']
        if league_type not in valid_league_types:
            raise ValueError(f"league_type must be one of {valid_league_types}")

        # Validate balance method
        valid_balance_methods = ["uniform", "complex"]
        if balance_method not in valid_balance_methods:
            raise ValueError(f"balance_method must be one of {valid_balance_methods}")

        # Validate overfit_test parameter
        valid_overfit_tests = [None, "low_risk", "high_risk", "no_coord", "yes_coord", "greedy_planning", "cluster_planning", "noisy_actions","very_noisy_actions", "stable_actions", "max_greedy", "periodic_stopping", "slow_decisions"]
        if overfit_test not in valid_overfit_tests:
            raise ValueError(f"overfit_test must be one of {valid_overfit_tests}")

        self.mode_selector_options = {
            'baseline': ["none"],
            'vanilla': ["none", "heuristic"],
            'strategy_diverse': ["heuristic"],
            'strategy_diverse_nohighrisk': ["heuristic"],
            'strategy_diverse_nolowrisk': ["heuristic"],
            'strategy_diverse_nonoisy': ["heuristic"]
        }
        self.risk_tolerance_options = {
            'baseline': ["none"],
            "vanilla": ["none"],
            "strategy_diverse": ["low", "medium", "high", "max_greedy"],
            "strategy_diverse_nohighrisk": ["low", "medium", "max_greedy"],
            "strategy_diverse_nolowrisk": ["high", "medium", "max_greedy"],
            "strategy_diverse_nonoisy": ["low", "medium", "high", "max_greedy"]
        }
        self.spatial_coord_options = {
            'baseline': [False],
            "vanilla": [False],
            "strategy_diverse": [False, True],
            "strategy_diverse_nohighrisk": [False, True],
            "strategy_diverse_nolowrisk": [False, True],
            "strategy_diverse_nonoisy": [False, True]
        }
        self.action_stability_options = {
            'baseline': ["stable"],
            'vanilla': ["stable"],
            'strategy_diverse': ["stable", "noisy", "very_noisy", "periodic_stopping"],
            'strategy_diverse_nohighrisk': ["stable", "noisy", "very_noisy", "periodic_stopping"],
            'strategy_diverse_nolowrisk': ["stable", "noisy", "very_noisy", "periodic_stopping"],
            'strategy_diverse_nonoisy': ["stable"]
        }
        self.planning_horizon_options = {
            'baseline': ["greedy"],
            'vanilla': ["greedy"],
            'strategy_diverse': ["greedy", "clusters"],
            'strategy_diverse_nohighrisk': ["greedy", "clusters"],
            'strategy_diverse_nolowrisk': ["greedy", "clusters"],
            'strategy_diverse_nonoisy': ["greedy", "clusters"]
        }

        self.decision_speed_options = {
            'baseline': ["fast"],
            'vanilla': ["fast"],
            'strategy_diverse': ["fast", "slow"],
            'strategy_diverse_nohighrisk': ["fast", "slow"],
            'strategy_diverse_nolowrisk': ["fast", "slow"],
            'strategy_diverse_nonoisy': ["fast", "slow"]
        }

        print(f"\nTeammateManager initialized: \n        league_type: {league_type}\n        balance_method: {balance_method}\n        selfplay_checkpoint_dir: {selfplay_checkpoint_dir}")

    def select_random_teammate(self):
        """Select a teammate based on league type and balance method configuration"""

        # Handle overfit test cases
        if self.overfit_test is not None:
            return self._create_overfit_test_teammate()

        elif self.balance_method == "uniform":
            return self._select_uniform_teammate()

        else:
            raise ValueError(f"Unknown balance_method: {self.balance_method}")


    def _select_uniform_teammate(self):
        """Original uniform random selection method"""

        if self.league_type == "selfplay":
            return self._create_selfplay_teammate()
        elif self.league_type == "baseline":
            return self._create_baseline_teammate()

        elif self.league_type == "vanilla":
            # 25% selfplay, 25% heuristic , 50% RL
            prob = random.random()
            if prob < 0.25:
                print(f'[_select_uniform_teammate] Creating selfplay teammate')
                try:
                    return self._create_selfplay_teammate()
                except:
                    print('Could not create selfplay, creating vanilla heuristic')
                    return self._create_vanilla_heuristic_teammate()
            elif prob < 0.50:
                print(f'[_select_uniform_teammate] Creating selfplay teammate')
                return self._create_vanilla_heuristic_teammate()
            else:
                print(f'[_select_uniform_teammate] Creating pretrained RL teammate')
                return self._create_pretrained_rl_teammate()

        elif self.league_type == "strategy_diverse":
            return self._create_strategy_diverse_heuristic_teammate() # TEMP

            prob = random.random()
            if prob < 0.25:
                print(f'[_select_uniform_teammate] Creating selfplay teammate')
                try:
                    return self._create_selfplay_teammate()
                except:
                    print('Could not create selfplay, creating diverse heuristic')
                    return self._create_strategy_diverse_heuristic_teammate()
            elif prob < 0.75:
                print(f'[_select_uniform_teammate] Creating selfplay teammate')
                return self._create_strategy_diverse_heuristic_teammate()
            else:
                print(f'[_select_uniform_teammate] Creating pretrained RL teammate')
                return self._create_pretrained_rl_teammate()

        elif self.league_type in ['mixed50', 'mixed25', 'mixed75']:
            ratio = int(self.league_type[-2:])/100
            if random.random() < ratio:
                return self._create_selfplay_teammate()
                # if random.random() < 0.25:
                #     print(f'[Teammate Manager - {self.league_type}] Creating selfplay teammate')
                #     return self._create_selfplay_teammate()
                # else:
                #     print(f'[Teammate Manager - {self.league_type}] Creating pretrained RL teammate')
                #     return self._create_pretrained_rl_teammate()
            else:
                print(f'[Teammate Manager - {self.league_type}] Creating strategy heuristic teammate')
                return self._create_strategy_diverse_heuristic_teammate()

        elif self.league_type == 'fcp':
            if random.random() < 0.75:
                return self._create_pretrained_rl_teammate()
            else:
                return self._create_selfplay_teammate()

        else:
            raise ValueError(f"Unknown league_type: {self.league_type}")


    def _create_selfplay_teammate(self):
        """Create self-play teammate by loading a previous checkpoint of the current agent"""
        return self._create_rl_teammate("selfplay")

    def _create_pretrained_rl_teammate(self):
        """Create pretrained RL mode selector teammate"""
        return self._create_rl_teammate("pretrained")

    def _create_rl_teammate(self, teammate_type):
        """
        Create RL teammate by loading a checkpoint (either selfplay or pretrained)

        Args:
            teammate_type (str): "selfplay" or "pretrained"
        """
        import os
        import glob
        import random
        from stable_baselines3 import PPO

        # Set directory and fallback name based on type
        if teammate_type == "selfplay":
            checkpoint_dir = self.selfplay_checkpoint_dir
            fallback_prefix = "SelfPlay"
            selection_strategy_enabled = True  # Only selfplay uses strategy selection
        elif teammate_type == "pretrained":
            checkpoint_dir = self.pretrained_teammate_dir
            fallback_prefix = "Pretrained"
            selection_strategy_enabled = False  # Pretrained uses simple random selection
        else:
            raise ValueError(f"teammate_type must be 'selfplay' or 'pretrained', got {teammate_type}")


        # Validation checks
        if checkpoint_dir is None:
            print(f"Warning: No {teammate_type}_checkpoint_dir specified, falling back to baseline teammate")
            raise ValueError

        if not os.path.exists(checkpoint_dir):
            print(f"Warning: {teammate_type.title()} directory {checkpoint_dir} does not exist, falling back to baseline")
            raise ValueError

        # Find all checkpoint files
        checkpoint_patterns = [
            os.path.join(checkpoint_dir, "*.zip"),
            os.path.join(checkpoint_dir, "**/*.zip"),
            os.path.join(checkpoint_dir, "checkpoint_*.zip"),
            os.path.join(checkpoint_dir, "model_*.zip"),
        ]

        normstats_patterns = [
            # Pattern: strat4L_0723_1646_seed69_checkpoint_vecnormalize_2496_steps.pkl
            os.path.join(checkpoint_dir, "*.pkl"),
            os.path.join(checkpoint_dir, "**/*.pkl"),
            os.path.join(checkpoint_dir, "*_steps.pkl"),
        ]

        all_checkpoints = []
        for pattern in checkpoint_patterns:
            all_checkpoints.extend(glob.glob(pattern, recursive=True))

        all_normstats = []
        for pattern in normstats_patterns:
            all_normstats.extend(glob.glob(pattern, recursive=True))

        # Remove duplicates and sort by modification time (newest first)
        all_checkpoints = list(set(all_checkpoints))
        if not all_checkpoints:
            if teammate_type == "selfplay":
                return self._create_current_teammate_copy()
            else:
                print(f"Warning: No {teammate_type} files found in {checkpoint_dir}, falling back to baseline")
                teammate = self._create_baseline_teammate()
                teammate.name = f"{fallback_prefix}_NoCheckpoints_Fallback"
                return teammate

        all_checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        all_normstats.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        # Select checkpoint based on type
        if selection_strategy_enabled:  # Selfplay strategy
            selection_strategy = random.random()
            #selection_strategy = 0.4 # TODO temp force

            if selection_strategy < 0.5:
                # Select from most recent 3 checkpoints
                #recent_checkpoints = all_checkpoints[:min(3, len(all_checkpoints))]
                recent_checkpoints = all_checkpoints[:min(2, len(all_checkpoints))]
                selected_checkpoint = random.choice(recent_checkpoints)
                strategy_name = "Recent3"
            elif selection_strategy < 0.8:
                # Select from recent 25% of checkpoints
                recent_count = max(1, len(all_checkpoints) // 4)
                recent_checkpoints = all_checkpoints[:recent_count]
                selected_checkpoint = random.choice(recent_checkpoints)
                strategy_name = "Recent25pct"
            else:
                # Select randomly from any checkpoint
                selected_checkpoint = random.choice(all_checkpoints)
                strategy_name = "Random"
        else:  # Pretrained simple selection
            selected_checkpoint = random.choice(all_checkpoints)
            strategy_name = "Random"

        try:
            # Load the selected checkpoint
            print(f"\nLoading {teammate_type} checkpoint: {os.path.basename(selected_checkpoint)}" + (f" (strategy: {strategy_name})" if selection_strategy_enabled else ""))
            model = PPO.load(selected_checkpoint)

            # Find the corresponding norm stats .pkl file for the selected checkpoint. If not found, set to None.
            import re
            checkpoint_filename = os.path.basename(selected_checkpoint)
            norm_stats_path = None

            # Strip off the model suffix
            # Extract step count to match against _vecnormalize_{steps}_steps.pkl
            step_match = re.search(r'checkpoint_(\d+)_steps(?:_model)?\.zip$', checkpoint_filename)
            norm_stats_path = None

            if step_match:
                step_count = step_match.group(1)
                checkpoint_prefix = checkpoint_filename.split("checkpoint_")[0].rstrip("_")

                # Construct expected vecnormalize filename
                expected_vecnorm_filename = f"{checkpoint_prefix}_checkpoint_vecnormalize_{step_count}_steps.pkl"

                for stat_path in all_normstats:
                    if os.path.basename(stat_path) == expected_vecnorm_filename:
                        norm_stats_path = stat_path
                        break

            # Create RL teammate policy using the loaded model
            rl_teammate = RLTeammatePolicy(
                model=model,
                env=None,
                local_search_policy=self.subpolicies.get('local_search'),
                go_to_highvalue_policy=self.subpolicies.get('go_to_threat'),
                change_region_subpolicy=self.subpolicies.get('change_region'),
                norm_stats_path=norm_stats_path
            )

            # Pass current normalization stats if available
            if rl_teammate.norm_stats is None and hasattr(self, 'obs_rms') and hasattr(self, 'ret_rms'):
                print('No teammate normstats file found. Using live stats')
                print(f'Debug: norm_stats_path = {norm_stats_path}')
                rl_teammate.set_live_normalization_stats(self.obs_rms, self.ret_rms)

            # Set name based on type
            checkpoint_name = os.path.splitext(os.path.basename(selected_checkpoint))[0]
            if teammate_type == "selfplay":
                rl_teammate.name = f"SelfPlay_{strategy_name}_{checkpoint_name}"
            else:
                rl_teammate.name = f"Pretrained_{checkpoint_name}"

            self.current_teammate = rl_teammate
            return rl_teammate

        except Exception as e:
            print(f"Error loading {teammate_type} checkpoint {selected_checkpoint}: {e}")
            print("Falling back to baseline teammate")
            teammate = self._create_baseline_teammate()
            teammate.name = f"{fallback_prefix}_LoadError_Fallback"
            return teammate


    def _create_overfit_test_teammate(self):
        """Create teammate with specific configuration for overfit testing"""

        if self.overfit_test == "low_risk":
            #print(f'[_create_overfit_test_teammate] Creating low risk teammate')
            mode_selector = "heuristic"
            risk_tolerance = "low"
            spatial_coord = False  # Default spatial coordination
            action_stability = "stable"  # Default for overfit tests
            planning_horizon = "greedy"

        elif self.overfit_test == "high_risk":
            #print(f'[_create_overfit_test_teammate] Creating high risk teammate')
            mode_selector = "heuristic"
            risk_tolerance = "high"
            spatial_coord = False  # Default spatial coordination
            action_stability = "stable"  # Default for overfit tests
            planning_horizon = "greedy"

        elif self.overfit_test == "max_greedy":
            # print(f'[_create_overfit_test_teammate] Creating high risk teammate')
            mode_selector = "heuristic"
            risk_tolerance = "max_greedy"
            spatial_coord = False  # Default spatial coordination
            action_stability = "stable"  # Default for overfit tests
            planning_horizon = "greedy"

        ###########################

        elif self.overfit_test == "slow_decisions":
            # print(f'[_create_overfit_test_teammate] Creating high risk teammate')
            mode_selector = "heuristic"
            risk_tolerance = "low"
            spatial_coord = False  # Default spatial coordination
            action_stability = "stable"  # Default for overfit tests
            planning_horizon = "greedy"
            decision_speed = 'slow'


        elif self.overfit_test == "no_coord":
            #print(f'[_create_overfit_test_teammate] Creating no spatial coordination teammate')
            mode_selector = "heuristic"
            risk_tolerance = "low"  # Default risk tolerance
            spatial_coord = False
            action_stability = "stable"  # Default for overfit tests
            planning_horizon = "greedy"
            decision_speed = 'fast'

        elif self.overfit_test == "yes_coord":
            #print(f'[_create_overfit_test_teammate] Creating high spatial coordination teammate')
            mode_selector = "heuristic"
            risk_tolerance = "low"  # Default risk tolerance
            spatial_coord = True
            action_stability = "stable"  # Default for overfit tests
            planning_horizon = "greedy"
            decision_speed = 'fast'

        elif self.overfit_test == "greedy_planning":
            # print(f'[_create_overfit_test_teammate] Creating high spatial coordination teammate')
            mode_selector = "heuristic"
            risk_tolerance = "low"  # Default risk tolerance
            spatial_coord = False
            action_stability = "stable"  # Default for overfit tests
            planning_horizon = "greedy"
            decision_speed = 'fast'

        elif self.overfit_test == "cluster_planning":
            # print(f'[_create_overfit_test_teammate] Creating high spatial coordination teammate')
            mode_selector = "heuristic"
            risk_tolerance = "low"  # Default risk tolerance
            spatial_coord = False
            action_stability = "stable"  # Default for overfit tests
            planning_horizon = 'clusters'
            decision_speed = 'fast'

        elif self.overfit_test == 'noisy_actions':
            #print(f'[_create_overfit_test_teammate] Creating noisy action teammate')
            mode_selector = "heuristic"
            risk_tolerance = "low"  # Default risk tolerance
            spatial_coord = False
            planning_horizon = "greedy"
            action_stability = "noisy"
            decision_speed = 'fast'

        elif self.overfit_test == 'periodic_stopping':
            #print(f'[_create_overfit_test_teammate] Creating noisy action teammate')
            mode_selector = "heuristic"
            risk_tolerance = "low"  # Default risk tolerance
            spatial_coord = False
            planning_horizon = "greedy"
            action_stability = "periodic_stopping"
            decision_speed = 'fast'


        elif self.overfit_test == 'very_noisy_actions':
            #print(f'[_create_overfit_test_teammate] Creating noisy action teammate')
            mode_selector = "heuristic"
            risk_tolerance = "low"  # Default risk tolerance
            spatial_coord = False
            planning_horizon = "greedy"
            action_stability = "very_noisy"
            decision_speed = 'fast'

        elif self.overfit_test == 'stable_actions':
            #print(f'[_create_overfit_test_teammate] Creating stable action teammate')
            mode_selector = "heuristic"
            risk_tolerance = "low"  # Default risk tolerance
            spatial_coord = False
            planning_horizon = "greedy"
            action_stability = "stable"  # Default for overfit tests
            decision_speed = 'fast'

        else:
            raise ValueError(f"Unknown overfit_test value: {self.overfit_test}")

        target_search_policy = TargetSearchLocalTSP(
            search_radius=1000,
            spatial_coord=spatial_coord,
            model_path=None,
            norm_stats_filepath=None,
            search_method=planning_horizon
        )

        # if planning_horizon == 'cluster_planning':
        #     target_search_policy = TargetSearchLocalTSP(
        #         search_radius=1000,
        #         spatial_coord=spatial_coord,
        #         model_path=None,
        #         norm_stats_filepath=None,
        #         search_method='clusters'
        #     )
        # elif planning_horizon == 'greedy_planning':
        #     target_search_policy = TargetSearchLocalTSP(
        #         search_radius=1000,
        #         spatial_coord=spatial_coord,
        #         model_path=None,
        #         norm_stats_filepath=None,
        #         search_method='greedy'
        #     )

        # elif planning_horizon == 'short':
        #     target_search_policy = self.subpolicies.get('local_search')
        # elif planning_horizon == 'medium':
        #     if spatial_coord == 'true':
        #         target_search_policy = self.subpolicies.get('local_tsp_yescoord')
        #     else:
        #         target_search_policy = self.subpolicies.get('local_tsp_nocoord')
        # elif planning_horizon == 'long':
        #     if spatial_coord == 'true':
        #         target_search_policy = self.subpolicies.get('global_tsp_yescoord')
        #     else:
        #         target_search_policy = self.subpolicies.get('global_tsp_nocoord')
        # else:
        #     raise ValueError(f"Unknown planning_horizon value: {planning_horizon}")

        heuristic_agent = HeuristicAgent(
            mode_selector=mode_selector,
            risk_tolerance=risk_tolerance,
            spatial_coord=spatial_coord,
        )

        teammate = GenericTeammatePolicy(
            env=None,
            local_search_policy=target_search_policy,
            go_to_highvalue_policy=self.subpolicies.get('go_to_threat'),
            change_region_subpolicy=self.subpolicies.get('change_region'),
            mode_selector_agent=heuristic_agent,
            use_collision_avoidance=False,
            action_stability=action_stability,
            decision_speed=decision_speed
        )

        teammate.name = f"OverfitTest_{self.overfit_test}_{mode_selector}-MS_{risk_tolerance}-risk_{planning_horizon}-planninghorizon_{action_stability}-stability"
        self.current_teammate = teammate
        return teammate

    def get_current_teammate_checkpoint_info(self):
        """Get checkpoint information for the current teammate"""
        if (self.current_teammate and
                hasattr(self.current_teammate, 'name') and
                'SelfPlay' in self.current_teammate.name):
            return self.current_teammate.name
        return None

    def _find_normalization_stats(self, checkpoint_path, teammate_type):
        """
        Find the normalization stats file corresponding to a checkpoint
        """
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]

        # Common patterns for normalization stats files
        possible_patterns = [
            os.path.join(checkpoint_dir, f"{checkpoint_name}_norm_stats.npy"), # Same name as checkpoint but with different extension
            os.path.join(checkpoint_dir, f"{checkpoint_name}local_search_norm_stats.npy"),

            os.path.join(os.path.dirname(checkpoint_dir), f"{checkpoint_name}_norm_stats.npy"), # Look in parent directory
            os.path.join(os.path.dirname(checkpoint_dir), f"{checkpoint_name}local_search_norm_stats.npy"),

            os.path.join(checkpoint_dir, "norm_stats.npy"), # Look for generic norm stats in the same directory
            os.path.join(checkpoint_dir, "local_search_norm_stats.npy"),

            os.path.join(os.path.dirname(checkpoint_dir), "norm_stats.npy"), # Look in parent directory for generic files
            os.path.join(os.path.dirname(checkpoint_dir), "local_search_norm_stats.npy"),
        ]

        for pattern in possible_patterns: # Try each pattern and return the first one that exists
            if os.path.exists(pattern):
                print(f"[TeammateManager] Found normalization stats for {teammate_type}: {pattern}")
                return pattern

        print(f"[TeammateManager] No normalization stats found for {teammate_type} checkpoint {checkpoint_path}")
        return None

    def set_normalization_stats(self, obs_rms, ret_rms):
        """Store normalization stats to pass to teammates"""
        self.obs_rms = obs_rms
        self.ret_rms = ret_rms

        if (hasattr(self, 'current_teammate') and
                self.current_teammate is not None and
                hasattr(self.current_teammate, 'set_live_normalization_stats')):
            #print(f"[TeammateManager] Updating existing teammate {self.current_teammate.name} with new stats")
            self.current_teammate.set_live_normalization_stats(obs_rms, ret_rms)

    def set_current_model(self, model):
        """Update the reference to the current model during training"""
        #print(f'%%%%%%%%%% TEAMMATE MANAGER current model set to {model}')
        self.current_model = model


    def _create_current_teammate_copy(self):
        """Create a copy of the current teammate for self-play when no checkpoints exist yet."""
        print("[TeammateManager] No selfplay checkpoints found, creating copy of current teammate")

        #print(f'Teammate manager current model is {self.current_model}')

        if self.current_model is not None:
            #print("%%%%%%%%%%  [TeammateManager] Using current model for teammate copy")
            current_teammate = RLTeammatePolicy(
                model=self.current_model,  # Use the current model directly
                env=None,
                local_search_policy=self.subpolicies.get('local_search'),
                go_to_highvalue_policy=self.subpolicies.get('go_to_threat'),
                change_region_subpolicy=self.subpolicies.get('change_region'),
            )

            # Pass live normalization stats if available
            if hasattr(self, 'obs_rms') and hasattr(self, 'ret_rms'):
                current_teammate.set_live_normalization_stats(self.obs_rms, self.ret_rms)

            current_teammate.name = "SelfPlay_CurrentModelCopy"
            self.current_teammate = current_teammate
            return current_teammate
        else:
            print("[TeammateManager] No current model available, creating baseline teammate")
            teammate = self._create_baseline_teammate()
            teammate.name = "SelfPlay_BaselineCopy_NoCurrentModel"
            self.current_teammate = teammate
            return teammate


    def _create_baseline_teammate(self):
        """Create baseline teammate: always heuristic with conservative settings"""
        mode_selector = random.choice(self.mode_selector_options['baseline'])
        risk_tolerance = random.choice(self.risk_tolerance_options['baseline'])
        spatial_coord = random.choice(self.spatial_coord_options['baseline'])
        planning_horizon = random.choice(self.planning_horizon_options['baseline'])
        action_stability = random.choice(self.action_stability_options['baseline'])
        decision_speed = random.choice(self.decision_speed_options['baseline'])

        target_search_policy = TargetSearchLocalTSP(
            search_radius=1000,
            spatial_coord=spatial_coord,
            model_path=None,
            norm_stats_filepath=None,
            search_method=planning_horizon
        )

        heuristic_agent = HeuristicAgent(
            mode_selector=mode_selector,
            risk_tolerance=risk_tolerance,
            spatial_coord=spatial_coord,
        )

        teammate = GenericTeammatePolicy(
            env=None,
            local_search_policy=target_search_policy,
            go_to_highvalue_policy=self.subpolicies.get('go_to_threat'),
            change_region_subpolicy=self.subpolicies.get('change_region'),
            mode_selector_agent=heuristic_agent,
            use_collision_avoidance=False,
            action_stability=action_stability,
            decision_speed=decision_speed
        )

        teammate.name = "Baseline_Greedy_noMS_lowrisk_nospatialcoord"
        self.current_teammate = teammate
        return teammate

    def _create_vanilla_heuristic_teammate(self):
        """Create vanilla teammate: varied mode_selector, conservative spatial/risk settings"""
        # Randomly sample mode_selector
        mode_selector = random.choice(self.mode_selector_options['vanilla'])
        risk_tolerance = random.choice(self.risk_tolerance_options['vanilla'])
        spatial_coord = random.choice(self.spatial_coord_options['vanilla'])
        planning_horizon = random.choice(self.planning_horizon_options['vanilla'])
        action_stability = random.choice(self.action_stability_options['vanilla'])
        decision_speed = random.choice(self.decision_speed_options['vanilla'])

        target_search_policy = TargetSearchLocalTSP(
            search_radius=1000,
            spatial_coord=spatial_coord,
            model_path=None,
            norm_stats_filepath=None,
            search_method=planning_horizon
        )

        heuristic_agent = HeuristicAgent(
            mode_selector=mode_selector,
            risk_tolerance=risk_tolerance,
            spatial_coord=spatial_coord,
        )

        teammate = GenericTeammatePolicy(
            env=None,
            local_search_policy=target_search_policy,
            go_to_highvalue_policy=self.subpolicies.get('go_to_threat'),
            change_region_subpolicy=self.subpolicies.get('change_region'),
            mode_selector_agent=heuristic_agent,
            use_collision_avoidance=False,
            action_stability=action_stability,
            decision_speed=decision_speed
        )

        teammate.name = f"Vanilla_{mode_selector}MS_norisk_nospatialcoord"
        self.current_teammate = teammate
        return teammate

    def _create_strategy_diverse_heuristic_teammate(self):
        """Create strategy diverse teammate: all parameters randomly sampled"""
        # Randomly sample all parameters
        mode_selector = random.choice(self.mode_selector_options['strategy_diverse'])
        risk_tolerance = random.choice(self.risk_tolerance_options['strategy_diverse'])
        spatial_coord = random.choice(self.spatial_coord_options['strategy_diverse'])
        action_stability = random.choice(self.action_stability_options['strategy_diverse'])
        planning_horizon = random.choice(self.planning_horizon_options['strategy_diverse'])
        decision_speed = random.choice(self.decision_speed_options['strategy_diverse'])

        target_search_policy = TargetSearchLocalTSP(
            search_radius=1000,
            spatial_coord=spatial_coord,
            model_path=None,
            norm_stats_filepath=None,
            search_method=planning_horizon
        )

        heuristic_agent = HeuristicAgent(
            mode_selector=mode_selector,
            risk_tolerance=risk_tolerance,
            spatial_coord=spatial_coord,
        )

        teammate = GenericTeammatePolicy(
            env=None,
            local_search_policy=target_search_policy,
            go_to_highvalue_policy=self.subpolicies.get('go_to_threat'),
            change_region_subpolicy=self.subpolicies.get('change_region'),
            mode_selector_agent=heuristic_agent,
            use_collision_avoidance=False,
            action_stability=action_stability,
            decision_speed=decision_speed
        )

        teammate.name = f"Diverse_{mode_selector}MS_{risk_tolerance}risk_{spatial_coord}spatial_{action_stability}stability"
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


class RLTeammatePolicy(TeammatePolicy):
    """
    Teammate policy that uses a trained RL model for mode selection
    """
    def __init__(self,
                 model,
                 env,
                 local_search_policy,
                 go_to_highvalue_policy,
                 change_region_subpolicy,
                 use_collision_avoidance: bool = False,
                 norm_stats_path: str = None,
                 ):

        self.model = model
        self.env = env
        self.use_collision_avoidance = use_collision_avoidance

        #self.norm_stats = None
        if norm_stats_path and os.path.exists(norm_stats_path):
            try:
                #self.norm_stats = np.load(norm_stats_path, allow_pickle=True).item()
                import pickle
                with open(norm_stats_path, 'rb') as f:
                    self.norm_stats = pickle.load(f)
                print(f"[RLTeammatePolicy] Loaded normalization stats from {norm_stats_path}")
            except Exception as e:
                print(f"[RLTeammatePolicy] Failed to load norm stats from {norm_stats_path}: {e}")
                self.norm_stats = None
        else:
            self.norm_stats = None

        self.local_search_policy = local_search_policy
        self.go_to_highvalue_policy = go_to_highvalue_policy
        self.change_region_subpolicy = change_region_subpolicy

        # Initialize normalization stats
        #self.norm_stats = None
        self.live_obs_rms = None  # For live stats from training env
        self.live_ret_rms = None

        # Default name
        self.name = "RL_Teammate"

        # Track last observation for potential debugging
        self.last_observation = None


    def choose_subpolicy(self, observation, current_subpolicy):
        """Choose subpolicy using the trained RL model"""
        try:
            self.last_observation = observation
            normalized_obs = self._normalize_observation(observation) # Apply normalization if available

            action, _ = self.model.predict(normalized_obs, deterministic=False) # Use the RL model to predict the action (subpolicy choice)

            # Ensure action is a valid subpolicy choice (0, 1, 2, or 3)
            if hasattr(action, 'item'):  # Handle numpy scalars
                action = action.item()
            action = int(action)
            action = max(0, min(3, action))  # Clamp to valid range

            return action

        except Exception as e:
            print(f"[RLTeammatePolicy] Error in choose_subpolicy: {e}")
            #print(f"[RLTeammatePolicy] Falling back to local search (subpolicy 0)")
            return 0  # Fallback to local search

    def set_live_normalization_stats(self, obs_rms, ret_rms):
        """Set live normalization stats from the training environment"""
        self.live_obs_rms = obs_rms
        self.live_ret_rms = ret_rms
        #print(f"[RLTeammatePolicy] Received live normalization stats")

    def _normalize_observation(self, observation):
        """Apply normalization to observation if stats are available"""
        if self.norm_stats is not None:
            try:
                obs_mean = self.norm_stats.obs_rms.mean
                obs_var = self.norm_stats.obs_rms.var
                #print('norm stats means:')
                #print(obs_mean)

                if observation.shape != obs_mean.shape:
                    #print(f'OBSERVATION IS WRONG SHAPE: {observation.shape} vs {obs_mean.shape}')
                    observation = observation[-1]

                epsilon = 1e-8
                clip_obs = 10.0


                normalized_obs = np.clip(
                    (observation - obs_mean) / np.sqrt(obs_var + epsilon),
                    -clip_obs,
                    clip_obs
                )

                return normalized_obs.astype(np.float32)
            except Exception as e:
                print(f"[RLTeammatePolicy] Error normalizing observation: {e}")

        elif self.live_obs_rms is not None:
            try:
                obs_mean = self.live_obs_rms.mean
                obs_var = self.live_obs_rms.var
                epsilon = 1e-8  # Same as VecNormalize default
                clip_obs = 10.0  # Same as VecNormalize default

                # Apply the exact same normalization formula as VecNormalize
                normalized_obs = np.clip(
                    (observation - obs_mean) / np.sqrt(obs_var + epsilon),
                    -clip_obs,
                    clip_obs
                )
                return normalized_obs.astype(np.float32)  # Same dtype as VecNormalize
            except Exception as e:
                print(f"[RLTeammatePolicy] Error using live normalization: {e}")

        # Fallback to file-based stats if live stats fail

        return observation

    def reset(self):
        """Reset any internal state"""
        self.last_observation = None

    def near_a_threat(self):
        """Return true if near threat and need to call evade"""
        if self.env is None:
            return False

        # Implementation would depend on environment structure
        # For now, return False as placeholder
        return False

class HeuristicAgent:
    """
    Heuristic agent that chooses subpolicies based on risk tolerance and spatial coordination settings.

    Subpolicies:
    0 = localsearch
    1 = changeregion
    2 = gotothreat
    """

    def __init__(self,
                 mode_selector="heuristic",
                 risk_tolerance="medium",
                 spatial_coord=False):
        """
        Initialize the heuristic agent.

        Args:
            mode_selector (str): "none" to always choose localsearch, or "heuristic" for decision logic
            risk_tolerance (str): "low", "medium", "high", or " - controls gotothreat usage
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

        #self.action_stability = action_stability

        # Action stability parameters
        self.jitter_frequency = 0.15  # 15% chance to jitter each step when noisy
        self.jitter_steps = [2, -2]  # Jitter by ±1 direction step

        # Validate configuration
        valid_risk_levels = ["low", "medium", "high", "extreme", "none", "max_greedy"]
        valid_spatial_levels = [False, True]
        valid_mode_selectors = ["none", "heuristic", "none"]
        #valid_stability_levels = ["stable", "noisy"]

        # Parameters for periodic stopping
        self.noise_state = {
            'step_counter': 0,
            'pause_counter': 0,
            'at_target_pause': 0,
            'oscillate_flag': False,
        }
        self.stop_every_n_steps = 50  # Period for stopping
        self.stop_duration = 5  # How many steps to "stop"
        self.at_target_delay = 3  # Steps to pause at target


        # if action_stability not in valid_stability_levels:
        #     raise ValueError(f"action_stability must be one of {valid_stability_levels}")
        if risk_tolerance not in valid_risk_levels:
            raise ValueError(f"risk_tolerance must be one of {valid_risk_levels}")
        if spatial_coord not in valid_spatial_levels:
            raise ValueError(f"spatial_coord must be one of {valid_spatial_levels}")
        if mode_selector not in valid_mode_selectors:
            raise ValueError(f"mode_selector must be one of {valid_mode_selectors}")

    def choose_subpolicy(self, env, agent_id=0):
        """Choose a subpolicy and potentially apply action jitter"""
        # Get the base subpolicy choice using existing logic
        base_subpolicy = self._choose_base_subpolicy(env, agent_id)
        return base_subpolicy


    def _choose_base_subpolicy(self, env, agent_id=0):
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


        if self.risk_tolerance == "max_greedy":
            # Decide whether closest unknown target or threat is closer
            agent = env.agents[env.aircraft_ids[agent_id]]
            agent_pos = np.array([agent.x, agent.y])

            # Get unknown targets and threats
            unknown_targets = env.targets[env.targets[:, 2] < 1.0, 3:5]  # shape (n,2)
            #unknown_threats = env.threats[env.threat_identified < 1.0, 3:5]
            unknown_threats = env.threats[~env.threat_identified]

            closest_target_dist = np.inf
            closest_threat_dist = np.inf

            if len(unknown_targets) > 0:
                target_dists = np.linalg.norm(unknown_targets - agent_pos, axis=1)
                closest_target_dist = np.min(target_dists)

            if len(unknown_threats) > 0:
                threat_dists = np.linalg.norm(unknown_threats - agent_pos, axis=1)
                closest_threat_dist = np.min(threat_dists)

            if closest_target_dist <= closest_threat_dist:
                # Go toward target
                self._update_tracking(0)  # local search
                return 0
            else:
                # Go toward threat
                self._update_tracking(2)
                return 2

        # If mode_selector is "none", always choose localsearch
        if self.mode_selector == "none":
            self._update_tracking(0)
            #print(f"[HeuristicAgent] mode_selector=none -> localsearch(0)")
            return 0

        # Check if we should choose gotothreat based on risk tolerance and detections

        should_go_to_threat = self._should_go_to_threat(env)
        if should_go_to_threat:
            self._update_tracking(2)
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
            #action_name = ["localsearch", "changeregion", "gotothreat"][choice]
            #print(f"[HeuristicAgent] {reason} -> {action_name}({choice})")
            return choice

        # Choose between localsearch (0) and changeregion (1) based on spatial coordination
        choice = self._choose_search_strategy(env, agent_id)

        # Apply changeregion cooldown
        if choice in [1,4,5,6] and self.changeregion_cooldown > 0:
            choice = 0  # Force localsearch if changeregion is in cooldown
            reason = "changeregion_in_cooldown"
        else:
            # # Add debugging for the search strategy choice
            # if self.spatial_coord == "none":
            #     reason = "spatial_coord=none"
            # elif self.spatial_coord == "some":
            #     reason = f"spatial_coord=some, target_rich_hysteresis={self.currently_consider_target_rich}"
            # elif self.spatial_coord == "high":
            #     same_quadrant = self._agents_in_same_quadrant(env, agent_id)
            #     reason = f"spatial_coord=high, same_quadrant={same_quadrant}"
            #else:
            reason = "default"

        # Start cooldown if switching away from changeregion
        if self.last_subpolicy in [1,4,5,6] and choice != self.last_subpolicy:
            self.changeregion_cooldown = self.changeregion_cooldown_duration

        self._update_tracking(choice)
        #action_name = ["localsearch", "changeregion", "gotothreat"][choice]
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
        Note: gotothreat is now action 2 in the new action space
        """
        detections = env.num_threats_identified
        #print(f'detections: {detections}')

        should_go = False
        if self.risk_tolerance == "low":
            should_go = False
        elif self.risk_tolerance == "medium":
            should_go = detections == 0
        elif self.risk_tolerance == "high":
            should_go = detections <= 1
        elif self.risk_tolerance == "extreme":
            should_go = detections <= 1

        # Debug logging
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0

        #if self._debug_counter % 10 == 0:  # Log every 50 calls
            #print(f"[HeuristicAgent] Risk: {self.risk_tolerance}, Detections: {detections}, Should go to threat: {should_go}")

        return should_go


    def _choose_search_strategy(self, env, agent_id):
        """Choose between local search and specific quadrant goto policies"""
        if self.spatial_coord == False:
            return 0  # Always choose localsearch
        return 0

    def _find_best_quadrant(self, env, agent_id):
        """Find the quadrant with the most unknown targets that's not occupied by agents"""
        if env.config['num_aircraft'] < 2:
            # No teammate, just find quadrant with most targets
            quadrant_counts = self._get_unknown_targets_per_quadrant(env)
            max_targets = max(quadrant_counts.values())
            for quadrant_name, count in quadrant_counts.items():
                if count == max_targets:
                    quadrant_name_to_id = {"NW": 0, "NE": 1, "SW": 2, "SE": 3}
                    return quadrant_name_to_id.get(quadrant_name, 0)
            return 0

        # Get agent and teammate positions
        agent_quadrant = self._get_agent_quadrant(env, agent_id)
        teammate_id = 1 if agent_id == 0 else 0
        teammate_quadrant = self._get_agent_quadrant(env, teammate_id)

        # Get target counts per quadrant
        quadrant_counts = self._get_unknown_targets_per_quadrant(env)

        # Find quadrant with most targets that's not occupied
        quadrant_name_to_id = {"NW": 0, "NE": 1, "SW": 2, "SE": 3}

        # Sort quadrants by target count (descending)
        sorted_quadrants = sorted(quadrant_counts.items(), key=lambda x: x[1], reverse=True)

        # Choose first quadrant that's not occupied by either agent
        for quadrant_name, target_count in sorted_quadrants:
            if target_count > 0 and quadrant_name != agent_quadrant and quadrant_name != teammate_quadrant:
                return quadrant_name_to_id.get(quadrant_name, 0)

        # If all good quadrants are occupied, just go to the one with most targets
        if sorted_quadrants:
            best_quadrant_name = sorted_quadrants[0][0]
            return quadrant_name_to_id.get(best_quadrant_name, 0)

        return 0  # Default to NW if no targets found

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


class GenericTeammatePolicy(TeammatePolicy):
    def __init__(self,
                 env,
                 local_search_policy: SubPolicy,
                 go_to_highvalue_policy: SubPolicy,
                 change_region_subpolicy: SubPolicy,
                 mode_selector_agent: HeuristicAgent = None,
                 use_collision_avoidance: bool = False,
                 action_stability='stable',
                 decision_speed='fast'
                 ):

        self.env = env
        self.mode_selector_agent = mode_selector_agent
        self.use_collision_avoidance = use_collision_avoidance
        self.action_stability = action_stability
        self.decision_speed = decision_speed

        self.local_search_policy = local_search_policy
        self.go_to_highvalue_policy = go_to_highvalue_policy
        self.change_region_subpolicy = change_region_subpolicy

        # For human control (if needed)
        self.key_to_action = {1: 0, 2: 1, 3: 2}  # Updated to use numbers instead of pygame keys

        # Default name
        self.name = "Generic_Teammate"

    def _normalize_observation(self, observation):
        """Passthrough"""
        return observation

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
            raise ValueError
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
            #print('[GoToNearestThreat]: Using provided model for inference')
        else:
            self.model = None
            #print('[GoToNearestThreat]: No model provided, using internal heuristic')

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
        if not isinstance(action, tuple):
            return action, None
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

        if not isinstance(action, tuple):
            return action, None
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
            #print('[LocalSearch] Using provided model for inference')
        else:
            self.model = None
            #print('[LocalSearch] No model provided, using internal heuristic')

        self.norm_statistics_path = norm_stats_filepath
        if norm_stats_filepath:
            self.norm_stats_filepath = norm_stats_filepath
            #print(f'Loaded training normalization stats from {norm_stats_filepath}')
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


import itertools

class TargetSearchLocalTSP(SubPolicy):
    """
    Sub-policy that uses TSP optimization to find the best route through unknown targets
    within a 200-pixel radius, considering teammate's position and greedy search behavior.
    """

    def __init__(self,
                 search_radius=200, spatial_coord=False, search_method="greedy",
                 model_path: str = None, norm_stats_filepath: str = None):
        super().__init__("target_search_local_tsp")
        self.search_radius = search_radius
        self.spatial_coord = spatial_coord

        self.search_method = search_method  # "greedy" or "clusters"

        # Add clustering parameters
        self.discount_factor = 0.9  # Exponential decay for later targets
        self.cluster_threshold = 100

        self.recalculation_period = 1  # Recalculate TSP every N steps
        self.teammate_prediction_steps = 5  # How many steps ahead to predict teammate movement

        # TSP-related state
        self.current_waypoints = []  # Current sequence of waypoints
        self.current_waypoint_index = 0  # Which waypoint we're heading to
        self.steps_since_recalculation = 0
        self.last_known_targets = set()  # Track which targets we've seen before

        # Teammate prediction
        self.teammate_last_positions = []  # Track teammate movement for prediction
        self.teammate_prediction_history_length = 3

        # Fallback to original LocalSearch behavior
        self.fallback_policy = LocalSearch(model_path, norm_stats_filepath)

        # Direction mapping for discrete actions
        self.directions = np.array([
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

        # Anti-oscillation state
        self._last_action = None
        self._action_repeat_count = 0
        self._max_repeat_count = 3

    def act(self, observation, env=None, agent_id=0):
        """
        Main action method that either follows TSP waypoints or falls back to local search
        """
        if env is None:
            # Fallback to original local search if no environment provided
            return self.fallback_policy.act(observation, env, agent_id), None

        # Check if we need to evade threats first
        # if self.near_threat(env, agent_id):
        #     return self.compute_tangential_escape_action(env, agent_id), None

        # Update teammate position tracking
        self._update_teammate_tracking(env, agent_id)

        # Check if we need to recalculate TSP route
        if (self.steps_since_recalculation >= self.recalculation_period or
                len(self.current_waypoints) == 0 or
                self._targets_changed(env)):
            self._recalculate_tsp_route(env, agent_id)
            self.steps_since_recalculation = 0

        # Follow current TSP route or fallback to local search
        if len(self.current_waypoints) > 0 and self.current_waypoint_index < len(self.current_waypoints):
            action = self._navigate_to_current_waypoint(env, agent_id)
        else:
            # No TSP route available, use fallback
            action = self.fallback_policy.act(observation, env, agent_id)

        self.steps_since_recalculation += 1
        return action, None

    def _update_teammate_tracking(self, env, agent_id):
        """Update teammate position history for movement prediction"""
        if env.config['num_aircraft'] < 2:
            return

        teammate_id = 1 if agent_id == 0 else 0
        teammate_pos = np.array([
            env.agents[env.aircraft_ids[teammate_id]].x,
            env.agents[env.aircraft_ids[teammate_id]].y
        ])

        self.teammate_last_positions.append(teammate_pos)
        if len(self.teammate_last_positions) > self.teammate_prediction_history_length:
            self.teammate_last_positions.pop(0)

    def _targets_changed(self, env):
        """Check if the set of unknown targets has changed significantly"""
        current_targets = set()
        target_positions = env.targets[:env.config['num_targets'], 3:5]
        target_info_levels = env.targets[:env.config['num_targets'], 2]

        agent_pos = np.array([
            env.agents[env.aircraft_ids[0]].x,
            env.agents[env.aircraft_ids[0]].y
        ])

        for i, (pos, info_level) in enumerate(zip(target_positions, target_info_levels)):
            if info_level < 1.0:  # Unknown target
                distance = np.linalg.norm(pos - agent_pos)
                if distance <= self.search_radius:
                    current_targets.add(i)

        # Check if targets changed significantly
        changed = len(current_targets.symmetric_difference(self.last_known_targets)) > 0
        self.last_known_targets = current_targets
        return changed

    def _recalculate_tsp_route(self, env, agent_id):
        """Recalculate the optimal TSP route through nearby unknown targets"""
        agent_pos = np.array([
            env.agents[env.aircraft_ids[agent_id]].x,
            env.agents[env.aircraft_ids[agent_id]].y
        ])

        # Get unknown targets within radius
        nearby_targets = self._get_nearby_unknown_targets(env, agent_pos)

        if len(nearby_targets) == 0:
            self.current_waypoints = []
            self.current_waypoint_index = 0
            return

        # Predict where teammate will search and filter out those targets
        if self.spatial_coord == True:
            #teammate_will_visit = self._predict_teammate_targets(env, agent_id)
            teammate_will_visit = self._predict_teammate_targets_dynamic(env, agent_id)
            filtered_targets = [t for t in nearby_targets if t['id'] not in teammate_will_visit]
        else:
            filtered_targets = nearby_targets

        if len(filtered_targets) == 0:
            filtered_targets = nearby_targets

        # Choose solving method based on search_method parameter
        if self.search_method == "clusters":
            self.current_waypoints = self._solve_tsp_with_clustering(agent_pos, filtered_targets)
        elif self.search_method == 'early_weighted':
            self.current_waypoints = self._solve_weighted_tsp(agent_pos, filtered_targets)
        else:  # "greedy" (default behavior)
            if len(filtered_targets) == 1:
                self.current_waypoints = [filtered_targets[0]['position']]
            elif len(filtered_targets) <= 8:
                self.current_waypoints = self._solve_tsp_exact(agent_pos, filtered_targets)
            else:
                self.current_waypoints = self._solve_tsp_heuristic(agent_pos, filtered_targets)

        self.current_waypoint_index = 0
        #print(f"[TSP] Calculated route with {len(self.current_waypoints)} waypoints using {self.search_method} method")

    def _get_nearby_unknown_targets(self, env, agent_pos):
        """Get all unknown targets within search radius"""
        targets = []
        target_positions = env.targets[:env.config['num_targets'], 3:5]
        target_info_levels = env.targets[:env.config['num_targets'], 2]

        for i, (pos, info_level) in enumerate(zip(target_positions, target_info_levels)):
            if info_level < 1.0:  # Unknown target
                distance = np.linalg.norm(pos - agent_pos)
                if distance <= self.search_radius:
                    targets.append({
                        'id': i,
                        'position': pos.copy(),
                        'distance': distance
                    })

        return targets

    def _solve_tsp_with_clustering(self, start_pos, targets):
        """
        Solve TSP using clustering approach to maximize early target acquisition.
        Enhanced to better prioritize dense clusters.
        """
        if len(targets) == 0:
            return []

        if len(targets) == 1:
            return [targets[0]['position']]

        # Step 1: Identify clusters
        clusters = self._identify_target_clusters(targets)

        # Step 2: Calculate enhanced cluster values
        cluster_values = []
        for cluster_id, cluster_targets in clusters.items():
            cluster_center = self._calculate_cluster_center(cluster_targets)
            distance_from_start = np.linalg.norm(cluster_center - start_pos)

            # Enhanced value calculation
            cluster_size = len(cluster_targets)
            cluster_density = self._calculate_cluster_density(cluster_targets)

            # Prioritize larger, denser clusters
            base_value = cluster_size * cluster_density

            # Apply distance penalty (but don't let it dominate)
            distance_penalty = max(0.1, 1.0 / (1.0 + distance_from_start / 200.0))

            # Apply time-based discount more conservatively
            estimated_visit_time = distance_from_start / 50.0
            time_discount = self.discount_factor ** (estimated_visit_time * 0.5)  # Reduced impact

            final_value = base_value * distance_penalty * time_discount

            cluster_values.append({
                'id': cluster_id,
                'targets': cluster_targets,
                'center': cluster_center,
                'value': final_value,
                'distance': distance_from_start,
                'size': cluster_size,
                'density': cluster_density
            })

        # Step 3: Sort by value (higher is better)
        cluster_values.sort(key=lambda c: c['value'], reverse=True)

        # Debug output
        #print(f"[TSP Clusters] Found {len(cluster_values)} clusters:")
        #for i, cluster in enumerate(cluster_values):
            #print(f"  Cluster {i}: size={cluster['size']}, density={cluster['density']:.2f}, "
             #     f"distance={cluster['distance']:.1f}, value={cluster['value']:.2f}")

        # Step 4: Build route visiting high-value clusters first
        route = []
        current_pos = start_pos

        for cluster_info in cluster_values:
            cluster_targets = cluster_info['targets']

            # Solve TSP within this cluster
            if len(cluster_targets) == 1:
                cluster_route = [cluster_targets[0]['position']]
            else:
                cluster_route = self._solve_cluster_internal_tsp(current_pos, cluster_targets)

            route.extend(cluster_route)

            # Update current position
            if cluster_route:
                current_pos = cluster_route[-1]

        return route

    def _calculate_cluster_density(self, cluster_targets):
        """
        Calculate cluster density as targets per unit area.
        Higher density = more tightly packed targets.
        """
        if len(cluster_targets) <= 1:
            return 1.0

        positions = np.array([t['position'] for t in cluster_targets])

        # Calculate bounding box area
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)

        width = max_coords[0] - min_coords[0]
        height = max_coords[1] - min_coords[1]

        # Avoid division by zero
        area = max((width + 50) * (height + 50), 1000)  # Add padding and minimum area

        density = len(cluster_targets) / area * 10000  # Scale for readability

        return density

    def _identify_target_clusters(self, targets):
        """
        Group targets into clusters using a more robust density-based approach.
        """
        if len(targets) <= 1:
            return {0: targets}

        # Convert to numpy array for easier distance calculations
        positions = np.array([t['position'] for t in targets])

        # Use a simple agglomerative clustering approach
        clusters = {}
        cluster_id = 0
        assigned = [False] * len(targets)

        for i, target in enumerate(targets):
            if assigned[i]:
                continue

            # Start new cluster
            current_cluster = [target]
            assigned[i] = True
            cluster_positions = [target['position']]

            # Iteratively add nearby targets
            added_target = True
            while added_target:
                added_target = False

                for j, candidate in enumerate(targets):
                    if assigned[j]:
                        continue

                    # Check if candidate is close to any target in current cluster
                    min_dist_to_cluster = min(
                        np.linalg.norm(candidate['position'] - cluster_pos)
                        for cluster_pos in cluster_positions
                    )

                    if min_dist_to_cluster <= self.cluster_threshold:
                        current_cluster.append(candidate)
                        cluster_positions.append(candidate['position'])
                        assigned[j] = True
                        added_target = True

            clusters[cluster_id] = current_cluster
            cluster_id += 1

        return clusters

    def _calculate_cluster_center(self, cluster_targets):
        """Calculate the geometric center of a cluster"""
        positions = np.array([t['position'] for t in cluster_targets])
        return np.mean(positions, axis=0)

    def _solve_cluster_internal_tsp(self, entry_point, cluster_targets):
        """
        Solve TSP within a single cluster, starting from entry_point.
        Uses nearest neighbor heuristic for efficiency.
        """
        if len(cluster_targets) == 0:
            return []

        if len(cluster_targets) == 1:
            return [cluster_targets[0]['position']]

        # Use nearest neighbor starting from entry point
        target_positions = [t['position'] for t in cluster_targets]
        unvisited = list(range(len(target_positions)))
        route = []
        current_pos = entry_point

        while unvisited:
            # Find nearest unvisited target
            distances = [np.linalg.norm(target_positions[i] - current_pos) for i in unvisited]
            nearest_idx = unvisited[np.argmin(distances)]

            route.append(target_positions[nearest_idx])
            current_pos = target_positions[nearest_idx]
            unvisited.remove(nearest_idx)

        return route

    def _solve_weighted_tsp(self, start_pos, targets):
        """
        Solve TSP using weighted approach that favors visiting targets earlier.
        Uses discount factor to weight target values by visit order.
        """
        if len(targets) == 0:
            return []

        if len(targets) == 1:
            return [targets[0]['position']]

        target_positions = [start_pos] + [t['position'] for t in targets]
        n = len(target_positions)

        # Create distance matrix
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    distances[i, j] = np.linalg.norm(target_positions[i] - target_positions[j])

        if len(targets) <= 8:  # Use exact solution for small problems
            return self._solve_weighted_tsp_exact(start_pos, targets, distances, target_positions)
        else:  # Use heuristic for larger problems
            return self._solve_weighted_tsp_heuristic(start_pos, targets, distances, target_positions)

    def _solve_weighted_tsp_exact(self, start_pos, targets, distances, target_positions):
        """Solve weighted TSP exactly using brute force for small problems"""
        import itertools

        n = len(target_positions)
        best_value = float('-inf')
        best_route = None

        # Try all permutations (excluding start position)
        for perm in itertools.permutations(range(1, n)):  # Start from 1 to exclude start position
            route = [0] + list(perm)  # Add start position at beginning

            # Calculate weighted value (higher is better)
            total_value = 0
            cumulative_distance = 0

            for i in range(len(route) - 1):
                cumulative_distance += distances[route[i], route[i + 1]]
                # Each target gets value based on when it's reached (earlier = higher value)
                if i > 0:  # Skip start position
                    visit_order = i  # 1st target has order 1, 2nd has order 2, etc.
                    target_value = self.discount_factor ** (visit_order - 1)
                    total_value += target_value

            if total_value > best_value:
                best_value = total_value
                best_route = route

        # Convert back to waypoints (excluding start position)
        if best_route:
            return [target_positions[i] for i in best_route[1:]]
        else:
            return [t['position'] for t in targets]

    def _solve_weighted_tsp_heuristic(self, start_pos, targets, distances, target_positions):
        """
        Solve weighted TSP using greedy heuristic that considers both distance and discount factor.
        At each step, choose the target that maximizes (discounted_value / distance_cost).
        """
        n = len(target_positions)
        unvisited = list(range(1, n))  # Exclude start position (index 0)
        route = []
        current_pos_idx = 0
        visit_order = 1

        while unvisited:
            best_ratio = float('-inf')
            best_target = None

            for target_idx in unvisited:
                # Calculate distance cost
                distance_cost = distances[current_pos_idx, target_idx]

                # Calculate discounted value for this target if visited at current order
                target_value = self.discount_factor ** (visit_order - 1)

                # Calculate value-to-cost ratio (higher is better)
                if distance_cost > 0:
                    ratio = target_value / distance_cost
                else:
                    ratio = float('inf')  # If distance is 0, this target is infinitely good

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_target = target_idx

            # Visit the best target
            route.append(target_positions[best_target])
            current_pos_idx = best_target
            unvisited.remove(best_target)
            visit_order += 1

        return route

    def _predict_teammate_targets_dynamic(self, env, agent_id, max_targets_to_predict=6):
        """
        Truly dynamic prediction that simulates greedy nearest-neighbor search.

        Key insight: Don't predict the N closest targets to start position.
        Instead, simulate the teammate's actual search sequence:
        1. Go to nearest target from current position
        2. From that target, go to nearest remaining target
        3. Repeat until done

        This accounts for targets becoming closer/farther as teammate moves.
        """
        if env.config['num_aircraft'] < 2:
            return set()

        teammate_id = 1 if agent_id == 0 else 0
        teammate_pos = np.array([
            env.agents[env.aircraft_ids[teammate_id]].x,
            env.agents[env.aircraft_ids[teammate_id]].y
        ])

        # Get all unknown targets
        unknown_targets = self._get_all_unknown_targets(env)
        if not unknown_targets:
            return set()

        # Simulate greedy nearest-neighbor search sequence
        predicted_targets = []  # Use list to maintain order
        current_pos = teammate_pos.copy()
        remaining_targets = [t for t in unknown_targets]  # Copy the list

        # Simulate the search sequence
        for step in range(max_targets_to_predict):
            if not remaining_targets:
                break

            # Find the nearest target from current position
            nearest_target = None
            min_distance = float('inf')

            for target in remaining_targets:
                distance = np.linalg.norm(target['position'] - current_pos)
                if distance < min_distance:
                    min_distance = distance
                    nearest_target = target

            if nearest_target is None:
                break

            # Add this target to prediction
            predicted_targets.append(nearest_target['id'])

            # Update current position to this target's location
            current_pos = nearest_target['position'].copy()

            # Remove this target from remaining targets
            remaining_targets = [t for t in remaining_targets if t['id'] != nearest_target['id']]

            #print(f"[Prediction] Step {step + 1}: Teammate will visit target {nearest_target['id']} at {nearest_target['position']}")

        #print(f"[Prediction] Final sequence: {predicted_targets}")
        return set(predicted_targets)

    def _estimate_teammate_velocity(self):
        """Estimate teammate's current velocity from position history"""
        if len(self.teammate_last_positions) < 2:
            return np.array([0.0, 0.0])

        # Use multiple recent positions for better velocity estimation
        if len(self.teammate_last_positions) >= 3:
            # Average velocity over last few steps for smoothing
            velocities = []
            for i in range(1, min(4, len(self.teammate_last_positions))):
                vel = self.teammate_last_positions[-i] - self.teammate_last_positions[-i - 1]
                velocities.append(vel)
            return np.mean(velocities, axis=0)
        else:
            return self.teammate_last_positions[-1] - self.teammate_last_positions[-2]

    def _get_all_unknown_targets(self, env):
        """Get all unknown targets in the environment"""
        targets = []
        target_positions = env.targets[:env.config['num_targets'], 3:5]
        target_info_levels = env.targets[:env.config['num_targets'], 2]

        for i, (pos, info_level) in enumerate(zip(target_positions, target_info_levels)):
            if info_level < 1.0:  # Unknown target
                targets.append({
                    'id': i,
                    'position': pos.copy(),
                })
        return targets

    def _find_nearest_target(self, current_pos, targets):
        """Find the nearest target from current position"""
        if not targets:
            return None

        min_distance = float('inf')
        nearest_target = None

        for target in targets:
            distance = np.linalg.norm(target['position'] - current_pos)
            if distance < min_distance:
                min_distance = distance
                nearest_target = target

        return nearest_target
    #
    #
    # def _predict_teammate_targets(self, env, agent_id):
    #     """Predict which targets the teammate will likely visit based on greedy search"""
    #     if env.config['num_aircraft'] < 2 or len(self.teammate_last_positions) < 2:
    #         return set()
    #
    #     teammate_id = 1 if agent_id == 0 else 0
    #     teammate_pos = np.array([
    #         env.agents[env.aircraft_ids[teammate_id]].x,
    #         env.agents[env.aircraft_ids[teammate_id]].y
    #     ])
    #
    #     # Predict teammate movement direction
    #     teammate_velocity = np.array([0.0, 0.0])
    #     if len(self.teammate_last_positions) >= 2:
    #         teammate_velocity = self.teammate_last_positions[-1] - self.teammate_last_positions[-2]
    #
    #     # Predict teammate position in the future
    #     predicted_pos = teammate_pos + teammate_velocity * self.teammate_prediction_steps
    #
    #     # Find targets the teammate is likely to visit (closest targets to predicted position)
    #     targets_teammate_will_visit = set()
    #     target_positions = env.targets[:env.config['num_targets'], 3:5]
    #     target_info_levels = env.targets[:env.config['num_targets'], 2]
    #
    #     teammate_target_distances = []
    #     for i, (pos, info_level) in enumerate(zip(target_positions, target_info_levels)):
    #         if info_level < 1.0:  # Unknown target
    #             distance_to_predicted = np.linalg.norm(pos - predicted_pos)
    #             distance_to_current = np.linalg.norm(pos - teammate_pos)
    #             teammate_target_distances.append((i, min(distance_to_predicted, distance_to_current)))
    #
    #     # Assume teammate will go for closest 2-3 targets
    #     teammate_target_distances.sort(key=lambda x: x[1])
    #     max_teammate_targets = min(8, len(teammate_target_distances))
    #
    #     for i in range(max_teammate_targets):
    #         targets_teammate_will_visit.add(teammate_target_distances[i][0])
    #
    #     return targets_teammate_will_visit

    def _solve_tsp_exact(self, start_pos, targets):
        """Solve TSP exactly using brute force for small problems"""
        if len(targets) <= 1:
            return [t['position'] for t in targets]

        target_positions = [start_pos] + [t['position'] for t in targets]
        n = len(target_positions)

        # Create distance matrix
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    distances[i, j] = np.linalg.norm(target_positions[i] - target_positions[j])

        # Try all permutations (excluding start position)
        best_distance = float('inf')
        best_route = None

        for perm in itertools.permutations(range(1, n)):  # Start from 1 to exclude start position
            route = [0] + list(perm)  # Add start position at beginning
            total_distance = 0

            for i in range(len(route) - 1):
                total_distance += distances[route[i], route[i + 1]]

            if total_distance < best_distance:
                best_distance = total_distance
                best_route = route

        # Convert back to waypoints (excluding start position)
        if best_route:
            return [target_positions[i] for i in best_route[1:]]
        else:
            return [t['position'] for t in targets]

    def _solve_tsp_heuristic(self, start_pos, targets):
        """Solve TSP using nearest neighbor heuristic for larger problems"""
        if len(targets) == 0:
            return []

        target_positions = [t['position'] for t in targets]
        unvisited = list(range(len(targets)))
        route = []
        current_pos = start_pos

        while unvisited:
            # Find nearest unvisited target
            distances = [np.linalg.norm(target_positions[i] - current_pos) for i in unvisited]
            nearest_idx = unvisited[np.argmin(distances)]

            route.append(target_positions[nearest_idx])
            current_pos = target_positions[nearest_idx]
            unvisited.remove(nearest_idx)

        return route

    def _navigate_to_current_waypoint(self, env, agent_id):
        """Navigate to the current waypoint in the TSP route"""
        if (self.current_waypoint_index >= len(self.current_waypoints)):
            return 0

        agent_pos = np.array([
            env.agents[env.aircraft_ids[agent_id]].x,
            env.agents[env.aircraft_ids[agent_id]].y
        ])

        target_pos = self.current_waypoints[self.current_waypoint_index]

        # Check if we've reached the current waypoint
        distance_to_waypoint = np.linalg.norm(target_pos - agent_pos)
        if distance_to_waypoint <= 30.0:  # Waypoint reached threshold
            self.current_waypoint_index += 1
            if self.current_waypoint_index >= len(self.current_waypoints):
                return 0  # All waypoints visited
            target_pos = self.current_waypoints[self.current_waypoint_index]

        # Calculate direction to target
        direction_to_target = target_pos - agent_pos
        target_norm = np.linalg.norm(direction_to_target)

        if target_norm == 0:
            return 0

        direction_to_target_norm = direction_to_target / target_norm

        # Find best matching action
        dot_products = np.dot(self.directions, direction_to_target_norm)
        best_action = np.argmax(dot_products)

        # Anti-oscillation logic
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
        return np.int32(best_action)

    # Threat avoidance methods (copied from LocalSearch)
    def near_threat(self, env, agent_id=0):
        """Check if the agent is near a threat and should automatically switch to evade mode"""
        agent_pos = np.array([env.agents[env.aircraft_ids[agent_id]].x,
                              env.agents[env.aircraft_ids[agent_id]].y])

        for threat_idx in range(len(env.threats)):
            threat_pos = np.array([env.threats[threat_idx, 0], env.threats[threat_idx, 1]])
            distance_to_threat = np.sqrt(np.sum((threat_pos - agent_pos) ** 2))

            threat_radius = env.config['threat_radius']
            warning_radius = threat_radius * 1.7

            if distance_to_threat <= warning_radius:
                return True
        return False

    def compute_tangential_escape_action(self, env, agent_id=0):
        """Compute escape action when near threat (simplified version)"""
        agent_pos = np.array([env.agents[env.aircraft_ids[agent_id]].x,
                              env.agents[env.aircraft_ids[agent_id]].y])

        # Find nearest threat
        nearest_threat_pos = None
        min_distance = float('inf')

        for threat_idx in range(len(env.threats)):
            threat_pos = np.array([env.threats[threat_idx, 0], env.threats[threat_idx, 1]])
            distance = np.sqrt(np.sum((threat_pos - agent_pos) ** 2))
            if distance < min_distance:
                min_distance = distance
                nearest_threat_pos = threat_pos

        if nearest_threat_pos is None:
            return 0

        # Move directly away from nearest threat
        escape_direction = agent_pos - nearest_threat_pos
        escape_norm = np.linalg.norm(escape_direction)

        if escape_norm > 0:
            escape_direction = escape_direction / escape_norm
            dot_products = np.dot(self.directions, escape_direction)
            return np.int32(np.argmax(dot_products))

        return 0

    def reset_evade_state(self):
        """Reset evade-related state"""
        pass  # TSP policy doesn't maintain evade state like LocalSearch


class ChangeRegions(SubPolicy):
    """Sub-policy that moves to a specific region of the map"""

    def __init__(self, model_path=None):
        super().__init__(f"change_region")
        if model_path is not None:
            self.model = PPO.load(model_path)
            #print('[ChangeRegions]: Using provided model for inference')
        else:
            self.model = None
            #print('[ChangeRegions]: No model provided, using internal heuristic')

        self.update_rate = 10 # Recalculate every 10 steps to reduce computation cost
        self.steps_since_update = 0

        self.target_region = None
        self.arrival_threshold = 0.25


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

class RecordedTrajectoryTeammate(TeammatePolicy):
    def __init__(self, trajectory_file, env, timescale_correction: int = 1):
        """
        Load a recorded human trajectory from a JSON file where each entry has 'human_position'.
        Optionally subsample positions by taking every `timescale_correction`th timestep.
        """
        import json

        self.env = env

        with open(trajectory_file, 'r') as f:
            data = json.load(f)

        # Handle both {"timesteps": [...]} and raw list formats
        if isinstance(data, dict) and "timesteps" in data:
            timesteps = data["timesteps"]
        elif isinstance(data, list):
            timesteps = data
        else:
            raise ValueError(f"Unrecognized trajectory file format: {type(data)} keys={list(data) if isinstance(data, dict) else 'N/A'}")

        # Extract and optionally downsample positions
        raw_positions = [entry["human_position"] for entry in timesteps]
        raw_actions = [entry["human_custom_waypoint"] for entry in timesteps]
        if timescale_correction > 1:
            raw_positions = raw_positions[::timescale_correction]
            raw_actions = raw_actions[::timescale_correction]


        self.position_trajectory = raw_positions
        self.action_trajectory = raw_actions
        self.last_action = None
        print(f'%%%%% Action trajectory = {self.action_trajectory}')
        self.index = 0
        self.name = "Recorded_Human_Teammate"
        self.timescale_correction = timescale_correction

    def choose_subpolicy(self, *args, **kwargs):
        return 0  # Not used

    def reset(self):
        self.index = 0

    def get_action(self):
        """Return the next recorded position (x, y)."""

        self.current_location = self.env.agents[self.env.aircraft_ids[1]].x, self.env.agents[self.env.aircraft_ids[1]].y

        if self.index < len(self.action_trajectory):
            #print(f'\n\n In recorded teammate get action: action_trajectory is {self.action_trajectory}')
            if self.action_trajectory[self.index] is None:
                if self.last_action is None:
                    self.last_action = self.current_location
                action = self.last_action
                print(f'Action at index {self.index} is None, using last action {self.last_action}')
            else:
                action = tuple(self.action_trajectory[self.index])
                print(f'Action at index {self.index} is {action}')
            self.index += 1
            self.last_action = action
            return action
        else:
            # Hold at last known position if trajectory is exhausted
            return self.action_trajectory[-1]
