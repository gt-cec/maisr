import random
import numpy as np
from abc import ABC, abstractmethod
from stable_baselines3 import PPO
from policies.sub_policies import LocalSearch, ChangeRegions, GoToNearestThreat


class TeammatePolicy(ABC):
    """Abstract base class for teammate policies"""

    @abstractmethod
    def get_action(self, observation, env_state):
        pass

    @abstractmethod
    def reset(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass


class RLTeammatePolicy(TeammatePolicy):
    """RL-trained teammate policy"""

    def __init__(self, model_path, policy_name):
        self.model = PPO.load(model_path)
        self._name = f"RL_{policy_name}"

    def get_action(self, observation, env_state):
        action, _ = self.model.predict(observation, deterministic=True)
        return action

    def reset(self):
        pass

    @property
    def name(self):
        return self._name


class HeuristicTeammatePolicy(TeammatePolicy):
    """Heuristic-based teammate policy"""

    def __init__(self, strategy_type, config=None):
        self.strategy_type = strategy_type
        self.config = config or {}
        self._name = f"Heuristic_{strategy_type}"
        self.current_mode = 0  # 0: local_search, 1: change_region, 2: go_to_threat
        self.steps_since_mode_change = 0
        self.mode_duration = self.config.get('mode_duration', 50)

    def get_action(self, observation, env_state):
        if self.strategy_type == "aggressive":
            return self._aggressive_strategy(observation, env_state)
        elif self.strategy_type == "conservative":
            return self._conservative_strategy(observation, env_state)
        elif self.strategy_type == "adaptive":
            return self._adaptive_strategy(observation, env_state)
        else:
            return self._random_strategy(observation, env_state)

    def _aggressive_strategy(self, observation, env_state):
        """Always prioritize going to threats"""
        return 2  # go_to_threat mode

    def _conservative_strategy(self, observation, env_state):
        """Prefer local search, avoid threats"""
        threats_nearby = self._check_threats_nearby(env_state)
        if threats_nearby:
            return 1  # change_region to escape
        return 0  # local_search

    def _adaptive_strategy(self, observation, env_state):
        """Switch modes based on game state"""
        targets_remaining = env_state.get('targets_remaining', 0)
        detection_risk = env_state.get('detection_risk', 0)

        if detection_risk > 0.7:
            return 1  # change_region
        elif targets_remaining > 5:
            return 0  # local_search
        else:
            return 2  # go_to_threat

    def _random_strategy(self, observation, env_state):
        """Random mode switching with some persistence"""
        self.steps_since_mode_change += 1

        if self.steps_since_mode_change >= self.mode_duration:
            self.current_mode = random.randint(0, 2)
            self.steps_since_mode_change = 0

        return self.current_mode

    def _check_threats_nearby(self, env_state):
        """Check if threats are within danger zone"""
        # Implement based on your environment's threat detection logic
        return env_state.get('threat_distance', float('inf')) < 100

    def reset(self):
        self.current_mode = 0
        self.steps_since_mode_change = 0

    @property
    def name(self):
        return self._name


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