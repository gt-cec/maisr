import gymnasium as gym
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union


class MAISRCurriculumWrapper(gym.Wrapper):
    """
    A wrapper for the MAISR environment that implements difficulty levels for curriculum learning.

    Attributes:
        difficulty (int): Difficulty level (0, 1, 2, or 3) affecting various environment parameters
        base_config (dict): Original configuration parameters
        current_config (dict): Modified configuration based on difficulty level
    """

    def __init__(self, env, difficulty: int = 0, base_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the wrapper with a difficulty level.

        Args:
            env: The MAISR environment to wrap
            difficulty: Integer from 0-3 representing difficulty level
            base_config: Optional base configuration to override environment defaults
        """
        super().__init__(env)

        if difficulty not in [0, 1, 2, 3]:
            raise ValueError(f"Difficulty must be 0, 1, 2, or 3, got {difficulty}")

        self.difficulty = difficulty
        self.base_config = base_config if base_config is not None else env.config.copy()
        self.current_config = self.base_config.copy()

        # Apply difficulty modifiers to config
        self._apply_difficulty()

        # Reset the environment with the new config
        self._update_env_config()

    def _apply_difficulty(self):
        """Apply difficulty settings to the environment configuration."""

        # Common settings across all difficulties
        self.current_config = self.base_config.copy()

        if self.difficulty == 0:  # Easiest
            # Fewer targets to identify
            self.current_config['num ships'] = 5

            # Slower detection probability
            self.env.prob_detect = 0.0005  # Lower probability of being detected

            # Faster info gathering
            self.env.steps_for_lowqual_info = 2 * 60  # 2 seconds for low quality
            self.env.steps_for_highqual_info = 5 * 60  # 5 seconds for high quality

            # More time to complete
            self.current_config['time limit'] = 240  # 4 minutes

        elif self.difficulty == 1:  # Medium
            self.current_config['num ships'] = 8

            self.env.prob_detect = 0.001

            self.env.steps_for_lowqual_info = 3 * 60  # 3 seconds
            self.env.steps_for_highqual_info = 6 * 60  # 6 seconds

            self.current_config['time limit'] = 210  # 3.5 minutes

        elif self.difficulty == 2:  # Hard
            self.current_config['num ships'] = 10

            self.env.prob_detect = 0.00133333  # Default value

            self.env.steps_for_lowqual_info = 3 * 60  # Default
            self.env.steps_for_highqual_info = 7 * 60  # Default

            self.current_config['time limit'] = 180  # 3 minutes

        elif self.difficulty == 3:  # Very Hard
            self.current_config['num ships'] = 15

            self.env.prob_detect = 0.002  # Higher probability

            self.env.steps_for_lowqual_info = 4 * 60  # 4 seconds
            self.env.steps_for_highqual_info = 9 * 60  # 9 seconds

            self.current_config['time limit'] = 150  # 2.5 minutes

    def _update_env_config(self):
        """Update the wrapped environment's configuration."""
        self.env.config = self.current_config

    def reset(self, **kwargs):
        """Reset the environment with the current difficulty settings."""
        # Ensure config is applied before reset
        self._update_env_config()
        return self.env.reset(**kwargs)

    def set_difficulty(self, difficulty: int):
        """
        Change the difficulty level of the environment.

        Args:
            difficulty: New difficulty level (0-3)
        """
        if difficulty not in [0, 1, 2, 3]:
            raise ValueError(f"Difficulty must be 0, 1, 2, or 3, got {difficulty}")

        self.difficulty = difficulty
        self._apply_difficulty()
        # No need to reset immediately - will happen on next reset call

    def get_current_config(self) -> Dict[str, Any]:
        """Return the current configuration based on difficulty."""
        return self.current_config.copy()

    def step(self, action):
        """Take a step in the environment using the action."""
        # Simply pass through to the wrapped env
        return self.env.step(action)