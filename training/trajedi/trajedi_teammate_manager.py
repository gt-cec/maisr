"""
TrajeDi Teammate Manager

Extends the existing TeammateManager to support TrajeDi's fixed teammate assignment.
In TrajeDi, teammates are set explicitly (BR trains with pop, pop trains with BR)
rather than being selected randomly from a league.
"""

import random
from typing import Optional

from utility.league_management import (
    TeammateManager,
    RLTeammatePolicy,
    LocalSearch,
    GoToNearestThreat,
    ChangeRegions,
)


class TrajeDiTeammateManager(TeammateManager):
    """
    Teammate manager for TrajeDi training.

    Supports two modes:
    - Fixed teammate: returns the same RLTeammatePolicy every episode (used during
      TrajeDi phases where BR<->pop pairing is deterministic)
    - Fallback to selfplay: creates a copy of the current model (used during init)
    """

    def __init__(self, selfplay_checkpoint_dir: str = None):
        # Initialize parent with selfplay league type
        super().__init__(
            league_type="selfplay",
            balance_method="uniform",
            selfplay_checkpoint_dir=selfplay_checkpoint_dir or "",
            pretrained_teammate_dir="",
            overfit_test=None,
            verbose=False,
        )

        self._fixed_teammate: Optional[RLTeammatePolicy] = None
        self._use_fixed = False

    def set_fixed_teammate(self, teammate: RLTeammatePolicy):
        """
        Set a fixed teammate that will be returned on every select_random_teammate() call.

        Args:
            teammate: The RLTeammatePolicy to use as the fixed teammate
        """
        self._fixed_teammate = teammate
        self._use_fixed = True
        self.current_teammate = teammate

    def clear_fixed_teammate(self):
        """Remove fixed teammate, reverting to normal selfplay selection."""
        self._fixed_teammate = None
        self._use_fixed = False

    def select_random_teammate(self):
        """
        Override: If a fixed teammate is set, always return that.
        Otherwise fall back to selfplay behavior.
        """
        if self._use_fixed and self._fixed_teammate is not None:
            self.current_teammate = self._fixed_teammate
            return self._fixed_teammate

        # Fallback: use current model copy (selfplay)
        return self._create_selfplay_teammate()

    def reset_for_episode(self):
        """
        Override: Don't re-select teammate if fixed mode is active.
        """
        self.episode_count += 1

        if self._use_fixed and self._fixed_teammate is not None:
            self.current_teammate = self._fixed_teammate
            if hasattr(self.current_teammate, 'reset'):
                self.current_teammate.reset()
            return

        # Fallback to parent behavior
        self.select_random_teammate()
        if self.current_teammate and hasattr(self.current_teammate, 'reset'):
            self.current_teammate.reset()

    def create_rl_teammate_from_model(self, model, obs_rms=None, ret_rms=None, name="TrajeDi_Teammate"):
        """
        Create an RLTeammatePolicy from a PPO model.

        Args:
            model: SB3 PPO model
            obs_rms: Observation running mean/std stats (from VecNormalize)
            ret_rms: Return running mean/std stats (from VecNormalize)
            name: Name for logging

        Returns:
            RLTeammatePolicy instance
        """
        rl_teammate = RLTeammatePolicy(
            model=model,
            env=None,
            local_search_policy=self.subpolicies.get('local_search'),
            go_to_highvalue_policy=self.subpolicies.get('go_to_threat'),
            change_region_subpolicy=self.subpolicies.get('change_region'),
        )
        rl_teammate.name = name

        if obs_rms is not None and ret_rms is not None:
            rl_teammate.set_live_normalization_stats(obs_rms, ret_rms)

        return rl_teammate
