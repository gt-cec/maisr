"""
TrajeDi Diversity Callback for SB3 PPO

Hooks into PPO's rollout collection to add JSD diversity bonus to population agents'
reward buffers. This replaces TrajeDi's direct backprop through the payoff matrix
with a reward shaping approach compatible with policy gradient methods.
"""

from typing import List, Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class TrajeDiDiversityCallback(BaseCallback):
    """
    SB3 callback that adds JSD diversity reward bonus to population agents.

    Hook point: _on_rollout_end() -- after PPO collects a rollout and computes
    returns/advantages, but before the policy gradient update.

    Steps:
    1. Read rollout_buffer.observations (n_steps, n_envs, obs_dim)
    2. Compute per-state JSD across all population peers
    3. Add +div_factor * jsd to rollout_buffer.rewards
    4. Recompute returns and advantages using last_values from model
    """

    def __init__(
        self,
        diversity_computer,
        peer_models: List[PPO],
        div_factor: float = 5.0,
        wandb_run=None,
        pool_idx: int = 0,
        pop_idx: int = 0,
        verbose: int = 0,
    ):
        """
        Args:
            diversity_computer: DiversityComputer instance for JSD computation
            peer_models: List of all population PPO models (including self)
            div_factor: Scaling factor for diversity bonus
            wandb_run: Optional wandb run for logging
            pool_idx: Pool/seed index for logging
            pop_idx: Population member index for logging
        """
        super().__init__(verbose)
        self.diversity_computer = diversity_computer
        self.peer_models = peer_models
        self.div_factor = div_factor
        self.wandb_run = wandb_run
        self.pool_idx = pool_idx
        self.pop_idx = pop_idx

        # Stored during _on_rollout_start for recomputation
        self._last_dones = None

    def _on_rollout_start(self) -> None:
        """Capture initial state for advantage recomputation."""
        pass

    def _on_step(self) -> bool:
        """Store the latest dones for advantage recomputation."""
        # Track episode_dones from the last step, used for recomputing advantages
        if "dones" in self.locals:
            self._last_dones = self.locals["dones"].copy()
        return True

    def _on_rollout_end(self) -> None:
        """
        Called after PPO collects a full rollout buffer and computes returns/advantages,
        but before the gradient update.

        Injects JSD diversity bonus into the rewards and recomputes advantages.
        """
        buffer = self.model.rollout_buffer

        # Get observations from buffer: shape is (buffer_size, n_envs, *obs_shape)
        # In SB3, buffer_size = n_steps, and observations are stored per-env
        observations = buffer.observations
        n_steps = buffer.buffer_size
        n_envs = buffer.n_envs

        if observations.ndim == 3:
            obs_dim = observations.shape[2]
            obs_flat = observations.reshape(-1, obs_dim)
        elif observations.ndim == 2:
            # Already flat (buffer_size * n_envs, obs_dim)
            obs_flat = observations
            obs_dim = observations.shape[1]
        else:
            # Unexpected shape, skip
            return

        # Convert to torch tensor
        obs_tensor = torch.as_tensor(obs_flat, dtype=torch.float32).to(self.model.device)

        # Compute JSD across all population peers
        jsd_flat = self.diversity_computer.compute_jsd(obs_tensor, self.peer_models)
        jsd_np = jsd_flat.cpu().numpy()

        # Reshape to match buffer rewards shape: (n_steps, n_envs)
        jsd_reshaped = jsd_np.reshape(n_steps, n_envs)

        # Add diversity bonus to rewards
        buffer.rewards += self.div_factor * jsd_reshaped

        # Recompute returns and advantages with the modified rewards.
        # We need last_values (V(s_T)) and dones at the end of the rollout.
        # Recompute last_values from the model using the last observation.
        with torch.no_grad():
            last_obs = torch.as_tensor(self.model._last_obs, dtype=torch.float32).to(self.model.device)
            last_values = self.model.policy.predict_values(last_obs)
            last_values = last_values.flatten()

        # Get the dones from the last step of the rollout
        if self._last_dones is not None:
            last_dones = self._last_dones
        else:
            # Fallback: use the last episode_starts in buffer (inverted)
            last_dones = np.zeros(n_envs)

        buffer.compute_returns_and_advantage(
            last_values=last_values,
            dones=last_dones,
        )

        # Store mean JSD on the model for logging
        mean_jsd = float(jsd_np.mean())
        self.model._last_mean_jsd = mean_jsd

        # Log to wandb
        if self.wandb_run is not None:
            self.wandb_run.log({
                f"trajedi/jsd_pool{self.pool_idx}_pop{self.pop_idx}": mean_jsd,
                f"trajedi/div_bonus_pool{self.pool_idx}_pop{self.pop_idx}": float(self.div_factor * mean_jsd),
            })

        if self.verbose > 0:
            print(
                f"  [DivCallback] Pool {self.pool_idx} Pop {self.pop_idx}: "
                f"mean_jsd={mean_jsd:.4f}, bonus={self.div_factor * mean_jsd:.4f}"
            )
