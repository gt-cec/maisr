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

    Injects diversity bonus after rollout collection but before gradient update.
    """

    def __init__(
            self,
            diversity_computer: DiversityComputer,
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

        self._last_dones = None

    def _on_step(self) -> bool:
        """Store the latest dones for advantage recomputation."""
        if "dones" in self.locals:
            self._last_dones = self.locals["dones"].copy()
        return True

    def _on_rollout_end(self) -> None:
        """
        Inject JSD diversity bonus into rewards and recompute advantages.
        """
        buffer = self.model.rollout_buffer
        n_steps = buffer.buffer_size
        n_envs = buffer.n_envs

        # Get observations and actions from buffer
        observations = buffer.observations  # (n_steps, n_envs, obs_dim)
        actions = buffer.actions  # (n_steps, n_envs)

        # Flatten for processing
        if observations.ndim == 3:
            obs_dim = observations.shape[2]
            obs_flat = observations.reshape(-1, obs_dim)
            actions_flat = actions.reshape(-1)
        elif observations.ndim == 2:
            obs_flat = observations
            actions_flat = actions
            obs_dim = observations.shape[1]
        else:
            return

        # Convert to torch tensors
        obs_tensor = torch.as_tensor(obs_flat, dtype=torch.float32).to(self.model.device)
        actions_tensor = torch.as_tensor(actions_flat, dtype=torch.long).to(self.model.device)

        # Compute JSD with temporal discounting
        jsd_flat = self.diversity_computer.compute_jsd(
            obs_tensor, actions_tensor, self.peer_models, n_envs
        )
        jsd_np = jsd_flat.cpu().numpy()

        # Reshape to match buffer rewards shape: (n_steps, n_envs)
        jsd_reshaped = jsd_np.reshape(n_steps, n_envs)

        # Add diversity bonus to rewards
        buffer.rewards += self.div_factor * jsd_reshaped

        # Recompute returns and advantages with modified rewards
        with torch.no_grad():
            last_obs = torch.as_tensor(self.model._last_obs, dtype=torch.float32).to(self.model.device)
            last_values = self.model.policy.predict_values(last_obs)
            last_values = last_values.flatten()

        # Get dones from last step
        if self._last_dones is not None:
            last_dones = self._last_dones
        else:
            last_dones = np.zeros(n_envs)

        buffer.compute_returns_and_advantage(
            last_values=last_values,
            dones=last_dones,
        )

        # Store mean JSD for logging
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
                f"mean_jsd={mean_jsd:.4f}, bonus={self.div_factor * mean_jsd:.4f}, gamma={self.diversity_computer.gamma}"
            )