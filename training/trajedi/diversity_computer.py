"""
DiversityComputer: Jensen-Shannon Divergence with Temporal Discounting

Computes trajectory diversity metrics for the TrajeDi algorithm. This module
implements JSD computation with temporal discounting to measure behavioral
diversity between policies without requiring trajectory sampling.

The full TrajeDi objective from Eq. 5:
    JSDγ = - 1/n ∑_i ∑_τ P(τ|πi) ∑_t (1/T) log(δ̂_t(τ) / δ_i,t(τ))

where δ_i,t(τ) = ∏_{t'=0}^T [π_i(a_t'|τ_t')]^γ^|t-t'|
"""

from typing import List

import torch
from stable_baselines3 import PPO


class DiversityComputer:
    """
    Computes Jensen-Shannon Divergence (JSD) with temporal discounting.

    Implements the full TrajeDi objective from Eq. 5:
    JSDγ = - 1/n ∑_i ∑_τ P(τ|πi) ∑_t (1/T) log(δ̂_t(τ) / δ_i,t(τ))

    where δ_i,t(τ) = ∏_{t'=0}^T [π_i(a_t'|τ_t')]^γ^|t-t'|
    """

    def __init__(self, gamma: float = 0.5, epsilon: float = 1e-8):
        """
        Args:
            gamma: Temporal discounting factor
                - gamma=1: Full trajectory-level diversity (most sensitive)
                - gamma=0: Action-level diversity (most stringent)
                - gamma in (0,1): Interpolates between the two
            epsilon: Small constant for numerical stability
        """
        self.gamma = gamma
        self.epsilon = epsilon

    @torch.no_grad()
    def get_action_probs(self, model: PPO, observations: torch.Tensor) -> torch.Tensor:
        """
        Get action probability distribution from a PPO model.

        Args:
            model: SB3 PPO model
            observations: (batch_size, obs_dim) tensor

        Returns:
            (batch_size, n_actions) tensor of action probabilities
        """
        policy = model.policy
        features = policy.extract_features(observations, policy.pi_features_extractor)
        latent_pi = policy.mlp_extractor.forward_actor(features)
        logits = policy.action_net(latent_pi)
        probs = torch.softmax(logits, dim=-1)
        return probs

    def compute_local_action_kernel(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            models: List[PPO],
            timestep_idx: int,
            n_steps: int,
    ) -> torch.Tensor:
        """
        Compute local action kernel δ_i,t(τ) for each policy.

        δ_i,t(τ) = ∏_{t'=0}^T [π_i(a_t'|obs_t')]^γ^|t-t'|

        Args:
            observations: (n_steps, obs_dim) - single trajectory
            actions: (n_steps,) - actions taken
            models: List of PPO models
            timestep_idx: Current timestep t
            n_steps: Total trajectory length T

        Returns:
            (n_models,) tensor of kernel values, one per policy
        """
        kernels = []

        for model in models:
            # Get action probs for all timesteps: (n_steps, n_actions)
            all_probs = self.get_action_probs(model, observations)

            # Get probability of actual actions taken: (n_steps,)
            action_probs = all_probs[torch.arange(n_steps), actions]
            action_probs = action_probs.clamp(min=self.epsilon)

            # Apply temporal discounting: γ^|t - t'|
            temporal_distances = torch.abs(torch.arange(n_steps, device=observations.device) - timestep_idx)
            discount_factors = self.gamma ** temporal_distances.float()

            # Compute kernel: ∏_{t'} [π(a_t'|obs_t')]^γ^|t-t'|
            # In log space: ∑_{t'} γ^|t-t'| * log(π(a_t'|obs_t'))
            log_kernel = (discount_factors * torch.log(action_probs)).sum()
            kernel = torch.exp(log_kernel)

            kernels.append(kernel)

        return torch.stack(kernels)

    def compute_jsd_trajectory(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            models: List[PPO],
    ) -> torch.Tensor:
        """
        Compute JSD for a single trajectory using local action kernels.

        Args:
            observations: (n_steps, obs_dim) - single trajectory observations
            actions: (n_steps,) - single trajectory actions
            models: List of PPO models

        Returns:
            (n_steps,) tensor of JSD values, one per timestep
        """
        n_steps = observations.shape[0]
        n_models = len(models)

        if n_models < 2:
            return torch.zeros(n_steps, device=observations.device)

        jsd_per_timestep = []

        for t in range(n_steps):
            # Compute local action kernel for each policy at timestep t
            # kernels: (n_models,)
            kernels = self.compute_local_action_kernel(
                observations, actions, models, timestep_idx=t, n_steps=n_steps
            )

            # Average kernel: δ̂_t(τ) = (1/n) ∑_i δ_i,t(τ)
            avg_kernel = kernels.mean()

            # JSD contribution at timestep t:
            # ∑_i (1/n) log(δ̂_t(τ) / δ_i,t(τ))
            # = (1/n) ∑_i [log(δ̂_t) - log(δ_i,t)]
            avg_kernel_clamped = avg_kernel.clamp(min=self.epsilon)
            kernels_clamped = kernels.clamp(min=self.epsilon)

            jsd_t = (torch.log(avg_kernel_clamped) - torch.log(kernels_clamped)).mean()
            jsd_per_timestep.append(jsd_t)

        return torch.stack(jsd_per_timestep)

    def compute_jsd(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            models: List[PPO],
            n_envs: int,
    ) -> torch.Tensor:
        """
        Compute JSD for a batch of parallel trajectories.

        Args:
            observations: (n_steps * n_envs, obs_dim) flattened observations
            actions: (n_steps * n_envs,) flattened actions
            models: List of PPO models
            n_envs: Number of parallel environments

        Returns:
            (n_steps * n_envs,) tensor of JSD values
        """
        if len(models) < 2:
            return torch.zeros(observations.shape[0], device=observations.device)

        # Reshape to (n_steps, n_envs, obs_dim)
        obs_dim = observations.shape[-1]
        n_total = observations.shape[0]
        n_steps = n_total // n_envs

        obs_reshaped = observations.reshape(n_steps, n_envs, obs_dim)
        actions_reshaped = actions.reshape(n_steps, n_envs)

        # Compute JSD for each environment trajectory separately
        jsd_all_envs = []

        for env_idx in range(n_envs):
            obs_traj = obs_reshaped[:, env_idx, :]  # (n_steps, obs_dim)
            actions_traj = actions_reshaped[:, env_idx]  # (n_steps,)

            jsd_traj = self.compute_jsd_trajectory(obs_traj, actions_traj, models)
            jsd_all_envs.append(jsd_traj)

        # Stack back: (n_steps, n_envs)
        jsd_reshaped = torch.stack(jsd_all_envs, dim=1)

        # Flatten: (n_steps * n_envs,)
        return jsd_reshaped.reshape(-1)
