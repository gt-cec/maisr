"""
TrajeDi-PPO: Trajectory Diversity with SB3 PPO for MAISR

Adapts the TrajeDi algorithm to train a population of SB3 PPO actor-critic agents.
Each seed maintains a Best Response (BR) agent and a population of diverse agents.
The BR trains by playing with population members as teammates. Population agents
train with a JSD diversity incentive to ensure behavioral variety.
"""

import copy
import json
import os
import random
from datetime import datetime
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from base_env import MaisrEnv
from utility.league_management import LocalSearch, GoToNearestThreat, ChangeRegions, RLTeammatePolicy
from utility.localsearch_training_wrapper import MaisrLocalSearchWrapper
from training.trajedi.trajedi_teammate_manager import TrajeDiTeammateManager
from training.trajedi.trajedi_callbacks import TrajeDiDiversityCallback


class DiversityComputer:
    """
    Computes Jensen-Shannon Divergence (JSD) across a population of PPO policies.

    JSD = H(avg_policy) - avg(H(individual_policies))

    Higher JSD means more behavioral diversity in the population.
    """

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    @torch.no_grad()
    def get_action_probs(self, model: PPO, observations: torch.Tensor) -> torch.Tensor:
        """
        Get action probability distribution from a PPO model for given observations.

        Args:
            model: SB3 PPO model
            observations: (batch_size, obs_dim) tensor of observations

        Returns:
            (batch_size, n_actions) tensor of action probabilities
        """
        policy = model.policy
        features = policy.extract_features(observations, policy.pi_features_extractor)
        latent_pi = policy.mlp_extractor.forward_actor(features)
        logits = policy.action_net(latent_pi)
        probs = torch.softmax(logits, dim=-1)
        return probs

    def compute_jsd(
        self,
        observations: torch.Tensor,
        models: List[PPO]
    ) -> torch.Tensor:
        """
        Compute per-state JSD across population action distributions.

        Args:
            observations: (batch_size, obs_dim) tensor
            models: List of PPO models (population members)

        Returns:
            (batch_size,) tensor of JSD values per state
        """
        if len(models) < 2:
            return torch.zeros(observations.shape[0], device=observations.device)

        # Collect action probs from each model
        all_probs = []
        for model in models:
            probs = self.get_action_probs(model, observations)
            all_probs.append(probs)

        # Stack: (n_models, batch_size, n_actions)
        stacked = torch.stack(all_probs, dim=0)

        # Average policy: (batch_size, n_actions)
        avg_probs = stacked.mean(dim=0)

        # H(avg_policy) per state
        avg_probs_clamped = avg_probs.clamp(min=self.epsilon)
        H_avg = -(avg_probs_clamped * torch.log(avg_probs_clamped)).sum(dim=-1)

        # avg H(individual policies) per state
        individual_entropies = []
        for probs in all_probs:
            probs_clamped = probs.clamp(min=self.epsilon)
            H_i = -(probs_clamped * torch.log(probs_clamped)).sum(dim=-1)
            individual_entropies.append(H_i)

        avg_H = torch.stack(individual_entropies, dim=0).mean(dim=0)

        # JSD = H(avg) - avg(H)
        jsd = H_avg - avg_H
        return jsd


class PopulationPool:
    """
    A pool containing one Best Response (BR) agent and N population agents.
    Each agent is an SB3 PPO model with its own VecEnv and TrajeDiTeammateManager.

    Uses DummyVecEnv (not SubprocVecEnv) so teammate managers remain in-process
    and can be modified directly between training phases.
    """

    def __init__(
        self,
        seed_idx: int,
        br_agent: PPO,
        br_env: VecNormalize,
        br_teammate_manager: TrajeDiTeammateManager,
        pop_agents: List[PPO],
        pop_envs: List[VecNormalize],
        pop_teammate_managers: List[TrajeDiTeammateManager],
    ):
        self.seed_idx = seed_idx
        self._br_agent = br_agent
        self._br_env = br_env
        self._br_tm = br_teammate_manager
        self._pop_agents = pop_agents
        self._pop_envs = pop_envs
        self._pop_tms = pop_teammate_managers

    @property
    def br_agent(self) -> PPO:
        return self._br_agent

    @property
    def br_env(self) -> VecNormalize:
        return self._br_env

    @property
    def br_tm(self) -> TrajeDiTeammateManager:
        return self._br_tm

    @property
    def pop_agents(self) -> List[PPO]:
        return self._pop_agents

    @property
    def pop_envs(self) -> List[VecNormalize]:
        return self._pop_envs

    @property
    def pop_tms(self) -> List[TrajeDiTeammateManager]:
        return self._pop_tms

    @property
    def all_agents(self) -> List[PPO]:
        return [self._br_agent] + self._pop_agents

    @property
    def all_envs(self) -> List[VecNormalize]:
        return [self._br_env] + self._pop_envs

    def sync_normalization(self):
        """Sync obs_rms from BR's VecNormalize to all population agents."""
        br_obs_rms = self._br_env.obs_rms
        br_ret_rms = self._br_env.ret_rms

        for pop_env in self._pop_envs:
            pop_env.obs_rms = copy.deepcopy(br_obs_rms)
            pop_env.ret_rms = copy.deepcopy(br_ret_rms)

    def save_pool(self, save_dir: str, round_num: int):
        """Save all models and normalization stats."""
        pool_dir = os.path.join(save_dir, f"pool_{self.seed_idx}", f"round_{round_num}")
        os.makedirs(pool_dir, exist_ok=True)

        # Save BR
        self._br_agent.save(os.path.join(pool_dir, "br_model.zip"))
        self._br_env.save(os.path.join(pool_dir, "br_vecnormalize.pkl"))

        # Save population agents
        for i, (agent, env) in enumerate(zip(self._pop_agents, self._pop_envs)):
            agent.save(os.path.join(pool_dir, f"pop_{i}_model.zip"))
            env.save(os.path.join(pool_dir, f"pop_{i}_vecnormalize.pkl"))

    @staticmethod
    def load_pool(load_dir: str, seed_idx: int, round_num: int, env_factory_fn) -> "PopulationPool":
        """Load a saved pool from disk."""
        pool_dir = os.path.join(load_dir, f"pool_{seed_idx}", f"round_{round_num}")

        # Load BR
        br_tm = TrajeDiTeammateManager()
        br_env = env_factory_fn(seed_idx, "br", br_tm)
        br_env = VecNormalize.load(os.path.join(pool_dir, "br_vecnormalize.pkl"), venv=br_env)
        br_agent = PPO.load(os.path.join(pool_dir, "br_model.zip"), env=br_env)
        br_tm.set_current_model(br_agent)

        # Load population
        pop_agents = []
        pop_envs = []
        pop_tms = []
        i = 0
        while os.path.exists(os.path.join(pool_dir, f"pop_{i}_model.zip")):
            pop_tm = TrajeDiTeammateManager()
            pop_env = env_factory_fn(seed_idx, f"pop_{i}", pop_tm)
            pop_env = VecNormalize.load(
                os.path.join(pool_dir, f"pop_{i}_vecnormalize.pkl"), venv=pop_env
            )
            pop_agent = PPO.load(os.path.join(pool_dir, f"pop_{i}_model.zip"), env=pop_env)
            pop_tm.set_current_model(pop_agent)
            pop_agents.append(pop_agent)
            pop_envs.append(pop_env)
            pop_tms.append(pop_tm)
            i += 1

        return PopulationPool(seed_idx, br_agent, br_env, br_tm, pop_agents, pop_envs, pop_tms)


class TrajeDiPPOTrainer:
    """
    Main trainer that orchestrates TrajeDi training with SB3 PPO agents.

    For each round, per pool:
      1. Train each population agent paired with BR, adding JSD diversity bonus
      2. Train BR agent paired with a random population member (pure reward)
      3. Periodically evaluate cross-play between BR agents from different seeds
    """

    def __init__(
        self,
        env_config: dict,
        trajedi_config: dict,
        run_name: str,
        project_name: str = "maisr-trajedi",
        machine_name: str = "machine",
        wandb_run=None,
    ):
        self.env_config = env_config
        self.trajedi_config = trajedi_config
        self.run_name = run_name
        self.project_name = project_name
        self.machine_name = machine_name
        self.wandb_run = wandb_run

        # TrajeDi params
        self.n_seeds = trajedi_config["n_seeds"]
        self.n_populations = trajedi_config["n_populations"]
        self.div_factor = trajedi_config["div_factor"]
        self.training_rounds = trajedi_config["training_rounds"]
        self.steps_per_phase = trajedi_config["steps_per_phase"]
        self.eval_frequency = trajedi_config["eval_frequency"]
        self.n_eval_episodes = trajedi_config["n_eval_episodes"]
        self.n_envs_per_agent = trajedi_config["n_envs_per_agent"]
        self.div_factor_schedule = trajedi_config.get("div_factor_schedule", "constant")

        self.diversity_computer = DiversityComputer()
        self.pools: List[PopulationPool] = []

        # Output directories
        self.output_dir = f"outputs/{run_name}"
        for subfolder in ["checkpoints", "trained_models", "vecnorm_stats", "logs"]:
            os.makedirs(os.path.join(self.output_dir, subfolder), exist_ok=True)

    def _make_env(self, seed: int, tag: str, teammate_manager: TrajeDiTeammateManager):
        """Create a single wrapped MAISR environment."""
        def _init():
            base_env = MaisrEnv(
                config=self.env_config,
                render_mode="headless",
                run_name=self.run_name,
                tag=tag,
                seed=seed,
                save_episode_plots=False,
            )
            local_search_policy = LocalSearch()
            go_to_highvalue_policy = GoToNearestThreat(model_path=None)
            change_region_subpolicy = ChangeRegions(model_path=None)

            wrapped_env = MaisrLocalSearchWrapper(
                base_env,
                self.env_config["obs_noise_std_localsearch"],
                local_search_policy,
                go_to_highvalue_policy,
                change_region_subpolicy,
                None,
                teammate_manager=teammate_manager,
            )
            wrapped_env = Monitor(wrapped_env)
            wrapped_env.reset()
            return wrapped_env

        return _init

    def _make_vec_env(
        self,
        seed_idx: int,
        agent_tag: str,
        teammate_manager: TrajeDiTeammateManager,
    ) -> VecNormalize:
        """
        Create a DummyVecEnv with VecNormalize for one agent.

        Uses DummyVecEnv (not SubprocVecEnv) so the teammate manager reference
        stays in-process and can be modified between training phases.
        """
        base_seed = self.env_config["seed"] + seed_idx * 1000
        env_fns = [
            self._make_env(
                seed=base_seed + i,
                tag=f"s{seed_idx}_{agent_tag}_e{i}",
                teammate_manager=teammate_manager,
            )
            for i in range(self.n_envs_per_agent)
        ]

        vec_env = DummyVecEnv(env_fns)
        vec_env = VecMonitor(vec_env)
        vec_env = VecNormalize(vec_env)
        vec_env.training = True
        vec_env.norm_reward = True
        return vec_env

    def _create_ppo_model(self, env: VecNormalize, seed: int) -> PPO:
        """Create a PPO model with the standard MAISR architecture."""
        policy_kwargs = dict(
            activation_fn=torch.nn.Tanh,
            net_arch=dict(
                pi=[self.env_config["network_size"]] * self.env_config["network_numlayers"],
                vf=[self.env_config["network_size"]] * self.env_config["network_numlayers"],
            ),
        )

        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=0,
            batch_size=self.env_config["batch_size"],
            n_steps=self.steps_per_phase // self.n_envs_per_agent,
            learning_rate=self.env_config["lr"],
            seed=seed,
            device="cpu",
            gamma=self.env_config["gamma"],
            ent_coef=self.env_config["entropy_regularization"],
            clip_range=self.env_config["clip_range"],
        )
        return model

    def _build_pools(self):
        """Create all pools with BR + population agents."""
        print(f"\n[TrajeDi] Building {self.n_seeds} pools, each with 1 BR + {self.n_populations} pop agents...")

        for seed_idx in range(self.n_seeds):
            base_seed = self.env_config["seed"] + seed_idx * 1000

            # Create teammate manager for BR (teammates will be pop agents)
            br_tm = TrajeDiTeammateManager(
                selfplay_checkpoint_dir=os.path.join(self.output_dir, "checkpoints"),
            )
            br_env = self._make_vec_env(seed_idx, "br", br_tm)
            br_agent = self._create_ppo_model(br_env, seed=base_seed)

            # Set up BR's teammate manager with a reference to the model
            br_tm.set_current_model(br_agent)
            br_tm._create_selfplay_teammate()
            br_tm.current_teammate.env = br_env

            # Create population agents
            pop_agents = []
            pop_envs = []
            pop_tms = []
            for pop_idx in range(self.n_populations):
                pop_seed = base_seed + 100 + pop_idx

                pop_tm = TrajeDiTeammateManager(
                    selfplay_checkpoint_dir=os.path.join(self.output_dir, "checkpoints"),
                )
                pop_env = self._make_vec_env(seed_idx, f"pop{pop_idx}", pop_tm)
                pop_agent = self._create_ppo_model(pop_env, seed=pop_seed)

                pop_tm.set_current_model(pop_agent)
                pop_tm._create_selfplay_teammate()
                pop_tm.current_teammate.env = pop_env

                pop_agents.append(pop_agent)
                pop_envs.append(pop_env)
                pop_tms.append(pop_tm)

            pool = PopulationPool(
                seed_idx, br_agent, br_env, br_tm,
                pop_agents, pop_envs, pop_tms
            )
            self.pools.append(pool)

            print(f"  Pool {seed_idx}: BR(seed={base_seed}) + {self.n_populations} pop agents created")

    def _set_teammate(
        self,
        teammate_manager: TrajeDiTeammateManager,
        teammate_model: PPO,
        training_env: VecNormalize,
        name: str = "TrajeDi_Teammate",
    ):
        """
        Set a specific PPO model as the fixed teammate via the teammate manager.

        Creates an RLTeammatePolicy wrapping the teammate model and sets it as
        the fixed teammate on the manager, so all subsequent env resets use it.

        Args:
            teammate_manager: The TrajeDiTeammateManager controlling the env's teammate
            teammate_model: PPO model to use as teammate
            training_env: VecNormalize env (for normalization stats)
            name: Name for logging
        """
        rl_teammate = teammate_manager.create_rl_teammate_from_model(
            model=teammate_model,
            obs_rms=training_env.obs_rms,
            ret_rms=training_env.ret_rms,
            name=name,
        )
        teammate_manager.set_fixed_teammate(rl_teammate)

    def _get_current_div_factor(self, round_num: int) -> float:
        """Get diversity factor, potentially with schedule."""
        if self.div_factor_schedule == "constant":
            return self.div_factor
        elif self.div_factor_schedule == "linear_warmup":
            warmup_rounds = min(20, self.training_rounds // 5)
            if round_num < warmup_rounds:
                return self.div_factor * (round_num / warmup_rounds)
            return self.div_factor
        return self.div_factor

    def _train_pool_round(self, pool: PopulationPool, round_num: int):
        """
        Train one round for a single pool.

        Phase 1: Train each population agent with diversity bonus (paired with BR)
        Phase 2: Train BR agent against a random population member (pure reward)
        """
        # Sync normalization stats from BR to all pop agents
        pool.sync_normalization()

        current_div_factor = self._get_current_div_factor(round_num)

        # === PHASE 1: Train population agents with diversity bonus ===
        for pop_idx, (pop_agent, pop_env, pop_tm) in enumerate(
            zip(pool.pop_agents, pool.pop_envs, pool.pop_tms)
        ):
            # Set BR as teammate for this population agent
            self._set_teammate(pop_tm, pool.br_agent, pop_env, name=f"BR_seed{pool.seed_idx}")

            # Create diversity callback using ALL population models for JSD
            diversity_callback = TrajeDiDiversityCallback(
                diversity_computer=self.diversity_computer,
                peer_models=pool.pop_agents,
                div_factor=current_div_factor,
                wandb_run=self.wandb_run,
                pool_idx=pool.seed_idx,
                pop_idx=pop_idx,
            )

            # Train with diversity bonus
            pop_agent.learn(
                total_timesteps=self.steps_per_phase,
                callback=diversity_callback,
                reset_num_timesteps=False,
            )

        # === PHASE 2: Train BR against random pop member (pure reward) ===
        random_pop_idx = random.randrange(len(pool.pop_agents))
        random_pop_agent = pool.pop_agents[random_pop_idx]
        self._set_teammate(
            pool.br_tm, random_pop_agent, pool.br_env,
            name=f"Pop{random_pop_idx}_seed{pool.seed_idx}"
        )

        pool.br_agent.learn(
            total_timesteps=self.steps_per_phase,
            reset_num_timesteps=False,
        )

    def _evaluate_self_play(self) -> Dict[int, float]:
        """Evaluate BR_i with itself as teammate (self-play score)."""
        results = {}
        for pool in self.pools:
            # Set BR as its own teammate
            self._set_teammate(
                pool.br_tm, pool.br_agent, pool.br_env,
                name=f"BR_self_seed{pool.seed_idx}"
            )

            pool.br_env.training = False
            pool.br_env.norm_reward = False

            try:
                mean_reward, _ = evaluate_policy(
                    pool.br_agent, pool.br_env, n_eval_episodes=self.n_eval_episodes
                )
                results[pool.seed_idx] = mean_reward
            except Exception as e:
                print(f"  [Eval] Self-play eval failed for pool {pool.seed_idx}: {e}")
                results[pool.seed_idx] = 0.0
            finally:
                pool.br_env.training = True
                pool.br_env.norm_reward = True

        return results

    def _evaluate_cross_play(self) -> Dict[Tuple[int, int], float]:
        """
        Evaluate cross-play: BR_i as Agent 1 with BR_j as Agent 2 for all i != j.
        Uses pool i's environment with pool j's BR as teammate.
        """
        results = {}
        for i, pool_i in enumerate(self.pools):
            for j, pool_j in enumerate(self.pools):
                if i == j:
                    continue

                # Set BR_j as teammate in pool_i's env
                self._set_teammate(
                    pool_i.br_tm, pool_j.br_agent, pool_i.br_env,
                    name=f"XP_BR{j}_in_env{i}"
                )

                pool_i.br_env.training = False
                pool_i.br_env.norm_reward = False

                try:
                    mean_reward, _ = evaluate_policy(
                        pool_i.br_agent, pool_i.br_env, n_eval_episodes=self.n_eval_episodes
                    )
                    results[(i, j)] = mean_reward
                except Exception as e:
                    print(f"  [Eval] Cross-play eval failed for ({i},{j}): {e}")
                    results[(i, j)] = 0.0
                finally:
                    pool_i.br_env.training = True
                    pool_i.br_env.norm_reward = True

        return results

    def _log_metrics(self, round_num: int, sp_results: dict, xp_results: dict, jsd_values: list):
        """Log metrics to wandb and console."""
        metrics = {}

        # Self-play metrics
        sp_scores = list(sp_results.values())
        if sp_scores:
            metrics["trajedi/mean_self_play_reward"] = np.mean(sp_scores)
            for seed_idx, score in sp_results.items():
                metrics[f"trajedi/self_play_seed{seed_idx}"] = score

        # Cross-play metrics
        xp_scores = list(xp_results.values())
        if xp_scores:
            metrics["trajedi/mean_cross_play_reward"] = np.mean(xp_scores)
            for (i, j), score in xp_results.items():
                metrics[f"trajedi/cross_play_{i}_vs_{j}"] = score

        # SP-XP gap
        if sp_scores and xp_scores:
            metrics["trajedi/sp_xp_gap"] = np.mean(sp_scores) - np.mean(xp_scores)

        # JSD metrics
        if jsd_values:
            metrics["trajedi/mean_jsd"] = np.mean(jsd_values)

        metrics["trajedi/round"] = round_num
        metrics["trajedi/div_factor"] = self._get_current_div_factor(round_num)

        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=round_num)

        # Console output
        sp_str = f"{np.mean(sp_scores):.2f}" if sp_scores else "N/A"
        xp_str = f"{np.mean(xp_scores):.2f}" if xp_scores else "N/A"
        jsd_str = f"{np.mean(jsd_values):.4f}" if jsd_values else "N/A"
        print(
            f"  Round {round_num}/{self.training_rounds} | "
            f"SP: {sp_str} | XP: {xp_str} | JSD: {jsd_str}"
        )

    def train(self):
        """Main training loop."""
        print(f"\n{'='*80}")
        print(f"TrajeDi-PPO Training")
        print(f"  Seeds: {self.n_seeds}, Pop size: {self.n_populations}, Div factor: {self.div_factor}")
        print(f"  Rounds: {self.training_rounds}, Steps/phase: {self.steps_per_phase}")
        print(f"{'='*80}\n")

        # Build all pools
        self._build_pools()

        jsd_history = []

        for round_num in range(1, self.training_rounds + 1):
            print(f"\n--- Round {round_num}/{self.training_rounds} ---")

            # Train each pool
            for pool in self.pools:
                print(f"  Training pool {pool.seed_idx}...")
                self._train_pool_round(pool, round_num)

            # Evaluate periodically
            if round_num % self.eval_frequency == 0 or round_num == 1:
                print(f"\n  [Eval] Running evaluations...")
                sp_results = self._evaluate_self_play()
                xp_results = self._evaluate_cross_play()

                # Collect JSD values from diversity callbacks (last recorded)
                jsd_values = []
                for pool in self.pools:
                    for pop_agent in pool.pop_agents:
                        if hasattr(pop_agent, '_last_mean_jsd'):
                            jsd_values.append(pop_agent._last_mean_jsd)

                jsd_history.extend(jsd_values)
                self._log_metrics(round_num, sp_results, xp_results, jsd_values)

            # Save checkpoints periodically
            if round_num % (self.eval_frequency * 2) == 0:
                for pool in self.pools:
                    pool.save_pool(os.path.join(self.output_dir, "checkpoints"), round_num)
                print(f"  [Save] Checkpoints saved for round {round_num}")

        # Final save
        print(f"\n{'='*80}")
        print(f"Training complete. Saving final models...")
        print(f"{'='*80}")

        for pool in self.pools:
            pool.save_pool(os.path.join(self.output_dir, "trained_models"), self.training_rounds)

        # Final evaluation
        print("\nFinal evaluation:")
        sp_results = self._evaluate_self_play()
        xp_results = self._evaluate_cross_play()
        self._log_metrics(
            self.training_rounds, sp_results, xp_results,
            jsd_history[-self.n_seeds * self.n_populations:] if jsd_history else []
        )

        # Close environments
        for pool in self.pools:
            for env in pool.all_envs:
                try:
                    env.close()
                except Exception:
                    pass

        return sp_results, xp_results
