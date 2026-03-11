"""
Generic Agent Evaluation Script for MAISR

This script provides a flexible, plugin-based framework for evaluating different
agent types (RL/PPO, BC, heuristic) against various teammates (held-out RL agents,
recorded human trajectories).

Architecture:
- AgentLoader (ABC): Plugin interface for loading different agent types
- TeammateProvider (ABC): Plugin interface for evaluation partners
- BCTeammatePolicy: Wrapper for BC models to act as teammates
- EvaluationRunner: Orchestrates evaluation loops
- ResultsAggregator: Computes statistics from raw results
- Visualizer: Generates plots and exports data

Usage:
    Edit the main() function to configure which agents and teammates to evaluate,
    then run: python evaluate_agent.py
"""
import ctypes
import os
import glob
import json
import re
import traceback
from abc import ABC, abstractmethod
import gymnasium as gym
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# MAISR imports
from base_env import MaisrEnv
from utility.localsearch_training_wrapper import MaisrLocalSearchWrapper
from utility.league_management import TeammatePolicy, RLTeammatePolicy, RecordedTrajectoryTeammate
from utility.config_management import load_env_config

import pygame

import torch.serialization

torch.serialization.add_safe_globals([
    gym.spaces.box.Box,
    gym.spaces.discrete.Discrete,
    gym.spaces.multi_discrete.MultiDiscrete,
    gym.spaces.multi_binary.MultiBinary,
    gym.spaces.dict.Dict,
    gym.spaces.tuple.Tuple,
])
torch.serialization.add_safe_globals([
    np.ndarray,
    np.dtype,                 # general dtype constructor
    np.float32, np.float64,   # common scalar types
])
try:
    torch.serialization.add_safe_globals([type(np.dtype("float32"))])
except Exception:
    pass
# Allow-list all dtype class variants present in this numpy build
dtype_classes = {type(np.dtype(name)) for name in ["float32","float64","int64","int32","uint8","bool"]}
torch.serialization.add_safe_globals(list(dtype_classes))
torch.serialization.add_safe_globals([
    np._core.multiarray._reconstruct,
    np._core.multiarray.scalar
])
# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class AgentSpec:
    """Unified specification for any agent type"""
    agent_id: str              # Unique identifier
    agent_type: str            # 'rl', 'bc', 'heuristic'
    display_name: str          # Human-readable name
    model: Any                 # The actual model object
    norm_stats: Optional[str] = None  # Path to VecNormalize stats (for RL only)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional info


@dataclass
class EvaluationConfig:
    """Configuration for evaluation run"""
    # Environment
    config_file: str = 'configs/Monolith_index_August.json'
    render: str = 'headless'

    # Episode counts
    num_episodes_rl: int = 100
    num_episodes_human: int = 1
    max_human_trajectories: int = 250

    # Output
    output_dir: str = 'evaluation_results'
    save_plots: bool = True
    save_json: bool = True
    plot_types: List[str] = field(default_factory=lambda: ['bar_comparison', 'metric_breakdown'])


# ============================================================================
# AGENT LOADERS
# ============================================================================

class AgentLoader(ABC):
    """Abstract base class for loading different agent types"""

    @abstractmethod
    def load_agents(self) -> List[AgentSpec]:
        """Load all agents from source and return as AgentSpec list"""
        pass

    @abstractmethod
    def create_policy(self, agent_spec: AgentSpec, env) -> TeammatePolicy:
        """Wrap agent for evaluation in MAISR environment"""
        pass


class RLAgentLoader(AgentLoader):
    """Loads PPO/RL agents with VecNormalize stats"""

    def __init__(self, agent_dir: str, pattern: str = '*_model.zip', label: str = 'rl'):
        self.agent_dir = agent_dir
        self.pattern = pattern
        self.label = label

    def load_agents(self) -> List[AgentSpec]:
        """Load RL agents matching pattern in agent_dir"""
        agents = []
        model_paths = glob.glob(os.path.join(self.agent_dir, self.pattern))

        for model_path in model_paths:
            # Find corresponding VecNormalize stats
            base_name = os.path.basename(model_path).replace('_model.zip', '')
            norm_path = os.path.join(self.agent_dir, f"{base_name}_vecnormalize.pkl")

            if not os.path.exists(norm_path):
                print(f"Warning: No VecNormalize stats found for {base_name}, skipping")
                continue

            # Load model
            try:
                model = PPO.load(model_path)
                agent_id = f"{self.label}_{base_name}"

                agents.append(AgentSpec(
                    agent_id=agent_id,
                    agent_type='rl',
                    display_name=base_name,
                    model=model,
                    norm_stats=norm_path,
                    metadata={'model_path': model_path, 'norm_path': norm_path}
                ))
                print(f"Loaded RL agent: {base_name}")
            except Exception as e:
                print(f"Error loading RL agent {base_name}: {e}")

        return agents

    def create_policy(self, agent_spec: AgentSpec, env) -> TeammatePolicy:
        """Create RLTeammatePolicy wrapper"""
        return RLTeammatePolicy(
            model=agent_spec.model,
            env=env,
            norm_stats_path=agent_spec.norm_stats,
            local_search_policy=None,
            go_to_highvalue_policy=None,
            change_region_subpolicy=None
        )



class BCAgentLoader(AgentLoader):
    """Loads Behavioral Cloning models saved as:
       - <stem>.pth  (policy.state_dict)
       - <stem>.json (minimal metadata, incl. net_arch)
    """

    def __init__(
        self,
        model_dir: str,
        pattern: str = "bc_policy*.pth",
        name_from_filename: bool = True,
        models: Optional[Dict[str, str]] = None,
    ):
        self.model_dir = model_dir
        self.pattern = pattern
        self.name_from_filename = name_from_filename
        self.models = models  # Optional: explicit {name: path_to_pth_or_stem} mapping

    def _resolve_pair(self, path_or_stem: str) -> Tuple[str, str]:
        """
        Returns (pth_path, json_path). Accepts:
          - /path/to/foo.pth
          - /path/to/foo.json
          - /path/to/foo        (stem)
        """
        base, ext = os.path.splitext(path_or_stem)
        if ext == ".pth":
            pth_path = path_or_stem
            json_path = base + ".json"
        elif ext == ".json":
            json_path = path_or_stem
            pth_path = base + ".pth"
        else:
            pth_path = path_or_stem + ".pth"
            json_path = path_or_stem + ".json"
        return pth_path, json_path

    def load_agents(self) -> List[AgentSpec]:
        agents: List[AgentSpec] = []

        # Build list of (display_name, pth_path, json_path)
        if self.models:
            pairs = []
            for name, path_or_stem in self.models.items():
                pth_path, json_path = self._resolve_pair(path_or_stem)
                pairs.append((name, pth_path, json_path))
        else:
            paths = glob.glob(os.path.join(self.model_dir, self.pattern))
            pairs = []
            for pth_path in paths:
                display_name = self._extract_name(pth_path)
                json_path = os.path.splitext(pth_path)[0] + ".json"
                pairs.append((display_name, pth_path, json_path))

        for display_name, pth_path, json_path in pairs:
            try:
                if not os.path.exists(pth_path):
                    raise FileNotFoundError(f"Missing weights file: {pth_path}")

                # JSON is optional but recommended; we can default net_arch if missing
                meta_json: Dict[str, Any] = {}
                if os.path.exists(json_path):
                    with open(json_path, "r") as f:
                        meta_json = json.load(f)

                # Load weights-only state dict (PyTorch 2.6+ safe)
                state_dict = torch.load(pth_path, weights_only=True, map_location="cpu")

                # net_arch: prefer JSON, else fall back to a safe default you used in training
                net_arch = meta_json.get("net_arch", [64, 64])

                # IMPORTANT: use env spaces when creating the policy.
                # This loader does NOT know env until create_policy(), so store pieces now.
                agent_id = f"bc_{display_name}"
                metadata = self._extract_metadata(pth_path)
                metadata.update({"weights_path": pth_path, "json_path": json_path})
                if meta_json:
                    metadata.update({"bc_meta": meta_json})

                # Store state_dict + net_arch temporarily; we'll build the actual policy in create_policy(env)
                agents.append(AgentSpec(
                    agent_id=agent_id,
                    agent_type="bc",
                    display_name=display_name,
                    model={"state_dict": state_dict, "net_arch": net_arch},  # placeholder
                    norm_stats=None,
                    metadata=metadata,
                ))
                print(f"Queued BC agent (weights-only): {display_name}")

            except Exception as e:
                print(f"Error loading BC agent {display_name}: {e}")
                print(traceback.format_exc())

        return agents

    def _extract_name(self, path: str) -> str:
        basename = os.path.basename(path).replace(".pth", "").replace(".pt", "")
        return basename if self.name_from_filename else basename

    def _extract_metadata(self, path: str) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {"model_path": path}
        basename = os.path.basename(path)

        lr_match = re.search(r"lr([\d.e-]+)", basename)
        if lr_match:
            metadata["learning_rate"] = float(lr_match.group(1))

        bs_match = re.search(r"batch(\d+)|bs(\d+)", basename)
        if bs_match:
            metadata["batch_size"] = int(next(g for g in bs_match.groups() if g is not None))

        ep_match = re.search(r"epochs(\d+)|ep(\d+)", basename)
        if ep_match:
            metadata["epochs"] = int(next(g for g in ep_match.groups() if g is not None))

        s_match = re.search(r"seed(\d+)|s(\d+)", basename)
        if s_match:
            metadata["seed"] = int(next(g for g in s_match.groups() if g is not None))

        return metadata

    def create_policy(self, agent_spec: AgentSpec, env) -> "BCTeammatePolicy":
        """
        Reconstruct the ActorCriticPolicy using the *current* env spaces,
        then load the saved state_dict.
        """
        payload = agent_spec.model
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise ValueError("BC AgentSpec.model expected to be a dict with {'state_dict', 'net_arch'}")

        state_dict = payload["state_dict"]
        net_arch = payload.get("net_arch", [64, 64])

        policy = ActorCriticPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda _: 0.0,   # inference only
            net_arch=net_arch,
        )
        policy.load_state_dict(state_dict)
        policy.eval()

        return BCTeammatePolicy(
            model=policy,
            env=env,
            name=agent_spec.display_name,
        )



class HeuristicAgentLoader(AgentLoader):
    """Creates heuristic teammates with configurable parameters"""

    def __init__(self, heuristic_configs: List[Dict[str, Any]]):
        self.heuristic_configs = heuristic_configs

    def load_agents(self) -> List[AgentSpec]:
        """Create heuristic agents from configs"""
        agents = []

        for config in self.heuristic_configs:
            # Import here to avoid circular dependency
            from utility.league_management import ConfigurableHeuristicTeammate

            name = config.get('name', 'heuristic')
            heuristic = ConfigurableHeuristicTeammate(**config.get('params', {}))

            agents.append(AgentSpec(
                agent_id=f"heuristic_{name}",
                agent_type='heuristic',
                display_name=name,
                model=heuristic,
                metadata=config
            ))
            print(f"Created heuristic agent: {name}")

        return agents

    def create_policy(self, agent_spec: AgentSpec, env) -> TeammatePolicy:
        """Return the heuristic teammate directly (already a TeammatePolicy)"""
        return agent_spec.model


# ============================================================================
# BC TEAMMATE POLICY WRAPPER
# ============================================================================

class BCTeammatePolicy(TeammatePolicy):
    """Wrapper for BC policies to work as MAISR teammates

    Mirrors RLTeammatePolicy structure but without VecNormalize stats.
    BC models are trained on raw observations and predict actions directly.
    """

    def __init__(self, model: ActorCriticPolicy, env, name: str):
        self.model = model
        self.env = env
        self.name = name
        self.last_observation = None
        self.device = next(model.parameters()).device

    def choose_subpolicy(self, observation, current_subpolicy):
        """Get action from BC policy

        Args:
            observation: Raw observation from environment
            current_subpolicy: Ignored (BC doesn't use subpolicies)

        Returns:
            action: Discrete action (0-15 for Discrete16)
        """
        # Store observation for get_action()
        self.last_observation = observation

        # Convert observation to tensor
        obs_tensor = torch.as_tensor(observation).unsqueeze(0).float()
        obs_tensor = obs_tensor.to(self.device)

        # Get action from policy (deterministic for evaluation)
        with torch.no_grad():
            action, _, _ = self.model.forward(obs_tensor, deterministic=True)

        return action.cpu().numpy()[0]

    def get_action(self):
        """Called by environment wrapper to get teammate action

        Returns:
            action: Discrete action for current observation
        """
        if self.last_observation is None:
            # Return neutral action if no observation yet
            return 0

        # Convert observation to tensor
        obs_tensor = torch.as_tensor(self.last_observation).unsqueeze(0).float()
        obs_tensor = obs_tensor.to(self.device)

        # Get action from policy
        with torch.no_grad():
            action, _, _ = self.model.forward(obs_tensor, deterministic=True)

        return action.cpu().numpy()[0]

    def reset(self):
        """Reset policy state between episodes"""
        self.last_observation = None


# ============================================================================
# TEAMMATE PROVIDERS
# ============================================================================

class TeammateProvider(ABC):
    """Abstract base class for providing evaluation teammates"""

    @abstractmethod
    def get_all_teammates(self) -> List[Tuple[str, Any, str]]:
        """Returns list of (teammate_id, teammate_object, teammate_type) tuples"""
        pass

    @abstractmethod
    def reset_teammate(self, teammate):
        """Reset teammate state between episodes"""
        pass


class RLTeammateProvider(TeammateProvider):
    """Provides RL agents as evaluation teammates"""

    def __init__(self, agent_loader: RLAgentLoader):
        self.agent_loader = agent_loader
        self.agents = None

    def get_all_teammates(self) -> List[Tuple[str, Any, str]]:
        """Load and return all RL teammates"""
        if self.agents is None:
            self.agents = self.agent_loader.load_agents()

        teammates = []
        for agent_spec in self.agents:
            teammates.append((
                agent_spec.agent_id,
                agent_spec,  # Pass AgentSpec, will be wrapped later
                'rl_teammate'
            ))

        return teammates

    def reset_teammate(self, teammate):
        """Reset RL teammate state"""
        if hasattr(teammate, 'reset'):
            teammate.reset()


class HumanTrajectoryProvider(TeammateProvider):
    """Provides recorded human trajectories as evaluation teammates"""

    def __init__(self, trajectory_dir: str, patterns: List[str],
                 max_trajectories: int = 250, timescale_correction: int = 10):
        self.trajectory_dir = trajectory_dir
        self.patterns = patterns
        self.max_trajectories = max_trajectories
        self.timescale_correction = timescale_correction
        self.trajectories = None

    def get_all_teammates(self) -> List[Tuple[str, str, str]]:
        """Find and return all human trajectory files"""
        if self.trajectories is None:
            self.trajectories = []

            for pattern in self.patterns:
                full_pattern = os.path.join(self.trajectory_dir, pattern)
                files = glob.glob(full_pattern, recursive=True)

                for filepath in files[:self.max_trajectories]:
                    # Extract trajectory ID from filename
                    basename = os.path.basename(filepath)
                    traj_id = basename.replace('.json', '')

                    self.trajectories.append((
                        traj_id,
                        filepath,  # Pass filepath, will load trajectory later
                        'human_teammate'
                    ))

            print(f"Found {len(self.trajectories)} human trajectories")

        return self.trajectories

    def reset_teammate(self, teammate):
        """Reset trajectory teammate state"""
        if hasattr(teammate, 'reset'):
            teammate.reset()

    def load_trajectory(self, filepath: str) -> RecordedTrajectoryTeammate:
        """Load a trajectory file and create RecordedTrajectoryTeammate

        Args:
            filepath: Path to trajectory JSON file

        Returns:
            RecordedTrajectoryTeammate instance
        """
        # Extract level from filename (e.g., timesteps_A1_... -> level 1)
        basename = os.path.basename(filepath)
        level_match = re.search(r'_[A-Z](\d)', basename)
        level = int(level_match.group(1)) if level_match else 99

        # Load and subsample trajectory data
        with open(filepath, 'r') as f:
            full_data = json.load(f)

        # Apply timescale correction (subsample by timescale_correction)
        subsampled_data = full_data[::self.timescale_correction]

        return RecordedTrajectoryTeammate(
            trajectory_data=subsampled_data,
            name=basename,
            level=level
        )


# ============================================================================
# EVALUATION RUNNER
# ============================================================================

class EvaluationRunner:
    """Orchestrates evaluation loops for agents against teammates"""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.env_config = load_env_config(config.config_file)

        # Setup pygame if rendering
        if config.render == 'human':
            pygame.init()
            self.window = pygame.display.set_mode((800, 800))
            self.clock = pygame.time.Clock()
        else:
            self.window = None
            self.clock = None

    def evaluate_agent_vs_teammates(self, agent_spec: AgentSpec,
                                    teammates: List[TeammateProvider],
                                    num_episodes: int) -> pd.DataFrame:
        """Main entry point for evaluation

        Args:
            agent_spec: Agent to evaluate
            teammates: List of teammate providers
            num_episodes: Number of episodes per agent-teammate pair

        Returns:
            DataFrame with all episode results
        """
        all_results = []

        for teammate_provider in teammates:
            print(f"\nEvaluating against {teammate_provider.__class__.__name__}...")

            teammate_list = teammate_provider.get_all_teammates()

            for teammate_id, teammate_obj, teammate_type in teammate_list:
                print(f"  Teammate: {teammate_id}")

                # Determine episodes based on teammate type
                n_episodes = (self.config.num_episodes_human
                            if teammate_type == 'human_teammate'
                            else num_episodes)

                for episode in range(n_episodes):
                    # Run single episode
                    result = self._run_single_episode(
                        agent_spec=agent_spec,
                        teammate_obj=teammate_obj,
                        teammate_type=teammate_type,
                        teammate_provider=teammate_provider,
                        episode=episode
                    )

                    if result is not None:
                        result.update({
                            'agent_id': agent_spec.agent_id,
                            'agent_type': agent_spec.agent_type,
                            'display_name': agent_spec.display_name,
                            'teammate_id': teammate_id,
                            'teammate_type': teammate_type,
                            'episode': episode
                        })
                        all_results.append(result)

        return pd.DataFrame(all_results)

    def _run_single_episode(self, agent_spec: AgentSpec, teammate_obj: Any,
                           teammate_type: str, teammate_provider: TeammateProvider,
                           episode: int) -> Optional[Dict]:
        """Run a single evaluation episode

        Args:
            agent_spec: Agent being evaluated
            teammate_obj: Teammate object (AgentSpec, filepath, etc.)
            teammate_type: Type of teammate ('rl_teammate', 'human_teammate')
            teammate_provider: Provider that created this teammate
            episode: Episode number

        Returns:
            Dictionary with episode metrics, or None if episode failed
        """
        # Create teammate policy
        if teammate_type == 'rl_teammate':
            # teammate_obj is an AgentSpec for RL agents
            teammate_loader = RLAgentLoader(agent_dir='', pattern='')
            teammate_policy = teammate_loader.create_policy(teammate_obj, None)
            level = 99  # Default level for RL teammates
        elif teammate_type == 'human_teammate':
            # teammate_obj is a filepath for human trajectories
            teammate_policy = teammate_provider.load_trajectory(teammate_obj)
            level = teammate_policy.level
        else:
            print(f"Unknown teammate type: {teammate_type}")
            return None

        # Create environment with teammate
        env_fns = [self._make_wrapped_env(teammate_policy)]
        env = DummyVecEnv(env_fns)

        # Apply VecNormalize for RL agents only
        if agent_spec.agent_type == 'rl' and agent_spec.norm_stats:
            env = self._load_vecnormalize_wrapper(agent_spec.norm_stats, env)

        # Access underlying environments
        base_env = env.envs[0].env
        wrapper_env = env.envs[0]

        # Activate teammate
        base_env.teammate_active = True
        wrapper_env.teammate_active = True
        wrapper_env.teammate_policy = teammate_policy
        wrapper_env.current_teammate = teammate_policy

        # Set level for human trajectories
        if teammate_type == 'human_teammate':
            base_env.level = level

        # Run episode
        obs = env.reset()
        done = False
        episode_reward = 0
        num_steps = 0
        target_ids = -1
        threat_ids = -1

        while not done:
            # Handle pygame events
            if self.config.render == 'human':
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        env.close()
                        return None

            # Get action from agent
            if agent_spec.agent_type == 'bc':

                if isinstance(agent_spec.model, dict) and "state_dict" in agent_spec.model:
                    payload = agent_spec.model
                    state_dict = payload["state_dict"]
                    net_arch = payload.get("net_arch", [64, 64])

                    policy = ActorCriticPolicy(
                        observation_space=env.observation_space,
                        action_space=env.action_space,
                        lr_schedule=lambda _: 0.0,  # inference only
                        net_arch=net_arch,
                    )
                    policy.load_state_dict(state_dict)
                    policy.eval()
                    agent_spec.model = policy  # overwrite placeholder with real policy

                # BC policy needs tensor input
                obs_tensor = torch.as_tensor(obs).float()
                with torch.no_grad():
                    action, _, _ = agent_spec.model.forward(obs_tensor, deterministic=True)
                action = action.cpu().numpy()
            else:
                # RL and heuristic agents use predict
                action, _ = agent_spec.model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, done, info = env.step(action)
            if self.config.render == 'human':
                env.envs[0].env.render()
                env.envs[0].env.clock.tick(60)

            # Extract metrics
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            target_ids = int(info[0].get('target_ids', -1))
            threat_ids = int(info[0].get('threat_ids', -1))
            num_steps += 1

        env.close()

        return {
            'reward': episode_reward,
            'target_ids': target_ids,
            'threat_ids': threat_ids,
            'num_steps': num_steps
        }

    def _make_wrapped_env(self, teammate_policy: TeammatePolicy):
        """Create wrapped MAISR environment with teammate support"""
        def _init():

            if self.config.render == 'human':
                if hasattr(ctypes, 'windll') and hasattr(ctypes.windll,'user32'): ctypes.windll.user32.SetProcessDPIAware()
                pygame.display.init()
                pygame.font.init()
                self.clock = pygame.time.Clock()

                window_width, window_height = 1000, 1100
                self.window = pygame.display.set_mode((window_width, window_height))

            else:
                pygame.font.init()
                self.window = None
                self.clock = None

            base_env = MaisrEnv(
                config=self.env_config,
                render_mode=self.config.render,
                window=self.window,
                clock=self.clock,
                tag='evalagent0'
            )
            wrapped_env = MaisrLocalSearchWrapper(
                env=base_env,
                teammate_policy=teammate_policy,
                obs_noise_std=0.01
                #teammate_active=True
            )
            return wrapped_env
        return _init

    def _load_vecnormalize_wrapper(self, norm_stats_path: str, env):
        """Load and apply VecNormalize wrapper"""
        env = VecNormalize.load(norm_stats_path, env)
        env.training = False
        env.norm_reward = False
        return env


# ============================================================================
# RESULTS AGGREGATOR
# ============================================================================

class ResultsAggregator:
    """Computes statistics from raw evaluation results"""

    def compute_statistics(self, df: pd.DataFrame) -> Dict:
        """Compute summary statistics from results DataFrame

        Args:
            df: DataFrame with columns: agent_id, agent_type, display_name,
                teammate_id, teammate_type, episode, reward, target_ids,
                threat_ids, num_steps

        Returns:
            Dictionary with nested statistics by agent and metric
        """
        stats = {
            'by_agent': {},
            'by_metric': {}
        }

        # Group by agent and teammate type
        for agent_id in df['agent_id'].unique():
            agent_df = df[df['agent_id'] == agent_id]
            agent_stats = {}

            for teammate_type in ['rl_teammate', 'human_teammate']:
                teammate_df = agent_df[agent_df['teammate_type'] == teammate_type]

                if len(teammate_df) > 0:
                    agent_stats[teammate_type] = {
                        'mean_reward': teammate_df['reward'].mean(),
                        'std_reward': teammate_df['reward'].std(),
                        'mean_target_ids': teammate_df['target_ids'].mean(),
                        'std_target_ids': teammate_df['target_ids'].std(),
                        'mean_threat_ids': teammate_df['threat_ids'].mean(),
                        'std_threat_ids': teammate_df['threat_ids'].std(),
                        'mean_num_steps': teammate_df['num_steps'].mean(),
                        'std_num_steps': teammate_df['num_steps'].std(),
                        'n': len(teammate_df)
                    }

            display_name = agent_df['display_name'].iloc[0]
            stats['by_agent'][display_name] = agent_stats

        # Compute overall metrics
        for metric in ['reward', 'target_ids', 'threat_ids', 'num_steps']:
            stats['by_metric'][metric] = {
                'overall_mean': df[metric].mean(),
                'overall_std': df[metric].std(),
                'by_teammate_type': {}
            }

            for teammate_type in ['rl_teammate', 'human_teammate']:
                teammate_df = df[df['teammate_type'] == teammate_type]
                if len(teammate_df) > 0:
                    stats['by_metric'][metric]['by_teammate_type'][teammate_type] = {
                        'mean': teammate_df[metric].mean(),
                        'std': teammate_df[metric].std()
                    }

        return stats

    def export_to_json(self, stats: Dict, filepath: str):
        """Save statistics as JSON with metadata

        Args:
            stats: Statistics dictionary from compute_statistics()
            filepath: Output JSON file path
        """
        output = {
            'timestamp': datetime.now().isoformat(),
            'statistics': stats
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Statistics saved to: {filepath}")


# ============================================================================
# VISUALIZER
# ============================================================================

class Visualizer:
    """Generates plots and exports results"""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        os.makedirs(os.path.join(config.output_dir, 'plots'), exist_ok=True)

    def plot_bar_comparison(self, stats: Dict):
        """Create side-by-side bar chart comparing performance vs RL/human teammates

        Args:
            stats: Statistics dictionary from ResultsAggregator
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        agents = list(stats['by_agent'].keys())

        # Left plot: RL teammates
        rl_means = []
        rl_stds = []
        for agent in agents:
            if 'rl_teammate' in stats['by_agent'][agent]:
                rl_means.append(stats['by_agent'][agent]['rl_teammate']['mean_reward'])
                rl_stds.append(stats['by_agent'][agent]['rl_teammate']['std_reward'])
            else:
                rl_means.append(0)
                rl_stds.append(0)

        x = np.arange(len(agents))
        ax1.bar(x, rl_means, yerr=rl_stds, capsize=5, alpha=0.7, color='steelblue')
        ax1.set_xlabel('Agent')
        ax1.set_ylabel('Mean Reward')
        ax1.set_title('Performance vs RL Teammates')
        ax1.set_xticks(x)
        ax1.set_xticklabels(agents, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)

        # Right plot: Human teammates
        human_means = []
        human_stds = []
        for agent in agents:
            if 'human_teammate' in stats['by_agent'][agent]:
                human_means.append(stats['by_agent'][agent]['human_teammate']['mean_reward'])
                human_stds.append(stats['by_agent'][agent]['human_teammate']['std_reward'])
            else:
                human_means.append(0)
                human_stds.append(0)

        ax2.bar(x, human_means, yerr=human_stds, capsize=5, alpha=0.7, color='coral')
        ax2.set_xlabel('Agent')
        ax2.set_ylabel('Mean Reward')
        ax2.set_title('Performance vs Human Teammates')
        ax2.set_xticks(x)
        ax2.set_xticklabels(agents, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if self.config.save_plots:
            output_path = os.path.join(self.config.output_dir, 'plots', 'bar_comparison.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Bar comparison plot saved to: {output_path}")

        plt.close()

    def plot_metric_breakdown(self, df: pd.DataFrame):
        """Create detailed breakdown plots for targets, threats, steps

        Args:
            df: Raw results DataFrame
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics = [
            ('reward', 'Reward'),
            ('target_ids', 'Target IDs'),
            ('threat_ids', 'Threat IDs'),
            ('num_steps', 'Number of Steps')
        ]

        for ax, (metric, label) in zip(axes.flat, metrics):
            # Create box plot grouped by agent and teammate type
            data_by_agent = []
            labels = []
            positions = []

            agents = df['display_name'].unique()
            pos = 0

            for agent in agents:
                agent_df = df[df['display_name'] == agent]

                # RL teammates
                rl_data = agent_df[agent_df['teammate_type'] == 'rl_teammate'][metric]
                if len(rl_data) > 0:
                    data_by_agent.append(rl_data)
                    labels.append(f"{agent}\n(RL)")
                    positions.append(pos)
                    pos += 1

                # Human teammates
                human_data = agent_df[agent_df['teammate_type'] == 'human_teammate'][metric]
                if len(human_data) > 0:
                    data_by_agent.append(human_data)
                    labels.append(f"{agent}\n(Human)")
                    positions.append(pos)
                    pos += 1

                pos += 0.5  # Gap between agents

            bp = ax.boxplot(data_by_agent, positions=positions, widths=0.6, patch_artist=True)

            # Color boxes
            for i, box in enumerate(bp['boxes']):
                if '(RL)' in labels[i]:
                    box.set_facecolor('steelblue')
                else:
                    box.set_facecolor('coral')

            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel(label)
            ax.set_title(f'{label} Distribution')
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if self.config.save_plots:
            output_path = os.path.join(self.config.output_dir, 'plots', 'metric_breakdown.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Metric breakdown plot saved to: {output_path}")

        plt.close()

    def export_results(self, df: pd.DataFrame, stats: Dict):
        """Export results to CSV and JSON

        Args:
            df: Raw results DataFrame
            stats: Statistics dictionary
        """
        # Save raw results
        csv_path = os.path.join(self.config.output_dir, 'results_raw.csv')
        df.to_csv(csv_path, index=False)
        print(f"Raw results saved to: {csv_path}")

        # Save statistics
        if self.config.save_json:
            json_path = os.path.join(self.config.output_dir, 'results_stats.json')
            aggregator = ResultsAggregator()
            aggregator.export_to_json(stats, json_path)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Configure and run agent evaluation.
    Edit this function to specify which agents and teammates to evaluate.
    """

    render = 'headless' # Or 'human'

    # Setup output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"evaluation_results/bc_sweep_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Configure evaluation settings
    config = EvaluationConfig(
        config_file='configs/main_config.json',
        render=render,
        num_episodes_rl=100,
        num_episodes_human=1,
        max_human_trajectories=250,
        output_dir=output_dir,
        save_plots=True,
        save_json=True,
        plot_types=['bar_comparison', 'metric_breakdown']
    )


    # ============================== CONFIGURE AGENTS TO EVALUATE ======================================================
    # Example 1: Evaluate all BC models from hyperparameter sweep
    bc_loader = BCAgentLoader(
        model_dir='training/bc',
        pattern='bc_policy*.pth',  # All sweep results
        name_from_filename=True
    )



    # ============================================= CONFIGURE HELD-OUT TEAMMATES =======================================
    # Example 1: Held-out RL agents
    heldout_rl_loader = RLAgentLoader(
        agent_dir='experiments/user_study/saved_agents',
        label='heldout_rl'
    )
    rl_teammate_provider = RLTeammateProvider(heldout_rl_loader)

    # Example 2: Human trajectories
    human_traj_provider = HumanTrajectoryProvider(
        trajectory_dir='userstudy_logs',
        patterns=[
            'subject_*/timestep_data/timesteps_[ABC][13457]_*.json',  # Dual
            'subject_*/timestep_data/timesteps_[PS][13457]_*.json'     # Solo
        ],
        max_trajectories=250,
        timescale_correction=10
    )

    # ============================================= RUN EVALUATION =============================================
    runner = EvaluationRunner(config)

    # Load agents to evaluate
    agents = bc_loader.load_agents()
    print(f"\n{'='*60}")
    print(f"Loaded {len(agents)} agents to evaluate")
    for agent in agents:
        print(f"  - {agent.display_name} ({agent.agent_type})")
    print(f"{'='*60}")

    # Run evaluations
    all_results = []
    for agent_spec in agents:
        print(f"\n{'='*60}")
        print(f"Evaluating: {agent_spec.display_name}")
        print(f"{'='*60}")

        results_df = runner.evaluate_agent_vs_teammates(
            agent_spec=agent_spec,
            teammates=[rl_teammate_provider, human_traj_provider],
            num_episodes=config.num_episodes_rl
        )
        all_results.append(results_df)

    # ===== AGGREGATE AND VISUALIZE =====
    combined_df = pd.concat(all_results, ignore_index=True)

    aggregator = ResultsAggregator()
    stats = aggregator.compute_statistics(combined_df)

    visualizer = Visualizer(config)
    visualizer.plot_bar_comparison(stats)
    visualizer.plot_metric_breakdown(combined_df)
    visualizer.export_results(combined_df, stats)

    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
