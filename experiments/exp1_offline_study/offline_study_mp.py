import ctypes
import glob
import os
import json
import random
import pickle
from datetime import datetime
from multiprocessing import Pool, cpu_count
import multiprocessing

import pygame
import re

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.policies import ActorCriticPolicy

from base_env import MaisrEnv
from utility.localsearch_training_wrapper import MaisrLocalSearchWrapper
from utility.config_management import load_env_config
from utility.league_management import RLTeammatePolicy

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import gymnasium as gym

from gymnasium import ObservationWrapper

def get_base_env(env):
    """Unwrap until we reach the actual MaisrEnv (has a 'config' dict attribute)."""
    e = env.envs[0].env
    while not hasattr(e, 'config'):
        e = e.env
    return e

class TruncateObsWrapper(ObservationWrapper):
    """Truncates observation to the first `target_dim` elements."""
    def __init__(self, env, target_dim):
        super().__init__(env)
        original_space = env.observation_space
        self.target_dim = target_dim
        self.observation_space = gym.spaces.Box(
            low=original_space.low[:target_dim],
            high=original_space.high[:target_dim],
            dtype=original_space.dtype
        )

    def observation(self, obs):
        return obs[:self.target_dim]

# ──────────────────────────────────────────────────────────────────────────────
# Agent loading (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def populate_agent_list(agent_dir, label='nolabel'):
    model_patterns = [
        os.path.join(agent_dir, "*_model.zip"),
        os.path.join(agent_dir, "**/*_model.zip")
    ]
    normstats_patterns = [
        os.path.join(agent_dir, "*_vecnormalize.pkl"),
        os.path.join(agent_dir, "**/*_vecnormalize.pkl")
    ]

    all_checkpoints = []
    for pattern in model_patterns:
        all_checkpoints.extend(glob.glob(pattern, recursive=True))

    all_normstats = []
    for pattern in normstats_patterns:
        all_normstats.extend(glob.glob(pattern, recursive=True))

    all_checkpoints = list(set(all_checkpoints))
    all_checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    all_normstats.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    # Return (model_path, vecnorm_path, name) — paths only, NOT loaded objects.
    # Workers will load models themselves to avoid pickle issues.
    agent_list = []
    for agent_model_filename in all_checkpoints:
        name = label + agent_model_filename
        prefix = agent_model_filename.replace('_model.zip', '')
        expected_vecnorm_filename = f"{prefix}_vecnormalize.pkl"
        norm_stats_path = expected_vecnorm_filename if os.path.exists(expected_vecnorm_filename) else None
        agent_list.append((agent_model_filename, norm_stats_path, name))

    print(f'Populated {len(agent_list)} {label} agents from directory {agent_dir}')
    return agent_list


def populate_agent_list_bc(agent_dir, label='nolabel'):
    """Returns (pth_path, json_path, name) tuples — paths only."""

    json_patterns = [
        os.path.join(agent_dir, "*.json"),
        os.path.join(agent_dir, "**/*.json"),
    ]
    json_files = []
    for pattern in json_patterns:
        json_files.extend(glob.glob(pattern, recursive=True))

    policy_json_files = []
    for jp in json_files:
        try:
            with open(jp, "r") as f:
                meta = json.load(f)
            if isinstance(meta, dict) and "obs_space" in meta and "act_space" in meta and "net_arch" in meta:
                policy_json_files.append(jp)
        except Exception:
            continue

    policy_json_files = list(set(policy_json_files))
    policy_json_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    agent_list = []
    for json_path in policy_json_files:
        prefix = os.path.splitext(json_path)[0]
        pth_path = prefix + ".pth"
        if not os.path.exists(pth_path):
            print(f"[populate_agent_list_bc] Skipping (missing weights): {json_path}")
            continue
        name = f"{label}{prefix}"
        # Store (pth_path, json_path, name) — json_path plays role of "vecnorm" slot
        # but we tag it so workers know it's a BC agent
        agent_list.append((pth_path, json_path, name))

    print(f"Populated {len(agent_list)} {label} BC agents from directory {agent_dir}")
    return agent_list


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────

def load_vecnormalize_wrapper(vecnorm_path, env):
    if vecnorm_path:
        vec_normalize = VecNormalize.load(vecnorm_path, venv=env)
        vec_normalize.training = False
        vec_normalize.norm_reward = False
    else:
        vec_normalize = env
    return vec_normalize


def make_wrapped_env(config, clock, window, teammate_policy, run_name='no_name', target_obs_dim=None):
    def _init():
        base_env = MaisrEnv(
            config=config,
            clock=clock,
            window=window,
            render_mode='human' if window else 'headless',
            run_name=run_name,
            tag=f'offlinestudy0',
            running_experiment=False
        )
        wrapped_env = MaisrLocalSearchWrapper(
            base_env,
            config['obs_noise_std_localsearch'],
            local_search_policy=None,
            go_to_highvalue_policy=None,
            change_region_subpolicy=None,
            evade_policy=None,
            teammate_policy=None,
        )
        wrapped_env.reset()
        if target_obs_dim is not None and wrapped_env.observation_space.shape[0] != target_obs_dim:
            print(f"[TruncateObsWrapper] Trimming obs {wrapped_env.observation_space.shape[0]} -> {target_obs_dim}")
            wrapped_env = TruncateObsWrapper(wrapped_env, target_obs_dim)
        return wrapped_env
    return _init


def run_single_rl_eval(env, agent_model, render):
    base_env = get_base_env(env)
    base_env.config['force_specific_level'] = 99

    obs = env.reset()
    done = False
    episode_reward, num_steps, target_ids, threat_ids = 0, 0, 0, 0
    while not done:
        agent_action, _ = agent_model.predict(obs, deterministic=True)
        obses, rewards, dones, infos = env.step([agent_action])
        obs = obses[0]
        reward = rewards[0]
        info = infos[0]
        done = dones[0]
        target_ids = int(info.get('target_ids', -1))
        threat_ids = int(info.get('threat_ids', -1))
        if render:
            base_env.render()
            base_env.clock.tick(60)
        episode_reward += reward
        num_steps += 1
    return episode_reward, target_ids, threat_ids, num_steps


def run_single_human_eval(env, agent_model, human_trajectory_file, level, render):
    base_env = get_base_env(env)
    base_env.level_idx = level
    base_env.config['force_specific_level'] = level
    obs = env.reset()
    done = False
    episode_reward, num_steps, target_ids, threat_ids = 0, 0, 0, 0
    step_idx = 0
    with open(human_trajectory_file, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and "timesteps" in data:
        timesteps = data["timesteps"]
    elif isinstance(data, list):
        timesteps = data
    waypoints = [entry["human_custom_waypoint"] for entry in timesteps]
    waypoints = waypoints[::10]
    current_pos = base_env.agents[base_env.aircraft_ids[1]].x, base_env.agents[base_env.aircraft_ids[1]].y
    waypoints = [wp if wp is not None else current_pos for wp in waypoints]
    while not done:
        if step_idx < len(waypoints):
            base_env.agents[base_env.aircraft_ids[1]].waypoint_override = tuple(waypoints[step_idx])
        else:
            base_env.agents[base_env.aircraft_ids[1]].waypoint_override = tuple(waypoints[-1])
        agent_action, _ = agent_model.predict(obs, deterministic=True)
        obses, rewards, dones, infos = env.step([agent_action])
        obs = obses[0]
        reward = rewards[0]
        info = infos[0]
        done = dones[0]
        if render:
            base_env.render()
            base_env.clock.tick(30)
        target_ids = int(info.get('target_ids', -1))
        threat_ids = int(info.get('threat_ids', -1))
        episode_reward += reward
        num_steps += 1
        step_idx += 1
    return episode_reward, target_ids, threat_ids, num_steps


# ──────────────────────────────────────────────────────────────────────────────
# Worker function — runs all evals for ONE testing agent
# ──────────────────────────────────────────────────────────────────────────────

def _load_agent_from_path(agent_path, vecnorm_or_json_path):
    """Load either an SB3 PPO model or a BC ActorCriticPolicy from file paths."""
    # Detect BC agent by checking if the "vecnorm" slot holds a .json file
    if vecnorm_or_json_path and vecnorm_or_json_path.endswith('.json'):
        json_path = vecnorm_or_json_path
        with open(json_path, 'r') as f:
            meta = json.load(f)

        def _space_from_json(space_spec):
            stype = space_spec.get("type")
            if stype == "Box":
                return gym.spaces.Box(low=-np.inf, high=np.inf,
                                      shape=tuple(space_spec["shape"]), dtype=np.float32)
            if stype == "Discrete":
                return gym.spaces.Discrete(int(space_spec["n"]))
            raise ValueError(f"Unsupported space type: {stype}")

        obs_space = _space_from_json(meta["obs_space"])
        act_space = _space_from_json(meta["act_space"])
        arch = meta["net_arch"]
        net_arch = dict(pi=arch, vf=arch) if isinstance(arch, list) and all(isinstance(x, int) for x in arch) else arch

        try:
            state_dict = torch.load(agent_path, weights_only=True, map_location="cpu")
        except TypeError:
            state_dict = torch.load(agent_path, map_location="cpu")

        policy = ActorCriticPolicy(
            observation_space=obs_space,
            action_space=act_space,
            lr_schedule=lambda _: 0.0,
            net_arch=net_arch,
        )
        policy.load_state_dict(state_dict)
        policy.eval()
        return policy, None  # BC agents have no vecnorm
    else:
        model = PPO.load(agent_path)
        return model, vecnorm_or_json_path  # return vecnorm path unchanged


def eval_single_testing_agent(args):
    """
    Worker function — each process runs this for one testing agent.

    Args is a dict containing everything needed to reproduce the eval for
    this agent, using only serialisable types (strings, ints, dicts, lists).

    Returns (rl_results_slice, human_results_slice) as plain dicts.
    """
    (agent_path, agent_vecnorm_path, agent_name,
     heldout_agent_paths,          # list of (model_path, vecnorm_path, name)
     dual_trajectory_files,
     solo_trajectory_files,
     config,
     num_episodes,
     num_human_episodes,
     render) = args

    # ── pygame must be initialised in each subprocess ──────────────────────
    pygame.font.init()
    clock = None
    window = None

    # ── Load this testing agent fresh in this process ──────────────────────
    agent_model, agent_vecnorm = _load_agent_from_path(agent_path, agent_vecnorm_path)

    temp_teammate_policy = RLTeammatePolicy(agent_model, None, None, None, None,
                                            norm_stats_path=agent_vecnorm)
    target_obs_dim = None
    if agent_vecnorm:
        with open(agent_vecnorm, 'rb') as f:
            saved_vecnorm = pickle.load(f)
        vecnorm_obs_dim = saved_vecnorm.observation_space.shape[0]
        target_obs_dim = vecnorm_obs_dim  # trim env obs to match vecnorm

    env_fns = [make_wrapped_env(config, clock, window, temp_teammate_policy, target_obs_dim=target_obs_dim)]
    env = DummyVecEnv(env_fns)

    try:
        env = load_vecnormalize_wrapper(agent_vecnorm, env)
    except AssertionError as e:
        print(f"\n[OBS SPACE MISMATCH] Agent: {agent_name}")
        print(f"  model path:  {agent_path}")
        print(f"  vecnorm:     {agent_vecnorm}")
        print(f"  env obs shape: {env.observation_space.shape}")
        if agent_vecnorm:
            with open(agent_vecnorm, 'rb') as f:
                saved_vecnorm = pickle.load(f)
            print(f"  vecnorm obs shape: {saved_vecnorm.observation_space.shape}")
        raise RuntimeError(f"Obs space mismatch for agent '{agent_name}' — see above") from e

    base_env = get_base_env(env)
    wrapper_env = env.envs[0]
    base_env.teammate_active = True
    wrapper_env.teammate_active = True

    print(f'\n[PID {os.getpid()}] Evaluating testing agent {agent_name}')

    rl_results_slice = {}
    human_results_slice = {}

    # ── RL teammate evals ───────────────────────────────────────────────────
    for (tm_path, tm_vecnorm_path, teammate_name) in heldout_agent_paths:
        teammate_model, _ = _load_agent_from_path(tm_path, tm_vecnorm_path)

        print(f'[PID {os.getpid()}] {agent_name} vs {teammate_name} for {num_episodes} episodes')
        for run in range(num_episodes):
            agent_basename = os.path.basename(agent_name[len('testagent'):])
            teammate_basename = os.path.basename(teammate_name[len('heldout'):])
            if agent_basename == teammate_basename:
                print(f'[PID {os.getpid()}] Skipping self-play: {agent_name} vs {teammate_name}')
                continue

            teammate = RLTeammatePolicy(teammate_model, env, None, None, None,
                                        norm_stats_path=tm_vecnorm_path)
            wrapper_env.teammate_policy = teammate
            wrapper_env.current_teammate = teammate

            reward, target_ids, threat_ids, num_steps = run_single_rl_eval(env, agent_model, render)
            rl_results_slice[(agent_name, teammate_name, run)] = (reward, target_ids, threat_ids, num_steps)

    # ── Human trajectory evals ─────────────────────────────────────────────
    all_human_trajectories = dual_trajectory_files + solo_trajectory_files
    random.seed(config['seed'])
    sampled = random.sample(all_human_trajectories,
                            min(num_human_episodes, len(all_human_trajectories)))
    for trajectory_file in sampled:
        for run in range(1):
            level = int(trajectory_file.split('_')[4][1])
            teammate_name = trajectory_file
            print(f'[PID {os.getpid()}] Human eval level={level} traj={trajectory_file}')
            reward, target_ids, threat_ids, num_steps = run_single_human_eval(
                env, agent_model, trajectory_file, level, render)
            human_results_slice[(agent_name, teammate_name, run)] = (reward, target_ids, threat_ids, num_steps)

    return rl_results_slice, human_results_slice


# ──────────────────────────────────────────────────────────────────────────────
# Plotting (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(rl_results, human_results):

    def build_unique_display_names(agent_names):
        def clean(n):
            b = os.path.basename(n)
            return re.sub(r"\.(zip|pt|pth|tar|gz)$", "", b)
        cleaned = [clean(n) for n in agent_names]
        tokenized = [c.split("_") for c in cleaned]
        max_len = max(len(t) for t in tokenized)
        padded = [t + [""] * (max_len - len(t)) for t in tokenized]
        varying_positions = [i for i in range(max_len)
                             if len(set(tokens[i] for tokens in padded)) > 1]
        if not varying_positions:
            return {name: cleaned[i][-16:] for i, name in enumerate(agent_names)}
        labels = {}
        for i, name in enumerate(agent_names):
            tokens = padded[i]
            diff_tokens = [tokens[pos] for pos in varying_positions if tokens[pos]]
            label = "_".join(diff_tokens)
            label = label.replace("learningrate", "lr").replace("batchsize", "batch")
            labels[name] = label
        return labels

    all_agents = set()
    for (agent_name, _, _) in rl_results.keys():
        all_agents.add(agent_name)
    for (agent_name, _, _) in human_results.keys():
        all_agents.add(agent_name)
    display_map = build_unique_display_names(list(all_agents)) if len(all_agents) > 1 else None

    def extract_agent_info(agent_name, display_map=None):
        base = os.path.basename(agent_name)
        base = base.replace("_model.zip", "").replace(".zip", "").replace(".pth", "").replace(".pt", "")
        name_l = base.lower()
        if "mixed75" in name_l:
            agent_type = "mixed75"
        elif "selfplay" in name_l:
            agent_type = "selfplay"
        elif "fcp" in name_l:
            agent_type = "fcp"
        elif "strategyfinetuned" in name_l or "strat" in name_l:
            agent_type = "strat-finetuned"
        elif "bc" in name_l:
            agent_type = 'bc'
        else:
            agent_type = "unknown"
        if display_map is None:
            display = agent_type
        else:
            unique_part = display_map.get(agent_name, base[-20:])
            display = f"{agent_type}-{unique_part}"
        return agent_type, display

    rl_data_rows, human_data_rows = [], []
    for (agent_name, teammate_name, run), (reward, target_ids, threat_ids, num_steps) in rl_results.items():
        agent_type, agent_display = extract_agent_info(agent_name, display_map)
        rl_data_rows.append({"agent": agent_display, "agent_type": agent_type,
                              "teammate": teammate_name, "run": run, "reward": reward,
                              "target_ids": target_ids, "threat_ids": threat_ids, "num_steps": num_steps})
    for (agent_name, teammate_name, run), (reward, target_ids, threat_ids, num_steps) in human_results.items():
        agent_type, agent_display = extract_agent_info(agent_name, display_map)
        human_data_rows.append({"agent": agent_display, "agent_type": agent_type,
                                 "teammate": teammate_name, "run": run, "reward": reward,
                                 "target_ids": target_ids, "threat_ids": threat_ids, "num_steps": num_steps})

    rl_df = pd.DataFrame(rl_data_rows)
    human_df = pd.DataFrame(human_data_rows)

    rl_stats = rl_df.groupby(["agent", "agent_type"])["reward"].agg(["mean", "std", "count"]).reset_index()
    human_stats = human_df.groupby(["agent", "agent_type"])["reward"].agg(["mean", "std", "count"]).reset_index()
    rl_target_stats = rl_df.groupby(["agent", "agent_type"])["target_ids"].agg(["mean", "std", "count"]).reset_index()
    human_target_stats = human_df.groupby(["agent", "agent_type"])["target_ids"].agg(["mean", "std", "count"]).reset_index()
    rl_threat_stats = rl_df.groupby(["agent", "agent_type"])["threat_ids"].agg(["mean", "std", "count"]).reset_index()
    human_threat_stats = human_df.groupby(["agent", "agent_type"])["threat_ids"].agg(["mean", "std", "count"]).reset_index()

    agent_colors = {"bc": "#4C72B0", "fcp": "#2E86AB", "mixed75": "#A23B72",
                    "selfplay": "#F18F01", "strat-finetuned": "#C73E1D", "unknown": "#808080"}
    agent_labels = {"fcp": "FCP", "bc": "BC", "mixed75": "Strat-FCP",
                    "selfplay": "SP", "strat-finetuned": "Strat-SP", "unknown": "Unknown"}

    for df in [rl_stats, human_stats, rl_target_stats, human_target_stats, rl_threat_stats, human_threat_stats]:
        df.sort_values(["agent_type", "agent"], inplace=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    for col, (stats, title) in enumerate([
        (rl_stats, "Performance with Held-Out RL Teammates"),
        (human_stats, "Performance with Recorded Human Teammates"),
    ]):
        colors = [agent_colors.get(t, "#808080") for t in stats["agent_type"]]
        axes[0, col].bar(range(len(stats)), stats["mean"], yerr=stats["std"], capsize=5, alpha=0.9, color=colors)
        axes[0, col].set_title(title, fontsize=16)
        axes[0, col].set_xlabel("Testing agent")
        axes[0, col].set_ylabel("Average reward")
        axes[0, col].set_xticks(range(len(stats)))
        axes[0, col].set_xticklabels(stats["agent"], rotation=45, ha="right")
        axes[0, col].set_ylim(0, 45)
        axes[0, col].grid(axis="y", alpha=0.3)

    for col, (stats, title) in enumerate([
        (rl_target_stats, "Target IDs with Held-Out RL Teammates"),
        (human_target_stats, "Target IDs with Recorded Human Teammates"),
    ]):
        colors = [agent_colors.get(t, "#808080") for t in stats["agent_type"]]
        axes[1, col].bar(range(len(stats)), stats["mean"], yerr=stats["std"], capsize=5, alpha=0.9, color=colors)
        axes[1, col].set_title(title, fontsize=16)
        axes[1, col].set_xlabel("Testing agent")
        axes[1, col].set_ylabel("Average target IDs")
        axes[1, col].set_xticks(range(len(stats)))
        axes[1, col].set_xticklabels(stats["agent"], rotation=45, ha="right")
        axes[1, col].grid(axis="y", alpha=0.3)

    for col, (stats, title) in enumerate([
        (rl_threat_stats, "Threat IDs with Held-Out RL Teammates"),
        (human_threat_stats, "Threat IDs with Recorded Human Teammates"),
    ]):
        colors = [agent_colors.get(t, "#808080") for t in stats["agent_type"]]
        axes[2, col].bar(range(len(stats)), stats["mean"], yerr=stats["std"], capsize=5, alpha=0.9, color=colors)
        axes[2, col].set_title(title, fontsize=16)
        axes[2, col].set_xlabel("Testing agent")
        axes[2, col].set_ylabel("Average threat IDs")
        axes[2, col].set_xticks(range(len(stats)))
        axes[2, col].set_xticklabels(stats["agent"], rotation=45, ha="right")
        axes[2, col].grid(axis="y", alpha=0.3)

    legend_handles, legend_labels_list = [], []
    for t in ["bc", "fcp", "mixed75", "selfplay", "strat-finetuned", "unknown"]:
        if rl_stats["agent_type"].eq(t).any() or human_stats["agent_type"].eq(t).any():
            legend_handles.append(plt.Line2D([0], [0], marker="s", color="w",
                                             markerfacecolor=agent_colors[t], markersize=10))
            legend_labels_list.append(agent_labels[t])
    fig.legend(legend_handles, legend_labels_list, loc="upper center", ncol=len(legend_labels_list))
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    for label, stats in [#("RL AGENT PERFORMANCE SUMMARY", rl_stats),
                          ("RL AGENT TARGET IDs", rl_target_stats),
                          ("RL AGENT THREAT IDs", rl_threat_stats),
                          #("HUMAN TRAJECTORY PERFORMANCE SUMMARY", human_stats),
                          ("HUMAN TRAJECTORY TARGET IDs", human_target_stats),
                          ("HUMAN TRAJECTORY THREAT IDs", human_threat_stats)]:
        print(f'\n====== {label} ======')
        col = "reward" if "PERFORMANCE" in label else ("target_ids" if "TARGET" in label else "threat_ids")
        # stats already have mean/std/count from groupby
        for _, row in stats.iterrows():
            print(f'{row["agent"]}: {col.replace("_ids","").capitalize()}={row["mean"]:.2f} ± {row["std"]:.2f} (n={row["count"]})')


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    multiprocessing.set_start_method('spawn', force=True)

    config_filename = '../../configs/main_config.json'
    config = load_env_config(config_filename)
    config['use_stuck_detection'] = False
    config['prob_detect'] = 0
    config['action_type'] = 'Discrete16'

    render = False
    num_episodes = 2
    num_human_episodes = 2

    # ── Pygame (main process only, for non-worker use) ─────────────────────
    pygame.font.init()

    # ── Directories ────────────────────────────────────────────────────────
    testing_agent_dir = 'testing_agents/rl'
    heldout_agent_dir = 'heldout_agents'
    human_trajectory_dir = 'heldout_humans'

    # Populate as (path, vecnorm_path, name) — no loaded objects
    testing_agents = populate_agent_list(testing_agent_dir, label='testagent')
    heldout_agent_paths = populate_agent_list(heldout_agent_dir, label='heldout') + populate_agent_list(testing_agent_dir, label='testagent')
    print(f'Heldout agent paths:\n{heldout_agent_paths}')

    dual_trajectory_files = glob.glob(f"{human_trajectory_dir}/subject_*/timestep_data/timesteps_[ABC][13457]_*.json")
    solo_trajectory_files = glob.glob(f"{human_trajectory_dir}/subject_*/timestep_data/timesteps_[PS][13457]_*.json")
    if not dual_trajectory_files:
        raise ValueError("[Human trajectory loading] No dual trajectories found")
    if not solo_trajectory_files:
        raise ValueError("[Human trajectory loading] No solo trajectories found")
    print(f'[Human trajectories] {len(dual_trajectory_files)} dual + {len(solo_trajectory_files)} solo')

    # ── Build worker args — one per testing agent ──────────────────────────
    worker_args = [
        (agent_path, agent_vecnorm_path, agent_name,
         heldout_agent_paths,
         dual_trajectory_files,
         solo_trajectory_files,
         config,
         num_episodes,
         num_human_episodes,
         render)
        for (agent_path, agent_vecnorm_path, agent_name) in testing_agents
    ]

    # ── Choose number of workers ───────────────────────────────────────────
    # Cap at number of testing agents (no point spawning more workers than tasks)
    # and leave one core free for the OS.  Tune NUM_WORKERS to your machine.
    NUM_WORKERS = min(len(testing_agents), max(1, cpu_count() - 1))
    print(f'\nLaunching {NUM_WORKERS} workers for {len(testing_agents)} testing agents '
          f'(machine has {cpu_count()} logical cores)\n')

    # ── Run in parallel ────────────────────────────────────────────────────
    rl_results = {}
    human_results = {}

    with Pool(processes=NUM_WORKERS) as pool:
        for rl_slice, human_slice in pool.map(eval_single_testing_agent, worker_args):
            rl_results.update(rl_slice)
            human_results.update(human_slice)

    # ── Plot + summarise ───────────────────────────────────────────────────
    plot_results(rl_results, human_results)


if __name__ == '__main__':
    main()
