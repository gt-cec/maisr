import ctypes
import glob
import os

import json
from datetime import datetime

import pygame
import re

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.policies import ActorCriticPolicy

from base_env import MaisrEnv
from utility.localsearch_training_wrapper import MaisrLocalSearchWrapper
from utility.config_management import load_env_config
from utility.league_management import (RLTeammatePolicy)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import gymnasium as gym


def populate_agent_list(agent_dir, label='nolabel'):
    """
    Loads a collection of RL agents to be evaluated in the offline evaluation.
    Returns a list containing tuples containing:
        * The model object for the agent's policy
        * The vecnormalize stats to load when running the agent
        * The agent's name
    """

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

    all_checkpoints = list(set(all_checkpoints))  # Remove duplicates and sort by modification time (newest first)
    all_checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    all_normstats.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    agent_list = []
    for agent_model_filename in all_checkpoints:
        model = PPO.load(agent_model_filename)
        name = label + agent_model_filename

        # Extract the prefix by removing '_model.zip' suffix
        prefix = agent_model_filename.replace('_model.zip', '')
        expected_vecnorm_filename = f"{prefix}_vecnormalize.pkl"

        norm_stats_path = None
        if os.path.exists(expected_vecnorm_filename):
            norm_stats_path = expected_vecnorm_filename

        agent_list.append((model, norm_stats_path, name))

    print(f'Populated {len(agent_list)} {label} agents from directory {agent_dir}')
    return agent_list

def populate_agent_list_bc(agent_dir, label='nolabel'):
    """
    Loads a collection of BC policies saved as:
      - <prefix>.json  (policy metadata: obs/act spaces, net_arch, etc.)
      - <prefix>.pth   (torch state_dict weights)

    Returns a list of tuples:
        (policy, norm_stats_path=None, name)
    """

    def _space_from_json(space_spec: dict):
        stype = space_spec.get("type", None)

        if stype == "Box":
            shape = tuple(space_spec["shape"])
            # Low/high aren't stored in your example json; use unbounded float32.
            # If your obs is bounded, add "low"/"high" fields in json and use them here.
            return gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=shape,
                dtype=np.float32,
            )

        if stype == "Discrete":
            return gym.spaces.Discrete(int(space_spec["n"]))

        raise ValueError(f"Unsupported space type in BC json: {stype} (spec={space_spec})")

    # Find json files recursively; each should have a matching .pth with same prefix
    json_patterns = [
        os.path.join(agent_dir, "*.json"),
        os.path.join(agent_dir, "**/*.json"),
    ]

    json_files = []
    for pattern in json_patterns:
        json_files.extend(glob.glob(pattern, recursive=True))

    # Filter out non-policy jsons if needed; here we keep those with required keys
    policy_json_files = []
    for jp in json_files:
        try:
            with open(jp, "r") as f:
                meta = json.load(f)
            if isinstance(meta, dict) and "obs_space" in meta and "act_space" in meta and "net_arch" in meta:
                policy_json_files.append(jp)
        except Exception:
            continue

    # Sort newest first (similar to populate_agent_list)
    policy_json_files = list(set(policy_json_files))
    policy_json_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    agent_list = []

    for json_path in policy_json_files:
        prefix = os.path.splitext(json_path)[0]
        pth_path = prefix + ".pth"

        if not os.path.exists(pth_path):
            print(f"[populate_agent_list_bc] Skipping (missing weights): {json_path} -> expected {pth_path}")
            continue

        with open(json_path, "r") as f:
            meta = json.load(f)

        obs_space = _space_from_json(meta["obs_space"])
        act_space = _space_from_json(meta["act_space"])

        # Your bc_policy.json has net_arch like [64, 64] :contentReference[oaicite:2]{index=2}.
        # ActorCriticPolicy expects either shared arch list[int] or dict(pi=..., vf=...).
        # Use symmetric pi/vf MLPs by default.
        arch = meta["net_arch"]
        if isinstance(arch, list) and all(isinstance(x, int) for x in arch):
            net_arch = dict(pi=arch, vf=arch)
        else:
            # If you later store a richer SB3 net_arch object, pass through.
            net_arch = arch

        # Load weights
        try:
            state_dict = torch.load(pth_path, weights_only=True, map_location="cpu")
        except TypeError:
            # Older torch versions don't support weights_only
            state_dict = torch.load(pth_path, map_location="cpu")

        policy = ActorCriticPolicy(
            observation_space=obs_space,
            action_space=act_space,
            lr_schedule=lambda _: 0.0,  # inference only
            net_arch=net_arch,
        )
        policy.load_state_dict(state_dict)
        policy.eval()

        name = f"{label}{prefix}"
        agent_list.append((policy, None, name))

    print(f"Populated {len(agent_list)} {label} BC agents from directory {agent_dir}")
    return agent_list


def load_vecnormalize_wrapper(vecnorm_path, env):
    """Load saved VecNormalize wrapper with stats from training and apply it to the new environment."""
    print(f"Loading VecNormalize stats from: {vecnorm_path}")

    if vecnorm_path:
        vec_normalize = VecNormalize.load(vecnorm_path, venv=env)
        vec_normalize.training = False  # Disable further normalization updates
        vec_normalize.norm_reward = False
    else:
        vec_normalize = env
    return vec_normalize


def make_wrapped_env(config, clock, window, teammate_policy, run_name='no_name'):
    def _init():
        base_env = MaisrEnv(
            config=config,
            clock=clock,
            window=window,
            render_mode='human' if window else 'headless',
            run_name=run_name,
            tag=f'offlinestudy0',
            #agent_appearance='black',
            running_experiment=False
        )

        wrapped_env = MaisrLocalSearchWrapper(
            base_env,
            config['obs_noise_std_localsearch'],
            local_search_policy=None,  # subpolicies['local_search'],
            go_to_highvalue_policy=None,  # subpolicies['go_to_threat'],
            change_region_subpolicy=None,  # subpolicies['change_region'],
            evade_policy=None,  # EvadeDetection(model_path=None),
            teammate_policy=None,
        )

        #wrapped_env = Monitor(wrapped_env)
        wrapped_env.reset()
        return wrapped_env

    return _init


def run_single_rl_eval(env, agent_model, render):

    base_env = env.envs[0].env
    base_env.config['force_specific_level'] = 99

    obs = env.reset()
    done = False
    episode_reward, num_steps, target_ids, threat_ids = 0, 0, 0, 0

    while not done:

        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         done = True
        #         break
        #     elif event.type == pygame.KEYDOWN:
        #         if event.key == pygame.K_ESCAPE:
        #             done = True
        #             break

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

    base_env = env.envs[0].env

    base_env.level_idx = level
    base_env.config['force_specific_level'] = level

    obs = env.reset()
    done = False
    episode_reward, num_steps, target_ids, threat_ids = 0, 0, 0, 0
    step_idx = 0

    # Load trajectory
    with open(human_trajectory_file, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict) and "timesteps" in data:
        timesteps = data["timesteps"]
    elif isinstance(data, list):
        timesteps = data
    waypoints = [entry["human_custom_waypoint"] for entry in timesteps]
    waypoints = waypoints[::10]  # Timescale correction
    current_pos = base_env.agents[base_env.aircraft_ids[1]].x, base_env.agents[base_env.aircraft_ids[1]].y
    waypoints = [wp if wp is not None else current_pos for wp in waypoints]

    while not done:
        if step_idx < len(waypoints):
            base_env.agents[base_env.aircraft_ids[1]].waypoint_override = tuple(waypoints[step_idx])
        else:
            base_env.agents[base_env.aircraft_ids[1]].waypoint_override = tuple(waypoints[-1])  # hold last

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


def main():
    config_filename = '../../configs/main_config.json'
    config = load_env_config(config_filename)
    config['use_stuck_detection'] = False
    config['prob_detect'] = 0  # 0.0003
    config['action_type'] = 'Discrete16'

    render = False
    num_episodes = 3 # 250
    num_human_episodes = 3 # 250

    ####################################     Pygame setup     ####################################
    if render:
        if hasattr(ctypes, 'windll') and hasattr(ctypes.windll, 'user32'): ctypes.windll.user32.SetProcessDPIAware()
        pygame.display.init()
        pygame.font.init()
        clock = pygame.time.Clock()

        window_width, window_height = config['window_size'][0], config['window_size'][1]
        window = pygame.display.set_mode((window_width, window_height))

    else:
        pygame.font.init()
        window = None
        clock = None


    ####################################################################################################################
    ####################################     Instantiate testing agents     ############################################
    ####################################################################################################################

    testing_agent_dir = 'testing_agents/rl'  # Agents being tested
    heldout_agent_dir = 'heldout_agents' # Held out agents to test with
    human_trajectory_dir = 'heldout_humans' # human_trajectories_for_training # Held out humans to test with

    if testing_agent_dir == 'testing_agents/bc':
        testing_agents = populate_agent_list_bc(testing_agent_dir, label = 'testagent')
    else:
        testing_agents = populate_agent_list(testing_agent_dir, label='testagent')

    heldout_agents = populate_agent_list(heldout_agent_dir, label='heldout')
    print(f'Contents of heldout_agents:\n{heldout_agents}')


    ####################################################################################################################
    #################################### Load human trajectories #######################################################
    ####################################################################################################################

    dual_trajectory_files = glob.glob(f"{human_trajectory_dir}/subject_*/timestep_data/timesteps_[ABC][13457]_*.json")
    solo_trajectory_files = glob.glob(f"{human_trajectory_dir}/subject_*/timestep_data/timesteps_[PS][13457]_*.json")
    if not dual_trajectory_files:
        print(f"[Human trajectory loading] No dual trajectories found")
        raise ValueError
    if not solo_trajectory_files:
        print(f"[Human trajectory loading] No solo trajectories found")
        raise ValueError
    print(f'[Human trajectories] Loaded {len(dual_trajectory_files)} dual trajectories and {len(solo_trajectory_files)} solo trajectories')

    rl_results = {}
    human_results = {}
    for agent_tuple in testing_agents:
        agent_model, agent_vecnorm, agent_name = agent_tuple

        temp_teammate_policy = RLTeammatePolicy(agent_model, None, None, None, None, norm_stats_path=agent_vecnorm)

        env_fns = [make_wrapped_env(config, clock, window, temp_teammate_policy) for _ in range(1)]
        env = DummyVecEnv(env_fns)
        env = load_vecnormalize_wrapper(agent_vecnorm, env)
        print(f'\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        print(f'NEW TESTING AGENT, loaded new vecnorm stats from {agent_vecnorm}')
        print(f'&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n')
        base_env = env.envs[0].env
        wrapper_env = env.envs[0]
        base_env.teammate_active = True
        wrapper_env.teammate_active = True

        #################################### Run RL agent evals ####################################
        print(f'\n#########################################################')
        print(f'Evaluating testing agent {agent_name}')
        print(f'#########################################################\n')


        for teammate_tuple in heldout_agents:
            print(f'\n%%%%%% Evaluating {teammate_tuple[2]} with held-out RL teammates for {num_episodes} episodes')

            for run in range(num_episodes):
                teammate_model = teammate_tuple[0]
                teammate_vecnorm = teammate_tuple[1]
                teammate_name = teammate_tuple[2]

                agent_basename = os.path.basename(agent_name[len('testagent'):])
                teammate_basename = os.path.basename(teammate_name[len('heldout'):])
                if agent_basename == teammate_basename:
                    print(f'% Skipping self-play: {agent_name} vs {teammate_name}')
                    continue

                teammate = RLTeammatePolicy(teammate_model, env, None, None, None, norm_stats_path=teammate_vecnorm)
                #print(f'teammate is using model {teammate_model} and norm stats {teammate_vecnorm}')

                wrapper_env.teammate_policy = teammate
                wrapper_env.current_teammate = teammate

                #print(f'\nRunning single RL eval with agent {agent_name} (model {agent_model}, vecnorm stats {agent_vecnorm}). Teammate is {teammate_name} (model {teammate_model}, vecnorm {teammate_vecnorm})')
                reward, target_ids, threat_ids, num_steps = run_single_rl_eval(env, agent_model, render)
                rl_results[(agent_name, teammate_name, run)] = reward, target_ids, threat_ids, num_steps


        #################################### Run human trajectory evals ####################################
        all_human_trajectories = dual_trajectory_files + solo_trajectory_files
        for trajectory_file in all_human_trajectories[0:num_human_episodes]:
            for run in range(1):
                level = int(trajectory_file.split('_')[4][1])

                print(f'Running human eval with level {level} and trajectory {trajectory_file}')
                teammate_name = trajectory_file
                reward, target_ids, threat_ids, num_steps = run_single_human_eval(env, agent_model, trajectory_file, level, render)
                human_results[(agent_name, teammate_name, run)] = reward, target_ids, threat_ids, num_steps

    #print(f'====== ====== FINAL RESULTS ====== ====== ')

    def build_unique_display_names(agent_names):
        """
        Given a list of agent filenames, automatically extract the minimal
        distinguishing tokens between them.
        Returns: {original_name: short_unique_label}
        """

        # strip paths + extensions
        def clean(n):
            b = os.path.basename(n)
            return re.sub(r"\.(zip|pt|pth|tar|gz)$", "", b)

        cleaned = [clean(n) for n in agent_names]

        # tokenize on underscores (works well for ML checkpoint naming)
        tokenized = [c.split("_") for c in cleaned]

        # pad to equal length
        max_len = max(len(t) for t in tokenized)
        padded = [t + [""] * (max_len - len(t)) for t in tokenized]

        # find which token positions vary across agents
        varying_positions = []
        for i in range(max_len):
            column = [tokens[i] for tokens in padded]
            if len(set(column)) > 1:
                varying_positions.append(i)

        # fallback: if everything identical (rare but possible)
        if not varying_positions:
            return {name: cleaned[i][-16:] for i, name in enumerate(agent_names)}

        # build minimal distinguishing label
        labels = {}
        for i, name in enumerate(agent_names):
            tokens = padded[i]
            diff_tokens = [tokens[pos] for pos in varying_positions if tokens[pos]]
            label = "_".join(diff_tokens)

            # keep it readable
            label = label.replace("learningrate", "lr")
            label = label.replace("batchsize", "batch")

            labels[name] = label

        return labels

    all_agents = set()
    for (agent_name, _, _) in rl_results.keys():
        all_agents.add(agent_name)
    for (agent_name, _, _) in human_results.keys():
        all_agents.add(agent_name)
    display_map = None
    if len(all_agents) > 1:
        display_map = build_unique_display_names(list(all_agents))

    def extract_agent_info(agent_name: str, display_map=None):

        """Return a unique agent id + a coarse type label for coloring."""
        base = os.path.basename(agent_name)
        base = base.replace("_model.zip", "").replace(".zip", "").replace(".pth", "").replace(".pt", "")
        name_l = base.lower()

        # agent type (for colors/legend)
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

    # Separate RL and human results
    rl_data_rows = []
    human_data_rows = []

    for (agent_name, teammate_name, run), (reward, target_ids, threat_ids, num_steps) in rl_results.items():
        agent_type, agent_display = extract_agent_info(agent_name, display_map)
        rl_data_rows.append({
            "agent": agent_display,          # UNIQUE per testing agent
            "agent_type": agent_type,        # for coloring
            "teammate": teammate_name,
            "run": run,
            "reward": reward,
            "target_ids": target_ids,
            "threat_ids": threat_ids,
            "num_steps": num_steps,
        })

    for (agent_name, teammate_name, run), (reward, target_ids, threat_ids, num_steps) in human_results.items():
        agent_type, agent_display = extract_agent_info(agent_name, display_map)
        human_data_rows.append({
            "agent": agent_display,          # UNIQUE per testing agent
            "agent_type": agent_type,        # for coloring
            "teammate": teammate_name,
            "run": run,
            "reward": reward,
            "target_ids": target_ids,
            "threat_ids": threat_ids,
            "num_steps": num_steps,
        })

    rl_df = pd.DataFrame(rl_data_rows)
    human_df = pd.DataFrame(human_data_rows)

    # Calculate statistics PER TESTING AGENT (not collapsed by type)
    rl_stats = rl_df.groupby(["agent", "agent_type"])["reward"].agg(["mean", "std", "count"]).reset_index()
    human_stats = human_df.groupby(["agent", "agent_type"])["reward"].agg(["mean", "std", "count"]).reset_index()

    # Calculate target_ids and threat_ids statistics
    rl_target_stats = rl_df.groupby(["agent", "agent_type"])["target_ids"].agg(["mean", "std", "count"]).reset_index()
    human_target_stats = human_df.groupby(["agent", "agent_type"])["target_ids"].agg(
        ["mean", "std", "count"]).reset_index()
    rl_threat_stats = rl_df.groupby(["agent", "agent_type"])["threat_ids"].agg(["mean", "std", "count"]).reset_index()
    human_threat_stats = human_df.groupby(["agent", "agent_type"])["threat_ids"].agg(
        ["mean", "std", "count"]).reset_index()

    # Define consistent colors and label mapping for agent types
    agent_colors = {
        "bc": "#4C72B0",
        "fcp": "#2E86AB",
        "mixed75": "#A23B72",
        "selfplay": "#F18F01",
        "strat-finetuned": "#C73E1D",
        "unknown": "#808080",
    }

    agent_labels = {
        "fcp": "FCP",
        "bc": "BC",
        "mixed75": "Strat-FCP",
        "selfplay": "SP",
        "strat-finetuned": "Strat-SP",
        "unknown": "Unknown",
    }

    # Sort so bars are grouped nicely by type then name
    rl_stats = rl_stats.sort_values(["agent_type", "agent"])
    human_stats = human_stats.sort_values(["agent_type", "agent"])
    rl_target_stats = rl_target_stats.sort_values(["agent_type", "agent"])
    human_target_stats = human_target_stats.sort_values(["agent_type", "agent"])
    rl_threat_stats = rl_threat_stats.sort_values(["agent_type", "agent"])
    human_threat_stats = human_threat_stats.sort_values(["agent_type", "agent"])

    # Create 3x2 subplot grid (3 rows: reward, targets, threats; 2 cols: RL, Human)
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    # Row 1: Reward
    rl_colors = [agent_colors.get(t, "#808080") for t in rl_stats["agent_type"]]
    axes[0, 0].bar(range(len(rl_stats)), rl_stats["mean"], yerr=rl_stats["std"], capsize=5, alpha=0.9, color=rl_colors)
    axes[0, 0].set_title("Performance with Held-Out RL Teammates", fontsize=16)
    axes[0, 0].set_xlabel("Testing agent")
    axes[0, 0].set_ylabel("Average reward")
    axes[0, 0].set_xticks(range(len(rl_stats)))
    axes[0, 0].set_xticklabels(rl_stats["agent"], rotation=45, ha="right")
    axes[0, 0].set_ylim(0, 45)
    axes[0, 0].grid(axis="y", alpha=0.3)

    human_colors = [agent_colors.get(t, "#808080") for t in human_stats["agent_type"]]
    axes[0, 1].bar(range(len(human_stats)), human_stats["mean"], yerr=human_stats["std"], capsize=5, alpha=0.9,
                   color=human_colors)
    axes[0, 1].set_title("Performance with Recorded Human Teammates", fontsize=16)
    axes[0, 1].set_xlabel("Testing agent")
    axes[0, 1].set_ylabel("Average reward")
    axes[0, 1].set_xticks(range(len(human_stats)))
    axes[0, 1].set_xticklabels(human_stats["agent"], rotation=45, ha="right")
    axes[0, 1].set_ylim(0, 45)
    axes[0, 1].grid(axis="y", alpha=0.3)

    # Row 2: Target IDs
    rl_target_colors = [agent_colors.get(t, "#808080") for t in rl_target_stats["agent_type"]]
    axes[1, 0].bar(range(len(rl_target_stats)), rl_target_stats["mean"], yerr=rl_target_stats["std"], capsize=5,
                   alpha=0.9, color=rl_target_colors)
    axes[1, 0].set_title("Target IDs with Held-Out RL Teammates", fontsize=16)
    axes[1, 0].set_xlabel("Testing agent")
    axes[1, 0].set_ylabel("Average target IDs")
    axes[1, 0].set_xticks(range(len(rl_target_stats)))
    axes[1, 0].set_xticklabels(rl_target_stats["agent"], rotation=45, ha="right")
    axes[1, 0].grid(axis="y", alpha=0.3)

    human_target_colors = [agent_colors.get(t, "#808080") for t in human_target_stats["agent_type"]]
    axes[1, 1].bar(range(len(human_target_stats)), human_target_stats["mean"], yerr=human_target_stats["std"],
                   capsize=5, alpha=0.9, color=human_target_colors)
    axes[1, 1].set_title("Target IDs with Recorded Human Teammates", fontsize=16)
    axes[1, 1].set_xlabel("Testing agent")
    axes[1, 1].set_ylabel("Average target IDs")
    axes[1, 1].set_xticks(range(len(human_target_stats)))
    axes[1, 1].set_xticklabels(human_target_stats["agent"], rotation=45, ha="right")
    axes[1, 1].grid(axis="y", alpha=0.3)

    # Row 3: Threat IDs
    rl_threat_colors = [agent_colors.get(t, "#808080") for t in rl_threat_stats["agent_type"]]
    axes[2, 0].bar(range(len(rl_threat_stats)), rl_threat_stats["mean"], yerr=rl_threat_stats["std"], capsize=5,
                   alpha=0.9, color=rl_threat_colors)
    axes[2, 0].set_title("Threat IDs with Held-Out RL Teammates", fontsize=16)
    axes[2, 0].set_xlabel("Testing agent")
    axes[2, 0].set_ylabel("Average threat IDs")
    axes[2, 0].set_xticks(range(len(rl_threat_stats)))
    axes[2, 0].set_xticklabels(rl_threat_stats["agent"], rotation=45, ha="right")
    axes[2, 0].grid(axis="y", alpha=0.3)

    human_threat_colors = [agent_colors.get(t, "#808080") for t in human_threat_stats["agent_type"]]
    axes[2, 1].bar(range(len(human_threat_stats)), human_threat_stats["mean"], yerr=human_threat_stats["std"],
                   capsize=5, alpha=0.9, color=human_threat_colors)
    axes[2, 1].set_title("Threat IDs with Recorded Human Teammates", fontsize=16)
    axes[2, 1].set_xlabel("Testing agent")
    axes[2, 1].set_ylabel("Average threat IDs")
    axes[2, 1].set_xticks(range(len(human_threat_stats)))
    axes[2, 1].set_xticklabels(human_threat_stats["agent"], rotation=45, ha="right")
    axes[2, 1].grid(axis="y", alpha=0.3)

    # Legend (type -> color)
    legend_handles = []
    legend_labels = []
    for t in ["bc", "fcp", "mixed75", "selfplay", "strat-finetuned", "unknown"]:
        if (rl_stats["agent_type"].eq(t).any()) or (human_stats["agent_type"].eq(t).any()):
            legend_handles.append(plt.Line2D([0], [0], marker="s", color="w",
                                             markerfacecolor=agent_colors[t], markersize=10))
            legend_labels.append(agent_labels[t])
    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=len(legend_labels))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    print(f'\n====== RL AGENT PERFORMANCE SUMMARY ======')
    for _, row in rl_stats.iterrows():
        agent_label = agent_labels.get(row["agent_type"], row["agent"])
        print(f'{row["agent"]}: Reward={row["mean"]:.2f} ± {row["std"]:.2f} (n={row["count"]})')

    print(f'\n====== RL AGENT TARGET IDs ======')
    for _, row in rl_target_stats.iterrows():
        print(f'{row["agent"]}: Targets={row["mean"]:.2f} ± {row["std"]:.2f} (n={row["count"]})')

    print(f'\n====== RL AGENT THREAT IDs ======')
    for _, row in rl_threat_stats.iterrows():
        print(f'{row["agent"]}: Threats={row["mean"]:.2f} ± {row["std"]:.2f} (n={row["count"]})')

    print(f'\n====== HUMAN TRAJECTORY PERFORMANCE SUMMARY ======')
    for _, row in human_stats.iterrows():
        agent_label = agent_labels.get(row["agent_type"], row["agent"])
        print(f'{row["agent"]}: Reward={row["mean"]:.2f} ± {row["std"]:.2f} (n={row["count"]})')

    print(f'\n====== HUMAN TRAJECTORY TARGET IDs ======')
    for _, row in human_target_stats.iterrows():
        print(f'{row["agent"]}: Targets={row["mean"]:.2f} ± {row["std"]:.2f} (n={row["count"]})')

    print(f'\n====== HUMAN TRAJECTORY THREAT IDs ======')
    for _, row in human_threat_stats.iterrows():
        print(f'{row["agent"]}: Threats={row["mean"]:.2f} ± {row["std"]:.2f} (n={row["count"]})')

    # Calculate statistics PER TESTING AGENT (not collapsed by type)
    # rl_stats = rl_df.groupby(["agent", "agent_type"])["reward"].agg(["mean", "std", "count"]).reset_index()
    # human_stats = human_df.groupby(["agent", "agent_type"])["reward"].agg(["mean", "std", "count"]).reset_index()
    #
    # # Define consistent colors and label mapping for agent types
    # agent_colors = {
    #     "bc": "#4C72B0",
    #     "fcp": "#2E86AB",
    #     "mixed75": "#A23B72",
    #     "selfplay": "#F18F01",
    #     "strat-finetuned": "#C73E1D",
    #     "unknown": "#808080",
    # }
    #
    # agent_labels = {
    #     "fcp": "FCP",
    #     "bc": "BC",
    #     "mixed75": "Strat-FCP",
    #     "selfplay": "SP",
    #     "strat-finetuned": "Strat-SP",
    #     "unknown": "Unknown",
    # }
    #
    # # Sort so bars are grouped nicely by type then name
    # rl_stats = rl_stats.sort_values(["agent_type", "agent"])
    # human_stats = human_stats.sort_values(["agent_type", "agent"])
    #
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    #
    # # RL subplot
    # rl_colors = [agent_colors.get(t, "#808080") for t in rl_stats["agent_type"]]
    # ax1.bar(range(len(rl_stats)), rl_stats["mean"], yerr=rl_stats["std"], capsize=5, alpha=0.9, color=rl_colors)
    # ax1.set_title("Performance with Held-Out RL Teammates", fontsize=16)
    # ax1.set_xlabel("Testing agent")
    # ax1.set_ylabel("Average reward")
    # ax1.set_xticks(range(len(rl_stats)))
    # ax1.set_xticklabels(rl_stats["agent"], rotation=45, ha="right")
    # ax1.set_ylim(0, 45)
    # ax1.grid(axis="y", alpha=0.3)
    #
    # # Human subplot
    # human_colors = [agent_colors.get(t, "#808080") for t in human_stats["agent_type"]]
    # ax2.bar(range(len(human_stats)), human_stats["mean"], yerr=human_stats["std"], capsize=5, alpha=0.9, color=human_colors)
    # ax2.set_title("Performance with Recorded Human Teammates", fontsize=16)
    # ax2.set_xlabel("Testing agent")
    # ax2.set_ylabel("Average reward")
    # ax2.set_xticks(range(len(human_stats)))
    # ax2.set_xticklabels(human_stats["agent"], rotation=45, ha="right")
    # ax2.set_ylim(0, 45)  # (bugfix: this used to incorrectly set ax1 twice)
    # ax2.grid(axis="y", alpha=0.3)
    #
    # # Legend (type -> color)
    # legend_handles = []
    # legend_labels = []
    # for t in ["bc", "fcp", "mixed75", "selfplay", "strat-finetuned", "unknown"]:
    #     if (rl_stats["agent_type"].eq(t).any()) or (human_stats["agent_type"].eq(t).any()):
    #         legend_handles.append(plt.Line2D([0], [0], marker="s", color="w",
    #                                          markerfacecolor=agent_colors[t], markersize=10))
    #         legend_labels.append(agent_labels[t])
    # fig.legend(legend_handles, legend_labels, loc="upper center", ncol=len(legend_labels))
    #
    # plt.tight_layout(rect=[0, 0, 1, 0.92])
    # plt.show()
    #
    # print(f'\n====== RL AGENT PERFORMANCE SUMMARY ======')
    # for _, row in rl_stats.iterrows():
    #     agent_label = agent_labels.get(row["agent"], row["agent"])
    #     print(f'{agent_label}: {row["mean"]:.2f} ± {row["std"]:.2f} (n={row["count"]})')
    #
    # print(f'\n====== HUMAN TRAJECTORY PERFORMANCE SUMMARY ======')
    # for _, row in human_stats.iterrows():
    #     agent_label = agent_labels.get(row["agent"], row["agent"])
    #     print(f'{agent_label}: {row["mean"]:.2f} ± {row["std"]:.2f} (n={row["count"]})')


if __name__ == '__main__':
    main()