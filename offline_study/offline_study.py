import argparse
import ctypes
import glob
import json
import os

import pygame
import numpy as np
import random
import re

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from env_multi_new import MAISREnvVec
from utility.localsearch_training_wrapper import MaisrLocalSearchWrapper
from utility.config import subject_id
from utility.data_logging import load_env_config
from utility.league_management import (GenericTeammatePolicy, SubPolicy, LocalSearch, ChangeRegions, GoToNearestThreat, EvadeDetection, TeammateManager, RLTeammatePolicy)


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


def load_vecnormalize_wrapper(vecnorm_path, env):
    """Load saved VecNormalize wrapper with stats from training and apply it to the new environment."""
    print(f"Loading VecNormalize stats from: {vecnorm_path}")

    vec_normalize = VecNormalize.load(vecnorm_path, venv=env)
    vec_normalize.training = False  # Disable further normalization updates
    vec_normalize.norm_reward = False
    return vec_normalize


def make_wrapped_env(config, clock, window, teammate_policy, run_name='no_name'):
    def _init():
        base_env = MAISREnvVec(
            config=config,
            clock=clock,
            window=window,
            render_mode='human',
            run_name=f'user_study_subj{subject_id}',
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

    obs = env.reset()
    done = False
    episode_reward, num_steps, target_ids, threat_ids = 0, 0, 0, 0

    while not done:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
                    break

        agent_action, _ = agent_model.predict(obs, deterministic=True) # TODO Check

        obses, rewards, dones, infos = env.step([agent_action])

        obs = obses[0]
        reward = rewards[0]
        info = infos[0]
        done = dones[0]

        target_ids = int(info.get('target_ids', -1))
        threat_ids = int(info.get('threat_ids', -1))

        if render:
            base_env.render()
            base_env.clock.tick(30)

        episode_reward += reward
        num_steps += 1

    return episode_reward, target_ids, threat_ids, num_steps


def run_single_human_eval(env, agent_model, human_trajectory_file, level, render):
    # TODO NOT READY

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

        agent_action, _ = agent_model.predict(obs, deterministic=True) # TODO Check

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
                    break

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

    config_filename = '../configs/Monolith_index_August.json'
    config = load_env_config(config_filename)

    #config['tick_rate'] = tick_rate
    #config['game_speed'] /= time_factor
    #config['max_steps'] *= (1700 / 1500) * time_factor
    config['use_stuck_detection'] = False
    config['prob_detect'] = 0  # 0.0003
    config['action_type'] = 'Discrete16'

    num_repeats = 1 # How many times to run each level.
    render = False

    ####################################     Pygame setup     ####################################
    if render:
        if hasattr(ctypes, 'windll') and hasattr(ctypes.windll, 'user32'): ctypes.windll.user32.SetProcessDPIAware()
        pygame.display.init()
        pygame.font.init()
        clock = pygame.time.Clock()

        window_width, window_height = config['window_size'][0], config['window_size'][1]
        window = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption(f"MAISR User Study - Subject {subject_id}")
    else:
        pygame.font.init()
        window = None
        clock = None

    ####################################     Instantiate testing agents     ####################################
    testing_agent_dir = './offline_study_testing_agents'
    heldout_agent_dir = './heldout_agents'
    human_trajectory_dir = '../userstudy_logs' # human_trajectories
    # TODO populate

    testing_agents = populate_agent_list(testing_agent_dir, label='testagent')
    heldout_agents = populate_agent_list(heldout_agent_dir, label='heldout')

    print('Contents of testing_agents:')
    print(testing_agents)
    print('Contents of heldout_agents:')
    print(heldout_agents)

    # Now I have a list of (PPO model, vecnorm stat filename, name) tuples for my testing agents


    #################################### Load human trajectories ####################################
    dual_traj_pattern = f"{human_trajectory_dir}/subject_*/timestep_data/timesteps_[ABC][13457]_*.json"
    solo_traj_pattern = f"{human_trajectory_dir}/subject_*/timestep_data/timesteps_[PS][13457]_*.json"
    dual_trajectory_files = glob.glob(dual_traj_pattern)
    solo_trajectory_files = glob.glob(solo_traj_pattern)
    if not dual_trajectory_files:
        print(f"[Human trajectory loading] No dual trajectories found")
        raise ValueError
    if not solo_trajectory_files:
        print(f"[Human trajectory loading] No solo trajectories found")
        raise ValueError
    print(f'[Human trajectories] Loaded {len(dual_trajectory_files)} dual trajectories and {len(solo_trajectory_files)} solo trajectories')


    results = {}
    for agent_tuple in testing_agents:
        agent_model = agent_tuple[0]
        agent_vecnorm = agent_tuple[1]
        agent_name = agent_tuple[2]

        # TODO temp hack
        temp_teammate_policy = RLTeammatePolicy(agent_model, None, None, None, None, norm_stats_path=agent_vecnorm)

        env_fns = [make_wrapped_env(config, clock, window, temp_teammate_policy) for _ in range(1)]
        env = DummyVecEnv(env_fns)
        env = load_vecnormalize_wrapper(agent_vecnorm, env)
        print(f'Loaded vecnorm stats from {agent_vecnorm}')
        base_env = env.envs[0].env
        base_env.teammate_active = True

        #################################### Run RL agent evals ####################################
        for teammate_tuple in testing_agents + heldout_agents:
            for run in range(num_repeats):
                teammate_model = teammate_tuple[0]
                teammate_vecnorm = teammate_tuple[1]
                teammate_name = teammate_tuple[2]
                teammate = RLTeammatePolicy(teammate_model, env, None, None, None, norm_stats_path=teammate_vecnorm)

                base_env.teammate_policy = teammate
                base_env.current_teammate = teammate

                print(f'\nRunning single RL eval with agent {agent_name} (model {agent_model}, vecnorm stats {agent_vecnorm}). Teammate is {teammate_name} (model {teammate_model}, vecnorm {teammate_vecnorm})')
                reward, target_ids, threat_ids, num_steps = run_single_rl_eval(env, agent_model, render)
                results[(agent_name, teammate_name, run)] = reward, target_ids, threat_ids, num_steps


        #################################### Run human trajectory evals ####################################
        # for trajectory_file in dual_trajectory_files + solo_trajectory_files:
        #     for run in range(num_repeats):
        #         level = int(trajectory_file.split('_')[4][1])
        #
        #         print(f'Running human eval with level {level} and trajectory {trajectory_file}')
        #         teammate_name = trajectory_file
        #         reward, target_ids, threat_ids, num_steps = run_single_human_eval(env, agent_model, trajectory_file, level, render)
        #         results[(agent_name, teammate_name, run)] = reward, target_ids, threat_ids, num_steps

    print(f'====== ====== FINAL RESULTS ====== ====== ')
    print(results)


if __name__ == '__main__':
    main()