import ctypes
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from env_multi_new import MAISREnvVec
from training_wrappers.localsearch_training_wrapper import MaisrLocalSearchWrapper
from training_wrappers.modeselector_training_wrapper import MaisrModeSelectorWrapper
from utility.data_logging import load_env_config
from policies.league_management import GenericTeammatePolicy, SubPolicy, LocalSearch, ChangeRegions, GoToNearestThreat, \
    EvadeDetection, TeammateManager, HeuristicAgent, TargetSearchLocalTSP
import json
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import pickle
from collections import defaultdict


def debug_wrapper_structure(env, use_normalize):
    """Debug function to understand the wrapper structure and find the correct path to counters"""

    print("\n=== DEBUGGING WRAPPER STRUCTURE ===")

    if use_normalize:
        print("With VecNormalize:")
        print(f"env type: {type(env)}")

        if hasattr(env, 'envs'):
            print(f"env.envs[0] type: {type(env.envs[0])}")

            if hasattr(env.envs[0], 'env'):
                print(f"env.envs[0].env type: {type(env.envs[0].env)}")

                if hasattr(env.envs[0].env, 'env'):
                    print(f"env.envs[0].env.env type: {type(env.envs[0].env.env)}")

        # Test different access paths
        access_paths = [
            ("env", env),
            ("env.envs[0]", getattr(env, 'envs', [None])[0] if hasattr(env, 'envs') else None),
            ("env.envs[0].env", getattr(getattr(env, 'envs', [None])[0], 'env', None) if hasattr(env, 'envs') and len(
                env.envs) > 0 else None),
            ("env.envs[0].env.env",
             getattr(getattr(getattr(env, 'envs', [None])[0], 'env', None), 'env', None) if hasattr(env,
                                                                                                    'envs') and len(
                 env.envs) > 0 and hasattr(env.envs[0], 'env') else None)
        ]
    else:
        print("Without VecNormalize:")
        print(f"env type: {type(env)}")

        if hasattr(env, 'env'):
            print(f"env.env type: {type(env.env)}")

            if hasattr(env.env, 'env'):
                print(f"env.env.env type: {type(env.env.env)}")

        # Test different access paths
        access_paths = [
            ("env", env),
            ("env.env", getattr(env, 'env', None)),
            ("env.env.env", getattr(getattr(env, 'env', None), 'env', None) if hasattr(env, 'env') else None)
        ]

    print("\nChecking for target/threat counters at each level:")
    for path_name, obj in access_paths:
        if obj is None:
            print(f"{path_name}: None")
            continue

        has_threats = hasattr(obj, 'num_threats_identified')
        has_targets = hasattr(obj, 'targets_identified')
        obj_type = type(obj).__name__

        print(f"{path_name} ({obj_type}): threats={has_threats}, targets={has_targets}")

        if has_threats and has_targets:
            try:
                threat_val = getattr(obj, 'num_threats_identified')
                target_val = getattr(obj, 'targets_identified')
                print(f"  -> FOUND COUNTERS: threats={threat_val}, targets={target_val}")
                return obj  # Return the object that has the counters
            except Exception as e:
                print(f"  -> Error accessing counters: {e}")

    print("=== END DEBUGGING ===\n")
    return None

def find_model_files(base_path):
    """Find .zip and .pkl files in the specified directory"""
    import glob

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Directory not found: {base_path}")

    # Find .zip file (model)
    zip_files = glob.glob(os.path.join(base_path, "*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No .zip model file found in {base_path}")
    if len(zip_files) > 1:
        print(f"Warning: Multiple .zip files found in {base_path}, using first one: {zip_files[0]}")
    model_path = zip_files[0]

    # Find .pkl file (normalization stats)
    pkl_files = glob.glob(os.path.join(base_path, "*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl normalization stats file found in {base_path}")
    if len(pkl_files) > 1:
        print(f"Warning: Multiple .pkl files found in {base_path}, using first one: {pkl_files[0]}")
    norm_stats_path = pkl_files[0]

    return model_path, norm_stats_path

def calculate_spatial_coverage(positions, gameboard_size):
    """Calculate what percentage of the map was visited"""
    if not positions:
        return 0

    # Create a simple grid-based coverage metric
    grid_size = 20  # 20x20 grid
    visited_cells = set()
    cell_size = gameboard_size / grid_size

    for x, y in positions:
        # Convert to grid coordinates
        grid_x = int((x + gameboard_size / 2) // cell_size)
        grid_y = int((y + gameboard_size / 2) // cell_size)
        visited_cells.add((grid_x, grid_y))

    total_cells = grid_size * grid_size
    coverage_percentage = len(visited_cells) / total_cells * 100
    return min(coverage_percentage, 100.0)


def calculate_coordination_score(human_sequence, teammate_usage):
    """Calculate a simple coordination score between human and teammate"""
    if not human_sequence or not teammate_usage:
        return 0

    # Simple metric: how often they used different subpolicies
    total_teammate_steps = sum(teammate_usage.values())
    if total_teammate_steps == 0:
        return 0

    # Calculate diversity - higher score when using different subpolicies
    human_dominant = max(set(human_sequence), key=human_sequence.count)
    teammate_dominant = max(teammate_usage.keys(), key=lambda k: teammate_usage[k])

    coordination_score = 100 if human_dominant != teammate_dominant else 50
    return coordination_score


def get_counter_overfit_type(overfit_type):
    """Get the opposite overfit type for counter behavior"""
    counter_mapping = {
        'low_risk': 'high_risk',
        'high_risk': 'low_risk',
        'nospatial': 'highspatial',
        'highspatial': 'nospatial',
        'noisy_actions': 'stable_actions',
        'stable_actions': 'noisy_actions'
    }
    return counter_mapping[overfit_type]


def create_overfit_agent(overfit_type, subpolicies):
    """Create an overfit agent with specific configuration"""
    if overfit_type == "low_risk":
        mode_selector = "heuristic"
        risk_tolerance = "low"
        spatial_coord = "false"  # Default spatial coordination
        action_stability = "stable"  # Default for overfit tests
        planning_horizon = "long"

    elif overfit_type == "high_risk":
        mode_selector = "heuristic"
        risk_tolerance = "high"
        spatial_coord = "false"  # Default spatial coordination
        action_stability = "stable"  # Default for overfit tests
        planning_horizon = "long"

    elif overfit_type == "nospatial":
        mode_selector = "heuristic"
        risk_tolerance = "medium"
        planning_horizon = "short"
        action_stability = "stable"  # Default for overfit tests
        spatial_coord = "none"

    elif overfit_type == "highspatial":
        mode_selector = "heuristic"
        risk_tolerance = "medium"
        planning_horizon = "short"
        action_stability = "stable"  # Default for overfit tests
        spatial_coord = "high"

    elif overfit_type == 'noisy_actions':
        mode_selector = "heuristic"
        risk_tolerance = "medium"  # Default risk tolerance
        spatial_coord = "false"
        planning_horizon = "short"
        action_stability = "noisy"  # Default for overfit tests

    elif overfit_type == 'stable_actions':
        mode_selector = "heuristic"
        risk_tolerance = "medium"  # Default risk tolerance
        spatial_coord = "false"
        planning_horizon = "short"
        action_stability = "stable"  # Default for overfit tests

    else:
        raise ValueError(f"Unknown overfit_type: {overfit_type}")

    if planning_horizon == 'cluster_planning':
        target_search_policy = TargetSearchLocalTSP(
            search_radius=1000,
            spatial_coord=False,
            model_path=None,
            norm_stats_filepath=None,
            search_method='clusters'
        )
    elif planning_horizon == 'greedy_planning':
        target_search_policy = TargetSearchLocalTSP(
            search_radius=1000,
            spatial_coord=False,
            model_path=None,
            norm_stats_filepath=None,
            search_method='greedy'
        )

    elif planning_horizon == 'short':
        target_search_policy = subpolicies.get('local_search')
    elif planning_horizon == 'medium':
        if spatial_coord == 'true':
            target_search_policy = subpolicies.get('local_tsp_yescoord')
        else:
            target_search_policy = subpolicies.get('local_tsp_nocoord')
    elif planning_horizon == 'long':
        if spatial_coord == 'true':
            target_search_policy = subpolicies.get('global_tsp_yescoord')
        else:
            target_search_policy = subpolicies.get('global_tsp_nocoord')
    else:
        raise ValueError(f"Unknown planning_horizon value: {planning_horizon}")

    heuristic_agent = HeuristicAgent(
        mode_selector=mode_selector,
        risk_tolerance=risk_tolerance,
        spatial_coord=spatial_coord
    )

    # agent = GenericTeammatePolicy(
    #     env=None,
    #     local_search_policy=subpolicies.get('local_search'),
    #     go_to_highvalue_policy=subpolicies.get('go_to_threat'),
    #     change_region_subpolicy=subpolicies.get('change_region'),
    #     mode_selector_agent=heuristic_agent,
    #     use_collision_avoidance=False
    # )

    teammate = GenericTeammatePolicy(
        env=None,
        local_search_policy=target_search_policy,
        go_to_highvalue_policy=subpolicies.get('go_to_threat'),
        change_region_subpolicy=subpolicies.get('change_region'),
        mode_selector_agent=heuristic_agent,
        use_collision_avoidance=False,
        action_stability=action_stability
    )

    teammate.name = f"OverfitTeammate_{overfit_type}_{mode_selector}MS_{risk_tolerance}risk_{spatial_coord}spatial"
    return teammate


def run_episode_batch(env, agent, num_episodes, overfit_type, behavior_type, use_normalize):
    """Run a batch of episodes with the given agent configuration"""
    episode_data = []

    print(f"\nRunning {num_episodes} episodes for {overfit_type} agent with {behavior_type} behavior...")

    #debug_wrapper_structure(env, use_normalize)

    for episode in range(num_episodes):
        if use_normalize:
            obs = env.reset()
        else:
            obs, info = env.reset()
        episode_reward = 0
        raw_episode_reward = 0

        # Initialize tracking variables
        episode_steps = 0
        # if use_normalize:
        #     initial_threat_ids = env.envs[0].env.num_threats_identified
        #     initial_target_ids = env.envs[0].env.targets_identified
        # else:
        #     initial_threat_ids = env.env.num_threats_identified
        #     initial_target_ids = env.env.targets_identified

        # Subpolicy tracking
        # subpolicy_usage = {0: 0, 1: 0, 2: 0, 3: 0, 4:0, 5:0, 6:0, 7:0}
        # subpolicy_switches = 0
        # last_action = None
        # subpolicy_sequence = []
        #
        # # Teammate tracking
        # teammate_subpolicy_usage = {0: 0, 1: 0, 2: 0, 3: 0, 4:0, 5:0, 6:0, 7:0}
        # teammate_switches = 0
        # last_teammate_action = None

        # Performance tracking
        detection_events = []
        identification_events = []
        distance_traveled = 0
        last_position = None
        positions_visited = []
        teammate_positions_visited = []
        target_discovery_times = {}
        threat_discovery_times = {}

        # Timing tracking
        episode_start_time = pygame.time.get_ticks()

        done = False

        target_tracker = 0
        threat_tracker = 0
        while not done:
            # Handle pygame events (minimal for automated testing)
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        break
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        done = True
                        break

                if done:
                    break

            # Get action from agent
            action, _ = agent.predict(obs, deterministic=True)

            # Track subpolicy usage
            # subpolicy_usage[int(action)] += 1
            # subpolicy_sequence.append(int(action))
            # if last_action is not None and last_action != action:
            #     subpolicy_switches += 1
            # last_action = action

            # # Track teammate behavior
            # if use_normalize:
            #     if env.envs[0].env.config['num_aircraft'] == 2:
            #         ai_subpolicy_id, ai_subpolicy_name = env.envs[0].get_teammate_subpolicy_info()
            #         teammate_subpolicy_usage[ai_subpolicy_id] += 1
            #         if last_teammate_action is not None and last_teammate_action != ai_subpolicy_id:
            #             teammate_switches += 1
            #         last_teammate_action = ai_subpolicy_id
            # else:
            #     if env.env.config['num_aircraft'] == 2:
            #         ai_subpolicy_id, ai_subpolicy_name = env.get_teammate_subpolicy_info()
            #         teammate_subpolicy_usage[ai_subpolicy_id] += 1
            #         if last_teammate_action is not None and last_teammate_action != ai_subpolicy_id:
            #             teammate_switches += 1
            #         last_teammate_action = ai_subpolicy_id

            # Take step
            if use_normalize:
                obses, rewards, dones, infos = env.step([action])
                obs, reward, done, info = obses[0], rewards[0], dones[0], infos[0]
                try:
                    env.render()
                except:
                    pass

                raw_reward = env.envs[0].env.ep_reward
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                raw_reward = reward
                try:
                    env.render()
                except:
                    pass

            episode_reward += reward
            raw_episode_reward += raw_reward
            episode_steps += 1

            if 'new_target_ids' in info:
                target_tracker += info['new_target_ids']
            if 'new_threat_ids' in info:
                threat_tracker += info['new_threat_ids']

            # Track position and distance
            if use_normalize:
                current_pos = (env.envs[0].env.agents[env.envs[0].env.aircraft_ids[0]].x, env.envs[0].env.agents[env.envs[0].env.aircraft_ids[0]].y)
                positions_visited.append(current_pos)
                if env.envs[0].env.config['num_aircraft'] == 2:
                    teammate_current_pos = (
                        env.envs[0].env.agents[env.envs[0].env.aircraft_ids[1]].x, env.envs[0].env.agents[env.envs[0].env.aircraft_ids[1]].y)
                    teammate_positions_visited.append(teammate_current_pos)
            else:
                current_pos = (env.env.agents[env.env.aircraft_ids[0]].x, env.env.agents[env.env.aircraft_ids[0]].y)
                positions_visited.append(current_pos)
                if env.env.config['num_aircraft'] == 2:
                    teammate_current_pos = (
                    env.env.agents[env.env.aircraft_ids[1]].x, env.env.agents[env.env.aircraft_ids[1]].y)
                    teammate_positions_visited.append(teammate_current_pos)

            if last_position is not None:
                distance_traveled += math.sqrt((current_pos[0] - last_position[0]) ** 2 +
                                               (current_pos[1] - last_position[1]) ** 2)
            last_position = current_pos

            # Track events
            if 'new_detections' in info and info['new_detections'] > 0:
                detection_events.append({
                    'step': episode_steps,
                    'count': info['new_detections'],
                    'position': current_pos
                })

            if 'new_target_ids' in info and info['new_target_ids'] > 0:
                identification_events.append({
                    'step': episode_steps,
                    'type': 'target',
                    'count': info['new_target_ids']
                })
                target_discovery_times[len(target_discovery_times)] = episode_steps

            if 'new_threat_ids' in info and info['new_threat_ids'] > 0:
                identification_events.append({
                    'step': episode_steps,
                    'type': 'threat',
                    'count': info['new_threat_ids']
                })
                threat_discovery_times[len(threat_discovery_times)] = episode_steps

        # Calculate episode duration
        episode_end_time = pygame.time.get_ticks()
        episode_duration_ms = episode_end_time - episode_start_time

        threat_ids_gained = threat_tracker# - initial_threat_ids
        target_ids_gained = target_tracker# - initial_target_ids

        print(f'[DEBUG] In test suite, threat ids gained = {threat_ids_gained}, target_ids = {target_ids_gained}, reward = {episode_reward}')

        # Calculate average teammate distance
        avg_teammate_distance = 0
        if positions_visited and teammate_positions_visited:
            min_length = min(len(positions_visited), len(teammate_positions_visited))
            distances = []
            for i in range(min_length):
                agent_pos = positions_visited[i]
                teammate_pos = teammate_positions_visited[i]
                distance = math.sqrt((agent_pos[0] - teammate_pos[0]) ** 2 +
                                     (agent_pos[1] - teammate_pos[1]) ** 2)
                distances.append(distance)
            avg_teammate_distance = sum(distances) / len(distances) if distances else 0

        # Store comprehensive episode data
        if use_normalize:
            episode_info = {
                # Basic metrics
                'episode': episode + 1,
                'steps': episode_steps,
                'threat_ids': threat_ids_gained,
                'target_ids': target_ids_gained,
                'total_reward': episode_reward,
                'duration_ms': episode_duration_ms,
                'avg_teammate_distance': avg_teammate_distance,

                # Configuration
                'overfit_type': overfit_type,
                'behavior_type': behavior_type,

                # Subpolicy analytics
                #'subpolicy_usage': subpolicy_usage.copy(),
                #'subpolicy_switches': subpolicy_switches,
                # 'subpolicy_percentages': {
                #     k: (v / episode_steps * 100) if episode_steps > 0 else 0
                #     for k, v in subpolicy_usage.items()
                # },
                # 'subpolicy_sequence': subpolicy_sequence.copy(),

                # Teammate analytics
                # 'teammate_subpolicy_usage': teammate_subpolicy_usage.copy(),
                # 'teammate_switches': teammate_switches,
                # 'coordination_score': calculate_coordination_score(subpolicy_sequence, teammate_subpolicy_usage),
                'positions_visited': positions_visited,
                'teammate_positions_visited': teammate_positions_visited,

                # Performance metrics
                'distance_traveled': distance_traveled,
                'efficiency_score': target_ids_gained / max(episode_steps, 1),
                'spatial_coverage_percent': calculate_spatial_coverage(positions_visited,env.envs[0].env.config['gameboard_size']),
                'avg_distance_per_step': distance_traveled / max(episode_steps, 1),

                # Event tracking
                'detection_events': detection_events,
                'identification_events': identification_events,
                'total_detections': env.envs[0].env.detections,
                'num_detection_events': len(detection_events),

                # Timing analysis
                'target_discovery_times': target_discovery_times,
                'threat_discovery_times': threat_discovery_times,
                'time_to_first_target': min(target_discovery_times.values()) if target_discovery_times else None,
                'time_to_last_target': max(target_discovery_times.values()) if target_discovery_times else None,

                # Episode outcome
                'completed_successfully': env.envs[0].env.all_targets_identified,
                'termination_reason': 'success' if env.envs[0].env.all_targets_identified else 'failed' if env.envs[0].env.failed else 'timeout',

                # Environment state
                'final_threat_count': threat_ids_gained,
                'final_target_count': target_ids_gained,
                'num_targets_total': env.envs[0].env.config['num_targets'],
                'num_threats_total': env.envs[0].env.config['num_threats'],
                'gameboard_size': env.envs[0].env.config['gameboard_size'],
                'max_steps_allowed': env.envs[0].env.max_steps,
            }
        else:
            episode_info = {
                # Basic metrics
                'episode': episode + 1,
                'steps': episode_steps,
                'threat_ids': threat_ids_gained,
                'target_ids': target_ids_gained,
                'total_reward': episode_reward,
                'duration_ms': episode_duration_ms,
                'avg_teammate_distance': avg_teammate_distance,

                # Configuration
                'overfit_type': overfit_type,
                'behavior_type': behavior_type,

                # Subpolicy analytics
                # 'subpolicy_usage': subpolicy_usage.copy(),
                # 'subpolicy_switches': subpolicy_switches,
                # 'subpolicy_percentages': {
                #     k: (v / episode_steps * 100) if episode_steps > 0 else 0
                #     for k, v in subpolicy_usage.items()
                # },
                # 'subpolicy_sequence': subpolicy_sequence.copy(),

                # Teammate analytics
                # 'teammate_subpolicy_usage': teammate_subpolicy_usage.copy(),
                # 'teammate_switches': teammate_switches,
                # 'coordination_score': calculate_coordination_score(subpolicy_sequence, teammate_subpolicy_usage),
                'positions_visited': positions_visited,
                'teammate_positions_visited': teammate_positions_visited,

                # Performance metrics
                'distance_traveled': distance_traveled,
                'efficiency_score': target_ids_gained / max(episode_steps, 1),
                'spatial_coverage_percent': calculate_spatial_coverage(positions_visited, env.env.config['gameboard_size']),
                'avg_distance_per_step': distance_traveled / max(episode_steps, 1),

                # Event tracking
                'detection_events': detection_events,
                'identification_events': identification_events,
                'total_detections': env.env.detections,
                'num_detection_events': len(detection_events),

                # Timing analysis
                'target_discovery_times': target_discovery_times,
                'threat_discovery_times': threat_discovery_times,
                'time_to_first_target': min(target_discovery_times.values()) if target_discovery_times else None,
                'time_to_last_target': max(target_discovery_times.values()) if target_discovery_times else None,

                # Episode outcome
                'completed_successfully': env.env.all_targets_identified,
                'termination_reason': 'success' if env.env.all_targets_identified else
                'failed' if env.env.failed else 'timeout',

                # Environment state
                'final_threat_count': threat_ids_gained,
                'final_target_count': target_ids_gained,
                'num_targets_total': env.env.config['num_targets'],
                'num_threats_total': env.env.config['num_threats'],
                'gameboard_size': env.env.config['gameboard_size'],
                'max_steps_allowed': env.env.max_steps,
            }

        episode_data.append(episode_info)

        # Print progress
        if (episode + 1) % 5 == 0:
            print(f"  Completed {episode + 1}/{num_episodes} episodes")

    return episode_data


def calculate_summary_statistics(episode_data):
    """Calculate summary statistics for a batch of episodes"""
    if not episode_data:
        return {}

    # Extract metrics
    rewards = [ep['total_reward'] for ep in episode_data]
    target_ids = [ep['target_ids'] for ep in episode_data]
    threat_ids = [ep['threat_ids'] for ep in episode_data]
    avg_distances = [ep['avg_teammate_distance'] for ep in episode_data]
    steps = [ep['steps'] for ep in episode_data]
    efficiency_scores = [ep['efficiency_score'] for ep in episode_data]
    spatial_coverage = [ep['spatial_coverage_percent'] for ep in episode_data]
    success_rate = sum(1 for ep in episode_data if ep['completed_successfully']) / len(episode_data)

    return {
        'reward': {'mean': np.mean(rewards), 'std': np.std(rewards), 'values': rewards},
        'target_ids': {'mean': np.mean(target_ids), 'std': np.std(target_ids), 'values': target_ids},
        'threat_ids': {'mean': np.mean(threat_ids), 'std': np.std(threat_ids), 'values': threat_ids},
        'avg_teammate_distance': {'mean': np.mean(avg_distances), 'std': np.std(avg_distances),
                                  'values': avg_distances},
        'steps': {'mean': np.mean(steps), 'std': np.std(steps), 'values': steps},
        'efficiency_score': {'mean': np.mean(efficiency_scores), 'std': np.std(efficiency_scores),
                             'values': efficiency_scores},
        'spatial_coverage': {'mean': np.mean(spatial_coverage), 'std': np.std(spatial_coverage),
                             'values': spatial_coverage},
        'success_rate': success_rate,
        'num_episodes': len(episode_data)
    }


def create_comparison_plots(all_results, timestamp):
    """Create comparison plots for all overfit agents and behaviors"""

    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('Overfit Agent Performance: Aligned vs Counter Behavior', fontsize=16)

    overfit_types = ['low_risk', 'high_risk', 'noisy_actions', 'stable_actions'] # 'low_risk', 'high_risk'
    behavior_types = ['aligned', 'counter']

    metrics = ['reward', 'target_ids', 'avg_teammate_distance', 'steps', 'efficiency_score',
               'spatial_coverage', 'success_rate', 'threat_ids']

    # Plot 1: Reward comparison
    ax = axes[0, 0]
    x_pos = np.arange(len(overfit_types))
    width = 0.35

    aligned_rewards = [all_results[ot]['aligned']['reward']['mean'] for ot in overfit_types]
    counter_rewards = [all_results[ot]['counter']['reward']['mean'] for ot in overfit_types]
    aligned_stds = [all_results[ot]['aligned']['reward']['std'] for ot in overfit_types]
    counter_stds = [all_results[ot]['counter']['reward']['std'] for ot in overfit_types]

    ax.bar(x_pos - width / 2, aligned_rewards, width, label='Aligned', yerr=aligned_stds, capsize=5)
    ax.bar(x_pos + width / 2, counter_rewards, width, label='Counter', yerr=counter_stds, capsize=5)
    ax.set_xlabel('Agent Type')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Reward Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(overfit_types, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Target IDs comparison
    ax = axes[0, 1]
    aligned_targets = [all_results[ot]['aligned']['target_ids']['mean'] for ot in overfit_types]
    counter_targets = [all_results[ot]['counter']['target_ids']['mean'] for ot in overfit_types]
    aligned_stds = [all_results[ot]['aligned']['target_ids']['std'] for ot in overfit_types]
    counter_stds = [all_results[ot]['counter']['target_ids']['std'] for ot in overfit_types]

    ax.bar(x_pos - width / 2, aligned_targets, width, label='Aligned', yerr=aligned_stds, capsize=5)
    ax.bar(x_pos + width / 2, counter_targets, width, label='Counter', yerr=counter_stds, capsize=5)
    ax.set_xlabel('Agent Type')
    ax.set_ylabel('Mean Target IDs')
    ax.set_title('Target IDs Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(overfit_types, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Average teammate distance comparison
    ax = axes[0, 2]
    aligned_distances = [all_results[ot]['aligned']['avg_teammate_distance']['mean'] for ot in overfit_types]
    counter_distances = [all_results[ot]['counter']['avg_teammate_distance']['mean'] for ot in overfit_types]
    aligned_stds = [all_results[ot]['aligned']['avg_teammate_distance']['std'] for ot in overfit_types]
    counter_stds = [all_results[ot]['counter']['avg_teammate_distance']['std'] for ot in overfit_types]

    ax.bar(x_pos - width / 2, aligned_distances, width, label='Aligned', yerr=aligned_stds, capsize=5)
    ax.bar(x_pos + width / 2, counter_distances, width, label='Counter', yerr=counter_stds, capsize=5)
    ax.set_xlabel('Agent Type')
    ax.set_ylabel('Mean Teammate Distance')
    ax.set_title('Teammate Distance Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(overfit_types, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Success rate comparison
    ax = axes[1, 0]
    aligned_success = [all_results[ot]['aligned']['success_rate'] for ot in overfit_types]
    counter_success = [all_results[ot]['counter']['success_rate'] for ot in overfit_types]

    ax.bar(x_pos - width / 2, aligned_success, width, label='Aligned')
    ax.bar(x_pos + width / 2, counter_success, width, label='Counter')
    ax.set_xlabel('Agent Type')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(overfit_types, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Plot 5: Efficiency comparison
    ax = axes[1, 1]
    aligned_efficiency = [all_results[ot]['aligned']['efficiency_score']['mean'] for ot in overfit_types]
    counter_efficiency = [all_results[ot]['counter']['efficiency_score']['mean'] for ot in overfit_types]
    aligned_stds = [all_results[ot]['aligned']['efficiency_score']['std'] for ot in overfit_types]
    counter_stds = [all_results[ot]['counter']['efficiency_score']['std'] for ot in overfit_types]

    ax.bar(x_pos - width / 2, aligned_efficiency, width, label='Aligned', yerr=aligned_stds, capsize=5)
    ax.bar(x_pos + width / 2, counter_efficiency, width, label='Counter', yerr=counter_stds, capsize=5)
    ax.set_xlabel('Agent Type')
    ax.set_ylabel('Mean Efficiency Score')
    ax.set_title('Efficiency Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(overfit_types, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Steps comparison
    ax = axes[1, 2]
    aligned_steps = [all_results[ot]['aligned']['steps']['mean'] for ot in overfit_types]
    counter_steps = [all_results[ot]['counter']['steps']['mean'] for ot in overfit_types]
    aligned_stds = [all_results[ot]['aligned']['steps']['std'] for ot in overfit_types]
    counter_stds = [all_results[ot]['counter']['steps']['std'] for ot in overfit_types]

    ax.bar(x_pos - width / 2, aligned_steps, width, label='Aligned', yerr=aligned_stds, capsize=5)
    ax.bar(x_pos + width / 2, counter_steps, width, label='Counter', yerr=counter_stds, capsize=5)
    ax.set_xlabel('Agent Type')
    ax.set_ylabel('Mean Steps')
    ax.set_title('Steps Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(overfit_types, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 7: Performance difference (Aligned - Counter)
    ax = axes[2, 0]
    reward_diff = [all_results[ot]['aligned']['reward']['mean'] - all_results[ot]['counter']['reward']['mean']
                   for ot in overfit_types]
    colors = ['green' if diff > 0 else 'red' for diff in reward_diff]

    ax.bar(x_pos, reward_diff, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.set_xlabel('Agent Type')
    ax.set_ylabel('Reward Difference (Aligned - Counter)')
    ax.set_title('Performance Advantage of Aligned Behavior')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(overfit_types, rotation=45)
    ax.grid(True, alpha=0.3)

    # Plot 8: Spatial coverage comparison
    ax = axes[2, 1]
    aligned_coverage = [all_results[ot]['aligned']['spatial_coverage']['mean'] for ot in overfit_types]
    counter_coverage = [all_results[ot]['counter']['spatial_coverage']['mean'] for ot in overfit_types]
    aligned_stds = [all_results[ot]['aligned']['spatial_coverage']['std'] for ot in overfit_types]
    counter_stds = [all_results[ot]['counter']['spatial_coverage']['std'] for ot in overfit_types]

    ax.bar(x_pos - width / 2, aligned_coverage, width, label='Aligned', yerr=aligned_stds, capsize=5)
    ax.bar(x_pos + width / 2, counter_coverage, width, label='Counter', yerr=counter_stds, capsize=5)
    ax.set_xlabel('Agent Type')
    ax.set_ylabel('Mean Spatial Coverage (%)')
    ax.set_title('Spatial Coverage Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(overfit_types, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 9: Combined effectiveness score
    ax = axes[2, 2]
    # Calculate combined score: (reward * target_ids * success_rate) / steps
    aligned_combined = []
    counter_combined = []

    for ot in overfit_types:
        aligned_score = (all_results[ot]['aligned']['reward']['mean'] *
                         all_results[ot]['aligned']['target_ids']['mean'] *
                         all_results[ot]['aligned']['success_rate']) / max(all_results[ot]['aligned']['steps']['mean'],
                                                                           1)
        counter_score = (all_results[ot]['counter']['reward']['mean'] *
                         all_results[ot]['counter']['target_ids']['mean'] *
                         all_results[ot]['counter']['success_rate']) / max(all_results[ot]['counter']['steps']['mean'],
                                                                           1)
        aligned_combined.append(aligned_score)
        counter_combined.append(counter_score)

    ax.bar(x_pos - width / 2, aligned_combined, width, label='Aligned')
    ax.bar(x_pos + width / 2, counter_combined, width, label='Counter')
    ax.set_xlabel('Agent Type')
    ax.set_ylabel('Combined Effectiveness Score')
    ax.set_title('Overall Effectiveness Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(overfit_types, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    plot_filename = f"./logs/overfit_tests/overfit_comparison_plots_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Comparison plots saved to {plot_filename}")


def convert_to_json_serializable(obj):
    """Convert numpy types and other non-serializable objects to JSON-serializable types"""
    import numpy as np

    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


if __name__ == "__main__":
    print(f'Beginning main')
    # Configuration
    config_filename = 'configs/Monolith_R6H_july8.json'
    num_episodes = 50
    tick_rate = 120
    use_normalize = True
    render = False

    localsearch_model_path = None
    localsearch_normstats_path = 'trained_models/local_search_2000000.0timesteps_0.1threatpenalty_0615_1541_6envslocal_search_norm_stats.npy'

    # Test configuration
    overfit_agents = ['low_risk', 'high_risk', 'noisy_actions', 'stable_actions'] # # 'nospatial', 'highspatial'
    behavior_types = ['aligned', 'counter']

    model_and_stats_paths = {
        'low_risk': './R6H_saved/R6H_lowrisk',
        'high_risk': './R6H_saved/R6H_highrisk',
        'noisy_actions': './R6H_saved/R6H_noisy',
        'stable_actions': './R6H_saved/R6H_stable'
    }

    # model_path_dict = {
    #     'low_risk': './trained_models/overfit_tests/modeselector_OverfitV9_low_risk_shaping_ratio1_0630_1910_/checkpoints/maisr_checkpoint_modeselector_OverfitV9_low_risk_shaping_ratio1_0630_1910__262080_steps.zip',
    #     'high_risk': './trained_models/overfit_tests/modeselector_OverfitV9_high_risk_shaping_ratio1_0630_1609_/checkpoints/maisr_checkpoint_modeselector_OverfitV9_high_risk_shaping_ratio1_0630_1609__262080_steps.zip',
    #     'nospatial': './trained_models/overfit_tests/modeselector_OverfitV9_nospatial_shaping_ratio1_0630_2212_/checkpoints/maisr_checkpoint_modeselector_OverfitV9_nospatial_shaping_ratio1_0630_2212__262080_steps.zip',
    #     'highspatial': './trained_models/overfit_tests/modeselector_OverfitV9_highspatial_shaping_ratio1_0701_0114_/checkpoints/maisr_checkpoint_modeselector_OverfitV9_highspatial_shaping_ratio1_0701_0114__262080_steps.zip'
    # }
    #
    # norm_stats_path_dict = {
    #     'low_risk': './trained_models/overfit_tests/modeselector_OverfitV9_low_risk_shaping_ratio1_0630_1910_/checkpoints/maisr_checkpoint_modeselector_OverfitV9_low_risk_shaping_ratio1_0630_1910__vecnormalize_262080_steps.pkl',
    #     'high_risk': './trained_models/overfit_tests/modeselector_OverfitV9_high_risk_shaping_ratio1_0630_1609_/checkpoints/maisr_checkpoint_modeselector_OverfitV9_high_risk_shaping_ratio1_0630_1609__vecnormalize_262080_steps.pkl',
    #     'nospatial': './trained_models/overfit_tests/modeselector_OverfitV9_nospatial_shaping_ratio1_0630_2212_/checkpoints/maisr_checkpoint_modeselector_OverfitV9_nospatial_shaping_ratio1_0630_2212__vecnormalize_262080_steps.pkl',
    #     'highspatial': './trained_models/overfit_tests/modeselector_OverfitV9_highspatial_shaping_ratio1_0701_0114_/checkpoints/maisr_checkpoint_modeselector_OverfitV9_highspatial_shaping_ratio1_0701_0114__vecnormalize_262080_steps.pkl'
    # }

    config = load_env_config(config_filename)
    print(f'LOADED CONFIG {config_filename}')

    # Initialize pygame
    if render:
        pygame.display.init()
        pygame.font.init()
        clock = pygame.time.Clock()
        ctypes.windll.user32.SetProcessDPIAware()
        window_width, window_height = config['window_size'][0], config['window_size'][1]
        config['teammate_active_at_start'] = True
        config['tick_rate'] = tick_rate
        window = pygame.display.set_mode((window_width, window_height), flags=pygame.NOFRAME)
        pygame.display.set_caption("MAISR Overfit Testing")

        # Create base environment
        base_env = MAISREnvVec(
            config=config,
            clock=clock,
            window=window,
            render_mode='human',
            run_name='overfit_test',
            tag=f'overfit_analysis_0',
        )
    else:
        base_env = MAISREnvVec(
            config=config,
            render_mode='headless',
            run_name='overfit_test',
            tag=f'overfit_analysis_0',
        )

    # Create subpolicies
    subpolicies = {
        'local_search': LocalSearch(model_path=None),
        'change_region': ChangeRegions(model_path=None),
        'go_to_threat': GoToNearestThreat(model_path=None),
        'local_tsp_nocoord': TargetSearchLocalTSP(search_radius=200),
        'global_tsp_nocoord': TargetSearchLocalTSP(search_radius=1000),
        'local_tsp_yescoord': TargetSearchLocalTSP(search_radius=200, spatial_coord=True),
        'global_tsp_yescoord': TargetSearchLocalTSP(search_radius=1000, spatial_coord=True)
    }

    # Storage for all results
    all_results = {}
    all_episode_data = {}

    # Create output directory
    os.makedirs('./logs/overfit_tests', exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("STARTING OVERFIT AGENT TESTING")
    print("=" * 80)
    print(f"Testing {len(overfit_agents)} agent types with {len(behavior_types)} behavior types")
    print(f"Episodes per configuration: {num_episodes}")
    print(f"Total episodes: {len(overfit_agents) * len(behavior_types) * num_episodes}")


    # Main testing loop
    for overfit_type in overfit_agents:
        print(f"\n{'=' * 60}")
        print(f"TESTING AGENT TYPE: {overfit_type.upper()}")
        print(f"{'=' * 60}")

        all_results[overfit_type] = {}
        all_episode_data[overfit_type] = {}

        # Load the agent for this type
        base_path = model_and_stats_paths[overfit_type]
        model_path, norm_stats_path = find_model_files(base_path)
        #model_path = model_path_dict[overfit_type]
        agent = model = PPO.load(model_path)

        for behavior_type in behavior_types:
            print(f"\n--- Testing {behavior_type} behavior ---")

            # Set up teammate manager based on behavior type
            if behavior_type == 'aligned':
                teammate_overfit_type = overfit_type
            else:  # counter
                teammate_overfit_type = get_counter_overfit_type(overfit_type)

            print(f"Testing Agent overfit to: {overfit_type}")
            print(f"Teammate type: {teammate_overfit_type}")

            local_search_policy = LocalSearch()
            go_to_highvalue_policy = GoToNearestThreat(model_path=None)
            change_region_subpolicy = ChangeRegions(model_path=None)
            evade_policy = None

            teammate_manager = TeammateManager(
                    league_type='vanilla',
                    balance_method='uniform',
                    selfplay_checkpoint_dir=None,
                    pretrained_teammate_dir=None,
                    subpolicies=subpolicies,
                    overfit_test=teammate_overfit_type)

            # Create environment with appropriate teammate manager

            env = MaisrLocalSearchWrapper(
                base_env,
                config['obs_noise_std_localsearch'],
                local_search_policy,
                go_to_highvalue_policy,
                change_region_subpolicy,
                evade_policy,
                teammate_manager=teammate_manager
            )

            # env = MaisrLocalSearchWrapper(
            #     base_env,
            #     local_search_policy=LocalSearch(model_path=localsearch_model_path, norm_stats_filepath=localsearch_normstats_path),
            #     go_to_highvalue_policy=GoToNearestThreat(model_path=None),
            #     change_region_subpolicy=ChangeRegions(model_path=None),
            #     evade_policy=EvadeDetection(model_path=None),
            #     teammate_manager=teammate_manager
            # )

            env = DummyVecEnv([lambda: env])
            env = VecNormalize.load(norm_stats_path, env)
            env.training = False
            env.norm_reward = False

            print(f'Loaded norm stats from {norm_stats_path}')
            env.training = False
            env.norm_Reward = False

            # Run episodes
            episode_data = run_episode_batch(env, agent, num_episodes, overfit_type, behavior_type, use_normalize)

            # Calculate summary statistics
            summary_stats = calculate_summary_statistics(episode_data)

            # Store results
            all_results[overfit_type][behavior_type] = summary_stats
            all_episode_data[overfit_type][behavior_type] = episode_data

            # Print summary for this configuration
            print(f"\nSummary for {overfit_type} agent with {behavior_type} behavior:")
            print(f"  Mean reward: {summary_stats['reward']['mean']:.2f} ± {summary_stats['reward']['std']:.2f}")
            print(
                f"  Mean target IDs: {summary_stats['target_ids']['mean']:.2f} ± {summary_stats['target_ids']['std']:.2f}")
            print(
                f"  Mean threat IDs: {summary_stats['threat_ids']['mean']:.2f} ± {summary_stats['threat_ids']['std']:.2f}")
            print(
                f"  Mean teammate distance: {summary_stats['avg_teammate_distance']['mean']:.1f} ± {summary_stats['avg_teammate_distance']['std']:.1f}")
            print(f"  Success rate: {summary_stats['success_rate']:.2%}")
            print(
                f"  Mean efficiency: {summary_stats['efficiency_score']['mean']:.4f} ± {summary_stats['efficiency_score']['std']:.4f}")

    # Close environment
    base_env.close()

    print(f"\n{'=' * 80}")
    print("TESTING COMPLETED - GENERATING ANALYSIS")
    print(f"{'=' * 80}")

    # Create comprehensive comparison report
    print(f"\n=== COMPREHENSIVE COMPARISON REPORT ===")

    for overfit_type in overfit_agents:
        print(f"\n{overfit_type.upper()} AGENT ANALYSIS:")
        print(f"{'-' * 40}")

        aligned_stats = all_results[overfit_type]['aligned']
        counter_stats = all_results[overfit_type]['counter']

        # Calculate differences (aligned - counter)
        reward_diff = aligned_stats['reward']['mean'] - counter_stats['reward']['mean']
        target_diff = aligned_stats['target_ids']['mean'] - counter_stats['target_ids']['mean']
        distance_diff = aligned_stats['avg_teammate_distance']['mean'] - counter_stats['avg_teammate_distance']['mean']
        success_diff = aligned_stats['success_rate'] - counter_stats['success_rate']
        efficiency_diff = aligned_stats['efficiency_score']['mean'] - counter_stats['efficiency_score']['mean']

        print(
            f"Reward:     Aligned={aligned_stats['reward']['mean']:.2f}, Counter={counter_stats['reward']['mean']:.2f}, Diff={reward_diff:+.2f}")
        print(
            f"Targets:    Aligned={aligned_stats['target_ids']['mean']:.2f}, Counter={counter_stats['target_ids']['mean']:.2f}, Diff={target_diff:+.2f}")
        print(
            f"Distance:   Aligned={aligned_stats['avg_teammate_distance']['mean']:.1f}, Counter={counter_stats['avg_teammate_distance']['mean']:.1f}, Diff={distance_diff:+.1f}")
        print(
            f"Success:    Aligned={aligned_stats['success_rate']:.2%}, Counter={counter_stats['success_rate']:.2%}, Diff={success_diff:+.2%}")
        print(
            f"Efficiency: Aligned={aligned_stats['efficiency_score']['mean']:.4f}, Counter={counter_stats['efficiency_score']['mean']:.4f}, Diff={efficiency_diff:+.4f}")

        # Determine which behavior is better
        aligned_better = sum([reward_diff > 0, target_diff > 0, success_diff > 0, efficiency_diff > 0])
        counter_better = sum([reward_diff < 0, target_diff < 0, success_diff < 0, efficiency_diff < 0])

        if aligned_better > counter_better:
            print(f"✅ ALIGNED behavior performs better ({aligned_better}/4 metrics)")
        elif counter_better > aligned_better:
            print(f"❌ COUNTER behavior performs better ({counter_better}/4 metrics)")
        else:
            print(f"🟡 MIXED results - no clear winner")

    # Save detailed results
    print(f"\n=== SAVING RESULTS ===")

    # Convert to JSON-serializable format
    serializable_results = convert_to_json_serializable(all_results)
    serializable_episode_data = convert_to_json_serializable(all_episode_data)

    # Save summary statistics
    summary_filename = f"./logs/overfit_tests/overfit_summary_stats_{timestamp}.json"
    try:
        with open(summary_filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Summary statistics saved to {summary_filename}")
    except Exception as e:
        print(f"Error saving summary JSON: {e}")
        # Fallback to pickle
        pickle_filename = summary_filename.replace('.json', '.pkl')
        with open(pickle_filename, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"Saved as pickle instead: {pickle_filename}")

    # Save detailed episode data
    detailed_filename = f"./logs/overfit_tests/overfit_detailed_data_{timestamp}.json"
    try:
        with open(detailed_filename, 'w') as f:
            json.dump(serializable_episode_data, f, indent=2)
        print(f"Detailed episode data saved to {detailed_filename}")
    except Exception as e:
        print(f"Error saving detailed JSON: {e}")
        # Fallback to pickle
        pickle_filename = detailed_filename.replace('.json', '.pkl')
        with open(pickle_filename, 'wb') as f:
            pickle.dump(all_episode_data, f)
        print(f"Saved as pickle instead: {pickle_filename}")

    # Create and save comparison plots
    create_comparison_plots(all_results, timestamp)

    # Generate final summary table
    print(f"\n=== FINAL PERFORMANCE SUMMARY TABLE ===")
    print(
        f"{'Agent Type':<12} | {'Behavior':<8} | {'Reward':<8} | {'Targets':<7} | {'Success%':<8} | {'Efficiency':<10}")
    print(f"{'-' * 12} | {'-' * 8} | {'-' * 8} | {'-' * 7} | {'-' * 8} | {'-' * 10}")

    for overfit_type in overfit_agents:
        for behavior_type in behavior_types:
            stats = all_results[overfit_type][behavior_type]
            print(f"{overfit_type:<12} | {behavior_type:<8} | {stats['reward']['mean']:<8.2f} | "
                  f"{stats['target_ids']['mean']:<7.2f} | {stats['success_rate']:<8.1%} | "
                  f"{stats['efficiency_score']['mean']:<10.4f}")

    # Calculate and display correlation analysis
    print(f"\n=== CORRELATION ANALYSIS ===")

    # Analyze correlation between agent type characteristics and performance differences
    print(f"Performance advantage of aligned vs counter behavior:")

    reward_advantages = []
    target_advantages = []
    success_advantages = []

    for overfit_type in overfit_agents:
        reward_adv = all_results[overfit_type]['aligned']['reward']['mean'] - \
                     all_results[overfit_type]['counter']['reward']['mean']
        target_adv = all_results[overfit_type]['aligned']['target_ids']['mean'] - \
                     all_results[overfit_type]['counter']['target_ids']['mean']
        success_adv = all_results[overfit_type]['aligned']['success_rate'] - all_results[overfit_type]['counter'][
            'success_rate']

        reward_advantages.append(reward_adv)
        target_advantages.append(target_adv)
        success_advantages.append(success_adv)

        print(f"{overfit_type:<12}: Reward={reward_adv:+.2f}, Targets={target_adv:+.2f}, Success={success_adv:+.2%}")

    # Find best and worst performing configurations
    best_configs = []
    worst_configs = []

    for overfit_type in overfit_agents:
        for behavior_type in behavior_types:
            config_name = f"{overfit_type}_{behavior_type}"
            combined_score = (all_results[overfit_type][behavior_type]['reward']['mean'] +
                              all_results[overfit_type][behavior_type]['target_ids']['mean'] +
                              all_results[overfit_type][behavior_type]['success_rate'] * 10)
            best_configs.append((config_name, combined_score))

    best_configs.sort(key=lambda x: x[1], reverse=True)

    print(f"\n=== TOP 3 CONFIGURATIONS ===")
    for i, (config, score) in enumerate(best_configs[:3]):
        print(f"{i + 1}. {config} (Combined Score: {score:.2f})")

    print(f"\n=== BOTTOM 3 CONFIGURATIONS ===")
    for i, (config, score) in enumerate(best_configs[-3:]):
        print(f"{len(best_configs) - 2 + i}. {config} (Combined Score: {score:.2f})")

    print(f"\n{'=' * 80}")
    print("OVERFIT TESTING ANALYSIS COMPLETE")
    print(f"{'=' * 80}")
    print(f"Results saved to: ./logs/overfit_tests/")
    print(f"Timestamp: {timestamp}")
    print(f"Total episodes run: {len(overfit_agents) * len(behavior_types) * num_episodes}")

    pygame.quit()