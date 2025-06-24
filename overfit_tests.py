import ctypes
import pygame
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from env_multi_new import MAISREnvVec
from training_wrappers.modeselector_training_wrapper import MaisrModeSelectorWrapper
from utility.data_logging import load_env_config
from policies.league_management import GenericTeammatePolicy, SubPolicy, LocalSearch, ChangeRegions, GoToNearestThreat, \
    EvadeDetection, TeammateManager
import json
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict


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


def get_behavior_instructions(overfit_type, behavior_type):
    """Get instructions for human player based on overfit agent type and behavior type"""

    base_instructions = {
        'low_risk': {
            'aligned': "Play CONSERVATIVELY. Avoid threats, focus on safe target identification. Use Local Search (1) most of the time.",
            'counter': "Play AGGRESSIVELY. Actively seek threats, take risks. Use Go to Threat (3) frequently."
        },
        'high_risk': {
            'aligned': "Play AGGRESSIVELY. Take risks, actively engage threats. Use Go to Threat (3) frequently.",
            'counter': "Play CONSERVATIVELY. Avoid all risks, stay away from threats. Use Local Search (1) only."
        },
        'nospatial': {
            'aligned': "Ignore your teammate's location. Focus only on your local area. Use Local Search (1) primarily.",
            'counter': "Coordinate closely with your teammate. Avoid overlapping areas. Use Change Region (2) to spread out."
        },
        'highspatial': {
            'aligned': "Coordinate closely with your teammate. Spread out to different areas. Use Change Region (2) when teammate is nearby.",
            'counter': "Ignore your teammate completely. Focus only on your immediate area. Use Local Search (1) regardless of teammate location."
        }
    }

    return base_instructions[overfit_type][behavior_type]


def run_overfit_episode(env, overfit_type, behavior_type, episode_num, total_episodes):
    """Run a single episode with the specified overfit agent and human behavior"""

    obs = env.reset()[0]
    episode_reward = 0
    episode_steps = 0

    # Initialize tracking variables
    initial_threat_ids = env.env.num_threats_identified
    initial_target_ids = env.env.targets_identified

    # Subpolicy tracking for human (agent 0 - the controllable agent)
    human_subpolicy_usage = {0: 0, 1: 0, 2: 0, 3: 0}
    human_subpolicy_switches = 0
    last_human_action = None
    human_subpolicy_sequence = []

    # Teammate tracking for overfit agent (agent 1 - the teammate)
    overfit_subpolicy_usage = {0: 0, 1: 0, 2: 0, 3: 0}
    overfit_switches = 0
    last_overfit_action = None

    # Performance tracking
    detection_events = []
    identification_events = []
    distance_traveled = 0
    last_position = None
    human_positions_visited = []
    overfit_positions_visited = []

    # Get behavior instructions for display
    instructions = get_behavior_instructions(overfit_type, behavior_type)

    done = False
    human_action = 0  # Default to Local Search

    print(f"\n=== Episode {episode_num}/{total_episodes} ===")
    print(f"Overfit Agent: {overfit_type} | Behavior: {behavior_type}")
    print(f"Instructions: {instructions}")
    print("Controls: 1=Local Search, 2=Change Region, 3=Go to Threat, 4=Hold")
    print("Press ESC to quit\n")

    # Key mapping
    key_to_action = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2, pygame.K_4: 3}

    while not done:
        # Handle pygame events for human input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None  # Signal to quit
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None  # Signal to quit
                elif event.key in key_to_action:
                    human_action = key_to_action[event.key]

        # Track human subpolicy usage
        human_subpolicy_usage[human_action] += 1
        human_subpolicy_sequence.append(human_action)
        if last_human_action is not None and last_human_action != human_action:
            human_subpolicy_switches += 1
        last_human_action = human_action

        # Track overfit agent behavior (teammate)
        overfit_subpolicy_id, overfit_subpolicy_name = env.get_teammate_subpolicy_info()
        overfit_subpolicy_usage[overfit_subpolicy_id] += 1
        if last_overfit_action is not None and last_overfit_action != overfit_subpolicy_id:
            overfit_switches += 1
        last_overfit_action = overfit_subpolicy_id

        # Take step (human_action controls the primary agent)
        obs, reward, terminated, truncated, info = env.step(human_action)
        episode_reward += reward
        done = terminated or truncated
        episode_steps += 1

        # Track positions (human is agent 0, overfit agent is agent 1)
        human_pos = (
        env.env.agents[env.env.aircraft_ids[0]].x, env.env.agents[env.env.aircraft_ids[0]].y)  # Human is agent 0
        overfit_pos = (
        env.env.agents[env.env.aircraft_ids[1]].x, env.env.agents[env.env.aircraft_ids[1]].y)  # Overfit is agent 1

        human_positions_visited.append(human_pos)
        overfit_positions_visited.append(overfit_pos)

        if last_position is not None:
            distance_traveled += math.sqrt((human_pos[0] - last_position[0]) ** 2 +
                                           (human_pos[1] - last_position[1]) ** 2)
        last_position = human_pos

        # Track events
        if 'new_detections' in info and info['new_detections'] > 0:
            detection_events.append({
                'step': episode_steps,
                'count': info['new_detections'],
                'position': human_pos
            })

        if 'new_target_ids' in info and info['new_target_ids'] > 0:
            identification_events.append({
                'step': episode_steps,
                'type': 'target',
                'count': info['new_target_ids']
            })

        if 'new_threat_ids' in info and info['new_threat_ids'] > 0:
            identification_events.append({
                'step': episode_steps,
                'type': 'threat',
                'count': info['new_threat_ids']
            })

        # Render with behavior instructions overlay
        human_subpolicy_id, human_subpolicy_name = env.get_current_subpolicy_info()
        env.env.render_subpolicy_indicators(human_subpolicy_id, human_subpolicy_name,
                                            overfit_subpolicy_id, overfit_subpolicy_name)

        # Add behavior type indicator
        font = pygame.font.Font(None, 24)
        behavior_text = font.render(f"{behavior_type.upper()} BEHAVIOR", True, (255, 255, 255))
        overfit_text = font.render(f"Overfit: {overfit_type.upper()}", True, (255, 255, 0))

        env.env.window.blit(behavior_text, (10, 50))
        env.env.window.blit(overfit_text, (10, 75))

        pygame.display.flip()

    # Calculate final metrics
    final_threat_ids = env.env.num_threats_identified
    final_target_ids = env.env.targets_identified
    threat_ids_gained = final_threat_ids - initial_threat_ids
    target_ids_gained = final_target_ids - initial_target_ids

    # Calculate average distance between agents
    distances = []
    for i in range(min(len(human_positions_visited), len(overfit_positions_visited))):
        human_pos = human_positions_visited[i]
        overfit_pos = overfit_positions_visited[i]
        distance = math.sqrt((human_pos[0] - overfit_pos[0]) ** 2 +
                             (human_pos[1] - overfit_pos[1]) ** 2)
        distances.append(distance)

    avg_distance = sum(distances) / len(distances) if distances else 0

    # Store episode data
    episode_info = {
        'overfit_type': overfit_type,
        'behavior_type': behavior_type,
        'episode': episode_num,
        'steps': episode_steps,
        'threat_ids': threat_ids_gained,
        'target_ids': target_ids_gained,
        'total_reward': episode_reward,
        'avg_distance_between_agents': avg_distance,

        # Human behavior tracking
        'human_subpolicy_usage': human_subpolicy_usage.copy(),
        'human_subpolicy_switches': human_subpolicy_switches,
        'human_positions_visited': human_positions_visited,

        # Overfit agent tracking
        'overfit_subpolicy_usage': overfit_subpolicy_usage.copy(),
        'overfit_switches': overfit_switches,
        'overfit_positions_visited': overfit_positions_visited,

        # Performance metrics
        'distance_traveled': distance_traveled,
        'efficiency_score': target_ids_gained / max(episode_steps, 1),
        'spatial_coverage_percent': calculate_spatial_coverage(human_positions_visited,
                                                               env.env.config['gameboard_size']),
        'coordination_score': calculate_coordination_score(human_subpolicy_sequence, overfit_subpolicy_usage),

        # Event tracking
        'detection_events': detection_events,
        'identification_events': identification_events,
        'total_detections': env.env.detections,

        # Episode outcome
        'completed_successfully': env.env.all_targets_identified,
        'termination_reason': 'success' if env.env.all_targets_identified else
        'failed' if env.env.failed else 'timeout',
    }

    print(f"Episode completed: {target_ids_gained} targets, {threat_ids_gained} threats, reward: {episode_reward:.2f}")
    return episode_info


def calculate_summary_stats(episode_data):
    """Calculate mean and std for key metrics"""
    if not episode_data:
        return {}

    metrics = {
        'total_reward': [ep['total_reward'] for ep in episode_data],
        'target_ids': [ep['target_ids'] for ep in episode_data],
        'threat_ids': [ep['threat_ids'] for ep in episode_data],
        'avg_distance_between_agents': [ep['avg_distance_between_agents'] for ep in episode_data],
        'steps': [ep['steps'] for ep in episode_data],
        'efficiency_score': [ep['efficiency_score'] for ep in episode_data],
        'spatial_coverage_percent': [ep['spatial_coverage_percent'] for ep in episode_data],
        'coordination_score': [ep['coordination_score'] for ep in episode_data],
    }

    summary = {}
    for metric_name, values in metrics.items():
        summary[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values
        }

    return summary


def create_comparison_plots(all_results, timestamp):
    """Create comprehensive comparison plots"""

    # Set up the figure
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Overfit Agent Performance: Aligned vs Counter Behavior', fontsize=16)

    # Metrics to plot
    metrics = [
        ('total_reward', 'Total Reward'),
        ('target_ids', 'Target IDs Gained'),
        ('threat_ids', 'Threat IDs Gained'),
        ('avg_distance_between_agents', 'Avg Distance Between Agents'),
        ('steps', 'Episode Steps'),
        ('efficiency_score', 'Efficiency Score'),
        ('spatial_coverage_percent', 'Spatial Coverage %'),
        ('coordination_score', 'Coordination Score'),
    ]

    overfit_types = ['low_risk', 'high_risk', 'nospatial', 'highspatial']
    behavior_types = ['aligned', 'counter']
    colors = {'aligned': 'blue', 'counter': 'red'}

    # Plot each metric
    for idx, (metric_key, metric_title) in enumerate(metrics):
        if idx >= 8:  # Only 8 subplots available
            break

        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        # Prepare data for plotting
        x_positions = np.arange(len(overfit_types))
        width = 0.35

        aligned_means = []
        aligned_stds = []
        counter_means = []
        counter_stds = []

        for overfit_type in overfit_types:
            if overfit_type in all_results:
                aligned_data = all_results[overfit_type]['aligned']['summary'][metric_key]
                counter_data = all_results[overfit_type]['counter']['summary'][metric_key]

                aligned_means.append(aligned_data['mean'])
                aligned_stds.append(aligned_data['std'])
                counter_means.append(counter_data['mean'])
                counter_stds.append(counter_data['std'])
            else:
                aligned_means.append(0)
                aligned_stds.append(0)
                counter_means.append(0)
                counter_stds.append(0)

        # Create bar plot
        bars1 = ax.bar(x_positions - width / 2, aligned_means, width, yerr=aligned_stds,
                       label='Aligned', color=colors['aligned'], alpha=0.7)
        bars2 = ax.bar(x_positions + width / 2, counter_means, width, yerr=counter_stds,
                       label='Counter', color=colors['counter'], alpha=0.7)

        ax.set_xlabel('Overfit Agent Type')
        ax.set_ylabel(metric_title)
        ax.set_title(f'{metric_title}')
        ax.set_xticks(x_positions)
        ax.set_xticklabels([t.replace('_', ' ').title() for t in overfit_types], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Use the last subplot for a summary table
    ax = axes[2, 2]
    ax.axis('off')

    # Create summary table
    summary_text = "Performance Differences (Aligned - Counter):\n\n"
    for overfit_type in overfit_types:
        if overfit_type in all_results:
            reward_diff = (all_results[overfit_type]['aligned']['summary']['total_reward']['mean'] -
                           all_results[overfit_type]['counter']['summary']['total_reward']['mean'])
            target_diff = (all_results[overfit_type]['aligned']['summary']['target_ids']['mean'] -
                           all_results[overfit_type]['counter']['summary']['target_ids']['mean'])

            summary_text += f"{overfit_type.replace('_', ' ').title()}:\n"
            summary_text += f"  Reward: {reward_diff:+.2f}\n"
            summary_text += f"  Targets: {target_diff:+.2f}\n\n"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()

    # Save plot
    plot_filename = f"./logs/overfit_tests/overfit_comparison_plots_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Comparison plots saved to {plot_filename}")


def main():
    # Configuration
    config_filename = 'configs/june23_poc1_2ship.json'
    num_episodes_per_condition = 10  # 10 aligned + 10 counter = 20 total per overfit type
    tick_rate = 40

    # Overfit agent types to test
    overfit_types = ['low_risk', 'high_risk', 'nospatial', 'highspatial']
    behavior_types = ['aligned', 'counter']

    # Load config
    config = load_env_config(config_filename)
    print(f'LOADED CONFIG {config_filename}')

    # Setup pygame
    pygame.display.init()
    pygame.font.init()
    clock = pygame.time.Clock()
    ctypes.windll.user32.SetProcessDPIAware()
    window_width, window_height = config['window_size'][0], config['window_size'][1]
    config['tick_rate'] = tick_rate
    window = pygame.display.set_mode((window_width, window_height), flags=pygame.NOFRAME)
    pygame.display.set_caption("MAISR Overfit Agent Testing")

    # Create logs directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('./logs/overfit_tests', exist_ok=True)

    # Create base environment once (reuse for all episodes)
    base_env = MAISREnvVec(
        config=config,
        clock=clock,
        window=window,
        render_mode='human',
        run_name='overfit_test',
        tag=f'overfit_comparison',
    )

    # Setup subpolicies
    subpolicies = {
        'local_search': LocalSearch(model_path=None),
        'change_region': ChangeRegions(model_path=None),
        'go_to_threat': GoToNearestThreat(model_path=None)
    }

    # Store all results
    all_results = {}
    all_episode_data = []

    # Test each overfit agent type
    for overfit_type in overfit_types:
        print(f"\n{'=' * 60}")
        print(f"TESTING OVERFIT AGENT: {overfit_type.upper()}")
        print(f"{'=' * 60}")

        all_results[overfit_type] = {}

        # Test both behavior types (aligned and counter)
        for behavior_type in behavior_types:
            print(f"\n{'-' * 40}")
            print(f"BEHAVIOR TYPE: {behavior_type.upper()}")
            print(f"{'-' * 40}")

            episode_data = []

            # Run episodes for this condition
            for episode in range(num_episodes_per_condition):
                # Create teammate manager with overfit test parameter for this condition
                teammate_manager = TeammateManager(
                    league_type='strategy_diverse',
                    balance_method='uniform',
                    selfplay_checkpoint_dir=None,
                    pretrained_teammate_dir=None,
                    subpolicies=subpolicies,
                    overfit_test=overfit_type  # This creates the overfit teammate
                )

                # Create wrapped environment with the overfit teammate
                env = MaisrModeSelectorWrapper(
                    base_env,
                    local_search_policy=LocalSearch(model_path=None),
                    go_to_highvalue_policy=GoToNearestThreat(model_path=None),
                    change_region_subpolicy=ChangeRegions(model_path=None),
                    evade_policy=EvadeDetection(model_path=None),
                    teammate_manager=teammate_manager
                )

                # Run episode
                episode_info = run_overfit_episode(
                    env, overfit_type, behavior_type, episode + 1, num_episodes_per_condition
                )

                if episode_info is None:  # User quit
                    base_env.close()
                    return

                episode_data.append(episode_info)
                all_episode_data.append(episode_info)

            # Calculate summary statistics for this condition
            summary_stats = calculate_summary_stats(episode_data)

            all_results[overfit_type][behavior_type] = {
                'episodes': episode_data,
                'summary': summary_stats
            }

            # Print summary for this condition
            print(f"\n{behavior_type.upper()} BEHAVIOR SUMMARY:")
            print(f"  Reward: {summary_stats['total_reward']['mean']:.2f} ± {summary_stats['total_reward']['std']:.2f}")
            print(f"  Target IDs: {summary_stats['target_ids']['mean']:.2f} ± {summary_stats['target_ids']['std']:.2f}")
            print(f"  Threat IDs: {summary_stats['threat_ids']['mean']:.2f} ± {summary_stats['threat_ids']['std']:.2f}")
            print(
                f"  Avg Distance: {summary_stats['avg_distance_between_agents']['mean']:.1f} ± {summary_stats['avg_distance_between_agents']['std']:.1f}")

    # Save all episode data
    serializable_data = convert_to_json_serializable(all_episode_data)
    filename = f"./logs/overfit_tests/overfit_episode_data_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    print(f"\nAll episode data saved to {filename}")

    # Save summary results
    summary_filename = f"./logs/overfit_tests/overfit_summary_{timestamp}.json"
    summary_data = {}
    for overfit_type in all_results:
        summary_data[overfit_type] = {}
        for behavior_type in all_results[overfit_type]:
            summary_data[overfit_type][behavior_type] = all_results[overfit_type][behavior_type]['summary']

    with open(summary_filename, 'w') as f:
        json.dump(convert_to_json_serializable(summary_data), f, indent=2)
    print(f"Summary statistics saved to {summary_filename}")

    # Create comparison plots
    create_comparison_plots(all_results, timestamp)

    # Print final comparison summary
    print(f"\n{'=' * 80}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'=' * 80}")

    for overfit_type in overfit_types:
        if overfit_type in all_results:
            print(f"\n{overfit_type.replace('_', ' ').title()}:")

            aligned_reward = all_results[overfit_type]['aligned']['summary']['total_reward']['mean']
            counter_reward = all_results[overfit_type]['counter']['summary']['total_reward']['mean']
            reward_diff = aligned_reward - counter_reward

            aligned_targets = all_results[overfit_type]['aligned']['summary']['target_ids']['mean']
            counter_targets = all_results[overfit_type]['counter']['summary']['target_ids']['mean']
            target_diff = aligned_targets - counter_targets

            aligned_distance = all_results[overfit_type]['aligned']['summary']['avg_distance_between_agents']['mean']
            counter_distance = all_results[overfit_type]['counter']['summary']['avg_distance_between_agents']['mean']
            distance_diff = aligned_distance - counter_distance

            print(
                f"  Aligned behavior:  Reward={aligned_reward:.2f}, Targets={aligned_targets:.2f}, Distance={aligned_distance:.1f}")
            print(
                f"  Counter behavior:  Reward={counter_reward:.2f}, Targets={counter_targets:.2f}, Distance={counter_distance:.1f}")
            print(
                f"  Difference:        Reward={reward_diff:+.2f}, Targets={target_diff:+.2f}, Distance={distance_diff:+.1f}")

    # Clean up
    base_env.close()


if __name__ == "__main__":
    main()