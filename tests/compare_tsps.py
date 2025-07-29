import ctypes

import pygame
import numpy as np
import matplotlib.pyplot as plt
from env_multi_new import MAISREnvVec
from training_wrappers.localsearch_training_wrapper import MaisrLocalSearchWrapper
from utility.data_logging import load_env_config
from utility.league_management import (
    TeammateManager, TargetSearchLocalTSP, LocalSearch,
    ChangeRegions, GoToNearestThreat, EvadeDetection
)


def run_agent_comparison():
    """Compare three agent types: TSP-200, TSP-1000, and Greedy across 6 episodes each"""

    # Configuration
    config_filename = 'configs/july1_ls_2ship.json'
    num_episodes = 7
    tick_rate = 120  # Faster for batch testing

    # Agent configurations
    agent_configs = [
        {'name': 'Greedy', 'radius': 1000, 'use_tsp': True, 'search_method': 'greedy'},
        {'name': 'Clusters', 'radius': 1000, 'use_tsp': True, 'search_method': 'clusters'},
        {'name': 'Early_weighted', 'radius': 1000, 'use_tsp': True, 'search_method': 'early_weighted'},
        #{'name': 'Greedy', 'radius': 1000, 'use_tsp': False}  # radius doesn't matter for greedy

    ]

    # Load environment configuration
    config = load_env_config(config_filename)
    print(f'LOADED CONFIG {config_filename}')

    # Setup pygame (minimal for batch processing)
    pygame.display.init()
    pygame.font.init()
    clock = pygame.time.Clock()
    ctypes.windll.user32.SetProcessDPIAware()

    window_width, window_height = config['window_size'][0], config['window_size'][1]
    config['tick_rate'] = tick_rate
    window = pygame.display.set_mode((window_width, window_height), flags=pygame.NOFRAME)
    pygame.display.set_caption("Agent Comparison Test")

    # Storage for all paths and results
    all_paths = {}  # {agent_name: [episode_paths]}
    all_results = {}  # {agent_name: [episode_results]}
    all_targets = {}

    for agent_config in agent_configs:
        agent_name = agent_config['name']
        search_radius = agent_config['radius']
        search_method = agent_config['search_method']
        use_tsp = agent_config['use_tsp']

        print(f"\n{'=' * 50}")
        print(f"Testing {agent_name} (radius={search_radius}, use_tsp={use_tsp})")
        print(f"{'=' * 50}")

        all_paths[agent_name] = []
        all_results[agent_name] = []
        all_targets[agent_name] = []

        for episode in range(num_episodes):
            print(f"\n--- {agent_name} Episode {episode + 1}/{num_episodes} ---")

            # Create fresh environment for each episode
            base_env = MAISREnvVec(
                config=config,
                clock=clock,
                window=window,
                render_mode='human',
                run_name=f'{agent_name.lower()}_test',
                tag=f'{agent_name.lower()}_episode_{episode}',
            )

            # Create the Local TSP policy
            local_tsp_policy = TargetSearchLocalTSP(
                search_radius=search_radius,
                spatial_coord='false',
                model_path=None,
                norm_stats_filepath=None,
                search_method=search_method
            )

            # Create subpolicies for teammate
            subpolicies = {
                'local_search': LocalSearch(model_path=None),
                'change_region': ChangeRegions(model_path=None),
                'go_to_threat': GoToNearestThreat(model_path=None),
                'local_tsp_nocoord': local_tsp_policy
            }

            # Create teammate manager
            teammate_manager = TeammateManager(
                league_type=config['league_type'],
                balance_method='uniform',
                subpolicies=subpolicies,
                selfplay_checkpoint_dir=None,
                pretrained_teammate_dir=None
            )

            # Create wrapper
            env = MaisrLocalSearchWrapper(
                base_env,
                obs_noise_std=0.0,
                local_search_policy=local_tsp_policy,
                go_to_highvalue_policy=GoToNearestThreat(model_path=None),
                change_region_subpolicy=ChangeRegions(model_path=None),
                evade_policy=EvadeDetection(model_path=None),
                teammate_manager=teammate_manager
            )

            # Run episode
            obs = env.reset()[0]
            episode_reward = 0
            episode_steps = 0
            done = False

            episode_path = []
            waypoints_calculated = 0
            waypoints_reached = 0

            while not done and episode_steps < 1000:  # Max steps to prevent infinite loops
                # Handle pygame events (minimal)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            done = True
                            break

                if done:
                    break

                # Get action based on agent type
                if use_tsp:
                    action, _ = local_tsp_policy.act(obs, env=env.env, agent_id=0)
                else:
                    # Use greedy fallback policy
                    action, _ = local_tsp_policy.fallback_policy.act(obs, env=env.env, agent_id=0)

                # Track waypoint changes for TSP agents
                if use_tsp and hasattr(local_tsp_policy, 'current_waypoints'):
                    current_waypoint_count = len(local_tsp_policy.current_waypoints)
                    if current_waypoint_count > 0:
                        waypoints_calculated = max(waypoints_calculated, 1)

                # Step environment
                if isinstance(action, tuple):
                    action = action[0]

                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                episode_steps += 1

                # Record agent position
                agent_x = env.env.agents[env.env.aircraft_ids[0]].x
                agent_y = env.env.agents[env.env.aircraft_ids[0]].y
                episode_path.append((agent_x, agent_y))

                # Minimal rendering (every 10 steps to speed up)
                if episode_steps % 10 == 0:
                    env.render()
                    pygame.display.flip()
                    clock.tick(tick_rate)

            # Store results

            episode_targets = {
                'positions': env.env.targets[:, 3:5].copy(),  # x,y positions
                'identified': env.env.targets[:, 2].copy()  # identification status
            }


            episode_result = {
                'steps': episode_steps,
                'reward': episode_reward,
                'targets_found': env.env.targets_identified,
                'threats_identified': env.env.num_threats_identified,
                'waypoints_calculated': waypoints_calculated
            }

            all_paths[agent_name].append(episode_path)
            all_results[agent_name].append(episode_result)
            all_targets[agent_name].append(episode_targets)

            print(f"  Steps: {episode_steps}, Reward: {episode_reward:.2f}, "
                  f"Targets: {episode_result['targets_found']}, "
                  f"Threats: {episode_result['threats_identified']}")

            env.close()

    pygame.quit()

    # Create unified visualization
    create_comparison_plot(all_paths, all_results, all_targets, config)

    # Print summary statistics
    print_summary_statistics(all_results)


def create_comparison_plot(all_paths, all_results, all_targets, config):
    """Create a figure showing all three agents' paths overlaid for each episode"""

    # Create figure with subplots: 2 rows x 4 columns (7 episodes)
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    fig.suptitle('Agent Trajectory Comparison: All Agents Overlaid by Episode', fontsize=16)

    # Get gameboard boundaries for consistent scaling
    board_width = config['gameboard_size']
    board_height = config['gameboard_size']
    x_lim = [-board_width / 2, board_width / 2]
    y_lim = [-board_height / 2, board_height / 2]

    agent_names = ['Greedy', 'Clusters', 'Early_weighted']
    colors = ['blue', 'red', 'green']
    line_styles = ['-', '--', '--']

    # Flatten axes array for easier indexing
    axes_flat = axes.flatten()

    def calculate_path_length(path):
        """Calculate total distance traveled along path"""
        if len(path) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            total_distance += np.sqrt(dx ** 2 + dy ** 2)

        return total_distance

    for episode_idx in range(7):
        ax = axes_flat[episode_idx]

        # Plot targets for this episode using actual target data
        # Use target data from the first agent (should be same for all agents in same episode)
        if episode_idx < len(all_targets[agent_names[0]]):
            episode_targets = all_targets[agent_names[0]][episode_idx]
            target_positions = episode_targets['positions']
            target_identified = episode_targets['identified']

            # Separate identified and unidentified targets for different colors
            identified_mask = target_identified >= 1.0
            unidentified_mask = target_identified < 1.0

            # Plot unidentified targets (orange)
            if np.any(unidentified_mask):
                unidentified_pos = target_positions[unidentified_mask]
                ax.scatter(unidentified_pos[:, 0], unidentified_pos[:, 1],
                           color='orange', s=50, marker='o', alpha=0.7,
                           zorder=8, edgecolors='black', linewidth=1)

            # Plot identified targets (lime green)
            if np.any(identified_mask):
                identified_pos = target_positions[identified_mask]
                ax.scatter(identified_pos[:, 0], identified_pos[:, 1],
                           color='lime', s=50, marker='o', alpha=0.7,
                           zorder=8, edgecolors='black', linewidth=1)

        # Plot all three agents on the same subplot
        for agent_idx, agent_name in enumerate(agent_names):
            if episode_idx < len(all_paths[agent_name]):
                path = all_paths[agent_name][episode_idx]
                result = all_results[agent_name][episode_idx]

                if len(path) > 0:
                    # Calculate path length
                    path_length = calculate_path_length(path)

                    # Extract x, y coordinates
                    x_coords = [pos[0] for pos in path]
                    y_coords = [pos[1] for pos in path]

                    # Plot path with different colors and line styles
                    ax.plot(x_coords, y_coords, color=colors[agent_idx],
                            linestyle=line_styles[agent_idx], linewidth=2,
                            alpha=0.8,
                            label=f'{agent_name} (L={path_length:.0f})')

                    # Mark start point (same for all agents, so only do it once)
                    if agent_idx == 0:
                        ax.scatter(x_coords[0], y_coords[0], color='black',
                                   s=100, marker='o', zorder=10)

                    # Mark end points with agent-specific colors
                    ax.scatter(x_coords[-1], y_coords[-1], color=colors[agent_idx],
                               s=80, marker='X', zorder=9)

        # Styling
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Episode {episode_idx + 1}', fontsize=12)
        ax.legend(fontsize=8, loc='best')

        # Labels
        if episode_idx >= 4:  # Bottom row
            ax.set_xlabel('X Position')
        if episode_idx % 4 == 0:  # Left column
            ax.set_ylabel('Y Position')

    plt.tight_layout()
    plt.savefig('agent_trajectory_overlay.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nOverlay trajectory plot saved as 'agent_trajectory_overlay.png'")



def print_summary_statistics(all_results):
    """Print summary statistics for all agents"""

    print(f"\n{'=' * 60}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 60}")

    for agent_name, results in all_results.items():
        print(f"\n{agent_name}:")
        print("-" * 20)

        # Calculate averages
        avg_reward = np.mean([r['reward'] for r in results])
        avg_steps = np.mean([r['steps'] for r in results])
        avg_targets = np.mean([r['targets_found'] for r in results])
        avg_threats = np.mean([r['threats_identified'] for r in results])

        # Calculate std deviations
        std_reward = np.std([r['reward'] for r in results])
        std_steps = np.std([r['steps'] for r in results])
        std_targets = np.std([r['targets_found'] for r in results])
        std_threats = np.std([r['threats_identified'] for r in results])

        print(f"  Reward:    {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"  Steps:     {avg_steps:.1f} ± {std_steps:.1f}")
        print(f"  Targets:   {avg_targets:.1f} ± {std_targets:.1f}")
        print(f"  Threats:   {avg_threats:.1f} ± {std_threats:.1f}")

        # Episode breakdown
        print("  Episodes:  ", end='')
        for i, r in enumerate(results):
            print(f"E{i + 1}:R{r['reward']:.1f}", end="  " if i < len(results) - 1 else "\n")


if __name__ == "__main__":
    try:
        run_agent_comparison()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        pygame.quit()
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback

        traceback.print_exc()
        pygame.quit()