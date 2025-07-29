import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pygame
from env_multi_new import MAISREnvVec
from training_wrappers.localsearch_training_wrapper import MaisrLocalSearchWrapper
from utility.data_logging import load_env_config
from utility.league_management import (
    TeammateManager, LocalSearch, ChangeRegions, GoToNearestThreat,
    TargetSearchLocalTSP
)


def create_subpolicies():
    """Create the subpolicy instances needed for the teammate manager"""
    return {
        'local_search': LocalSearch(model_path=None),
        'change_region': ChangeRegions(model_path=None),
        'go_to_threat': GoToNearestThreat(model_path=None),
        'local_tsp_nocoord': TargetSearchLocalTSP(
            search_radius=200,
            spatial_coord=False,
            search_method="greedy"
        ),
        'local_tsp_yescoord': TargetSearchLocalTSP(
            search_radius=200,
            spatial_coord=True,
            search_method="greedy"
        ),
        'global_tsp_nocoord': TargetSearchLocalTSP(
            search_radius=400,
            spatial_coord=False,
            search_method="greedy"
        ),
        'global_tsp_yescoord': TargetSearchLocalTSP(
            search_radius=400,
            spatial_coord=True,
            search_method="greedy"
        )
    }


def run_episode_and_collect_trajectory(env, max_steps=1000):
    """Run a single episode and collect agent trajectories"""
    obs = env.reset()[0]

    # Initialize trajectory storage
    trajectories = {
        'agent_0': {'x': [], 'y': [], 'actions': [], 'subpolicies': []},
        'agent_1': {'x': [], 'y': [], 'actions': [], 'subpolicies': []}
    }

    # Store environment state information
    targets_initial = []
    threats_initial = []

    # Get initial target and threat positions
    base_env = env.env if hasattr(env, 'env') else env
    for i in range(base_env.config['num_targets']):
        target_pos = base_env.targets[i, 3:5]  # x, y coordinates
        targets_initial.append(target_pos.copy())

    for i in range(len(base_env.threats)):
        threat_pos = base_env.threats[i, :2]  # x, y coordinates
        threats_initial.append(threat_pos.copy())

    step_count = 0
    done = False

    print(f"Starting episode with {len(targets_initial)} targets and {len(threats_initial)} threats")

    while not done and step_count < max_steps:
        # Record current positions
        for agent_idx in range(base_env.config['num_aircraft']):
            agent_id = base_env.aircraft_ids[agent_idx]
            agent_x = base_env.agents[agent_id].x
            agent_y = base_env.agents[agent_id].y

            trajectories[f'agent_{agent_idx}']['x'].append(agent_x)
            trajectories[f'agent_{agent_idx}']['y'].append(agent_y)

        # For the wrapped environment, we only need to provide action for agent 0
        # Agent 1 (teammate) is controlled by the teammate policy
        action = 0  # Default action (can be random or from a simple policy)

        # Take step
        obs, reward, terminated, truncated, info = env.step(action)

        # Record the action and subpolicy info if available
        trajectories['agent_0']['actions'].append(action)
        if hasattr(env, 'get_current_subpolicy_info'):
            try:
                subpolicy_id, subpolicy_name = env.get_current_subpolicy_info()
                trajectories['agent_0']['subpolicies'].append(subpolicy_name)
            except:
                trajectories['agent_0']['subpolicies'].append('Unknown')
        else:
            trajectories['agent_0']['subpolicies'].append('LocalSearch')

        # Record teammate subpolicy if available
        if hasattr(env, 'get_teammate_subpolicy_info'):
            try:
                teammate_subpolicy_id, teammate_subpolicy_name = env.get_teammate_subpolicy_info()
                trajectories['agent_1']['subpolicies'].append(teammate_subpolicy_name)
            except:
                trajectories['agent_1']['subpolicies'].append('Unknown')
        else:
            trajectories['agent_1']['subpolicies'].append('Heuristic')

        done = terminated or truncated
        step_count += 1

        if step_count % 100 == 0:
            print(f"Step {step_count}, Reward: {reward:.2f}, Targets identified: {info.get('target_ids', 0)}")

    print(f"Episode completed in {step_count} steps")
    print(f"Final reward: {reward:.2f}")
    print(f"Targets identified: {info.get('target_ids', 0)}")

    return trajectories, targets_initial, threats_initial, base_env.config


def plot_trajectory(trajectories, targets, threats, config, save_path=None):
    """Plot the agent trajectories with targets and threats"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Set up the plot
    gameboard_size = config['gameboard_size']
    half_size = gameboard_size / 2
    ax.set_xlim(-half_size, half_size)
    ax.set_ylim(-half_size, half_size)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Agent Trajectory with Noisy Actions Overfit Test', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')

    # Plot targets
    for i, target_pos in enumerate(targets):
        ax.scatter(target_pos[0], target_pos[1], c='green', s=100, marker='s',
                   alpha=0.7, label='Targets' if i == 0 else "")
        ax.annotate(f'T{i}', (target_pos[0], target_pos[1]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Plot threats with danger zones
    threat_radius = config.get('threat_radius', 50)
    for i, threat_pos in enumerate(threats):
        # Threat center
        ax.scatter(threat_pos[0], threat_pos[1], c='red', s=150, marker='X',
                   alpha=0.8, label='Threats' if i == 0 else "")

        # Threat danger zone
        circle = Circle((threat_pos[0], threat_pos[1]), threat_radius,
                        color='red', alpha=0.2, fill=True)
        ax.add_patch(circle)

        ax.annotate(f'Th{i}', (threat_pos[0], threat_pos[1]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8, color='red')

    # Plot agent trajectories
    colors = ['blue', 'orange']
    agent_names = ['Agent 0 (Human Control)', 'Agent 1 (Teammate)']

    for agent_idx in range(min(2, len(trajectories))):
        agent_key = f'agent_{agent_idx}'
        if agent_key in trajectories and len(trajectories[agent_key]['x']) > 0:
            x_traj = trajectories[agent_key]['x']
            y_traj = trajectories[agent_key]['y']

            # Plot trajectory line
            ax.plot(x_traj, y_traj, color=colors[agent_idx], alpha=0.6,
                    linewidth=2, label=f'{agent_names[agent_idx]} Path')

            # Plot start position
            ax.scatter(x_traj[0], y_traj[0], color=colors[agent_idx], s=150,
                       marker='o', edgecolors='black', linewidth=2,
                       label=f'{agent_names[agent_idx]} Start')

            # Plot end position
            ax.scatter(x_traj[-1], y_traj[-1], color=colors[agent_idx], s=150,
                       marker='*', edgecolors='black', linewidth=2,
                       label=f'{agent_names[agent_idx]} End')

            # Add trajectory direction arrows (every 50 steps)
            for i in range(0, len(x_traj) - 1, 50):
                dx = x_traj[i + 1] - x_traj[i]
                dy = y_traj[i + 1] - y_traj[i]
                if abs(dx) > 1 or abs(dy) > 1:  # Only draw if there's actual movement
                    ax.arrow(x_traj[i], y_traj[i], dx * 5, dy * 5,
                             head_width=10, head_length=15, fc=colors[agent_idx],
                             ec=colors[agent_idx], alpha=0.5)

    # Add quadrant lines
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Add quadrant labels
    offset = half_size * 0.8
    ax.text(-offset, offset, 'NW', ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    ax.text(offset, offset, 'NE', ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    ax.text(-offset, -offset, 'SW', ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    ax.text(offset, -offset, 'SE', ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

    # Create legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trajectory plot saved to: {save_path}")

    plt.show()


def analyze_trajectory_stats(trajectories):
    """Analyze and print trajectory statistics"""
    print("\n" + "=" * 50)
    print("TRAJECTORY ANALYSIS")
    print("=" * 50)

    for agent_idx in range(len(trajectories)):
        agent_key = f'agent_{agent_idx}'
        if agent_key not in trajectories:
            continue

        x_traj = np.array(trajectories[agent_key]['x'])
        y_traj = np.array(trajectories[agent_key]['y'])

        if len(x_traj) == 0:
            continue

        print(f"\nAgent {agent_idx}:")
        print(f"  Total steps: {len(x_traj)}")

        # Calculate total distance traveled
        if len(x_traj) > 1:
            distances = np.sqrt(np.diff(x_traj) ** 2 + np.diff(y_traj) ** 2)
            total_distance = np.sum(distances)
            avg_speed = np.mean(distances)
            print(f"  Total distance: {total_distance:.1f} pixels")
            print(f"  Average speed: {avg_speed:.1f} pixels/step")

        # Position statistics
        print(f"  X range: [{np.min(x_traj):.1f}, {np.max(x_traj):.1f}]")
        print(f"  Y range: [{np.min(y_traj):.1f}, {np.max(y_traj):.1f}]")
        print(f"  Start position: ({x_traj[0]:.1f}, {y_traj[0]:.1f})")
        print(f"  End position: ({x_traj[-1]:.1f}, {y_traj[-1]:.1f})")

        # Subpolicy usage if available
        if 'subpolicies' in trajectories[agent_key] and len(trajectories[agent_key]['subpolicies']) > 0:
            subpolicies = trajectories[agent_key]['subpolicies']
            unique_subpolicies, counts = np.unique(subpolicies, return_counts=True)
            print(f"  Subpolicy usage:")
            for subpolicy, count in zip(unique_subpolicies, counts):
                percentage = (count / len(subpolicies)) * 100
                print(f"    {subpolicy}: {count} steps ({percentage:.1f}%)")


def main():
    """Main function to run the overfit test and plot trajectory"""
    # Configuration
    config_filename = 'configs/Monolith_R5L_july8.json'  # Update path as needed
    overfit_test = "noisy_actions"  # This will test the fallback behavior

    try:
        # Load configuration
        config = load_env_config(config_filename)
        config['teammate_active_at_start'] = True
        print(f'Loaded config from {config_filename}')

        # Initialize pygame (needed for the environment)
        pygame.init()

        # Create subpolicies
        subpolicies = create_subpolicies()

        # Create teammate manager with overfit test
        teammate_manager = TeammateManager(
            league_type="strategy_diverse",  # Not used for overfit tests
            balance_method="uniform",  # Not used for overfit tests
            selfplay_checkpoint_dir=None,
            pretrained_teammate_dir=None,
            subpolicies=subpolicies,
            overfit_test=overfit_test
        )

        print(f"\nCreating environment with overfit_test='{overfit_test}'")

        # Create base environment
        base_env = MAISREnvVec(
            config=config,
            clock=None,  # No pygame clock needed for headless operation
            window=None,  # No rendering window
            render_mode='headless',  # Headless mode
            run_name='overfit_test_trajectory',
            tag='noisy_actions_test_0',
        )

        local_search_policy = LocalSearch()
        go_to_highvalue_policy = GoToNearestThreat(model_path=None)
        change_region_subpolicy = ChangeRegions(model_path=None)
        evade_policy = None

        # Create wrapped environment
        env = MaisrLocalSearchWrapper(
            base_env,
            config['obs_noise_std_localsearch'],
            local_search_policy,
            go_to_highvalue_policy,
            change_region_subpolicy,
            evade_policy,
            teammate_manager=teammate_manager
        )

        print("Environment created successfully")
        print(f"Teammate: {teammate_manager.current_teammate.name if teammate_manager.current_teammate else 'None'}")

        # Run episode and collect trajectory
        print("\nRunning episode...")
        trajectories, targets, threats, env_config = run_episode_and_collect_trajectory(env, max_steps=500)

        # Analyze trajectory statistics
        analyze_trajectory_stats(trajectories)

        # Plot trajectory
        print("\nGenerating trajectory plot...")
        save_path = f"trajectory_overfit_{overfit_test}.png"
        plot_trajectory(trajectories, targets, threats, env_config, save_path=save_path)

        print(f"\nScript completed successfully!")

    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {e}")
        print("Please update the config_filename path in the script")
    except Exception as e:
        print(f"Error during execution: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            env.close()
        except:
            pass
        pygame.quit()


if __name__ == "__main__":
    main()