import ctypes
import pygame
import numpy as np
from env_multi_new import MAISREnvVec
from training_wrappers.localsearch_training_wrapper import MaisrLocalSearchWrapper
from utility.data_logging import load_env_config
from policies.league_management import (
    TeammateManager, TargetSearchLocalTSP, LocalSearch,
    ChangeRegions, GoToNearestThreat, EvadeDetection
)


def run_local_tsp_test():
    """Test the local TSP heuristic agent with specified parameters"""

    # Configuration
    config_filename = 'configs/july1_ls_2ship.json'
    num_episodes = 7
    tick_rate = 10
    search_radius = 1000
    spatial_coord = 'false'  # coord = false

    # Load environment configuration
    config = load_env_config(config_filename)
    print(f'LOADED CONFIG {config_filename}')

    # Setup pygame
    pygame.display.init()
    pygame.font.init()
    font = pygame.font.Font(None, 36)
    clock = pygame.time.Clock()
    ctypes.windll.user32.SetProcessDPIAware()

    window_width, window_height = config['window_size'][0], config['window_size'][1]
    config['tick_rate'] = tick_rate
    window = pygame.display.set_mode((window_width, window_height), flags=pygame.NOFRAME)
    pygame.display.set_caption("Local TSP Heuristic Test")

    # Create base environment
    base_env = MAISREnvVec(
        config=config,
        clock=clock,
        window=window,
        render_mode='human',
        run_name='local_tsp_test',
        tag='tsp_heuristic_test_0',
    )

    # Create the Local TSP policy with specified parameters
    local_tsp_policy = TargetSearchLocalTSP(
        search_radius=search_radius,
        spatial_coord=spatial_coord,
        model_path=None,  # Use heuristic
        norm_stats_filepath=None
    )

    # Create other subpolicies for teammate
    subpolicies = {
        'local_search': LocalSearch(model_path=None),
        'change_region': ChangeRegions(model_path=None),
        'go_to_threat': GoToNearestThreat(model_path=None),
        'local_tsp_nocoord': local_tsp_policy  # Add our TSP policy
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
        obs_noise_std=0.0,  # No noise for testing
        local_search_policy=local_tsp_policy,  # Use TSP as main policy
        go_to_highvalue_policy=GoToNearestThreat(model_path=None),
        change_region_subpolicy=ChangeRegions(model_path=None),
        evade_policy=EvadeDetection(model_path=None),
        teammate_manager=teammate_manager
    )

    print(f"\nTesting Local TSP Heuristic:")
    print(f"  Search radius: {search_radius} pixels")
    print(f"  Spatial coordination: {spatial_coord}")
    print(f"  Episodes to run: {num_episodes}")

    # Run test episodes
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

        obs = env.reset()[0]
        episode_reward = 0
        episode_steps = 0
        done = False

        # Track TSP behavior
        waypoints_calculated = 0
        waypoints_reached = 0
        last_waypoint_count = 0

        agent_path = []
        while not done:
            # Handle pygame events
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

            # Get action from the TSP policy directly
            action, _ = local_tsp_policy.act(obs, env=env.env, agent_id=0)

            # Track TSP state changes
            current_waypoint_count = len(local_tsp_policy.current_waypoints)
            if current_waypoint_count != last_waypoint_count:
                if current_waypoint_count > last_waypoint_count:
                    waypoints_calculated += 1
                    print(f"  Step {episode_steps}: TSP calculated {current_waypoint_count} waypoints")
                last_waypoint_count = current_waypoint_count

            if hasattr(local_tsp_policy, 'current_waypoint_index'):
                if (local_tsp_policy.current_waypoint_index > 0 and
                        local_tsp_policy.current_waypoint_index != getattr(local_tsp_policy, '_last_waypoint_index',
                                                                           0)):
                    waypoints_reached += 1
                    print(f"  Step {episode_steps}: Reached waypoint {local_tsp_policy.current_waypoint_index}")
                local_tsp_policy._last_waypoint_index = local_tsp_policy.current_waypoint_index

            # Step environment with the action
            if isinstance(action, tuple):
                action = action[0]
            print(f'action is {action}')
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            episode_steps += 1

            # Record agent position for path rendering
            agent_x = env.env.agents[env.env.aircraft_ids[0]].x
            agent_y = env.env.agents[env.env.aircraft_ids[0]].y
            agent_path.append((agent_x, -agent_y))

            # Render the environment
            env.render()

            # Render agent path
            if len(agent_path) > 1:
                # Convert world coordinates to screen coordinates
                screen_path = []
                for pos in agent_path:
                    screen_x = int(pos[0] + config['gameboard_size'][0] // 2)
                    screen_y = int(config['gameboard_size'][1] // 2 - pos[1])
                    screen_path.append((screen_x, screen_y))

                # Draw path as connected lines
                if len(screen_path) > 1:
                    pygame.draw.lines(window, (255, 100, 100), False, screen_path, 2)  # Red path, 2px thick


            # Add debug information to display
            debug_text = [
                f"Episode: {episode + 1}/{num_episodes}",
                f"Step: {episode_steps}",
                f"Reward: {episode_reward:.2f}",
                f"Action: {action}",
                f"TSP Waypoints: {len(local_tsp_policy.current_waypoints)}",
                f"Current Waypoint Index: {getattr(local_tsp_policy, 'current_waypoint_index', 0)}",
                f"Waypoints Calculated: {waypoints_calculated}",
                f"Waypoints Reached: {waypoints_reached}",
                f"Search Radius: {search_radius}px",
                f"Spatial Coord: {spatial_coord}"
            ]

            # Display debug info
            for i, text in enumerate(debug_text):
                color = (255, 255, 255)  # White text
                if i == 3:  # Action line
                    color = (255, 255, 0)  # Yellow for action
                elif i >= 4 and i <= 7:  # TSP info lines
                    color = (0, 255, 255)  # Cyan for TSP info

                rendered_text = font.render(text, True, color)
                window.blit(rendered_text, (10, 10 + i * 25))

            pygame.display.flip()
            clock.tick(tick_rate)

        # Episode summary
        targets_found = env.env.targets_identified
        threats_identified = env.env.num_threats_identified

        print(f"Episode {episode + 1} completed:")
        print(f"  Steps: {episode_steps}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Targets found: {targets_found}")
        print(f"  Threats identified: {threats_identified}")
        print(f"  TSP calculations: {waypoints_calculated}")
        print(f"  Waypoints reached: {waypoints_reached}")

        if hasattr(local_tsp_policy, 'fallback_policy'):
            print(f"  Used fallback policy: {getattr(local_tsp_policy.fallback_policy, '_action_count', 0)} times")

    env.close()
    pygame.quit()

    print(f"\nLocal TSP Heuristic test completed!")
    print(f"Configuration tested:")
    print(f"  - Search radius: {search_radius} pixels")
    print(f"  - Spatial coordination: {spatial_coord}")
    print(f"  - Episodes run: {num_episodes}")


if __name__ == "__main__":
    try:
        run_local_tsp_test()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        pygame.quit()
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback

        traceback.print_exc()
        pygame.quit()