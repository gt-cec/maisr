import ctypes
import pygame
import numpy as np
from stable_baselines3 import PPO
from env_multi_new import MAISREnvVec
from training_wrappers.localsearch_training_wrapper import MaisrLocalSearchWrapper
from utility.data_logging import load_env_config
from utility.league_management import (GenericTeammatePolicy, LocalSearch,
                                       ChangeRegions, GoToNearestThreat, EvadeDetection,
                                       RLTeammatePolicy)


class HumanSubpolicyController:
    """Handles human input for subpolicy selection via keyboard and mouse clicks"""

    def __init__(self, env):
        self.env = env
        self.current_subpolicy = 0  # Default to local search
        self.last_click_time = 0
        self.click_cooldown = 200  # milliseconds between clicks
        self.custom_waypoint = None
        self.use_custom_waypoint = False

    def handle_click(self, mouse_pos):
        """Convert mouse click to custom waypoint and switch to custom mode"""
        current_time = pygame.time.get_ticks()

        # Prevent rapid clicking
        if current_time - self.last_click_time < self.click_cooldown:
            return None

        self.last_click_time = current_time

        # Get click position relative to game board
        click_x, click_y = mouse_pos
        gameboard_size = self.env.env.config['gameboard_size']

        # Check if click is within gameboard bounds
        if 0 <= click_x <= gameboard_size and 0 <= click_y <= gameboard_size:
            # Convert from screen coordinates to game coordinates
            map_half_size = gameboard_size / 2
            game_x = click_x - map_half_size
            game_y = click_y - map_half_size

            # Store the waypoint for custom control
            self.custom_waypoint = np.array([game_x/map_half_size, game_y/map_half_size], dtype=np.float32)
            print(f'Custom waypoint = {self.custom_waypoint}')
            self.env.human_custom_waypoint = self.custom_waypoint
            self.use_custom_waypoint = True

            print(f"Human clicked at screen ({click_x}, {click_y}) -> game waypoint ({game_x:.1f}, {game_y:.1f})")
            print("Switched to custom waypoint mode")
            return True

        return False

    def handle_keypress(self, key):
        """Handle keyboard input for subpolicy selection"""
        key_to_subpolicy = {
            pygame.K_1: 0,  # Local search
            pygame.K_2: 1,  # Goto NW
            pygame.K_3: 2,  # Go to threat
            pygame.K_4: 3,  # Hold
            pygame.K_5: 4,  # Goto NE
            pygame.K_6: 5,  # Goto SE
            pygame.K_7: 6,  # Goto SW
            pygame.K_8: 7,  # Custom waypoint mode
        }

        if key in key_to_subpolicy:
            self.current_subpolicy = key_to_subpolicy[key]
            if self.current_subpolicy == 7:
                self.use_custom_waypoint = True
            else:
                self.use_custom_waypoint = False

            subpolicy_names = ["Local Search", "Goto NW", "Go to Threat", "Hold",
                               "Goto NE", "Goto SE", "Goto SW", "Custom Waypoint"]
            print(f"Human selected subpolicy: {subpolicy_names[self.current_subpolicy]}")
            return True

        return False

    def get_current_action(self):
        """Get the current action for the human player"""
        return self.current_subpolicy  # Always return the current subpolicy

    def should_override_waypoint(self):
        """Check if we should override the agent's waypoint with custom waypoint"""
        return self.use_custom_waypoint and self.custom_waypoint is not None

    def get_custom_waypoint(self):
        """Get the custom waypoint coordinates"""
        return self.custom_waypoint


def create_pretrained_teammate(model_path, subpolicies):
    """Create a pretrained RL teammate from a model file"""
    try:
        print(f"Loading pretrained teammate model from: {model_path}")
        model = PPO.load(model_path)

        teammate = RLTeammatePolicy(
            model=model,
            env=None,
            local_search_policy=subpolicies['local_search'],
            go_to_highvalue_policy=subpolicies['go_to_threat'],
            change_region_subpolicy=subpolicies['change_region'],
            use_collision_avoidance=False
        )

        teammate.name = f"Pretrained_Human_Session"
        print(f"Successfully loaded pretrained teammate: {teammate.name}")
        return teammate

    except Exception as e:
        print(f"Error loading pretrained teammate: {e}")
        print("Falling back to heuristic teammate")

        # Fallback to heuristic teammate
        from utility.league_management import HeuristicAgent
        heuristic_agent = HeuristicAgent(
            mode_selector="heuristic",
            risk_tolerance="medium",
            spatial_coord="some"
        )

        teammate = GenericTeammatePolicy(
            env=None,
            local_search_policy=subpolicies['local_search'],
            go_to_highvalue_policy=subpolicies['go_to_threat'],
            change_region_subpolicy=subpolicies['change_region'],
            mode_selector_agent=heuristic_agent,
            use_collision_avoidance=False
        )

        teammate.name = "Heuristic_Fallback"
        return teammate


def draw_instructions(window, font):
    """Draw instructions for the human player"""
    instructions = [
        "HUMAN CONTROL INSTRUCTIONS:",
        "• Keys 1-7: Select subpolicy",
        "  1 = Local Search, 2 = Goto NW",
        "  3 = Go to Threat, 4 = Hold",
        "  5 = Goto NE, 6 = Goto SE, 7 = Goto SW",
        "• Click map: Direct waypoint control",
        "• Green targets = identified, Orange = unknown",
        "• Avoid gold threat circles",
        "• ESC = quit, SPACE = pause"
    ]

    y_offset = 20
    for instruction in instructions:
        color = (255, 255, 255) if instruction.startswith("HUMAN") else (200, 200, 200)
        text_surface = font.render(instruction, True, color)
        window.blit(text_surface, (1050, y_offset))
        y_offset += 25


def draw_status_info(window, font, episode, step_count, episode_reward, controller):
    """Draw current status information"""
    subpolicy_names = ["Local Search", "Goto NW", "Go to Threat", "Hold", "Goto NE", "Goto SE", "Goto SW"]
    current_mode = "Custom Waypoint" if controller.use_custom_waypoint else subpolicy_names[
        controller.current_subpolicy]

    status_info = [
        f"Episode: {episode + 1}",
        f"Step: {step_count}",
        f"Reward: {episode_reward:.2f}",
        f"Control Mode: {current_mode}",
        f"Custom Waypoint: {controller.custom_waypoint if controller.custom_waypoint is not None else 'None'}"
    ]

    y_offset = 250
    for info in status_info:
        text_surface = font.render(info, True, (255, 255, 255))
        window.blit(text_surface, (1050, y_offset))
        y_offset += 25


def main():
    # Configuration
    config_filename = 'configs/Monolith_R8H_july10.json'
    num_episodes = 5
    tick_rate = 120

    # Model paths - UPDATE THESE PATHS AS NEEDED
    pretrained_teammate_model_path = './trained_models/modeselector_poc1_2ship_0.0005lr_1024bs_0623_1424_16envs/maisr_checkpoint_modeselector_poc1_2ship_0.0005lr_1024bs_0623_1424_16envs_149760_steps.zip'
    localsearch_model_path = None  # Use heuristic
    localsearch_normstats_path = 'trained_models/local_search_2000000.0timesteps_0.1threatpenalty_0615_1541_6envslocal_search_norm_stats.npy'

    # Load configuration
    config = load_env_config(config_filename)
    config['tick_rate'] = tick_rate
    print(f'LOADED CONFIG {config_filename}')

    # Initialize pygame
    pygame.display.init()
    pygame.font.init()
    clock = pygame.time.Clock()
    ctypes.windll.user32.SetProcessDPIAware()

    window_width, window_height = config['window_size'][0]+500, config['window_size'][1]
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("MAISR Human-AI Cooperation")

    # Create font for instructions
    font = pygame.font.SysFont(None, 24)

    # Initialize subpolicies
    subpolicies = {
        'local_search': LocalSearch(
            model_path=localsearch_model_path,
            norm_stats_filepath=localsearch_normstats_path if localsearch_model_path else None
        ),
        'change_region': ChangeRegions(model_path=None),
        'go_to_threat': GoToNearestThreat(model_path=None)
    }

    # Create base environment
    base_env = MAISREnvVec(
        config=config,
        clock=clock,
        window=window,
        render_mode='human',
        run_name='human_ai_coop',
        tag='human_test_0',
    )

    # Create pretrained teammate
    pretrained_teammate = create_pretrained_teammate(pretrained_teammate_model_path, subpolicies)

    # Create wrapped environment
    env = MaisrLocalSearchWrapper(
        base_env,
        local_search_policy=subpolicies['local_search'],
        go_to_highvalue_policy=subpolicies['go_to_threat'],
        change_region_subpolicy=subpolicies['change_region'],
        evade_policy=EvadeDetection(model_path=None),
        teammate_policy=pretrained_teammate  # Use single teammate instead of manager
    )

    # Initialize human controller
    human_controller = HumanSubpolicyController(env)

    # Game loop
    episode_rewards = []

    try:
        for episode in range(num_episodes):
            print(f"\n{'=' * 50}")
            print(f"Starting Episode {episode + 1}/{num_episodes}")
            print(f"{'=' * 50}")

            obs = env.reset()[0]
            episode_reward = 0
            step_count = 0
            done = False
            paused = False

            while not done:
                current_time = pygame.time.get_ticks()

                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            done = True
                            break
                        elif event.key == pygame.K_SPACE:
                            paused = not paused
                            print("Game paused" if paused else "Game resumed")
                        else:
                            # Handle subpolicy selection
                            human_controller.handle_keypress(event.key)
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:  # Left click
                            human_controller.handle_click(event.pos)

                if done:
                    break

                if paused:
                    pygame.time.wait(50)
                    continue

                # Get human action (subpolicy selection)
                human_action = human_controller.get_current_action()

                # Override waypoint if human is in custom waypoint mode
                if human_controller.should_override_waypoint():
                    custom_waypoint = human_controller.get_custom_waypoint()
                    # Set the waypoint override for agent 0 before stepping
                    env.env.agents[env.env.aircraft_ids[0]].waypoint_override = tuple(custom_waypoint)

                # Take step in environment
                obs, reward, terminated, truncated, info = env.step(human_action)

                episode_reward += reward
                done = terminated or truncated
                step_count += 1

                # Render the environment
                env.render()

                # Draw additional UI elements
                draw_instructions(window, font)
                draw_status_info(window, font, episode, step_count, episode_reward,
                                 human_controller)

                # Draw subpolicy indicators
                agent0_subpolicy_id, agent0_subpolicy_name = env.get_current_subpolicy_info()
                agent1_subpolicy_id, agent1_subpolicy_name = env.get_teammate_subpolicy_info()

                env.env.render_subpolicy_indicators(
                    agent0_subpolicy_id, agent0_subpolicy_name,
                    agent1_subpolicy_id, agent1_subpolicy_name
                )

                # Update display
                pygame.display.flip()
                pygame.time.wait(50)
                clock.tick(tick_rate)

                # Print periodic status
                if step_count % 50 == 0:
                    print(f"Step {step_count}: Reward = {episode_reward:.2f}, "
                          f"Human subpolicy = {agent0_subpolicy_name}, "
                          f"AI subpolicy = {agent1_subpolicy_name}")

            episode_rewards.append(episode_reward)

            print(f"\nEpisode {episode + 1} Complete!")
            print(f"Final Reward: {episode_reward:.2f}")
            print(f"Steps Taken: {step_count}")
            print(f"Targets Identified: {info.get('target_ids', 0)}")
            print(f"Detections: {info.get('detections', 0)}")

            # Brief pause between episodes
            if episode < num_episodes - 1:
                print("Starting next episode in 1 seconds...")
                pygame.time.wait(1000)

    except KeyboardInterrupt:
        print("\nGame interrupted by user")

    finally:
        # Print summary
        print(f"\n{'=' * 50}")
        print("GAME SUMMARY")
        print(f"{'=' * 50}")
        print(f"Episodes Completed: {len(episode_rewards)}")
        if episode_rewards:
            print(f"Average Reward: {np.mean(episode_rewards):.2f}")
            print(f"Best Episode: {np.max(episode_rewards):.2f}")
            print(f"Episode Rewards: {[f'{r:.2f}' for r in episode_rewards]}")

        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()