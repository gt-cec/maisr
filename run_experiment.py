import argparse
import ctypes
import pygame
import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from env_multi_new import MAISREnvVec
from training_wrappers.localsearch_training_wrapper import MaisrLocalSearchWrapper
from training_wrappers.modeselector_training_wrapper import MaisrModeSelectorWrapper
from utility.data_logging import load_env_config
from policies.league_management import (GenericTeammatePolicy, SubPolicy, LocalSearch,
                                        ChangeRegions, GoToNearestThreat, EvadeDetection,
                                        TeammateManager, RLTeammatePolicy)


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
            self.custom_waypoint = np.array([game_x / map_half_size, game_y / map_half_size], dtype=np.float32)
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
        return self.current_subpolicy

    def should_override_waypoint(self):
        """Check if we should override the agent's waypoint with custom waypoint"""
        return self.use_custom_waypoint and self.custom_waypoint is not None

    def get_custom_waypoint(self):
        """Get the custom waypoint coordinates"""
        return self.custom_waypoint


def create_rl_teammate(model_path, subpolicies, agent_name):
    """Create an RL teammate from a model file"""
    try:
        print(f"Loading {agent_name} model from: {model_path}")
        model = PPO.load(model_path)

        teammate = RLTeammatePolicy(
            model=model,
            env=None,
            local_search_policy=subpolicies['local_search'],
            go_to_highvalue_policy=subpolicies['go_to_threat'],
            change_region_subpolicy=subpolicies['change_region'],
            use_collision_avoidance=False
        )

        teammate.name = agent_name
        print(f"Successfully loaded RL teammate: {teammate.name}")
        return teammate

    except Exception as e:
        print(f"Error loading {agent_name} model: {e}")
        print("Falling back to heuristic teammate")

        # Fallback to heuristic teammate
        from policies.league_management import HeuristicAgent
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

        teammate.name = f"Heuristic_Fallback_{agent_name}"
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


def draw_status_info(window, font, current_config, config_index, total_configs, step_count, episode_reward, controller,
                     current_agent_name):
    """Draw current status information"""
    subpolicy_names = ["Local Search", "Goto NW", "Go to Threat", "Hold", "Goto NE", "Goto SE", "Goto SW"]
    current_mode = "Custom Waypoint" if controller.use_custom_waypoint else subpolicy_names[
        controller.current_subpolicy]

    status_info = [
        f"Config: {current_config} ({config_index + 1}/{total_configs})",
        f"Current Agent: {current_agent_name}",
        f"Step: {step_count}",
        f"Reward: {episode_reward:.2f}",
        f"Control Mode: {current_mode}",
    ]

    y_offset = 280
    for info in status_info:
        text_surface = font.render(info, True, (255, 255, 255))
        window.blit(text_surface, (1050, y_offset))
        y_offset += 25


def run_single_episode(env, human_controller, config, config_index, total_configs, current_agent_name, window, font,
                       clock, tick_rate):
    """Run a single episode of the experiment"""
    print(f"\n{'=' * 50}")
    print(f"Starting Config: {config} ({config_index + 1}/{total_configs})")
    print(f"Agent: {current_agent_name}")
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
                return True, episode_reward, step_count  # Signal to quit experiment
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return True, episode_reward, step_count  # Signal to quit experiment
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("Game paused" if paused else "Game resumed")
                else:
                    # Handle subpolicy selection
                    human_controller.handle_keypress(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    human_controller.handle_click(event.pos)

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
        draw_status_info(window, font, config, config_index, total_configs, step_count,
                         episode_reward, human_controller, current_agent_name)

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

    print(f"\nConfig {config} Complete!")
    print(f"Final Reward: {episode_reward:.2f}")
    print(f"Steps Taken: {step_count}")
    print(f"Targets Identified: {info.get('target_ids', 0)}")
    print(f"Detections: {info.get('detections', 0)}")

    return False, episode_reward, step_count  # False = don't quit experiment


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run MAISR user study experiment')
    parser.add_argument('subject_id', type=int, help='Subject ID (integer)')
    parser.add_argument('--start_level', type=int, default=0,
                        help='Starting level index (default: 0)')
    args = parser.parse_args()

    print(f"Subject ID: {args.subject_id}")
    print(f"Starting from level: {args.start_level}")

    # Configuration
    config_filename = 'configs/Monolith_R8H_july10.json'
    tick_rate = 60

    # Define RL agent model paths - UPDATE THESE AS NEEDED
    agent_models = {
        'A': './trained_models/agent_A_model.zip',  # Replace with actual path
        'B': './trained_models/agent_B_model.zip',  # Replace with actual path
    }

    # Define subpolicy model paths
    localsearch_model_path = None  # Use heuristic
    localsearch_normstats_path = 'trained_models/local_search_2000000.0timesteps_0.1threatpenalty_0615_1541_6envslocal_search_norm_stats.npy'

    # Create experiment configuration list (agent + level combinations)
    # 7 rounds each with 2 agents = 14 total configurations
    config_list = []
    for agent in ['A', 'B']:
        for level in range(1, 8):  # Levels 1-7
            config_list.append(f'{agent}{level}')

    # Shuffle the configuration list for randomized order
    random.shuffle(config_list)
    print(f"Randomized configuration order: {config_list}")

    # If start_level is specified, start from that index
    if args.start_level > 0:
        if args.start_level >= len(config_list):
            print(f"Error: start_level {args.start_level} is >= total configs {len(config_list)}")
            return
        config_list = config_list[args.start_level:]
        print(f"Starting from level {args.start_level}: {config_list}")

    # Load configuration
    config = load_env_config(config_filename)
    config['tick_rate'] = tick_rate
    print(f'LOADED CONFIG {config_filename}')

    # Initialize pygame
    pygame.display.init()
    pygame.font.init()
    clock = pygame.time.Clock()
    ctypes.windll.user32.SetProcessDPIAware()

    window_width, window_height = config['window_size'][0] + 500, config['window_size'][1]
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption(f"MAISR User Study - Subject {args.subject_id}")

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

    # Store results
    experiment_results = []
    current_agents = {}

    try:
        # Main experiment loop
        for config_index, current_config in enumerate(config_list):
            # Parse agent and level from config string
            agent_letter = current_config[0]  # 'A' or 'B'
            level_number = int(current_config[1:])  # Level number

            print(f"\nPreparing for config: {current_config}")
            print(f"Agent: {agent_letter}, Level: {level_number}")

            # Load the appropriate RL agent if not already loaded
            if agent_letter not in current_agents:
                current_agents[agent_letter] = create_rl_teammate(
                    agent_models[agent_letter],
                    subpolicies,
                    f"Agent_{agent_letter}"
                )

            current_agent = current_agents[agent_letter]

            # Create base environment with the specific level
            base_env = MAISREnvVec(
                config=config,
                clock=clock,
                window=window,
                render_mode='human',
                run_name=f'user_study_subject_{args.subject_id}',
                tag=f'subject_{args.subject_id}_config_{current_config}',
            )

            # Set the specific level for this episode
            base_env.config['force_specific_level'] = level_number - 1  # Convert to 0-indexed

            # Create wrapped environment with current agent
            env = MaisrLocalSearchWrapper(
                base_env,
                local_search_policy=subpolicies['local_search'],
                go_to_highvalue_policy=subpolicies['go_to_threat'],
                change_region_subpolicy=subpolicies['change_region'],
                evade_policy=EvadeDetection(model_path=None),
                teammate_policy=current_agent
            )

            # Initialize human controller for this episode
            human_controller = HumanSubpolicyController(env)

            # Run the episode
            should_quit, episode_reward, step_count = run_single_episode(
                env, human_controller, current_config, config_index, len(config_list),
                current_agent.name, window, font, clock, tick_rate
            )

            # Store results
            result = {
                'config': current_config,
                'agent': agent_letter,
                'level': level_number,
                'reward': episode_reward,
                'steps': step_count,
                'config_index': config_index + args.start_level
            }
            experiment_results.append(result)

            # Clean up environment
            env.close()

            # Check if user wants to quit
            if should_quit:
                print("Experiment terminated by user")
                break

            # Brief pause between episodes (unless it's the last one)
            if config_index < len(config_list) - 1:
                print("Next episode starting in 2 seconds...")
                pygame.time.wait(2000)

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")

    finally:
        # Print experiment summary
        print(f"\n{'=' * 60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'=' * 60}")
        print(f"Subject ID: {args.subject_id}")
        print(f"Completed configurations: {len(experiment_results)}")

        if experiment_results:
            total_reward = sum(r['reward'] for r in experiment_results)
            avg_reward = total_reward / len(experiment_results)
            print(f"Total reward: {total_reward:.2f}")
            print(f"Average reward: {avg_reward:.2f}")

            print("\nDetailed results:")
            for result in experiment_results:
                print(f"  {result['config']}: Reward = {result['reward']:.2f}, "
                      f"Steps = {result['steps']}")

        pygame.quit()


if __name__ == "__main__":
    main()