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
from user_study.rl_data_logger import ExperimentDataLogger
from user_study.instructional_screens import ScreenManager, WorkloadSurveyScreen, TeammatePreferenceSurveyScreen, \
    InstructionSeriesManager
import webbrowser

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
            #self.custom_waypoint = np.array([game_x / map_half_size, game_y / map_half_size], dtype=np.float32)
            self.custom_waypoint = np.array([game_x, game_y], dtype=np.float32)
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


def create_rl_teammate(model_path, agent_name):
    """Create an RL teammate from a model file"""
    try:
        print(f"Loading {agent_name} model from: {model_path}")
        model = PPO.load(model_path)

        teammate = RLTeammatePolicy(
            model=model,
            env=None,
            local_search_policy=None, #subpolicies['local_search'],
            go_to_highvalue_policy=None, #subpolicies['go_to_threat'],
            change_region_subpolicy=None, #subpolicies['change_region'],
            use_collision_avoidance=False
        )

        teammate.name = agent_name
        print(f"Successfully loaded RL teammate: {teammate.name}")
        return teammate

    except Exception as e:
        print(f"Error loading {agent_name} model: {e}")


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
                     ):
    """Draw current status information"""
    subpolicy_names = ["Local Search", "Goto NW", "Go to Threat", "Hold", "Goto NE", "Goto SE", "Goto SW"]
    current_mode = "Custom Waypoint" if controller.use_custom_waypoint else subpolicy_names[
        controller.current_subpolicy]

    status_info = [
        f"Config: {current_config} ({config_index + 1}/{total_configs})",
        #f"Current Agent: {current_agent_name}",
        f"Step: {step_count}",
        f"Reward: {episode_reward:.2f}",
        f"Control Mode: {current_mode}",
    ]

    y_offset = 280
    for info in status_info:
        text_surface = font.render(info, True, (255, 255, 255))
        window.blit(text_surface, (1050, y_offset))
        y_offset += 25


def run_single_episode(env, human_controller, config, config_index, total_configs, agent_model, window, font,
                       clock, tick_rate, data_logger):
    """Run a single episode of the experiment"""
    print(f"\n{'=' * 50}")
    print(f"Starting Config: {config} ({config_index + 1}/{total_configs})")
    print(f"Agent: {agent_model}")
    print(f"{'=' * 50}")

    # Parse agent and level from config
    agent_letter = config[0]
    level_number = int(config[1:])
    data_logger.start_episode(config, agent_letter, level_number)

    obs = env.reset()[0]
    print(f'Obs: {obs} (shape {obs.shape}')
    episode_reward = 0
    step_count = 0
    done = False
    paused = False

    while not done:
        map_half_size = env.env.config['gameboard_size']
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

        # Get human action
        human_action = human_controller.get_current_action()
        #print(f'[Experiment loop] Human action is {human_action}')
        if human_controller.should_override_waypoint():
            custom_waypoint = human_controller.get_custom_waypoint()
            custom_waypoint = (int(custom_waypoint[0]), int(custom_waypoint[1]))
            #scaled_waypoint = (custom_waypoint[0]*map_half_size, custom_waypoint[1]*map_half_size)
            env.env.agents[env.env.aircraft_ids[1]].waypoint_override = custom_waypoint #tuple(custom_waypoint)
            print(f'[Experiment loop] Setting human waypoint to {custom_waypoint}')


        # Get agent action
        agent_action, _ = agent_model.predict(obs, deterministic=True)
        #print(f'[Experiment loop] Agent chose action {agent_action}')

        # Take step in environment
        obs, reward, terminated, truncated, info = env.step(agent_action)

        # Log timestep data
        data_logger.log_timestep(
            env=env,
            human_controller=human_controller,
            human_action=human_action,
            agent_action=agent_action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info
        )

        episode_reward += reward
        done = terminated or truncated
        step_count += 1

        # Render the environment
        env.render()

        # Draw additional UI elements
        draw_instructions(window, font)
        draw_status_info(window, font, config, config_index, total_configs, step_count, episode_reward, human_controller)

        # Update display
        pygame.display.flip()
        pygame.time.wait(50)
        clock.tick(tick_rate)

        # Print periodic status
        if step_count % 50 == 0:
            print(f"Step {step_count}: Reward = {episode_reward:.2f}, ")

    # End episode logging
    episode_summary = data_logger.end_episode(env, info)

    print(f"\nConfig {config} Complete!")
    print(f"Final Reward: {episode_reward:.2f}")
    print(f"Steps Taken: {step_count}")
    print(f"Targets Identified: {info.get('target_ids', 0)}")
    print(f"Detections: {info.get('detections', 0)}")

    return False, episode_reward, step_count  # False = don't quit experiment


def launch_survey_url(url, level_id: int, agent_type: str, subject_id: int = None):
    """Launch survey URL in default browser
    Args:
        level_id: Int, 0 to num_levels
        agent_type: Str ('fcp', 'self_play', 'strategy_diverse')
        subject_id: int

    """
    try:
        full_url = url + '?subject_id=' + str(subject_id) + '&scenario_number=' + str(level_id) + '&agent_type=' + str(agent_type)
        print(f"Opening survey URL: {full_url}")

        webbrowser.open_new_tab(full_url)
        return True
    except Exception as e:
        print(f"Error opening survey URL: {e}")
        return False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run MAISR user study experiment')
    parser.add_argument('subject_id', type=int, help='Subject ID (integer)')
    parser.add_argument('--start_level', type=int, default=0,
                        help='Starting level index (default: 0)')
    args = parser.parse_args()

    print(f"\n \n Subject ID: {args.subject_id}")
    print(f"Starting from level: {args.start_level}")

    # Configuration
    config_filename = 'configs/Monolith_R8H_july10.json'
    tick_rate = 20
    #survey_url = "https://gatech.co1.qualtrics.com/jfe/form/SV_egiLZSvblF8SVO6" # TODO

    # Define RL agent model paths - UPDATE THESE AS NEEDED
    agent_models = {
        'A': './user_study/saved_agents/test_agent_a.zip',  # Replace with actual path TODO
        'B': './user_study/saved_agents/test_agent_b.zip',  # Replace with actual path TODO
    }

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
    config['num_observed_targets'] = 4 # TODO TEMP
    print(f'LOADED CONFIG {config_filename}')

    # Initialize pygame
    pygame.display.init()
    pygame.font.init()
    clock = pygame.time.Clock()
    ctypes.windll.user32.SetProcessDPIAware()

    window_width, window_height = config['window_size'][0], config['window_size'][1]
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption(f"MAISR User Study - Subject {args.subject_id}")

    # Create font for instructions
    font = pygame.font.SysFont(None, 24)

    # Initialize subpolicies
    # subpolicies = {
    #     'local_search': LocalSearch(
    #         model_path=localsearch_model_path,
    #         norm_stats_filepath=localsearch_normstats_path if localsearch_model_path else None
    #     ),
    #     'change_region': ChangeRegions(model_path=None),
    #     'go_to_threat': GoToNearestThreat(model_path=None)
    # }

    # Store results
    experiment_results = []
    current_agents = {}

    data_logger = ExperimentDataLogger(args.subject_id)
    screen_manager = ScreenManager(window, clock)

    # Main experiment loop
    try:
        # Show instruction screens at the beginning
        print("Starting instruction screens...")
        instruction_manager = InstructionSeriesManager(
            window,
            clock,
            map_image_path="user_study/img/map_image.png",  # Update these paths
            sensor_image_path="user_study/img/sensor_image.png",  # to your actual
            hvt_image_path="user_study/img/threat_image.png"  # image files
        )

        instruction_result = instruction_manager.run_instruction_series()

        if instruction_result["action"] == "exit":
            print("User exited during instructions")
            return

        print("Instructions completed. Starting experiment...")

        for config_index, current_config in enumerate(config_list):
            agent_letter = current_config[0]  # 'A' or 'B'
            level_number = int(current_config[1:])  # Level number

            print(f"\nPreparing for config: {current_config}")
            print(f"Agent: {agent_letter}, Level: {level_number}")

            # Load the appropriate RL agent if not already loaded
            if agent_letter not in current_agents:
                current_agents[agent_letter] = PPO.load(agent_models[agent_letter])
                # current_agents[agent_letter] = create_rl_teammate(
                #     agent_models[agent_letter],
                #     #subpolicies,
                #     f"Agent_{agent_letter}"
                # )

            current_agent_name = current_agents[agent_letter]

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
                local_search_policy=None,#subpolicies['local_search'],
                go_to_highvalue_policy=None,#subpolicies['go_to_threat'],
                change_region_subpolicy=None,#subpolicies['change_region'],
                evade_policy=None,#EvadeDetection(model_path=None),
                teammate_policy=None,
                obs_noise_std=0.0
            )



            # Initialize human controller for this episode
            human_controller = HumanSubpolicyController(env)

            # Run the episode
            should_quit, episode_reward, step_count = run_single_episode(
                env, human_controller, current_config, config_index, len(config_list),
                current_agent_name, window, font, clock, tick_rate, data_logger
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

            # Launch survey URL after each episode
            print(f"\nEpisode {current_config} completed!")

            workload_survey_screen = WorkloadSurveyScreen(episode_config=current_config, window_width=window_width, window_height=window_height)
            workload_survey_result = screen_manager.show_screen(workload_survey_screen)
            #survey_launched = launch_survey_url(survey_url, current_config, args.subject_id)

            if workload_survey_result["action"] == "exit":
                print("Experiment terminated by user")
                break
            elif workload_survey_result["action"] == "continue":
                # Log the survey data
                survey_data = workload_survey_result.get("survey_data", {})
                print(f"Survey responses for {current_config}: {survey_data['responses']}")

                # Add survey data to your data logger
                if hasattr(data_logger, 'log_survey_data'):
                    data_logger.log_survey_data(survey_data)
                else:
                    # Fallback: save to file or print
                    print(f"Survey data: {survey_data}")

            # Show the teammate preference survey
            if config_index % 2 == 0:
                teammate_compare_survey = TeammatePreferenceSurveyScreen(window_width, window_height)
                teammate_compare_result = screen_manager.show_screen(teammate_compare_survey)

                if teammate_compare_result["action"] == "continue":
                    survey_data = teammate_compare_result["survey_data"]
                    data_logger.log_teammate_survey_data(survey_data)
                elif teammate_compare_result["action"] == "exit":
                    # Handle exit
                    pass

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
        data_logger.save_session_data()
        session_summary = data_logger.get_session_summary()

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

        # Print session summary from data logger
        if session_summary:
            print(f"\nSession Statistics:")
            print(f"  Total episodes: {session_summary['total_episodes']}")
            print(f"  Total session duration: {session_summary['total_session_duration']:.2f} seconds")
            print(f"  Average episode duration: {session_summary['average_episode_duration']:.2f} seconds")
            print(f"  Success rate: {session_summary['success_rate']:.2%}")
            print(f"  Total targets identified: {session_summary['total_targets_identified']}")
            print(f"  Total threats identified: {session_summary['total_threats_identified']}")

            if 'agent_performance' in session_summary:
                print(f"\nAgent Performance:")
                for agent, performance in session_summary['agent_performance'].items():
                    print(f"  Agent {agent}:")
                    print(f"    Episodes: {performance['episodes']}")
                    print(f"    Average reward: {performance['average_reward']:.2f}")
                    print(f"    Success rate: {performance['success_rate']:.2%}")
                    print(f"    Average targets identified: {performance['average_targets_identified']:.1f}")

        print(f"\nData saved to: {data_logger.output_dir}/subject_{args.subject_id}/")
        pygame.quit()


if __name__ == "__main__":
    main()