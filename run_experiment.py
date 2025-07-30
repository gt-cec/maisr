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
from utility.config import subject_id
from utility.data_logging import load_env_config
from utility.league_management import (GenericTeammatePolicy, SubPolicy, LocalSearch, ChangeRegions, GoToNearestThreat, EvadeDetection, TeammateManager, RLTeammatePolicy)
from user_study.rl_data_logger import ExperimentDataLogger
from user_study.instructional_screens import ScreenManager, WorkloadSurveyScreen, TeammatePreferenceSurveyScreen, \
    InstructionSeriesManager, FinalSummaryScreen, AfterPracticeScreen
from PIL import Image
from io import BytesIO
import sockets
import webbrowser
from stable_baselines3.common.vec_env import VecNormalize

#sockets.pyg = pygame

window = None

def load_vecnormalize_wrapper(vecnorm_path, env):
    """Load saved VecNormalize wrapper with stats from training and apply it to the new environment."""
    print(f"Loading VecNormalize stats from: {vecnorm_path}")

    vec_normalize = VecNormalize.load(vecnorm_path, venv=env)
    vec_normalize.training = False  # Disable further normalization updates
    vec_normalize.norm_reward = False
    return vec_normalize


def make_wrapped_env(config, clock, window, agent_appearance, subject_id, run_name='no_name'):
    def _init():
        base_env = MAISREnvVec(
            config=config,
            clock=clock,
            window=window,
            render_mode='human',
            run_name=f'user_study_subj{subject_id}',
            tag=f'userstudy_0',
            agent_appearance=agent_appearance,
            running_experiment=True
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
        gameboard_size = self.env.envs[0].env.config['gameboard_size']

        # Check if click is within gameboard bounds
        if 0 <= click_x <= gameboard_size and 0 <= click_y <= gameboard_size:
            # Convert from screen coordinates to game coordinates
            map_half_size = gameboard_size / 2
            game_x = click_x - map_half_size
            game_y = click_y - map_half_size

            # Store the waypoint for custom control
            #self.custom_waypoint = np.array([game_x / map_half_size, game_y / map_half_size], dtype=np.float32)
            self.custom_waypoint = np.array([game_x, game_y], dtype=np.float32)
            #print(f'Custom waypoint = {self.custom_waypoint}')
            self.env.human_custom_waypoint = (float(self.custom_waypoint[0]),float(self.custom_waypoint[1]))
            self.use_custom_waypoint = True

            #print(f"Human clicked at screen ({click_x}, {click_y}) -> game waypoint ({game_x:.1f}, {game_y:.1f})")
            #print("Switched to custom waypoint mode")
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


# def create_rl_teammate(model_path, agent_name):
#     """Create an RL teammate from a model file"""
#     try:
#         print(f"Loading {agent_name} model from: {model_path}")
#         model = PPO.load(model_path)
#
#         teammate = RLTeammatePolicy(
#             model=model,
#             env=None,
#             local_search_policy=None, #subpolicies['local_search'],
#             go_to_highvalue_policy=None, #subpolicies['go_to_threat'],
#             change_region_subpolicy=None, #subpolicies['change_region'],
#             use_collision_avoidance=False
#         )
#
#         teammate.name = agent_name
#         print(f"Successfully loaded RL teammate: {teammate.name}")
#         return teammate
#
#     except Exception as e:
#         print(f"Error loading {agent_name} model: {e}")

def draw_progress_bar(window, font, current_index, total_configs):
    """Draw a segmented progress bar with labels for each segment."""
    progress_bar_y_start = 1000
    progress_bar_height = 50
    progress_bar_margin = 10
    segment_width = (window.get_width() - 2 * progress_bar_margin) // total_configs
    segment_height = progress_bar_height - 2 * progress_bar_margin

    for i in range(total_configs):
        x = progress_bar_margin + i * segment_width
        y = progress_bar_y_start + progress_bar_margin

        # Highlight current segment in blue, completed ones in green, others in gray
        if i == current_index:
            color = (0, 128, 255)  # Blue for current
        elif i < current_index:
            color = (0, 255, 0)    # Green for completed
        else:
            color = (100, 100, 100)  # Gray for upcoming

        pygame.draw.rect(window, color, pygame.Rect(x, y, segment_width - 2, segment_height))

        # Label: "PRACTICE" for first, numbers for rest
        label = "PRAC" if i == 0 else str(i)
        label_surface = font.render(label, True, (0, 0, 0))
        label_rect = label_surface.get_rect(center=(x + segment_width // 2, y + segment_height // 2))
        window.blit(label_surface, label_rect)




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


def draw_status_info(window, font, current_config, config_index, total_configs, step_count, episode_reward, controller):
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


def draw_bottom_bar_info(window, font, threats_identified, targets_identified, detections, step_count, max_steps):
    """Draw bottom bar with score and countdown timer"""
    regular_targets = targets_identified
    high_value_targets = threats_identified

    # Compute score: 5 × (# of targets) + 30 × abs(2 - # of threats)
    score = 5 * regular_targets - 30 * abs(2 - high_value_targets)

    # Countdown timer: 971 steps ≈ 45 seconds
    total_seconds = 40
    time_left = max(0, total_seconds - int(step_count * (total_seconds / 971)))

    bottom_texts = [
        f"Regular: {regular_targets}/15",
        f"High-value: {high_value_targets}/4",
        f"Score: {score}",
        f"Time left: {time_left}s"
    ]

    x_start = 50
    y_pos = 1060  # Just above the progress bar
    spacing = 1000 / 4

    for i, text in enumerate(bottom_texts):
        text_surface = font.render(text, True, (0, 0, 0))
        window.blit(text_surface, (x_start + i * spacing, y_pos))


# def draw_bottom_bar_info(window, font, threats_identified, targets_identified, detections, step_count, max_steps):
#     """Draw regular target, high-value target, step count, and detections"""
#     regular_targets = targets_identified
#     high_value_targets = threats_identified
#     detections = detections
#
#     # Set total counts
#     total_regular_targets = 15
#     total_high_value_targets = 4
#
#     bottom_texts = [
#         f"Regular targets: {regular_targets}/{total_regular_targets}",
#         f"High-value targets: {high_value_targets}/{total_high_value_targets}",
#         f"Steps: {step_count}/{round(max_steps/35, 0)}",
#         f"Detections: {detections}"
#     ]
#
#     x_start = 50
#     y_pos = 1060  # Just above the progress bar
#     spacing = 1000/4
#
#     for i, text in enumerate(bottom_texts):
#         text_surface = font.render(text, True, (0, 0, 0))
#         window.blit(text_surface, (x_start + i * spacing, y_pos))


def run_single_episode(env, human_controller, config, config_index, total_configs, agent_model, window, font, clock, tick_rate, data_logger, time_factor, agent_letter):
    """Run a single episode of the experiment"""
    print(f"\n{'=' * 50}")
    print(f"Starting Config: {config} ({config_index + 1}/{total_configs})")
    print(f"Agent: {agent_model}")
    print(f"{'=' * 50}")

    sockets.human_controller = human_controller

    base_env = env.envs[0].env

    if agent_letter == 'S':
        base_env.agents[base_env.aircraft_ids[0]].is_visible = False

    # Parse agent and level from config
    agent_letter = config[0]
    level_number = int(config[1:])
    data_logger.start_episode(config, agent_letter, level_number)

    # Agent action rate (to speed up processing)
    last_agent_action = None

    #obs = env.reset()[0]
    obs = env.reset()
    #print(f'Obs: {obs} (shape {obs.shape}')
    episode_reward = 0
    step_count = 0
    done = False
    paused = False

    # Draw static labels once
    draw_bottom_bar_info(window, font, base_env.num_threats_identified, base_env.targets_identified, base_env.detections, 0, base_env.config['max_steps'])

    while not done:
        map_half_size = env.envs[0].env.config['gameboard_size']
        current_time = pygame.time.get_ticks()
        skip_round = False

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
                elif event.key == pygame.K_RETURN:
                    skip_round = True
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
        if human_controller.should_override_waypoint():
            custom_waypoint = human_controller.get_custom_waypoint()
            custom_waypoint = (int(custom_waypoint[0]), int(custom_waypoint[1]))
            base_env.agents[base_env.aircraft_ids[1]].waypoint_override = custom_waypoint

        # Get agent action
        #agent_action, _ = agent_model.predict(obs, deterministic=True)
        if last_agent_action is None or step_count % time_factor == 0:
            if agent_letter == 'S':
                agent_action = 8
            else:
                agent_action, _ = agent_model.predict(obs, deterministic=True)
            #print(f'Agent chose action {agent_action}')
        else:
            agent_action = last_agent_action

        last_agent_action = agent_action

        # Take step in environment
        obses, rewards, dones, infos = env.step([agent_action])
        obs = obses[0]
        reward = rewards[0]
        info = infos[0]
        done = dones[0] or (np.sum(base_env.threat_identified) >= 2.0 and base_env.targets_identified >= 15) or skip_round # TODO testing

        final_target_ids = base_env.targets_identified
        final_threat_ids = base_env.num_threats_identified

        # Log timestep data
        data_logger.log_timestep(
            env=env,
            human_controller=human_controller,
            human_action=human_action,
            agent_action=agent_action,
            reward=reward,
            terminated=done,
            truncated=False,
            info=info
        )

        episode_reward += reward
        step_count += 1

        # Render the environment
        env.render()
        draw_status_info(window, font, config, config_index, total_configs, step_count, episode_reward, human_controller)
        draw_progress_bar(window, font, config_index, total_configs)
        draw_bottom_bar_info(window, font, base_env.num_threats_identified, base_env.targets_identified,base_env.detections, step_count, base_env.config['max_steps'])

        # Update display
        pygame.display.flip()
        if step_count % 1 == 0:
            sockets.send_frame(window)
        clock.tick(tick_rate)

        # Print periodic status
        #if step_count % 50 == 0:
            #print(f"Step {step_count}: Reward = {episode_reward:.2f}, ")

    # End episode logging
    episode_summary = data_logger.end_episode(env, info, int(info.get('target_ids', -1)), int(info.get('threat_ids', -1)))

    print(f"\nConfig {config} Complete!")
    print(f"Final Reward: {episode_reward:.2f}")
    print(f"Steps Taken: {step_count}")
    print(f"Targets Identified: {info.get('target_ids', -1)}")
    print(f"Detections: {info.get('detections', -1)}")

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


def main(subject_id=None, start_level=None, skip_instructions=None,collect_solo_trajectories=False):

    print(f"\n \n Subject ID: {subject_id}")
    print(f"Starting from level: {start_level}")

    # Configuration
    config_filename = 'configs/Monolith_R8H_july10.json'
    tick_rate = 45
    time_factor = 20  # 20
    config = load_env_config(config_filename)

    config['tick_rate'] = tick_rate
    config['game_speed'] /= time_factor
    config['max_steps'] *= (1700 / 1500) * time_factor
    config['use_stuck_detection'] = False
    config['prob_detect'] = 0.0003
    print(f'LOADED CONFIG {config_filename}')

    if subject_id == 90:
        agent_a_name = 'index_selfplay'#strategy_trained'#'selfplay_trained_jul18'
        agent_b_name = 'index_selfplay'#'strategy_trained_jul18'
        config['action_type'] = 'target_index'

    elif subject_id == 91:
        agent_a_name = 'index_strategy'  # strategy_trained'#'selfplay_trained_jul18'
        agent_b_name = 'index_strategy'  # 'strategy_trained_jul18'
        config['action_type'] = 'target_index'

    elif subject_id == 92:
        agent_a_name = 'strategy_trained'  # '#'selfplay_trained_jul18'
        agent_b_name = 'strategy_trained'  # 'strategy_trained_jul18'

    elif subject_id == 93:
        agent_a_name = 'selfplay_seed77'  # strategy_trained'#'selfplay_trained_jul18'
        agent_b_name = 'selfplay_seed77'  # 'strategy_trained_jul18'

    else:
        agent_a_name = 'selfplay_seed77'  # strategy_trained'#'selfplay_trained_jul18'
        agent_b_name = 'selfplay_seed77'  # 'strategy_trained_jul18'


    # Define RL agent model paths
    agent_models = {
        'A': f'./user_study/saved_agents/{agent_a_name}_model.zip',
        'B': f'./user_study/saved_agents/{agent_b_name}_model.zip',
        'S': f'./user_study/saved_agents/{agent_b_name}_model.zip'
    }

    vecnorm_paths = {
        'A': f'./user_study/saved_agents/{agent_a_name}_vecnormalize.pkl',
        'B': f'./user_study/saved_agents/{agent_b_name}_vecnormalize.pkl',
        'S': f'./user_study/saved_agents/{agent_b_name}_vecnormalize.pkl'
    }

    levels = list(range(1, 8))
    random.shuffle(levels)
    config_list = [f'{agent}{level}' for level in levels for agent in ('A', 'B')]

    practice_level = levels[0]
    practice_config = [f'C{practice_level}']

    full_config_list = practice_config + config_list

    print(f"Randomized configuration order: {full_config_list}")

    # If start_level is specified, start from that index
    if start_level > 0:
        if start_level >= len(full_config_list):
            print(f"Error: start_level {start_level} is >= total configs {len(full_config_list)}")
            return
        full_config_list = full_config_list[start_level:]
        print(f"Starting from level {start_level}: {full_config_list}")

    if collect_solo_trajectories:
        solo_configs = ['S1', 'S2','S3','S4','S5','S6','S7']
        full_config_list.extend(solo_configs)

    # Initialize pygame
    # if using windows, set DPI awareness to avoid scaling issues
    if hasattr(ctypes, 'windll') and hasattr(ctypes.windll, 'user32'):
        ctypes.windll.user32.SetProcessDPIAware()
    else:
        print("Not running on Windows, skipping DPI awareness setting")
    pygame.display.init()
    pygame.font.init()
    clock = pygame.time.Clock()


    window_width, window_height = config['window_size'][0], config['window_size'][1]
    global window
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption(f"MAISR User Study - Subject {subject_id}")

    # Create font for instructions
    #font = pygame.font.SysFont(None, 24)
    font = pygame.font.Font('./user_study/AcPlus_IBM_VGA_8x16.ttf', 24)  # pygame.font.SysFont('Arial', 36, bold=True)

    # Store results
    experiment_results = []
    current_agents = {}

    data_logger = ExperimentDataLogger(subject_id)
    screen_manager = ScreenManager(window, clock)

    # Main experiment loop
    try:
        if not skip_instructions:
            print("Starting instruction screens...")
            instruction_manager = InstructionSeriesManager(
                window,
                clock,
                map_image_path="user_study/img/map_image.png",
                sensor_image_path="user_study/img/sensor_image.png",
                hvt_video_path="user_study/img/target_id_video.mp4",
                detection_video_path="user_study/img/detection_video.mp4",
                click_video_path="user_study/img/click_control.mp4", # TODO replace
                human_image_path="user_study/img/human_aircraft.png",
                teammate_image_path="user_study/img/teammates_image.png",
                sio = sockets
            )
            instruction_result = instruction_manager.run_instruction_series()

            if instruction_result["action"] == "exit":
                print("User exited during instructions")
                return
        else:
            print("Skipping instructional screens...")


        last_agent_appearance = None

        for config_index, current_config in enumerate(full_config_list):

            # Handle practice level special settings
            if config_index == 0:
                print(f"\nPreparing for practice level (config: {current_config})")
                agent_letter = 'A'  # Pick default practice agent
                level_number = 1
                agent_appearance = 'brown'  # Practice agent color
            else:
                agent_letter = current_config[0]  # 'A' or 'B'
                level_number = int(current_config[1:])
                if agent_letter == 'A':
                    agent_appearance = 'purple'
                elif agent_letter == 'B':
                    agent_appearance = 'green'
                elif agent_letter == 'S':
                    agent_appearance = 'brown'
                else:
                    raise ValueError(f"Unexpected agent letter {agent_letter}")

            print(f"\nPreparing for config: {current_config}")
            if agent_letter == 'S':
                print(f'%%% PILOT STUDY ONLY - RUNNING SOLO ROUND {level_number} %%%')
            print(f"Agent: {agent_letter}, Level: {level_number}")
            config['force_specific_level'] = level_number - 1  # Convert to 0-indexed

            # Build environment
            env_fns = [make_wrapped_env(config, clock, window, agent_appearance, subject_id) for _ in range(1)]
            env = DummyVecEnv(env_fns)
            vecnorm_path = vecnorm_paths[agent_letter]
            print(f'Loaded vecnorm stats from {vecnorm_path}')
            env = load_vecnormalize_wrapper(vecnorm_path, env)

            # Load RL agent
            if agent_letter not in current_agents:
                current_agents[agent_letter] = PPO.load(agent_models[agent_letter], env=env)
            current_agent_model = current_agents[agent_letter]

            # Initialize human controller for this episode
            human_controller = HumanSubpolicyController(env)
            sockets.human_controller = human_controller

            # Run the episode
            should_quit, episode_reward, step_count = run_single_episode(
                env, human_controller, current_config, config_index, len(full_config_list),
                current_agent_model, window, font, clock, tick_rate, data_logger, time_factor, agent_letter
            )
            if should_quit:
                print("Experiment terminated by user")
                break

            # After practice episode, show the after-practice screen
            if config_index == 0:
                after_practice_screen = AfterPracticeScreen(window.get_width(), window.get_height())
                after_practice_result = screen_manager.show_screen(after_practice_screen)
                if after_practice_result["action"] == "exit":
                    print("User exited after practice screen")
                    return
                continue  # skip survey for practice

            # Store results
            experiment_results.append({
                'config': current_config,
                'agent': agent_letter,
                'level': level_number,
                'reward': episode_reward,
                'steps': step_count,
                'config_index': config_index + start_level
            })

            env.close()

            # Workload survey and teammate survey logic (skip for practice)
            if not skip_instructions:
                workload_survey_screen = WorkloadSurveyScreen(
                    episode_config=current_config, window_width=window_width, window_height=window_height)
                workload_survey_result = screen_manager.show_screen(workload_survey_screen)
                if workload_survey_result["action"] == "exit":
                    print("Experiment terminated by user")
                    break
                elif workload_survey_result["action"] == "continue":
                    survey_data = workload_survey_result.get("survey_data", {})
                    print(f"Survey responses for {current_config}: {survey_data['responses']}")
                    if hasattr(data_logger, 'log_survey_data'):
                        data_logger.log_survey_data(survey_data)

            print(f'config index = {config_index}')
            print(f'(config_index + 1) % 2 == 0: {(config_index) % 2 == 0}')
            if (not skip_instructions) and config_index > 0 and (config_index) % 2 == 0:
                teammate_compare_survey = TeammatePreferenceSurveyScreen(
                    window_width, window_height, agent_appearance=agent_appearance, last_agent_appearance=last_agent_appearance)
                teammate_compare_result = screen_manager.show_screen(teammate_compare_survey)
                if teammate_compare_result["action"] == "continue":
                    survey_data = teammate_compare_result["survey_data"]
                    data_logger.log_teammate_survey_data(survey_data)

            last_agent_appearance = agent_appearance

            if config_index < len(full_config_list) - 1:
                print("Next episode starting in 2 seconds...")
                pygame.time.wait(100)

        # for config_index, current_config in enumerate(full_config_list):
        # #for config_index, current_config in enumerate(full_config_list[1:]):
        #     agent_letter = current_config[0]  # 'A' or 'B'
        #     level_number = int(current_config[1:])  # Level number
        #     if agent_letter == 'A':
        #         #if level_number % 2 == 0: # Even levels
        #         agent_appearance = 'purple'
        #         # else: # Odd levels
        #         #     agent_appearance = 'red'
        #     elif agent_letter == 'B': # Agent B
        #         # if level_number % 2 == 0: # Even levels
        #         #     agent_appearance = 'brown'
        #         # else: # Odd levels
        #         agent_appearance = 'green'
        #     else:
        #         raise ValueError(f'Agent letter is {agent_letter}')
        #
        #     print(f"\nPreparing for config: {current_config}")
        #     print(f"Agent: {agent_letter}, Level: {level_number}")
        #
        #     config['force_specific_level'] = level_number - 1  # Convert to 0-indexed
        #
        #     env_fns = [make_wrapped_env(config, clock, window, agent_appearance, subject_id) for _ in range(1)]
        #     env = DummyVecEnv(env_fns)
        #     vecnorm_path = vecnorm_paths[agent_letter]
        #     print(f'Loaded vecnorm stats from {vecnorm_path}')
        #     env = load_vecnormalize_wrapper(vecnorm_path, env)
        #
        #     # Load the appropriate agent
        #     if agent_letter not in current_agents:
        #         current_agents[agent_letter] = PPO.load(agent_models[agent_letter], env=env)
        #     current_agent_name = current_agents[agent_letter]
        #
        #     # Initialize human controller for this episode
        #     human_controller = HumanSubpolicyController(env)
        #     sockets.human_controller = human_controller
        #
        #     should_quit, episode_reward, step_count = run_single_episode(
        #         env, human_controller, current_config, config_index, len(full_config_list),
        #         current_agent_name, window, font, clock, tick_rate, data_logger, time_factor
        #     )
        #
        #     # Store results
        #     result = {
        #         'config': current_config,
        #         'agent': agent_letter,
        #         'level': level_number,
        #         'reward': episode_reward,
        #         'steps': step_count,
        #         'config_index': config_index + start_level
        #     }
        #     experiment_results.append(result)
        #
        #     # Clean up environment
        #     env.close()
        #
        #     print(f"\nEpisode {current_config} completed!")
        #
        #     if not skip_instructions:
        #         workload_survey_screen = WorkloadSurveyScreen(episode_config=current_config, window_width=window_width, window_height=window_height)
        #         workload_survey_result = screen_manager.show_screen(workload_survey_screen)
        #         #survey_launched = launch_survey_url(survey_url, current_config, args.subject_id)
        #
        #         if workload_survey_result["action"] == "exit":
        #             print("Experiment terminated by user")
        #             break
        #         elif workload_survey_result["action"] == "continue":
        #             # Log the survey data
        #             survey_data = workload_survey_result.get("survey_data", {})
        #             print(f"Survey responses for {current_config}: {survey_data['responses']}")
        #
        #             # Add survey data to your data logger
        #             if hasattr(data_logger, 'log_survey_data'):
        #                 data_logger.log_survey_data(survey_data)
        #             else:
        #                 # Fallback: save to file or print
        #                 print(f"Survey data: {survey_data}")
        #
        #     # Show the teammate preference survey
        #
        #     if (not skip_instructions) and config_index > 0 and (config_index+1) % 2 == 0:
        #         teammate_compare_survey = TeammatePreferenceSurveyScreen(window_width, window_height, agent_appearance=agent_appearance, last_agent_appearance=last_agent_appearance)
        #         teammate_compare_result = screen_manager.show_screen(teammate_compare_survey)
        #
        #         if teammate_compare_result["action"] == "continue":
        #             survey_data = teammate_compare_result["survey_data"]
        #             data_logger.log_teammate_survey_data(survey_data)
        #         elif teammate_compare_result["action"] == "exit": pass
        #     last_agent_appearance = agent_appearance
        #
        #     # Check if user wants to quit
        #     if should_quit:
        #         print("Experiment terminated by user")
        #         break
        #
        #     # Brief pause between episodes (unless it's the last one)
        #     if config_index < len(full_config_list) - 1:
        #         print("Next episode starting in 2 seconds...")
        #         pygame.time.wait(100)

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")

    finally:
        final_screen = FinalSummaryScreen(experiment_results, window.get_width(), window.get_height())
        screen_manager.show_screen(final_screen)

        data_logger.save_session_data()
        session_summary = data_logger.get_session_summary()

        # Print experiment summary
        print(f"\n{'=' * 60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'=' * 60}")
        print(f"Subject ID: {subject_id}")
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

        print(f"\nData saved to: {data_logger.output_dir}/subject_{subject_id}/")
        pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MAISR user study experiment')
    parser.add_argument('subject_id', type=int, help='Subject ID (integer)')
    parser.add_argument('--start_level', type=int, default=0, help='Starting level index (default: 0)')
    parser.add_argument('--skip', action='store_true', help='Skip instructional screens')
    parser.add_argument('--pilot', action='store_true', help='Set to true if running pilot studies. Appends solo configs after main rounds.')
    args = parser.parse_args()
    subject_id = args.subject_id
    start_level = args.start_level
    skip_instructions = args.skip
    main(subject_id=subject_id, start_level=start_level, skip_instructions=skip_instructions, collect_solo_trajectories = args.pilot)