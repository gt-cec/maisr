import argparse
import ctypes
import pygame
import numpy as np
import random

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from env_multi_new import MAISREnvVec
from server import socketio
from utility.localsearch_training_wrapper import MaisrLocalSearchWrapper
from utility.config import subject_id
from utility.data_logging import load_env_config
from utility.league_management import (GenericTeammatePolicy, SubPolicy, LocalSearch, ChangeRegions, GoToNearestThreat, EvadeDetection, TeammateManager, RLTeammatePolicy)
from user_study.rl_data_logger import ExperimentDataLogger
from user_study.instructional_screens import ScreenManager, WorkloadSurveyScreen, TeammatePreferenceSurveyScreen, \
    InstructionSeriesManager, FinalSummaryScreen, AfterPracticeScreen, SecondPracticeIntroScreen, InterScreen, \
    BeforeSoloScreen
from PIL import Image
from io import BytesIO

import webbrowser
from stable_baselines3.common.vec_env import VecNormalize


window = None

def draw_countdown_overlay(window, font, countdown_steps, countdown_length, map_rect=None):
    """Draw a 3-second countdown overlay with gray background and a black circle around the number."""
    # Calculate seconds left (3, 2, 1)
    seconds_total = 3
    steps_per_second = countdown_length // seconds_total
    seconds_left = seconds_total - (countdown_steps // steps_per_second)
    seconds_left = max(1, seconds_left)  # Ensure we display 1 at the last moment

    width, height = window.get_size()

    # Gray box overlay: either the whole screen or just the map area
    if map_rect is None:
        overlay_rect = pygame.Rect(0, 0, width, height)
    else:
        overlay_rect = map_rect

    overlay = pygame.Surface((overlay_rect.width, overlay_rect.height))
    overlay.set_alpha(180)  # 0=transparent, 255=opaque
    overlay.fill((100, 100, 100))
    window.blit(overlay, overlay_rect.topleft)

    # Big countdown font (ignore the passed font for size)
    big_font = pygame.font.Font('./user_study/AcPlus_IBM_VGA_8x16.ttf', 100)
    countdown_text = str(seconds_left)
    text_surface = big_font.render(countdown_text, True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=(500, 500))

    # Draw black circle behind text
    circle_radius = max(text_rect.width, text_rect.height) // 2# + 20
    pygame.draw.circle(window, (0, 0, 0), text_rect.center, circle_radius)

    # Draw the countdown text
    window.blit(text_surface, text_rect)




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

def draw_progress_bar(window, font, current_index, total_configs, start_level):
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
        if start_level == 0:
            label = "P1" if i == 0 else "P2" if i == 1 else str(i - 1)
        else:
            label = str(i)

        label_surface = font.render(label, True, (0, 0, 0))
        label_rect = label_surface.get_rect(center=(x + segment_width // 2, y + segment_height // 2))
        window.blit(label_surface, label_rect)



# def draw_bottom_bar_info(window, font, threats_identified, targets_identified, detections,
#                          step_count, max_steps, tick_rate):
#     """Draw bottom bar with score and countdown timer"""
#     regular_targets = targets_identified
#     high_value_targets = threats_identified
#
#     # Compute score: 5 × (# of targets) + 30 × abs(2 - # of threats)
#     score = 5 * regular_targets - 10 * abs(2 - high_value_targets)
#
#     # Timer: Counts down from 75 seconds
#     total_seconds = 75
#     elapsed_seconds = (step_count / 486) * 75
#     time_left = max(0, int(total_seconds - elapsed_seconds))
#
#     bottom_texts = [
#         f"Regular: {regular_targets}/15",
#         f"High-value: {high_value_targets}/2",
#         f"SCORE: {score}",
#         f"Steps: {step_count} / {int(round(max_steps/35, 0))}",
#     ]
#
#     x_positions = [10, 200, 450, 800]
#     y_pos = 1060
#
#     for i, text in enumerate(bottom_texts):
#         if (i == 0 and regular_targets == 15) or (i == 1 and high_value_targets == 2):
#             color = (0, 200, 0)
#         elif i == 1 and high_value_targets > 2:
#             color = (225, 0, 0)
#         else:
#             color = (0, 0, 0)
#         text_surface = font.render(text, True, color)
#         window.blit(text_surface, (x_positions[i], y_pos))
#
#     # Outline box around SCORE and timer
#     pygame.draw.rect(window, (0, 0, 0), pygame.Rect(440, 1050, 200, 40), width=3)

def draw_bottom_bar_info(window, font, threats_identified, targets_identified, detections,
                         step_count, max_steps, tick_rate):
    """Draw bottom bar with score and countdown timer"""
    regular_targets = targets_identified
    high_value_targets = threats_identified

    # Compute score: 5 × (# of targets) - 10 × |2 - (# of threats)|
    score = 5 * regular_targets - 10 * abs(2 - high_value_targets)

    # Timer: Counts down from 75 seconds
    total_seconds = 75
    elapsed_seconds = (step_count / 486) * 75
    time_left = max(0, int(total_seconds - elapsed_seconds))

    # Convert to M:SS format
    minutes = time_left // 60
    seconds = time_left % 60
    timer_text = f"{minutes}:{seconds:02d}"

    # --- Draw Top Timer ---
    timer_color = (255, 0, 0) if time_left < 20 else (0, 0, 0)
    big_font = pygame.font.Font('./user_study/AcPlus_IBM_VGA_8x16.ttf', 40)  # Bigger font
    timer_surface = big_font.render(timer_text, True, timer_color)
    timer_rect = timer_surface.get_rect(center=(500, 20))

    # Draw outline box around timer (bigger than text)
    padding_x, padding_y = 20, 10
    outline_rect = pygame.Rect(
        timer_rect.left - padding_x,
        timer_rect.top - padding_y,
        timer_rect.width + 2 * padding_x,
        timer_rect.height + 2 * padding_y - 10
    )
    pygame.draw.rect(window, (255, 255, 255), outline_rect)  # White background
    pygame.draw.rect(window, (0, 0, 0), outline_rect, width=4)  # Black border

    # Draw timer text on top
    window.blit(timer_surface, timer_rect)

    # --- Draw Bottom Bar Info ---
    bottom_texts = [
        f"Regular: {regular_targets}/15",
        f"High-value: {high_value_targets}/2",
        f"SCORE: {score}",
        #f"Steps: {step_count} / {int(round(max_steps/35, 0))}",
    ]

    x_positions = [10, 200, 450, 800]
    y_pos = 1060

    for i, text in enumerate(bottom_texts):
        if (i == 0 and regular_targets == 15) or (i == 1 and high_value_targets == 2):
            color = (0, 200, 0)
        elif i == 1 and high_value_targets > 2:
            color = (225, 0, 0)
        else:
            color = (0, 0, 0)
        text_surface = font.render(text, True, color)
        window.blit(text_surface, (x_positions[i], y_pos))

    # Outline box around SCORE
    pygame.draw.rect(window, (0, 0, 0), pygame.Rect(440, 1050, 140, 40), width=3)



def run_single_episode(env, human_controller, config, config_index, total_configs, agent_model, window, font, clock, tick_rate, data_logger, time_factor, agent_letter, agent_model_name, start_level, short_rounds, admin=False):
    """Run a single episode of the experiment"""
    #print(f"\n{'=' * 50}")
    print(f"Starting Config: {config} ({config_index + 1}/{total_configs})")
    print(f"Agent {agent_letter} (Model {agent_model_name}")
    print(f"{'=' * 50}")

    import sockets

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

    # Initialize frame management
    frame_skip_counter = 0
    frame_send_interval = 1  # Send every N frames (adjust for performance)

    # Draw static labels once
    draw_bottom_bar_info(window, font, base_env.num_threats_identified, base_env.targets_identified, base_env.detections, 0, base_env.config['max_steps'], tick_rate)

    countdown_length = 100
    countdown_steps = 0
    sockets.delta_manager.last_frame = None
    sockets.delta_manager.frame_count = 0
    sockets.delta_manager.last_full_frame = 0

    first_countdown_frame = True
    skip_round = False
    while not done:

        if countdown_steps <= countdown_length:
            env.render()
            draw_countdown_overlay(window, font, countdown_steps, countdown_length)
            pygame.display.flip()
            #pygame.time.wait(33)
            sockets.send_frame(window)
            countdown_steps += 1
            continue  # Skip the rest of the loop until countdown is done

        #sockets.send_frame(window)
        sockets.send_frame_with_delta(window, quality=75)  # will trigger full frame
        pygame.time.wait(50)

        map_half_size = env.envs[0].env.config['gameboard_size']
        current_time = pygame.time.get_ticks()


        # if first_frame:
        #     sockets.send_frame(window)
        #     first_frame = False

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True, episode_reward, step_count, 0, 0  # Signal to quit experiment
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE and admin:
                    return True, episode_reward, step_count, 0, 0  # Signal to quit experiment
                elif event.key == pygame.K_SPACE and admin:
                    paused = not paused
                    print("Game paused" if paused else "Game resumed")
                elif event.key == pygame.K_RETURN and admin:
                    print('SKIP ROUND')
                    skip_round = True
                else:
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
        if last_agent_action is None or step_count % time_factor == 0:
            if agent_letter == 'P':
                agent_action = 0
            else:
                agent_action, _ = agent_model.predict(obs, deterministic=True)
            #     with torch.no_grad():
            #         obs_tensor = torch.tensor(obs).unsqueeze(0).float()
            #         distribution = agent_model.policy.get_distribution(obs_tensor)
            #         action_probs = distribution.distribution.probs.squeeze(0).clone()
            #
            #         #tie_breaker = torch.tensor([0.1, 0.06, 0.04, 0.02, 0.00, 0.1, 0.1])
            #         #action_probs += tie_breaker
            #
            #         k = 7
            #         top_probs, top_indices = torch.topk(action_probs, k, largest=True)
            #
            #         top_probs_values = top_probs.squeeze().cpu().numpy()
            #         top_indices_values = top_indices.squeeze().cpu().numpy()
            #
            #         print(f"Top {k} probs: {top_probs_values}")
            #         print(f"Top {k} idx:   {top_indices_values}")
            #
            #         #agent_action = torch.argmax(action_probs).item()
            #
            #         if abs(top_probs_values[0] - top_probs_values[1]) <= 0.03:
            #             agent_action = min(top_indices_values[0], top_indices_values[1])
            #             print(f"Tie-breaker triggered: Prob diff {abs(top_probs_values[0] - top_probs_values[1]):.4f}, "
            #                   f"selected lower index {agent_action}")
            #
            #         else:
            #             agent_action = top_indices_values[0].item()  # highest probability

        else:
            agent_action = last_agent_action

        last_agent_action = agent_action

        # Take step in environment
        obses, rewards, dones, infos = env.step([agent_action])
        obs = obses[0]
        reward = rewards[0]
        info = infos[0]
        short_round_triggered = step_count > 5 and short_rounds
        done = dones[0] or (np.sum(base_env.threat_identified) >= 2.0 and base_env.targets_identified >= 15) or skip_round or short_round_triggered
        if skip_round:
            print(f'SKIP ROUND')
            done = True
            break

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
        #draw_status_info(window, font, config, config_index, total_configs, step_count, episode_reward, human_controller)
        draw_progress_bar(window, font, config_index, total_configs, start_level)
        draw_bottom_bar_info(window, font, base_env.num_threats_identified, base_env.targets_identified,base_env.detections, step_count, base_env.config['max_steps'], tick_rate)

        # Update display
        pygame.display.flip()

        if step_count % 120 == 0:
            sockets.send_frame(window)
        else:
            sockets.send_frame_with_delta(window, quality=65)
        clock.tick(tick_rate)

        # Print periodic status
        #if step_count % 50 == 0:
            #print(f"Step {step_count}: Reward = {episode_reward:.2f}, ")

    # End episode logging
    episode_summary = data_logger.end_episode(env, info, int(info.get('target_ids', -1)), int(info.get('threat_ids', -1)), agent_model_name)

    print(f"\nConfig {config} Complete!")
    print(f"    Final Reward: {episode_reward:.2f}")
    print(f"    Steps Taken: {step_count}")
    print(f"    Targets Identified: {info.get('target_ids', -1)}")
    print(f"    Threats Identified: {info.get('threat_ids', -1)}")
    #print(f"Detections: {info.get('detections', -1)}")

    return False, episode_reward, step_count, info.get('target_ids', -1), info.get('threat_ids', -1)


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


def main(subject_id=None, start_level=0, skip_instructions=None,collect_solo_trajectories=False, admin=False, run_third_agent = False, short_rounds = False):

    import sockets
    sockets.connect()
    #sio = socketio.Client()
    #sio.connect("http://99.45.36.114:5001")

    print(f"\n \n Subject ID: {subject_id}")
    print(f"Starting from level: {start_level}")

    # Configuration
    config_filename = 'configs/Monolith_index_August.json'
    tick_rate = 30
    time_factor = 10
    config = load_env_config(config_filename)

    config['tick_rate'] = tick_rate
    config['game_speed'] /= time_factor
    config['max_steps'] *= (1700 / 1500) * time_factor
    config['use_stuck_detection'] = False
    config['prob_detect'] = 0#0.0003
    config['action_type'] = 'Discrete16'
    #config['observe_teammate_priority'] = False # TODO switch to true with new agents
    print(f'LOADED CONFIG {config_filename}')

    agent_a_name = 'aug2b_finetuned'#'strategy_trained' #'M1S_indexstrategy'
    agent_b_name = 'aug2b_baseline'#'selfplay_seed77' #'M1S-2_selfplay_750ksteps' # 'M1S_indexselfplay'
    agent_c_name = 'aug2b_mixed75_35e5steps' #'M1S_indexmixed50'
    agent_s_name = 'bad_practice_agent'
    agent_p_name = 'bad_practice_agent'


    # Define RL agent model paths
    agent_models = {
        'A': f'./user_study/saved_agents/{agent_a_name}_model.zip',
        'B': f'./user_study/saved_agents/{agent_b_name}_model.zip',
        'C': f'./user_study/saved_agents/{agent_c_name}_model.zip',
        'S': f'./user_study/saved_agents/{agent_s_name}_model.zip', # Solo
        'P': f'./user_study/saved_agents/{agent_p_name}_model.zip' # Practice
    }

    vecnorm_paths = {
        'A': f'./user_study/saved_agents/{agent_a_name}_vecnormalize.pkl',
        'B': f'./user_study/saved_agents/{agent_b_name}_vecnormalize.pkl',
        'C': f'./user_study/saved_agents/{agent_c_name}_vecnormalize.pkl',
        'S': f'./user_study/saved_agents/{agent_s_name}_vecnormalize.pkl',
        'P': f'./user_study/saved_agents/{agent_p_name}_vecnormalize.pkl'
    }

    if subject_id % 2 == 0:
        if run_third_agent:
            #config_list = ['A1', 'B1', 'C7', 'B2', 'A2', 'C5', 'A3', 'B3', 'C1', 'B4', 'A4', 'C6', 'A5', 'B5', 'C3', 'B6', 'A6', 'C2', 'A7', 'B7', 'C4']
            config_list = ['A1', 'B1',  # A first
                           'C3', 'A3',  # C first
                           'S3', 'S7',
                           'B3', 'C1',  # B first
                           'A4', 'B4',  # A first
                           'C5', 'A5', # C first
                           'B5', 'C7', # B first
                           'A7', 'B7'] # A first

        else:
            config_list = ['A1', 'B1',
                           'B3', 'A3',
                           'A4', 'B4',
                           'S3', 'S7',
                           'B5', 'A5',
                           'A7', 'B7'
                           ]
    else:
        if run_third_agent:
            config_list = ['B1', 'A1',
                           'C3', 'B3',
                           'S3', 'S7',
                           'A3', 'C1',
                           'B4', 'A4',
                           'C5', 'B5',
                           'A5', 'C7',
                           'B7', 'A7']
        else:
            config_list = ['B1', 'A1',
                           'A3', 'B3',
                           'B4', 'A4',
                           'S3', 'S7',
                           'A5', 'B5',
                           'B7', 'A7'
                           ]  #

    practice_config = [f"P1", f"P4"]
    full_config_list = practice_config + config_list

    # If start_level is specified, start from that index
    if start_level > 0:
        if start_level >= len(full_config_list):
            print(f"Error: start_level {start_level} is >= total configs {len(full_config_list)}")
            return
        full_config_list = full_config_list[start_level:]
        print(f"Starting from level {start_level}: {full_config_list}")

    print(f"Randomized configuration order: {full_config_list}")

    # Initialize pygame
    if hasattr(ctypes, 'windll') and hasattr(ctypes.windll, 'user32'): ctypes.windll.user32.SetProcessDPIAware()
    pygame.display.init()
    pygame.font.init()
    clock = pygame.time.Clock()

    window_width, window_height = config['window_size'][0], config['window_size'][1]
    global window
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption(f"MAISR User Study - Subject {subject_id}")

    # Create font for instructions
    font = pygame.font.Font('./user_study/AcPlus_IBM_VGA_8x16.ttf', 26)  # pygame.font.SysFont('Arial', 36, bold=True)

    # Store results
    experiment_results = []
    current_agents = {}

    data_logger = ExperimentDataLogger(subject_id)
    screen_manager = ScreenManager(window, clock)

    # Main experiment loop
    try:
        if (not skip_instructions) and start_level == 0:
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

            if admin and instruction_result["action"] == "exit":
                print("User exited during instructions")
                return
        else:
            print("Skipping instructional screens...")


        last_agent_appearance = None

        appearance_map = {
            'A': 'red',
            'B': 'purple',
            'C': 'green',
            'S': 'invisible',
            'P': 'invisible'
        }

        before_solo_screen = BeforeSoloScreen(window_width=window_width, window_height=window_height, sio=sockets)

        for config_index, current_config in enumerate(full_config_list):

            agent_letter = current_config[0]  # 'A', 'B', or 'S'
            level_number = int(current_config[1:])

            if config_index == 0 and start_level == 0: # # Handle practice level special settings
                print(f"\nPreparing for practice level (config: {current_config})")
                agent_letter = 'P'
                level_number = 1
                agent_appearance = appearance_map.get(agent_letter, 'black')
            elif config_index == 1 and start_level == 0: # # Handle practice level special settings
                screen = SecondPracticeIntroScreen(window_width=window_width, window_height=window_height, sio=sockets)
                result = screen_manager.show_screen(screen)
                print(f"\nPreparing for practice level (config: {current_config})")
                agent_letter = 'P'
                level_number = 4
                agent_appearance = appearance_map.get(agent_letter, 'black')

            elif agent_letter == 'S':
                result = screen_manager.show_screen(before_solo_screen)
                print(f'\n Preparing for solo round')
                agent_appearance = 'invisible'
                before_solo_screen.second = True
            else:
                agent_appearance = appearance_map.get(agent_letter, 'black')

            print('\n\n=====================================================================')
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

            agent_model_name = agent_models[agent_letter] # For logging

            # Run the episode
            should_quit, episode_reward, step_count, target_ids, threat_ids, = run_single_episode(
                        env, human_controller, current_config, config_index, len(full_config_list),
                        current_agent_model, window, font, clock, tick_rate, data_logger, time_factor, agent_letter, agent_model_name, start_level, short_rounds, admin=admin)

            if should_quit:
                print("Experiment terminated by user")
                break

            # After second practice episode, show the after-practice screen
            if config_index == 1:
                after_practice_screen = AfterPracticeScreen(window.get_width(), window.get_height(), sio=sockets)
                after_practice_result = screen_manager.show_screen(after_practice_screen)
                continue  # skip survey for practice

            # Store results
            experiment_results.append({
                'config': current_config,
                'agent': agent_letter,
                'level': level_number,
                'target_ids': target_ids,
                'threat_ids': threat_ids,
                #'reward': episode_reward,
                'steps': step_count,
                'config_index': config_index + start_level
            })

            env.close()

            # Workload survey and teammate survey logic (skip for practice)
            level = agent_letter + str(level_number)

            levels_for_workload_survey = ['A1', 'B1', 'B3', 'A4', 'B5', 'B7', 'C5', 'C3', 'C4']
            if level in levels_for_workload_survey:
                workload_survey_screen = WorkloadSurveyScreen(
                    episode_config=current_config, window_width=window_width, window_height=window_height)
                workload_survey_result = screen_manager.show_screen(workload_survey_screen)
                if admin and workload_survey_result["action"] == "exit":
                    print("Experiment terminated by user")
                    break
                elif workload_survey_result["action"] == "continue":
                    survey_data = workload_survey_result.get("survey_data", {})
                    print(f"Survey responses for {current_config}: {survey_data['responses']}")
                    if hasattr(data_logger, 'log_survey_data'):
                        data_logger.log_survey_data(survey_data)

            if run_third_agent:
                if subject_id % 2 == 0:
                    levels_for_preference_survey = ['B1', 'A3', 'C1', 'B4', 'A5', 'C3', 'B7']
                else:
                    levels_for_preference_survey = ['A1', 'B3', 'C1', 'A4', 'B5', 'C3', 'A7']
            else:
                if subject_id % 2 == 0:
                    levels_for_preference_survey = ['B1', 'A3', 'B7', 'A5', 'B4']
                else:
                    levels_for_preference_survey = ['A1', 'B3', 'A7', 'B5', 'A4']


            if level in levels_for_preference_survey:
                teammate_compare_survey = TeammatePreferenceSurveyScreen(
                    window_width, window_height,
                    agent_appearance=agent_appearance,
                    last_agent_appearance=last_agent_appearance
                )

                teammate_compare_result = screen_manager.show_screen(teammate_compare_survey)
                if teammate_compare_result["action"] == "continue":
                    survey_data = teammate_compare_result["survey_data"]
                    survey_data['episode_config'] = current_config
                    survey_data['compared_agents'] = (agent_appearance, last_agent_appearance)
                    data_logger.log_teammate_survey_data(survey_data)

                data_logger.save_session_data()

            no_surveys = level not in levels_for_workload_survey and level not in levels_for_preference_survey
            if agent_letter not in ['P', 'S'] and no_surveys:
                inter_screen = InterScreen()
                screen_manager.show_screen(inter_screen)

            last_agent_appearance = agent_appearance

            if config_index < len(full_config_list) - 1:
                print("Next episode starting in 2 seconds...")
                pygame.time.wait(100)




    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")

    finally:
        final_screen = FinalSummaryScreen(experiment_results, window.get_width(), window.get_height())
        screen_manager.show_screen(final_screen)

        data_logger.save_session_data()
        session_summary = data_logger.get_session_summary()

        try:
            print("Emitting study_complete to server...")
            sockets.sio.emit('study_complete')  #{'subject_id': subject_id}
        except Exception as e:
            print(f"Failed to emit study_complete: {e}")

        # Print experiment summary
        print(f"\n{'=' * 60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'=' * 60}")
        print(f"Subject ID: {subject_id}")
        print(f"Completed configurations: {len(experiment_results)}")

        #if experiment_results:
            #total_reward = sum(r['reward'] for r in experiment_results)
            #avg_reward = total_reward / len(experiment_results)
            #print(f"Total reward: {total_reward:.2f}")
            #print(f"Average reward: {avg_reward:.2f}")

            # print("\nDetailed results:")
            # for result in experiment_results:
            #     print(f"  {result['config']}: Reward = {result['reward']:.2f}, "
            #           f"Steps = {result['steps']}")

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
                    #print(f"    Average reward: {performance['average_reward']:.2f}")
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
    parser.add_argument('--admin', action='store_true',help='Allows skipping sections with ENTER')
    parser.add_argument('--short', action='store_true', help='')
    args = parser.parse_args()

    subject_id = args.subject_id
    start_level = args.start_level
    skip_instructions = args.skip
    short_rounds = args.short



    main(subject_id=subject_id, start_level=start_level, skip_instructions=skip_instructions, collect_solo_trajectories = False, run_third_agent = True, short_rounds=short_rounds)
