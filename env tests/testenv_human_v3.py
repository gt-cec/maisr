import pygame
import os
import ctypes
import numpy as np

from env_combined import MAISREnvVec
from utility.data_logging import load_env_config
from config import x, y


def convert_mouse_to_game_coords(mouse_pos, gameboard_size):
    """
    Convert pygame mouse coordinates (0, gameboard_size) to game coordinates (-150, +150)
    
    Args:
        mouse_pos: (x, y) tuple of mouse position in pygame coordinates
        gameboard_size: Size of the gameboard (typically 300)
    
    Returns:
        (x, y) tuple in game coordinates (-150, +150)
    """
    mouse_x, mouse_y = mouse_pos
    map_half_size = gameboard_size / 2
    
    # Convert from pygame coordinates to game coordinates
    game_x = mouse_x - map_half_size
    game_y = mouse_y - map_half_size
    
    return game_x, game_y


def convert_game_to_action_space(game_coords, gameboard_size):
    """
    Convert game coordinates (-150, +150) to action space (-1, +1)
    
    Args:
        game_coords: (x, y) tuple in game coordinates
        gameboard_size: Size of the gameboard (typically 300)
    
    Returns:
        (x, y) tuple in action space (-1, +1)
    """
    game_x, game_y = game_coords
    map_half_size = gameboard_size / 2
    
    # Convert to action space
    action_x = game_x / map_half_size
    action_y = game_y / map_half_size
    
    # Clamp to valid action space
    action_x = max(-1.0, min(1.0, action_x))
    action_y = max(-1.0, min(1.0, action_y))
    
    return np.array([action_x, action_y], dtype=np.float32)


def render_reward_text(window, episode_reward, step_reward_components, cumulative_reward_components, gameboard_size, config):
    """
    Render the episode reward and reward components on the screen

    Args:
        window: pygame display surface
        episode_reward: Current episode total reward
        step_reward_components: Reward components for current step
        cumulative_reward_components: Cumulative reward components for episode
        gameboard_size: Size of the gameboard
    """
    font_large = pygame.font.SysFont(None, 20)
    font_small = pygame.font.SysFont(None, 18)

    # Main episode reward
    episode_text = f"Episode Reward: {episode_reward:.2f}"
    episode_surface = font_large.render(episode_text, True, (255, 255, 255))

    # Reward component texts
    component_texts = []
    component_texts.append("=== CUMULATIVE ===")
    component_texts.append(f"High Val Targets: {cumulative_reward_components.get('high val target id', 0):.2f}")
    component_texts.append(f"Regular Targets: {cumulative_reward_components.get('regular val target id', 0):.2f}")
    component_texts.append(f"Proximity: {cumulative_reward_components.get('proximity', 0):.2f}")
    component_texts.append(f"Early Finish: {cumulative_reward_components.get('early finish', 0):.2f}")
    component_texts.append(f"Waypoint-to-Nearest: {cumulative_reward_components.get('waypoint-to-nearest', 0):.2f}")
    component_texts.append("")
    component_texts.append("=== CURRENT STEP ===")

    component_texts.append(f"High Val Targets: {step_reward_components.get('high val target id', 0) * config["highqual_highvaltarget_reward"]:.2f}")
    component_texts.append(f"Regular Targets: {step_reward_components.get('regular val target id', 0) * config["highqual_regulartarget_reward"]:.2f}")
    component_texts.append(f"Proximity: {step_reward_components.get('proximity', 0):.2f}")
    component_texts.append(f"Early Finish: {step_reward_components.get('early finish', 0):.2f}")
    component_texts.append(f"Waypoint-to-Nearest: {step_reward_components.get('waypoint-to-nearest', 0):.2f}")

    # Calculate total height needed
    line_height = 20
    total_height = 50 + len(component_texts) * line_height + 20  # Episode text + components + padding

    # Create background
    # background_width = 300
    # background_rect = pygame.Rect(300, 10, background_width, total_height)
    # background_surface = pygame.Surface((background_width, total_height))
    # background_surface.set_alpha(0)
    # background_surface.fill((0, 0, 0))
    #
    # window.blit(background_surface, background_rect)

    # Render episode reward
    window.blit(episode_surface, (320, 320))

    # Render component breakdown
    y_offset = 60
    for text in component_texts:
        if text.startswith("==="):
            # Header text in yellow
            text_surface = font_small.render(text, True, (255, 255, 0))
        elif text == "":
            # Skip empty lines
            y_offset += line_height
            continue
        else:
            # Regular component text in white
            text_surface = font_small.render(text, True, (255, 255, 255))

        window.blit(text_surface, (10, y_offset+300))
        y_offset += line_height

# def render_reward_text(window, episode_reward, gameboard_size):
#     """
#     Render the cumulative reward text on the screen
#
#     Args:
#         window: pygame display surface
#         episode_reward: Current episode cumulative reward
#         gameboard_size: Size of the gameboard
#     """
#     font = pygame.font.SysFont(None, 20)
#     reward_text = f"Episode Reward: {episode_reward:.2f}"
#     text_surface = font.render(reward_text, True, (255, 255, 255))  # White text
#
#     # Position the text in the top-left corner with some padding
#     text_rect = text_surface.get_rect()
#     text_rect.topleft = (10, 310)
#
#     # Draw a semi-transparent background for better readability
#     background_rect = text_rect.copy()
#     background_rect.inflate(20, 10)  # Add padding around text
#     background_surface = pygame.Surface((background_rect.width, background_rect.height))
#     background_surface.set_alpha(256)  # Semi-transparent
#     background_surface.fill((0, 0, 0))  # Black background
#
#     window.blit(background_surface, background_rect)
#     window.blit(text_surface, text_rect)


def render_target_labels(window, env, gameboard_size):
    """
    Render target index and coordinates next to each target

    Args:
        window: pygame display surface
        env: The environment instance
        gameboard_size: Size of the gameboard
    """
    font = pygame.font.SysFont(None, 16)
    map_half_size = gameboard_size / 2

    for i in range(env.num_targets):
        target = env.targets[i]
        target_id = int(target[0])  # Target index
        target_x = target[3]  # Game coordinates
        target_y = target[4]  # Game coordinates

        # Convert game coordinates to screen coordinates
        screen_x = target_x + map_half_size
        screen_y = target_y + map_half_size

        # Create label text
        label_text = f"{target_id}: ({target_x:.0f},{target_y:.0f})"
        text_surface = font.render(label_text, True, (0, 0, 0))  # Black text

        # Position label slightly offset from target center
        label_x = screen_x + 15  # Offset to the right
        label_y = screen_y - 10  # Offset upward

        # Draw white background for better readability
        text_rect = text_surface.get_rect()
        text_rect.topleft = (label_x, label_y)
        background_rect = text_rect.copy()
        background_rect.inflate(4, 2)  # Small padding

        pygame.draw.rect(window, (255, 255, 255), background_rect)
        pygame.draw.rect(window, (0, 0, 0), background_rect, 1)  # Black border
        window.blit(text_surface, text_rect)


def render_current_action(window, current_action, gameboard_size):
    """
    Render the current action in the corner of the screen

    Args:
        window: pygame display surface
        current_action: Current action array
        gameboard_size: Size of the gameboard
    """
    font = pygame.font.SysFont(None, 20)
    action_text = f"Action: ({current_action[0]:.2f}, {current_action[1]:.2f})"
    text_surface = font.render(action_text, True, (255, 255, 255))  # White text

    # Position in bottom-left corner
    text_rect = text_surface.get_rect()
    text_rect.bottomleft = (10, gameboard_size + 45)

    # Draw semi-transparent background
    background_rect = text_rect.copy()
    background_rect.inflate(20, 10)
    background_surface = pygame.Surface((background_rect.width, background_rect.height))
    background_surface.set_alpha(256)
    background_surface.fill((0, 0, 0))

    window.blit(background_surface, background_rect)
    window.blit(text_surface, text_rect)

def main():
    
    pygame.init()
    clock = pygame.time.Clock()
    
    # Disable display scaling for high-res monitors
    ctypes.windll.user32.SetProcessDPIAware()
    
    # Load environment configuration
    #config_file = './config_files/humantest_config.json'
    config_file = '../config_files/rl_simpleoar.json'
    env_config = load_env_config(config_file)
    
    # Override config for human play
    env_config['obs_type'] = 'absolute'
    env_config['action_type'] = 'continuous-normalized'
    
    # Set up pygame window
    window_width, window_height = env_config['window_size'][0], env_config['window_size'][1]
    gameboard_size = env_config['gameboard_size']
    
    # Position window (from config.py)
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"
    window = pygame.display.set_mode((window_width, window_height), flags=pygame.NOFRAME)
    pygame.display.set_caption("MAISR Human Interface")
    
    # Create environment
    env = MAISREnvVec(
        config=env_config,
        clock=clock,
        window=window,
        render_mode='human',
        num_agents=1,
        run_name='human_player',
        tag='human_test'
    )
    
    print("Environment created. Click on the map to set waypoints!")
    print("Close the window or press ESC to exit.")
    
    running = True
    current_action = None
    
    while running:
        # Reset environment for new episode
        observation, info = env.reset()
        episode_reward = 0.0
        terminated, truncated = False, False
        current_action = np.array([0.0, 0.0], dtype=np.float32)

        cumulative_reward_components = {
            'high val target id': 0.0,
            'regular val target id': 0.0,
            'proximity': 0.0,
            'early finish': 0.0,
            'waypoint-to-nearest': 0.0
        }
        
        print(f"\n--- New Episode Started ---")
        
        # Episode loop
        while not (terminated or truncated) and running:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        mouse_pos = pygame.mouse.get_pos()
                        
                        # Only process clicks within the gameboard area
                        if 0 <= mouse_pos[0] <= gameboard_size and 0 <= mouse_pos[1] <= gameboard_size:
                            # Convert mouse position to game coordinates
                            game_coords = convert_mouse_to_game_coords(mouse_pos, gameboard_size)
                            
                            # Convert to action space
                            current_action = convert_game_to_action_space(game_coords, gameboard_size)
                            
                            print(f"Waypoint set: Mouse({mouse_pos[0]:.0f}, {mouse_pos[1]:.0f}) -> "
                                  f"Game({game_coords[0]:.1f}, {game_coords[1]:.1f}) -> "
                                  f"Action({current_action[0]:.3f}, {current_action[1]:.3f})")
            
            if not running:
                break
            
            action = current_action
            
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            # Update cumulative reward components
            step_reward_components = info.get('reward_components', {})
            for component, value in step_reward_components.items():
                if component in cumulative_reward_components:
                    if component == 'high val target id':
                        cumulative_reward_components[component] += value * env_config['highqual_highvaltarget_reward']
                    elif component == 'regular val target id':
                        cumulative_reward_components[component] += value * env_config['highqual_regulartarget_reward']
                    else:
                        cumulative_reward_components[component] += value
            # step_reward_components = info.get('reward_components', {})
            # for component, value in step_reward_components.items():
            #     if component in cumulative_reward_components:
            #         cumulative_reward_components[component] += value
            #
            env.render()
            
            # Render episode reward on top of the game
            #render_reward_text(window, episode_reward, gameboard_size)

            step_reward_components = info.get('reward_components', {})
            render_reward_text(window, episode_reward, step_reward_components, cumulative_reward_components,gameboard_size, env_config)

            render_target_labels(window, env, gameboard_size)
            render_current_action(window, current_action, gameboard_size)
            
            pygame.display.flip()
            clock.tick(10)  # 60 FPS
        
        if running:
            print(f"Episode finished! Episode reward: {episode_reward:.2f}")
            
            # Brief pause before next episode
            pygame.time.wait(10000)
    
    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()