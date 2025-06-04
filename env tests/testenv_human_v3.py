import pygame
import sys
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


def render_reward_text(window, episode_reward, gameboard_size):
    """
    Render the cumulative reward text on the screen
    
    Args:
        window: pygame display surface
        episode_reward: Current episode cumulative reward
        gameboard_size: Size of the gameboard
    """
    font = pygame.font.SysFont(None, 48)
    reward_text = f"Episode Reward: {episode_reward:.2f}"
    text_surface = font.render(reward_text, True, (255, 255, 255))  # White text
    
    # Position the text in the top-left corner with some padding
    text_rect = text_surface.get_rect()
    text_rect.topleft = (10, 10)
    
    # Draw a semi-transparent background for better readability
    background_rect = text_rect.copy()
    background_rect.inflate(20, 10)  # Add padding around text
    background_surface = pygame.Surface((background_rect.width, background_rect.height))
    background_surface.set_alpha(128)  # Semi-transparent
    background_surface.fill((0, 0, 0))  # Black background
    
    window.blit(background_surface, background_rect)
    window.blit(text_surface, text_rect)


def main():
    
    pygame.init()
    clock = pygame.time.Clock()
    
    # Disable display scaling for high-res monitors
    ctypes.windll.user32.SetProcessDPIAware()
    
    # Load environment configuration
    config_file = './config_files/rl_training_default.json'
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
            
            env.render()
            
            # Render episode reward on top of the game
            render_reward_text(window, episode_reward, gameboard_size)
            
            pygame.display.flip()
            clock.tick(60)  # 60 FPS
        
        if running:
            print(f"Episode finished! Episode reward: {episode_reward:.2f}")
            
            # Brief pause before next episode
            pygame.time.wait(200)
    
    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()