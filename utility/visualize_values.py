import math
import pygame


def get_directional_potential_gains(env, current_obs):
    """Calculate potential GAINS for all 8 possible actions from current state"""
    gains = {}

    # Get current potential
    current_potential = env.get_potential(current_obs)

    # Store current state
    original_agents = []
    for agent in env.agents:
        if hasattr(agent, 'x') and hasattr(agent, 'y'):
            original_agents.append((agent.x, agent.y))

    # Test each action direction
    for action in range(8):
        # Get the waypoint for this action
        waypoint = env.process_action(action)

        # Temporarily move agent to see what the potential would be
        if env.agents and len(env.aircraft_ids) > 0:
            agent = env.agents[env.aircraft_ids[0]]
            original_x, original_y = agent.x, agent.y

            # Move agent temporarily
            agent.x, agent.y = waypoint[0], waypoint[1]

            # Get new observation and potential
            temp_obs = env.get_observation()
            new_potential = env.get_potential(temp_obs)

            # Calculate gain (new - current)
            potential_gain = new_potential - current_potential
            gains[action] = potential_gain

            # Restore original position
            agent.x, agent.y = original_x, original_y

    return gains


def draw_value_arrows(window, env, potential_gains, agent_x, agent_y, map_half_size):
    """Draw arrows showing potential GAIN of each direction from agent position"""

    # Convert agent position to screen coordinates
    screen_x = agent_x + map_half_size
    screen_y = agent_y + map_half_size

    # Direction vectors for 8 actions (same as in process_action)
    direction_map = {
        0: (0, 1),  # up
        1: (1, 1),  # up-right
        2: (1, 0),  # right
        3: (1, -1),  # down-right
        4: (0, -1),  # down
        5: (-1, -1),  # down-left
        6: (-1, 0),  # left
        7: (-1, 1)  # up-left
    }

    # Find min/max gains for scaling
    if potential_gains:
        min_gain = min(potential_gains.values())
        max_gain = max(potential_gains.values())
        max_abs_gain = max(abs(min_gain), abs(max_gain))

        if max_abs_gain == 0:
            max_abs_gain = 1  # Avoid division by zero

    # Draw arrows for each direction
    for action, gain in potential_gains.items():
        dx, dy = direction_map[action]

        # Normalize direction
        length = math.sqrt(dx * dx + dy * dy)
        dx_norm = dx / length
        dy_norm = dy / length

        # Scale arrow length based on absolute gain (10-50 pixels)
        if max_abs_gain > 0:
            normalized_gain = abs(gain) / max_abs_gain
        else:
            normalized_gain = 0

        arrow_length = 15 + (normalized_gain * 35)

        # Calculate arrow end point
        end_x = screen_x + (dx_norm * arrow_length)
        end_y = screen_y - (dy_norm * arrow_length)  # Flip Y for pygame coordinates

        # Color based on gain value (green=positive gain, red=negative gain)
        if gain > 0:
            # Green intensity based on gain magnitude
            intensity = min(255, int(255 * (gain / max_abs_gain))) if max_abs_gain > 0 else 128
            color = (0, intensity, 0)
        elif gain < 0:
            # Red intensity based on gain magnitude
            intensity = min(255, int(255 * (abs(gain) / max_abs_gain))) if max_abs_gain > 0 else 128
            color = (intensity, 0, 0)
        else:
            # Neutral (no gain)
            color = (128, 128, 128)

        # Draw arrow line
        pygame.draw.line(window, color, (screen_x, screen_y), (end_x, end_y), 3)

        # Draw arrowhead
        arrow_size = 8
        # Calculate arrowhead points
        angle = math.atan2(-dy_norm, dx_norm)  # Flip Y for pygame
        left_angle = angle + 2.5
        right_angle = angle - 2.5

        left_x = end_x - arrow_size * math.cos(left_angle)
        left_y = end_y - arrow_size * math.sin(left_angle)

        font = pygame.font.Font(None, 18)
        gain_text = f"{gain:+.1f}"  # Show + or - sign
        text = font.render(gain_text, True, color)
        text_x = end_x + dx_norm * 15
        text_y = end_y - dy_norm * 15
        window.blit(text, (text_x - text.get_width() // 2, text_y - text.get_height() // 2))