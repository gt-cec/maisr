# agents.py: includes classes for the aircraft and ship agents

import pygame  # for rendering
import math  # for math functions
import random  # for random number generation
import numpy as np


# generic agent class
class Agent:
    def __init__(self, env, initial_direction=0, color=(0,0,0), scale=1, speed=1, agent_class="agent",policy=None):
        self.env = env
        self.agent_idx = len(env.agents)
        env.agents.append(self)  # add agent to the environment

        # Convert to centered coordinate system [-150, +150]
        map_half_size = env.config['gameboard_size'] / 2
        margin = env.config['gameboard_size'] * 0.03

        self.x = random.uniform(-map_half_size + margin, map_half_size - margin)
        self.y = random.uniform(-map_half_size + margin, map_half_size - margin)

        self.direction = 0
        self.initial_direction = initial_direction
        self.agent_class = agent_class
        self.color = color
        self.scale = scale  # scale of the agent's sprite
        self.width = 10 * self.scale
        self.speed = speed
        self.path = []
        self.waypoint_override = None  # if an extra waypoint is specified, agent will prioritize it
        self.target_point = None

    # draws the agent
    def draw(self, window, color_override=None):
        # Convert from centered coordinates to screen coordinates
        map_half_size = self.env.config['gameboard_size'] / 2
        screen_x = self.x + map_half_size
        screen_y = self.y + map_half_size
        pygame.draw.circle(window, self.color if color_override is None else color_override,
                           (int(screen_x), int(screen_y)), self.width)

    # move the agent towards the next waypoint
    def move(self):


        if self.waypoint_override is not None:
            self.target_point = self.waypoint_override
        else:
            self.target_point = (self.x,self.y) # Temporary hack, should loiter in place.

        dx, dy = self.target_point[0] - self.x, self.target_point[1] - self.y

        self.direction = math.atan2(dy, dx)
        dist = math.hypot(dx, dy)

        if dist > self.speed:  # threshold for reaching the waypoint location
            dx, dy = dx / dist, dy / dist
            self.x += dx * self.speed
            self.y += dy * self.speed


        else:
            self.x = self.target_point[0]
            self.y = self.target_point[1]
            # if at the agent-overriding waypoint, remove it
            if self.waypoint_override is not None:
                self.waypoint_override = None
            else:  # if at the path waypoint, remove it
                del self.path[0]
        return

    def distance(self, agent):
        return math.hypot(self.x - agent.x, self.y - agent.y)

# aircraft agent class
class Aircraft(Agent):
    def __init__(self, env, direction, color, speed=1, scale=1,max_health=10, flight_pattern="none", policy=None,is_visible=True):
        super().__init__(env, direction, color, scale, speed, agent_class="aircraft")
        #self.damage = 0  # damage taken by the aircraft
        self.max_health = max_health
        self.health_points = max_health
        self.speed = speed
        self.env.aircraft_ids.append(self.agent_idx)
        self.alive = True
        self.is_visible = is_visible
        self.show_agent_waypoint = env.show_agent_waypoint
        self.regroup_clicked = False
        #self.base_speed = speed

        self.smoothed_direction = 0.0  # Smoothed direction in radians
        self.direction_smoothing_factor = 0.02  # Lower = more smoothing
        self.smoothed_waypoint = None  # Smoothed waypoint position
        self.waypoint_smoothing_factor = 0.01  # Separate factor for waypoint smoothing


    def draw(self, window):
        if self.is_visible:
            # Convert from centered coordinates to screen coordinates for rendering
            map_half_size = self.env.config['gameboard_size'] / 2
            screen_x = self.x + map_half_size
            screen_y = self.y + map_half_size

            # Calculate all points using screen coordinates
            nose_point = (screen_x + math.cos(self.direction) * self.env.AIRCRAFT_NOSE_LENGTH,
                          screen_y + math.sin(self.direction) * self.env.AIRCRAFT_NOSE_LENGTH)
            tail_point = (screen_x - math.cos(self.direction) * self.env.AIRCRAFT_TAIL_LENGTH,
                          screen_y - math.sin(self.direction) * self.env.AIRCRAFT_TAIL_LENGTH)

            left_wingtip_point = (screen_x - math.cos(self.direction - math.pi / 2) * self.env.AIRCRAFT_WING_LENGTH,
                                  screen_y - math.sin(self.direction - math.pi / 2) * self.env.AIRCRAFT_WING_LENGTH)
            right_wingtip_point = (screen_x + math.cos(self.direction - math.pi / 2) * self.env.AIRCRAFT_WING_LENGTH,
                                   screen_y + math.sin(self.direction - math.pi / 2) * self.env.AIRCRAFT_WING_LENGTH)

            left_tail_point = (tail_point[0] - math.cos(self.direction - math.pi / 2) * self.env.AIRCRAFT_TAIL_WIDTH,
                               tail_point[1] - math.sin(self.direction - math.pi / 2) * self.env.AIRCRAFT_TAIL_WIDTH)
            right_tail_point = (tail_point[0] + math.cos(self.direction - math.pi / 2) * self.env.AIRCRAFT_TAIL_WIDTH,
                                tail_point[1] + math.sin(self.direction - math.pi / 2) * self.env.AIRCRAFT_TAIL_WIDTH)

            # Draw all the aircraft components
            pygame.draw.line(window, self.color, tail_point, nose_point, self.env.AIRCRAFT_LINE_WIDTH)
            pygame.draw.circle(window, self.color, nose_point, self.env.AIRCRAFT_LINE_WIDTH // 2)
            pygame.draw.line(window, self.color, left_tail_point, right_tail_point, self.env.AIRCRAFT_LINE_WIDTH)
            pygame.draw.circle(window, self.color, left_tail_point, self.env.AIRCRAFT_LINE_WIDTH // 2)
            pygame.draw.circle(window, self.color, right_tail_point, self.env.AIRCRAFT_LINE_WIDTH // 2)
            pygame.draw.line(window, self.color, left_wingtip_point, right_wingtip_point, self.env.AIRCRAFT_LINE_WIDTH)
            pygame.draw.circle(window, self.color, left_wingtip_point, self.env.AIRCRAFT_LINE_WIDTH // 2)
            pygame.draw.circle(window, self.color, right_wingtip_point, self.env.AIRCRAFT_LINE_WIDTH // 2)

            # Draw the engagement radius (using screen coordinates)
            pygame.draw.circle(window, self.color, (int(screen_x), int(screen_y)), self.env.AIRCRAFT_ENGAGEMENT_RADIUS,2)

            # Draw waypoint line and marker
            # Draw waypoint line and marker
            if self.target_point is not None:
                if self.show_agent_waypoint >= 1:
                    # Use smoothed waypoint for rendering if available, otherwise fall back to target_point
                    waypoint_to_draw = self.smoothed_waypoint if self.smoothed_waypoint is not None else self.target_point

                    # Calculate direction from aircraft to smoothed waypoint
                    dx = waypoint_to_draw[0] - self.x
                    dy = waypoint_to_draw[1] - self.y
                    distance = math.hypot(dx, dy)

                    if distance > 0:
                        # Extend the line to a fixed length (e.g., same as original target distance)
                        original_distance = math.hypot(self.target_point[0] - self.x, self.target_point[1] - self.y)
                        extension_factor = max(1.0, original_distance / distance) if distance > 0 else 1.0

                        extended_x = self.x + (dx / distance) * original_distance
                        extended_y = self.y + (dy / distance) * original_distance

                        # Convert to screen coordinates
                        target_screen_x = extended_x + map_half_size
                        target_screen_y = extended_y + map_half_size
                    else:
                        # Fallback if distance is zero
                        target_screen_x = waypoint_to_draw[0] + map_half_size
                        target_screen_y = waypoint_to_draw[1] + map_half_size

                    pygame.draw.line(window, (0, 0, 0), (screen_x, screen_y), (target_screen_x, target_screen_y), 2)
                    pygame.draw.rect(window, self.color, pygame.Rect(target_screen_x - 5, target_screen_y - 5, 10, 10))


    def draw_damage(self):
        target_rect = pygame.Rect((self.x, self.y), (0, 0)).inflate((self.env.AIRCRAFT_ISR_RADIUS * 2, self.env.AIRCRAFT_ISR_RADIUS * 2))
        shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        semicircle_points = [(self.env.AIRCRAFT_ISR_RADIUS + math.cos(self.direction + math.pi * i / 180) * self.env.AIRCRAFT_ISR_RADIUS, self.env.AIRCRAFT_ISR_RADIUS + math.sin(self.direction + math.pi * i / 180) * self.env.AIRCRAFT_ISR_RADIUS) for i in range(-90, 90 + 10, 10)]
        pygame.draw.polygon(shape_surf, (255,0,0), semicircle_points)

    def move(self):
        if self.waypoint_override is not None:
            self.target_point = self.waypoint_override
        else:
            self.target_point = (self.x, self.y)  # Temporary hack, should loiter in place.

        # Smooth the waypoint for rendering
        if self.target_point is not None:
            if self.smoothed_waypoint is None:
                # Initialize smoothed waypoint on first frame
                self.smoothed_waypoint = self.target_point
            else:
                # Smooth the waypoint position
                current_x, current_y = self.smoothed_waypoint
                target_x, target_y = self.target_point

                # Apply smoothing to both x and y coordinates
                smooth_x = current_x + (target_x - current_x) * self.waypoint_smoothing_factor
                smooth_y = current_y + (target_y - current_y) * self.waypoint_smoothing_factor

                self.smoothed_waypoint = (smooth_x, smooth_y)

        dx, dy = self.target_point[0] - self.x, self.target_point[1] - self.y

        # Calculate the raw direction from movement
        if dx != 0 or dy != 0:
            raw_direction = math.atan2(dy, dx)

            # Smooth the direction using exponential moving average
            # Handle angle wrapping (shortest angular distance)
            angle_diff = raw_direction - self.smoothed_direction
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            elif angle_diff < -math.pi:
                angle_diff += 2 * math.pi

            self.smoothed_direction += angle_diff * self.direction_smoothing_factor

            # Use smoothed direction for rendering
            self.direction = self.smoothed_direction

        dist = math.hypot(dx, dy)

        if dist > self.speed:  # threshold for reaching the waypoint location
            dx, dy = dx / dist, dy / dist
            new_x = self.x + dx * self.speed
            new_y = self.y + dy * self.speed

            # Ensure agent stays within centered coordinate bounds
            map_half_size = self.env.config['gameboard_size'] / 2
            self.x = np.clip(new_x, -map_half_size, map_half_size)
            self.y = np.clip(new_y, -map_half_size, map_half_size)
        else:
            # Ensure target point is within bounds when reaching it
            map_half_size = self.env.config['gameboard_size'] / 2
            self.x = np.clip(self.target_point[0], -map_half_size, map_half_size)
            self.y = np.clip(self.target_point[1], -map_half_size, map_half_size)

            # if at the agent-overriding waypoint, remove it
            if self.waypoint_override is not None:
                self.waypoint_override = None
            else:  # if at the path waypoint, remove it
                if self.path:  # Check if path exists before deleting
                    del self.path[0]
        return

    # def move(self):
    #     if self.waypoint_override is not None:
    #         self.target_point = self.waypoint_override
    #     else:
    #         self.target_point = (self.x, self.y)  # Temporary hack, should loiter in place.
    #
    #     dx, dy = self.target_point[0] - self.x, self.target_point[1] - self.y
    #     self.direction = math.atan2(dy, dx)
    #     dist = math.hypot(dx, dy)
    #
    #     if dist > self.speed:  # threshold for reaching the waypoint location
    #         dx, dy = dx / dist, dy / dist
    #         new_x = self.x + dx * self.speed
    #         new_y = self.y + dy * self.speed
    #
    #         # Ensure agent stays within centered coordinate bounds
    #         map_half_size = self.env.config['gameboard_size'] / 2
    #         self.x = np.clip(new_x, -map_half_size, map_half_size)
    #         self.y = np.clip(new_y, -map_half_size, map_half_size)
    #     else:
    #         # Ensure target point is within bounds when reaching it
    #         map_half_size = self.env.config['gameboard_size'] / 2
    #         self.x = np.clip(self.target_point[0], -map_half_size, map_half_size)
    #         self.y = np.clip(self.target_point[1], -map_half_size, map_half_size)
    #
    #         # if at the agent-overriding waypoint, remove it
    #         if self.waypoint_override is not None:
    #             self.waypoint_override = None
    #         else:  # if at the path waypoint, remove it
    #             if self.path:  # Check if path exists before deleting
    #                 del self.path[0]
    #     return