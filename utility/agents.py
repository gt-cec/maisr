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
            pygame.draw.circle(window, self.color, (int(screen_x), int(screen_y)), self.env.AIRCRAFT_ENGAGEMENT_RADIUS,
                               2)

            # Draw waypoint line and marker
            # if self.target_point is not None: # TODO temporarily removed for CNN observations
            #     if self.show_agent_waypoint >= 1:
            #         # Convert target point to screen coordinates
            #         target_screen_x = self.target_point[0] + map_half_size
            #         target_screen_y = self.target_point[1] + map_half_size
            #
            #         pygame.draw.line(window, (0, 0, 0), (screen_x, screen_y), (target_screen_x, target_screen_y), 2)
            #         pygame.draw.rect(window, self.color, pygame.Rect(target_screen_x - 5, target_screen_y - 5, 10, 10))

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

        dx, dy = self.target_point[0] - self.x, self.target_point[1] - self.y
        self.direction = math.atan2(dy, dx)
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