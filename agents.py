# agents.py: includes classes for the aircraft and ship agents

import pygame  # for rendering
import math  # for math functions
import random  # for random number generation
from autonomous_policy import AutonomousPolicy


# generic agent class
class Agent:
    def __init__(self, env, initial_direction=0, color=(0,0,0), scale=1, speed=1, agent_class="agent",policy=None):
        self.env = env
        self.agent_idx = len(env.agents)
        env.agents.append(self)  # add agent to the environment
        #self.x = random.randint(0, env.config["gameboard size"])
        #self.y = random.randint(0, env.config["gameboard size"])

        self.x = random.randint(env.config['gameboard border margin'], env.config['gameboard size'] - env.config["gameboard border margin"])
        self.y = random.randint(env.config['gameboard border margin'], env.config['gameboard size'] - env.config["gameboard border margin"])

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
        #self.policy = policy if policy else AutonomousPolicy

    # draws the agent
    def draw(self, window, color_override=None):
        pygame.draw.circle(window, self.color if color_override is None else color_override, (self.x, self.y), self.width)

    # move the agent towards the next waypoint
    def move(self):
        #if self.policy: # TODO testing
            #self.waypoint_override, self.direction = self.policy(self.env,self.agent_idx)

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
    def __init__(self, env, direction, color, speed=1, scale=1, prob_detect=0.1,max_health=4, flight_pattern="none", policy=None,is_visible=True):
        super().__init__(env, direction, color, scale, speed, agent_class="aircraft")
        self.damage = 0  # damage taken by the aircraft
        self.prob_detect = prob_detect # Probability of taking damage on a given timestep if inside a threat radius
        self.max_health = max_health
        self.health_points = max_health
        self.flight_pattern = flight_pattern
        self.env.aircraft_ids.append(self.agent_idx)
        self.policy = policy # Not implemented right now
        self.alive = True
        self.is_visible = is_visible
        self.show_agent_waypoint = env.show_agent_waypoint
        self.regroup_clicked = False
        self.base_speed = speed

    def draw(self, window):
        if not self.alive:
            return

        if self.is_visible:


            # draw the aircraft
            nose_point = (self.x + math.cos(self.direction) * self.env.AIRCRAFT_NOSE_LENGTH, self.y + math.sin(self.direction) * self.env.AIRCRAFT_NOSE_LENGTH)
            tail_point = (self.x - math.cos(self.direction) * self.env.AIRCRAFT_TAIL_LENGTH, self.y - math.sin(self.direction) * self.env.AIRCRAFT_TAIL_LENGTH)
            left_wingtip_point = (self.x - math.cos(self.direction - math.pi / 2) * self.env.AIRCRAFT_WING_LENGTH, self.y - math.sin(self.direction - math.pi / 2) * self.env.AIRCRAFT_WING_LENGTH)
            right_wingtip_point = (self.x + math.cos(self.direction - math.pi / 2) * self.env.AIRCRAFT_WING_LENGTH, self.y + math.sin(self.direction - math.pi / 2) * self.env.AIRCRAFT_WING_LENGTH)
            left_tail_point = (tail_point[0] - math.cos(self.direction - math.pi / 2) * self.env.AIRCRAFT_TAIL_WIDTH, tail_point[1] - math.sin(self.direction - math.pi / 2) * self.env.AIRCRAFT_TAIL_WIDTH)
            right_tail_point = (tail_point[0] + math.cos(self.direction - math.pi / 2) * self.env.AIRCRAFT_TAIL_WIDTH, tail_point[1] + math.sin(self.direction - math.pi / 2) * self.env.AIRCRAFT_TAIL_WIDTH)
            pygame.draw.line(window, self.color, tail_point, nose_point, self.env.AIRCRAFT_LINE_WIDTH)
            pygame.draw.circle(window, self.color, nose_point, self.env.AIRCRAFT_LINE_WIDTH / 2)
            pygame.draw.line(window, self.color, left_tail_point, right_tail_point, self.env.AIRCRAFT_LINE_WIDTH)
            pygame.draw.circle(window, self.color, left_tail_point, self.env.AIRCRAFT_LINE_WIDTH / 2)
            pygame.draw.circle(window, self.color, right_tail_point, self.env.AIRCRAFT_LINE_WIDTH / 2)
            pygame.draw.line(window, self.color, left_wingtip_point, right_wingtip_point, self.env.AIRCRAFT_LINE_WIDTH)
            pygame.draw.circle(window, self.color, left_wingtip_point, self.env.AIRCRAFT_LINE_WIDTH / 2)
            pygame.draw.circle(window, self.color, right_wingtip_point, self.env.AIRCRAFT_LINE_WIDTH / 2)
            # draw the engagement radius
            if not self.regroup_clicked: pygame.draw.circle(window, self.color, (self.x, self.y), self.env.AIRCRAFT_ENGAGEMENT_RADIUS, 2)
            # draw the ISR radius
            if not self.regroup_clicked:
                target_rect = pygame.Rect((self.x, self.y), (0, 0)).inflate((self.env.AIRCRAFT_ISR_RADIUS * 2, self.env.AIRCRAFT_ISR_RADIUS * 2))
                shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
                semicircle_points = [(self.env.AIRCRAFT_ISR_RADIUS + math.cos(self.direction + math.pi * i / 180) * self.env.AIRCRAFT_ISR_RADIUS, self.env.AIRCRAFT_ISR_RADIUS + math.sin(self.direction + math.pi * i / 180) * self.env.AIRCRAFT_ISR_RADIUS) for i in range(-90, 90+10, 10)]
                pygame.draw.polygon(shape_surf, self.color + (30,), semicircle_points)
                window.blit(shape_surf, target_rect)


            if self.target_point is not None:
                if self.show_agent_waypoint >= 1:
                    pygame.draw.line(window, (0, 0, 0), (self.x, self.y), (self.target_point[0], self.target_point[1]),2)  # Draw line from aircraft to waypoint
                    pygame.draw.rect(window, self.color, pygame.Rect(self.target_point[0]-5,self.target_point[1]-5,10,10)) # Draw box at waypoint location
                    #pygame.draw.rect(window, (0,0,0),pygame.Rect(self.target_point[0] - 3, self.target_point[1] - 3, 6, 6)) # Draw inner box at waypoint location


    # check if another agent is in the ISR range
    def in_isr_range(self, agent=None, distance=None) -> bool:
        if distance is None and agent is None:
            raise ValueError("Either distance or agent must be provided")
        return (not self.regroup_clicked) and ((math.hypot(agent.x - self.x, agent.y - self.y) if distance is None else distance) <= self.env.AIRCRAFT_ISR_RADIUS)

    # check if another agent is in the engagement range
    def in_engagement_range(self, agent=None, distance=None) -> bool:
        if distance is None and agent is None:
            raise ValueError("Either distance or agent must be provided")
        return (not self.regroup_clicked) and ((math.hypot(agent.x - self.x, agent.y - self.y) if distance is None else distance) <= self.env.AIRCRAFT_ENGAGEMENT_RADIUS)

    # check the waypoints and flight path
    def move(self):
        if not self.alive:
            return

        if self.damage >= 100 and self.env.config['infinite health'] == False:
            self.destroy()
            return

        if self.regroup_clicked: self.speed = self.base_speed * 2.5
        else: self.speed = self.base_speed

        if self.path == []: # Loiter in a holding pattern (TODO: Doesn't work)
            if self.direction >= 0:
                self.path.append((self.x, self.y))

        super().move()

    def destroy(self):
        if self.alive:
            print('Aircraft %s destroyed' % self.agent_idx)
            self.alive = False
            if self.agent_idx in self.env.aircraft_ids:
                self.env.aircraft_ids.remove(self.agent_idx)

    # utility function for convert x/y proportions to gameboard proportions
    def __flightplan_proportion_to_gameboard__(self, x, y):
        return (self.env.config["gameboard size"] * (x * (1 - 2 * self.env.FLIGHTPLAN_EDGE_MARGIN) + self.env.FLIGHTPLAN_EDGE_MARGIN), self.env.config["gameboard size"] * ((1 - y) * (1 - 2 * self.env.FLIGHTPLAN_EDGE_MARGIN) + self.env.FLIGHTPLAN_EDGE_MARGIN))


# ship agent class
class Ship(Agent):
    def __init__(self, env, direction=0, threat=-1, speed=-1, scale=1):
        super().__init__(env, direction, speed=speed, scale=scale, agent_class="ship")
        self.idx = -1  # ship index, can be the index of the agents array
        self.observed = False  # whether the ship has been observed
        self.observed_threat = False  # whether the threat level of the ship has been observed
        #self.neutral = False
        # set threat level of the ship
        if threat != -1:
            self.threat = threat
        else:
            #self.threat = random.randint(0, len(env.AGENT_THREAT_RADIUS) - 1)
            self.threat = random.choices([0, 1, 2, 3], weights=[0.60, 0.13, 0.13, 0.14])[0] # 60% chance of neutral, 40% chance of hostile

        if self.threat > 0:  # set color by threat level
            self.color = self.env.AGENT_COLOR_THREAT
        else:
            self.color = self.env.AGENT_COLOR_OBSERVED
            #self.neutral = True

        # generate the ship's speed
        if self.speed != -1:
            print("Note: Ship speed was manually specified.")
        elif "motion iteration" not in env.config:
            print("Note: Ship speed not provided and 'motion iteration' is not in the env config, defaulting to speed = 1")
            self.speed = 1
        else:
            if env.config["motion iteration"] == "F":
                self.speed = 0
            elif env.config["motion iteration"] == "G":
                self.speed = 5
            elif env.config["motion iteration"] == "H":
                self.speed = 10
            elif env.config["motion iteration"] == "I":
                self.speed = 15
            elif env.config["motion iteration"] == "J":
                r = random.random()
                if r <= 0.5:
                    self.speed = 0
                elif r <= 0.8:
                    self.speed = 5
                elif r <= 0.95:
                    self.speed = 10
                else:
                    self.speed = 15
            elif env.config["motion iteration"] == "K":
                r = random.random()
                if r <= 0.5:
                    self.speed = 15
                elif r <= 0.8:
                    self.speed = 10
                elif r <= 0.95:
                    self.speed = 5
                else:
                    self.speed = 0
        # generate the ship's waypoints
        self.add_random_waypoint()

    def draw(self, window):
        if not self.observed:
            display_color = self.env.AGENT_COLOR_UNOBSERVED
        else:
            if self.threat > 0:
                display_color = self.env.AGENT_COLOR_THREAT
            else:
                display_color = self.env.AGENT_COLOR_OBSERVED

        super().draw(window, color_override=self.env.AGENT_COLOR_UNOBSERVED if not self.observed else self.color) # TODO: The self.env here might be causing problems

        threat_radius = self.width * self.env.AGENT_THREAT_RADIUS[self.threat]
        possible_threat_radius = self.width * self.env.AGENT_THREAT_RADIUS[3]

        # Draw orange circle for any unidentified target (neutral or hostile)
        if not self.observed_threat or (self.observed_threat and self.threat == 0):
            pygame.draw.circle(window, self.env.AGENT_COLOR_UNOBSERVED,
                               (self.x, self.y),
                               possible_threat_radius * self.scale,
                               2)
        if self.observed_threat and self.threat == 0:
            pygame.draw.circle(window, (255,255,255),
                               (self.x, self.y),
                               possible_threat_radius * self.scale,
                               2)
        # If threat is observed and ship is hostile, show actual threat radius
        elif self.observed_threat and self.threat > 0:
            pygame.draw.circle(window, self.env.AGENT_COLOR_THREAT,(self.x, self.y),threat_radius * self.scale,2)


    def move(self):
        if self.path == []:
            self.add_random_waypoint()
        super().move()

    def add_random_waypoint(self):
        self.path.append((random.randint(0, self.env.config["gameboard size"]), random.randint(0, self.env.config["gameboard size"])))

    def in_weapon_range(self, agent=None, distance=None):
        if distance is None and agent is None:
            raise ValueError("Either distance or agent must be provided")
        return (math.hypot(agent.x - self.x,agent.y - self.y) if distance is None else distance) <= self.width * self.env.AGENT_THREAT_RADIUS[self.threat]

class Missile(Agent):
    def __init__(self, env, direction, color, speed=1, scale=1,max_health=4, flight_pattern="none", policy=None,is_visible=True, target_aircraft_id=None,prob_detect=0):
        super().__init__(env, direction, color, scale, speed, agent_class="missile")
        self.damage = 0  # damage taken by the aircraft
        self.max_health = max_health
        self.health_points = max_health
        self.flight_pattern = flight_pattern
        self.env.aircraft_ids.append(self.agent_idx)
        self.policy = policy # Not implemented right now
        self.alive = True
        self.is_visible = is_visible
        self.show_agent_waypoint = env.show_agent_waypoint
        self.regroup_clicked = False
        self.base_speed = speed
        self.target_aircraft_id = target_aircraft_id
        self.prob_detect = prob_detect
        self.spawn_time = env.display_time
        self.lifespan = 20000


    def draw(self, window):
        nose_point = (self.x + math.cos(self.direction) * self.env.AIRCRAFT_NOSE_LENGTH,self.y + math.sin(self.direction) * self.env.AIRCRAFT_NOSE_LENGTH)
        tail_point = (self.x - math.cos(self.direction) * self.env.AIRCRAFT_TAIL_LENGTH,
                      self.y - math.sin(self.direction) * self.env.AIRCRAFT_TAIL_LENGTH)
        if not self.alive: return
        if self.is_visible:
            if self.damage >= 100 and self.env.config['infinite health'] == False:
                self.destroy()
                pygame.draw.circle(window, self.color, nose_point, self.env.AIRCRAFT_LINE_WIDTH)
                return

            # draw the aircraft
            #left_wingtip_point = (self.x - math.cos(self.direction - math.pi / 2) * self.env.AIRCRAFT_WING_LENGTH, self.y - math.sin(self.direction - math.pi / 2) * self.env.AIRCRAFT_WING_LENGTH)
            #right_wingtip_point = (self.x + math.cos(self.direction - math.pi / 2) * self.env.AIRCRAFT_WING_LENGTH, self.y + math.sin(self.direction - math.pi / 2) * self.env.AIRCRAFT_WING_LENGTH)
            left_tail_point = (tail_point[0] - math.cos(self.direction - math.pi / 2) * self.env.AIRCRAFT_TAIL_WIDTH, tail_point[1] - math.sin(self.direction - math.pi / 2) * self.env.AIRCRAFT_TAIL_WIDTH)
            right_tail_point = (tail_point[0] + math.cos(self.direction - math.pi / 2) * self.env.AIRCRAFT_TAIL_WIDTH, tail_point[1] + math.sin(self.direction - math.pi / 2) * self.env.AIRCRAFT_TAIL_WIDTH)
            pygame.draw.line(window, self.color, tail_point, nose_point, self.env.AIRCRAFT_LINE_WIDTH)
            pygame.draw.circle(window, self.color, nose_point, self.env.AIRCRAFT_LINE_WIDTH / 2)
            pygame.draw.line(window, self.color, left_tail_point, right_tail_point, self.env.AIRCRAFT_LINE_WIDTH)

            #pygame.draw.circle(window, self.color, left_tail_point, self.env.AIRCRAFT_LINE_WIDTH / 2)
            #pygame.draw.circle(window, self.color, right_tail_point, self.env.AIRCRAFT_LINE_WIDTH / 2)
            #pygame.draw.line(window, self.color, left_wingtip_point, right_wingtip_point, self.env.AIRCRAFT_LINE_WIDTH)
            #pygame.draw.circle(window, self.color, left_wingtip_point, self.env.AIRCRAFT_LINE_WIDTH / 2)
            #pygame.draw.circle(window, self.color, right_wingtip_point, self.env.AIRCRAFT_LINE_WIDTH / 2)

            # draw the ISR radius
            # if not self.regroup_clicked:
            #     target_rect = pygame.Rect((self.x, self.y), (0, 0)).inflate((self.env.AIRCRAFT_ISR_RADIUS * 2, self.env.AIRCRAFT_ISR_RADIUS * 2))
            #     shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
            #     semicircle_points = [(self.env.AIRCRAFT_ISR_RADIUS + math.cos(self.direction + math.pi * i / 180) * self.env.AIRCRAFT_ISR_RADIUS, self.env.AIRCRAFT_ISR_RADIUS + math.sin(self.direction + math.pi * i / 180) * self.env.AIRCRAFT_ISR_RADIUS) for i in range(-90, 90+10, 10)]
            #     pygame.draw.polygon(shape_surf, self.color + (30,), semicircle_points)
            #     window.blit(shape_surf, target_rect)

            if self.target_point is not None:
                if self.show_agent_waypoint >= 1:
                    pygame.draw.line(window, (200, 0, 0), (self.x, self.y), (self.target_point[0], self.target_point[1]),2)  # Draw line from aircraft to waypoint
                    pygame.draw.rect(window, self.color, pygame.Rect(self.target_point[0]-5,self.target_point[1]-5,10,10)) # Draw box at waypoint location

    # check if another agent is in the ISR range
    def in_isr_range(self, agent=None, distance=None) -> bool:
        if distance is None and agent is None:
            raise ValueError("Either distance or agent must be provided")
        return (not self.regroup_clicked) and ((math.hypot(agent.x - self.x, agent.y - self.y) if distance is None else distance) <= self.env.AIRCRAFT_ISR_RADIUS)

    # check if another agent is in the engagement range
    def in_engagement_range(self, agent=None, distance=None) -> bool:
        if distance is None and agent is None:
            raise ValueError("Either distance or agent must be provided")
        return (not self.regroup_clicked) and ((math.hypot(agent.x - self.x, agent.y - self.y) if distance is None else distance) <= self.env.AIRCRAFT_ENGAGEMENT_RADIUS)

    # check the waypoints and flight path
    def move(self):
        if not self.alive:
            return

        if (self.env.display_time - self.spawn_time) > self.lifespan:
            self.destroy()
            return

        super().move()

    def destroy(self):
        if self.alive:
            print('Missile %s destroyed' % self.agent_idx)
            self.alive = False

            if self.agent_idx in self.env.aircraft_ids:
                self.env.aircraft_ids.remove(self.agent_idx)