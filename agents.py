# agents.py: includes classes for the aircraft and ship agents

import pygame  # for rendering
import math  # for math functions
import random  # for random number generation


# generic agent class
class Agent:
    def __init__(self, env, initial_direction=0, color=(0,0,0), scale=1, speed=1, agent_class="agent"):
        self.env = env
        self.agent_idx = len(env.agents)
        env.agents.append(self)  # add agent to the environment
        self.x = random.randint(0, env.config["gameboard size"])
        self.y = random.randint(0, env.config["gameboard size"])
        self.direction = 0
        self.initial_direction = initial_direction
        self.agent_class = agent_class
        self.color = color
        self.scale = scale  # scale of the agent's sprite
        self.width = 10 * self.scale
        self.speed = speed
        self.path = []
        self.waypoint_override = None  # if an extra waypoint is specified, agent will prioritize it

    # draws the agent
    def draw(self, window, color_override=None):
        pygame.draw.circle(window, self.color if color_override is None else color_override, (self.x, self.y), self.width)

    # move the agent towards the next waypoint
    def move(self):
        if self.waypoint_override is not None:
            target_point = self.waypoint_override
        elif self.path != []:
            target_point = self.path[0]
        else:
            return

        dx, dy = target_point[0] - self.x, target_point[1] - self.y
        dist = math.hypot(dx, dy)

        if dist > self.speed:  # threshold for reaching the waypoint location
            dx, dy = dx / dist, dy / dist
            self.x += dx * self.speed
            self.y += dy * self.speed
        else:
            self.x = target_point[0]
            self.y = target_point[1]
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
    def __init__(self, env, direction, color, speed=1, scale=1, flight_pattern="none"):
        super().__init__(env, direction, color, scale, speed, agent_class="aircraft")
        self.damage = 0  # damage taken by the aircraft
        self.flight_pattern = flight_pattern
        self.env.aircraft_ids.append(self.agent_idx)

    def draw(self, window):
        if len(self.path) > 0:
            self.direction = math.atan2(self.path[0][1] - self.y, self.path[0][0] - self.x)
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
        pygame.draw.circle(window, self.color, (self.x, self.y), self.env.AIRCRAFT_ENGAGEMENT_RADIUS, 2)
        # draw the ISR radius
        target_rect = pygame.Rect((self.x, self.y), (0, 0)).inflate((self.env.AIRCRAFT_ISR_RADIUS * 2, self.env.AIRCRAFT_ISR_RADIUS * 2))
        shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        semicircle_points = [(self.env.AIRCRAFT_ISR_RADIUS + math.cos(self.direction + math.pi * i / 180) * self.env.AIRCRAFT_ISR_RADIUS, self.env.AIRCRAFT_ISR_RADIUS + math.sin(self.direction + math.pi * i / 180) * self.env.AIRCRAFT_ISR_RADIUS) for i in range(-90, 90+10, 10)]
        pygame.draw.polygon(shape_surf, self.color + (30,), semicircle_points)
        window.blit(shape_surf, target_rect)

    # check if another agent is in the ISR range
    def in_isr_range(self, agent=None, distance=None) -> bool:
        if distance is None and agent is None:
            raise ValueError("Either distance or agent must be provided")
        return (math.hypot(agent.x - self.x, agent.y - self.y) if distance is None else distance) <= self.env.AIRCRAFT_ISR_RADIUS

    # check if another agent is in the engagement range
    def in_engagement_range(self, agent=None, distance=None) -> bool:
        if distance is None and agent is None:
            raise ValueError("Either distance or agent must be provided")
        return (math.hypot(agent.x - self.x, agent.y - self.y) if distance is None else distance) <= self.env.AIRCRAFT_ENGAGEMENT_RADIUS

    # check the waypoints and flight path
    def move(self):
        # if the path is empty, generate using the flight pattern
        if self.path == []:
            if self.flight_pattern in self.env.FLIGHTPLANS:
                for waypoint in self.env.FLIGHTPLANS[self.flight_pattern]:
                    self.path.append(self.__flightplan_proportion_to_gameboard__(waypoint[0], waypoint[1]))
            else:
                raise ValueError(f"Flight pattern ({self.flight_pattern}) is not defined in env.py!")

        super().move()

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
        # set threat level of the ship
        if threat != -1:
            self.threat = threat
        else:
            self.threat = random.randint(0, len(env.AGENT_THREAT_RADIUS) - 1)

        if self.threat > 0:  # set color by threat level
            self.color = self.env.AGENT_COLOR_THREAT
        else:
            self.color = self.env.AGENT_COLOR_OBSERVED
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
        super().draw(window, color_override=self.env.AGENT_COLOR_UNOBSERVED if not self.observed else self.color)
        # draw a red circle around the ship if it is a threat
        if self.observed_threat and self.threat > 0:
            threat_radius = self.width * self.env.AGENT_THREAT_RADIUS[self.threat]
            pygame.draw.circle(window, self.env.AGENT_COLOR_THREAT, (self.x, self.y), threat_radius * self.scale, 2)

    def move(self):
        if self.path == []:
            self.add_random_waypoint()
        super().move()

    def add_random_waypoint(self):
        self.path.append((random.randint(0, self.env.config["gameboard size"]), random.randint(0, self.env.config["gameboard size"])))

    def in_weapon_range(self, agent=None, distance=None):
        if distance is None and agent is None:
            raise ValueError("Either distance or agent must be provided")
        return (math.hypot(agent.x - self.x, agent.y - self.y) if distance is None else distance) <= self.env.AIRCRAFT_ENGAGEMENT_RADIUS

def target_id_policy(env,aircraft_id,quadrant='full'):
    """ Basic rule-based action policy for tactical HAI ISR project.
    Inputs:
        * Env: Game environment
        * aircraft_id: ID of the aircraft being moved
        * quadrant: Specifies whether agent is restricted to search in a specific map quadrant. Default is 'full' (all quadrants allowed). Alternatives are 'NW', 'NE', 'SW', 'SE' as strings.

    Returns: Waypoint to the nearest unknown target"""
    # TODO: Currently set to full game window, not just inside the green bounds (10% to 90% of gameboard size). I want to make the green bounds easily configurable before i include them here.
    gameboard_size = env.config["gameboard size"]
    quadrant_bounds = {'full':(0,gameboard_size,0,gameboard_size), 'NW':(0,gameboard_size*0.5,0,gameboard_size*0.5),'NE':(gameboard_size*0.5,gameboard_size,0,gameboard_size*0.5),'SW':(gameboard_size*0.5,gameboard_size,0,gameboard_size*0.5),'SE':(gameboard_size*0.5,gameboard_size,gameboard_size*0.5,gameboard_size)} # specifies (Min x, max x, min y, max y)

    current_state = env.get_state()
    current_target_distances = {} # Will be {agent_idx:distance}
    for ship_id in current_state['ships']:
        # Loops through all ships in the environment, calculates distance from current aircraft position, finds the
        # closest ship, and sets aircraft waypoint to that ship's location.
        if current_state['ships'][ship_id]['observed'] == False and (quadrant_bounds[quadrant][0] <= current_state['ships'][ship_id]['position'][0] <= quadrant_bounds[quadrant][1]) and (quadrant_bounds[quadrant][2] <= current_state['ships'][ship_id]['position'][1] <= quadrant_bounds[quadrant][3]):
            dist = math.hypot(env.agents[aircraft_id].x - env.agents[ship_id].x, env.agents[aircraft_id].y - env.agents[ship_id].y)
            current_target_distances[ship_id] = dist

    if current_target_distances:
        nearest_target_id = min(current_target_distances, key=current_target_distances.get)
        target_waypoint = tuple((env.agents[nearest_target_id].x, env.agents[nearest_target_id].y))
        print('Nearest unknown target is %s. Setting waypoint to %s' % (nearest_target_id, target_waypoint))
    else:
        target_waypoint = (gameboard_size*0.5,gameboard_size*0.5) # If no more targets, return to center of game board TODO: Make this more robust
    #target_direction = math.atan2(target_waypoint[1] - env.agents[aircraft_id].y, target_waypoint[0] - env.agents[aircraft_id].x)
    target_direction = math.atan2(target_waypoint[1] - env.agents[aircraft_id].y,
                                  target_waypoint[0] - env.agents[aircraft_id].x)
    return target_waypoint, target_direction

def wez_id_policy():
    pass
def mouse_waypoint_policy():
    pass