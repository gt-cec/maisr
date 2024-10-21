# agents.py: includes classes for the aircraft and ship agents

import pygame  # for rendering
import math  # for math functions
import random  # for random number generation
import heapq

# TODO: Ryan changed target_point to self.target_point in all instances (so it can be accessed inside the draw() method. Change back if it causes problems.

# generic agent class
class Agent:
    def __init__(self, env, initial_direction=0, color=(0,0,0), scale=1, speed=1, agent_class="agent",policy=None): # TODO policy kwarg is under test
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
        self.policy = policy if policy else target_id_policy

    # draws the agent
    def draw(self, window, color_override=None):
        pygame.draw.circle(window, self.color if color_override is None else color_override, (self.x, self.y), self.width)

    # move the agent towards the next waypoint
    def move(self):
        if self.policy: # TODO testing
            self.waypoint_override, self.direction = self.policy(self.env,self.agent_idx)

        if self.waypoint_override is not None:
            self.target_point = self.waypoint_override
        else:
            self.target_point = (self.x,self.y) # Temporary hack, should loiter in place.
        #elif self.direction > 0: self.target_point = (self.x,self.y-100)
        #else: self.target_point = (self.x,self.y+100)

        #elif self.path != []: # Original code
        #    self.target_point = self.path[0]
        #else:
        #    return

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
    def __init__(self, env, direction, color, speed=1, scale=1, flight_pattern="none", policy=None):
        super().__init__(env, direction, color, scale, speed, agent_class="aircraft")
        self.damage = 0  # damage taken by the aircraft
        self.flight_pattern = flight_pattern
        self.env.aircraft_ids.append(self.agent_idx)
        self.policy = policy
        self.alive = True

    def draw(self, window):
        if self.damage >= 100: # TODO: This only stops rendering the aircraft. Need to stop its movement too
            if self.alive == True:
                self.alive = False
                print('Aircraft %s destroyed' % self.agent_idx)
            return

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

        if self.target_point is not None:
            pygame.draw.line(window, (0, 0, 0), (self.x, self.y), (self.target_point[0], self.target_point[1]),2)  # Draw line from aircraft to waypoint
            pygame.draw.rect(window, self.color, pygame.Rect(self.target_point[0]-5,self.target_point[1]-5,10,10)) # Draw box at waypoint location
            #pygame.draw.rect(window, (0,0,0),pygame.Rect(self.target_point[0] - 3, self.target_point[1] - 3, 6, 6)) # Draw inner box at waypoint location


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
        #print(self.direction)
        if self.path == []: # Loiter in a holding pattern (TODO: Doesn't work)
            if self.direction >= 0:
                self.path.append((self.x, self.y))
            #elif math.pi < self.direction < 2*math.pi:
          #  elif self.direction < 0:
          #      self.path.append((self.x,self.y))

            #if self.flight_pattern in self.env.FLIGHTPLANS:
             #   for waypoint in self.env.FLIGHTPLANS[self.flight_pattern]:
              #      self.path.append(self.__flightplan_proportion_to_gameboard__(waypoint[0], waypoint[1]))
            #else:
             #   raise ValueError(f"Flight pattern ({self.flight_pattern}) is not defined in env.py!")

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
        self.neutral = False
        # set threat level of the ship
        if threat != -1:
            self.threat = threat
        else:
            self.threat = random.randint(0, len(env.AGENT_THREAT_RADIUS) - 1)

        if self.threat > 0:  # set color by threat level
            self.color = self.env.AGENT_COLOR_THREAT
        else:
            self.color = self.env.AGENT_COLOR_OBSERVED
            self.neutral = True
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
        super().draw(window, color_override=self.env.AGENT_COLOR_UNOBSERVED if not self.observed else self.color) # TODO: The self.env here might be causing problems

        threat_radius = self.width * self.env.AGENT_THREAT_RADIUS[self.threat]
        possible_threat_radius = self.width * self.env.AGENT_THREAT_RADIUS[3]
        if not self.observed_threat: # Draw orange circle around unknown targets, representing the worst possible threat radius
            pygame.draw.circle(window, self.env.AGENT_COLOR_UNOBSERVED, (self.x, self.y), possible_threat_radius * self.scale, 2)

        if self.neutral:  # TODO Added as a hack to fix potential threat ring drawing. Not working properly yet -  doesn't delete the ring around neutral targets once ID'd
            pygame.draw.circle(window, self.env.AGENT_COLOR_UNOBSERVED, (self.x, self.y),possible_threat_radius * self.scale, 2)

        if self.observed_threat and self.threat > 0 and not self.neutral: # draw a red circle around the ship if it is a threat (TODO: Added self.neutral check
            #pygame.draw.circle(window, (255,255,255), (self.x, self.y),possible_threat_radius * self.scale, 2)
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
        return (math.hypot(agent.x - self.x,agent.y - self.y) if distance is None else distance) <= self.width * self.env.AGENT_THREAT_RADIUS[self.threat]

def target_id_policy(env,aircraft_id,quadrant='full', id_type='target'):
    """
    Basic rule-based action policy for tactical HAI ISR project.
    Inputs:
        * Env: Game environment
        * aircraft_id: ID of the aircraft being moved
        * quadrant: Specifies whether agent is restricted to search in a specific map quadrant. Default is 'full' (all quadrants allowed). Alternatives are 'NW', 'NE', 'SW', 'SE' as strings.
        * id_type: 'target' (only ID unknown targets but not unknown WEZs of known hostiles) or 'wez' (ID unknown WEZs)
    Returns: Waypoint to the nearest unknown target
    """
    gameboard_size = env.config["gameboard size"] # TODO: Currently set to full game window, not just inside the green bounds (10% to 90% of gameboard size)
    quadrant_bounds = {'full':(0,gameboard_size,0,gameboard_size), 'NW':(0,gameboard_size*0.5,0,gameboard_size*0.5),'NE':(gameboard_size*0.5,gameboard_size,0,gameboard_size*0.5),'SW':(0,gameboard_size*0.5,gameboard_size*0.5,gameboard_size),'SE':(gameboard_size*0.5,gameboard_size,gameboard_size*0.5,gameboard_size)} # specifies (Min x, max x, min y, max y)

    current_state = env.get_state()
    current_target_distances = {} # Will be {agent_idx:distance}
    for ship_id in current_state['ships']:
        # Loops through all ships in the environment, calculates distance from current aircraft position, finds the closest unknown ship (or unknown WEZ), and sets aircraft waypoint to that ship's location.
        if id_type == 'target': # If set to target, only consider unknown targets
            if current_state['ships'][ship_id]['observed'] == False and (quadrant_bounds[quadrant][0] <= current_state['ships'][ship_id]['position'][0] <= quadrant_bounds[quadrant][1]) and (quadrant_bounds[quadrant][2] <= current_state['ships'][ship_id]['position'][1] <= quadrant_bounds[quadrant][3]):
                dist = math.hypot(env.agents[aircraft_id].x - env.agents[ship_id].x, env.agents[aircraft_id].y - env.agents[ship_id].y)
                current_target_distances[ship_id] = dist
        elif id_type == 'wez': # If set to wez, consider unknown targets AND known hostiles with unknown threat rings
            if (current_state['ships'][ship_id]['observed'] == False or current_state['ships'][ship_id]['observed threat'] == False) and (quadrant_bounds[quadrant][0] <= current_state['ships'][ship_id]['position'][0] <= quadrant_bounds[quadrant][1]) and (quadrant_bounds[quadrant][2] <= current_state['ships'][ship_id]['position'][1] <= quadrant_bounds[quadrant][3]):
                dist = math.hypot(env.agents[aircraft_id].x - env.agents[ship_id].x,env.agents[aircraft_id].y - env.agents[ship_id].y)
                current_target_distances[ship_id] = dist

    if current_target_distances:
        nearest_target_id = min(current_target_distances, key=current_target_distances.get)
        target_waypoint = tuple((env.agents[nearest_target_id].x, env.agents[nearest_target_id].y))
        #print('Nearest unknown target is %s. Setting waypoint to %s' % (nearest_target_id, target_waypoint))
    else: # If all targets ID'd, loiter in center of board or specified quadrant
        if quadrant == 'full': target_waypoint = (gameboard_size*0.5,gameboard_size*0.5) # If no more targets, return to center of game board TODO: Make this more robust
        elif quadrant == 'NW': target_waypoint = (gameboard_size*0.25,gameboard_size*0.25)
        elif quadrant == 'NE': target_waypoint = (gameboard_size * 0.75, gameboard_size * 0.25)
        elif quadrant == 'SW': target_waypoint = (gameboard_size * 0.25, gameboard_size * 0.75)
        elif quadrant == 'SE': target_waypoint = (gameboard_size * 0.75, gameboard_size * 0.75)

    target_direction = math.atan2(target_waypoint[1] - env.agents[aircraft_id].y,target_waypoint[0] - env.agents[aircraft_id].x)
    return target_waypoint, target_direction

def hold_policy(env,aircraft_id,quadrant='full',id_type='target'):
    # Note: kwargs not currently used.
    target_waypoint = env.agents[aircraft_id].x, env.agents[aircraft_id].y
    target_direction = math.atan2(target_waypoint[1] - env.agents[aircraft_id].y,target_waypoint[0] - env.agents[aircraft_id].x)
    return target_waypoint, target_direction

def autonomous_policy(env,aircraft_id,quadrant='full',id_type='target'):
    """
    "Autonomous" policy that chooses waypoints based on what it deems to be best.
    If damage <= 50, attempts to identify closest...
    """
    gameboard_size = env.config["gameboard size"] # TODO: Currently set to full game window, not just inside the green bounds (10% to 90% of gameboard size)
    quadrant_bounds = {'full':(0,gameboard_size,0,gameboard_size), 'NW':(0,gameboard_size*0.5,0,gameboard_size*0.5),'NE':(gameboard_size*0.5,gameboard_size,0,gameboard_size*0.5),'SW':(0,gameboard_size*0.5,gameboard_size*0.5,gameboard_size),'SE':(gameboard_size*0.5,gameboard_size,gameboard_size*0.5,gameboard_size)} # specifies (Min x, max x, min y, max y)

    current_state = env.get_state()
    current_target_distances = {} # Will be {agent_idx:distance}

    # Decide whether to ID targets or targets + WEZs
    if env.agents[aircraft_id].damage <= 50:
        print('Autonomous policy prioritizing target+WEZ search')
        id_type = 'wez'
    else:
        print('Autonomous policy prioritizing target search')
        id_type = 'target'

    # Determine which quadrant has most unknown targets (TODO: Very inefficient, combine with other for loop below
    ship_quadrants = {'NW':0,'NE':0,'SW':0,'SE':0,'full':0}  # For counting how many current unknown ships in each quadrant
    for ship_id in current_state['ships']:
        if current_state['ships'][ship_id]['observed'] == False:
            if current_state['ships'][ship_id]['position'][0] <= gameboard_size*0.5 and current_state['ships'][ship_id]['position'][1] <= gameboard_size*0.5:
                ship_quadrants['NW'] += 1
            elif current_state['ships'][ship_id]['position'][0] <= gameboard_size*0.5 and gameboard_size*0.5 <= current_state['ships'][ship_id]['position'][1] <= gameboard_size:
                ship_quadrants['SW'] += 1
            elif gameboard_size*0.5 <= current_state['ships'][ship_id]['position'][0] <= gameboard_size and gameboard_size * 0.5 <= current_state['ships'][ship_id]['position'][1] <= gameboard_size:
                ship_quadrants['SE'] += 1
            elif gameboard_size*0.5 <= current_state['ships'][ship_id]['position'][0] <= gameboard_size and current_state['ships'][ship_id]['position'][1] <= gameboard_size*0.5:
                ship_quadrants['NE'] += 1

    new_densest_quadrant = max(ship_quadrants, key=ship_quadrants.get) # Set search quadrant to the one with the most unknown ships
    if ship_quadrants[new_densest_quadrant] > 3 + ship_quadrants[quadrant]:  # TODO: Bug: Spamming console because quadrant always re-initializes as 'full'. Need to fix.
        #densest_quadrant = new_densest_quadrant
        quadrant = new_densest_quadrant
        print('Autonomous policy prioritizing quadrant %s' % (quadrant,))

    for ship_id in current_state['ships']:
        # Loops through all ships in the environment, calculates distance from current aircraft position, finds the closest unknown ship (or unknown WEZ), and sets aircraft waypoint to that ship's location.
        if id_type == 'target': # If set to target, only consider unknown targets
            if current_state['ships'][ship_id]['observed'] == False and (quadrant_bounds[quadrant][0] <= current_state['ships'][ship_id]['position'][0] <= quadrant_bounds[quadrant][1]) and (quadrant_bounds[quadrant][2] <= current_state['ships'][ship_id]['position'][1] <= quadrant_bounds[quadrant][3]):
                dist = math.hypot(env.agents[aircraft_id].x - env.agents[ship_id].x, env.agents[aircraft_id].y - env.agents[ship_id].y)
                current_target_distances[ship_id] = dist
        elif id_type == 'wez': # If set to wez, consider unknown targets AND known hostiles with unknown threat rings
            if (current_state['ships'][ship_id]['observed'] == False or current_state['ships'][ship_id]['observed threat'] == False) and (quadrant_bounds[quadrant][0] <= current_state['ships'][ship_id]['position'][0] <= quadrant_bounds[quadrant][1]) and (quadrant_bounds[quadrant][2] <= current_state['ships'][ship_id]['position'][1] <= quadrant_bounds[quadrant][3]):
                dist = math.hypot(env.agents[aircraft_id].x - env.agents[ship_id].x,env.agents[aircraft_id].y - env.agents[ship_id].y)
                current_target_distances[ship_id] = dist

    if current_target_distances:
        nearest_target_id = min(current_target_distances, key=current_target_distances.get)
        target_waypoint = tuple((env.agents[nearest_target_id].x, env.agents[nearest_target_id].y))
        #print('Nearest unknown target is %s. Setting waypoint to %s' % (nearest_target_id, target_waypoint))
    else: # If all targets ID'd, loiter in center of board or specified quadrant
        if quadrant == 'full': target_waypoint = (gameboard_size*0.5,gameboard_size*0.5) # If no more targets, return to center of game board TODO: Make this more robust
        elif quadrant == 'NW': target_waypoint = (gameboard_size*0.25,gameboard_size*0.25)
        elif quadrant == 'NE': target_waypoint = (gameboard_size * 0.75, gameboard_size * 0.25)
        elif quadrant == 'SW': target_waypoint = (gameboard_size * 0.25, gameboard_size * 0.75)
        elif quadrant == 'SE': target_waypoint = (gameboard_size * 0.75, gameboard_size * 0.75)

    target_direction = math.atan2(target_waypoint[1] - env.agents[aircraft_id].y,target_waypoint[0] - env.agents[aircraft_id].x)
    return target_waypoint, target_direction

def mouse_waypoint_policy(env,aircraft_id):
    # TODO Currently this is implemented in main.py. Might move it here.
    pass
    #return target_waypoint, target_direction


# NOTE: policies below are under test and not currently working.
def astar_policy(env, aircraft_id, target_waypoint):
    aircraft = env.agents[aircraft_id]
    start = (aircraft.x, aircraft.y)
    goal = target_waypoint

    def heuristic(a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def get_neighbors(pos):
        x, y = pos
        neighbors = [
            (x + 30, y), (x - 30, y), (x, y + 30), (x, y - 30),
            (x + 21, y + 21), (x - 21, y + 21), (x + 21, y - 21), (x - 21, y - 21)
        ]
        return [(nx, ny) for nx, ny in neighbors if
                0 <= nx < env.config["gameboard size"] and 0 <= ny < env.config["gameboard size"]]

    def is_valid(pos):
        x, y = pos
        for ship in env.agents:
            if ship.agent_class == "ship" and ship.threat > 0:
                dist = math.hypot(ship.x - x, ship.y - y)
                if dist <= ship.width * env.AGENT_THREAT_RADIUS[ship.threat]:
                    return False
        return True

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for neighbor in get_neighbors(current):
            if not is_valid(neighbor):
                continue

            tentative_g_score = g_score[current] + heuristic(current, neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found

def astar_waypoint_policy(env, aircraft_id, quadrant='full', id_type='target'):
    current_state = env.get_state()
    aircraft = env.agents[aircraft_id]

    # Use the existing target_id_policy to get the target waypoint
    target_waypoint, _ = target_id_policy(env, aircraft_id, quadrant, id_type)

    # Use A* to find a path to the target waypoint
    path = astar_policy(env, aircraft_id, target_waypoint)

    if path:
        # Return the next waypoint in the path
        next_waypoint = path[1] if len(path) > 1 else path[0]
        target_direction = math.atan2(next_waypoint[1] - aircraft.y, next_waypoint[0] - aircraft.x)
        return next_waypoint, target_direction
    else:
        # If no path is found, return the current position
        return (aircraft.x, aircraft.y), aircraft.direction

# TODO Testing
# This code works but runs pretty slowly, and can still get stuck between threats
def safe_target_id_policy(env, aircraft_id, quadrant='full', id_type='target'):
    aircraft = env.agents[aircraft_id]
    start = (aircraft.x, aircraft.y)

    # Get all valid targets
    valid_targets = get_valid_targets(env, aircraft_id, quadrant, id_type)

    if not valid_targets:
        return return_to_quadrant_center(env, quadrant), 0

    # Sort targets by distance
    sorted_targets = sorted(valid_targets, key=lambda t: math.hypot(t[0] - start[0], t[1] - start[1]))

    for target in sorted_targets:
        if is_path_safe(env, start, target):
            return target, math.atan2(target[1] - aircraft.y, target[0] - aircraft.x)

        safe_waypoint = find_safe_intermediate_waypoint(env, start, target)
        if safe_waypoint:
            return safe_waypoint, math.atan2(safe_waypoint[1] - aircraft.y, safe_waypoint[0] - aircraft.x)

    # If we can't find a safe path to any target, move towards the closest one as far as safely possible
    closest_target = sorted_targets[0]
    safe_waypoint = move_towards_safely(env, start, closest_target)
    return safe_waypoint, math.atan2(safe_waypoint[1] - aircraft.y, safe_waypoint[0] - aircraft.x)


def get_valid_targets(env, aircraft_id, quadrant, id_type):
    current_state = env.get_state()
    valid_targets = []

    for ship_id, ship in current_state['ships'].items():
        if (id_type == 'target' and not ship['observed']) or (
                id_type == 'wez' and (not ship['observed'] or not ship['observed threat'])):
            ship_pos = ship['position']
            if is_in_quadrant(ship_pos, quadrant, env.config['gameboard size']):
                valid_targets.append(ship_pos)

    return valid_targets


def is_in_quadrant(pos, quadrant, board_size):
    x, y = pos
    half_size = board_size / 2
    if quadrant == 'full':
        return True
    elif quadrant == 'NW':
        return x < half_size and y < half_size
    elif quadrant == 'NE':
        return x >= half_size and y < half_size
    elif quadrant == 'SW':
        return x < half_size and y >= half_size
    elif quadrant == 'SE':
        return x >= half_size and y >= half_size
    return False


def return_to_quadrant_center(env, quadrant):
    board_size = env.config['gameboard size']
    if quadrant == 'full':
        return (board_size / 2, board_size / 2)
    elif quadrant == 'NW':
        return (board_size / 4, board_size / 4)
    elif quadrant == 'NE':
        return (3 * board_size / 4, board_size / 4)
    elif quadrant == 'SW':
        return (board_size / 4, 3 * board_size / 4)
    elif quadrant == 'SE':
        return (3 * board_size / 4, 3 * board_size / 4)


def is_path_safe(env, start, end):
    steps = 10
    for i in range(steps + 1):
        t = i / steps
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        if not is_point_safe(env, (x, y)):
            return False
    return True


def is_point_safe(env, point):
    for agent in env.agents:
        if agent.agent_class == "ship" and agent.threat > 0:
            distance = math.hypot(point[0] - agent.x, point[1] - agent.y)
            if distance <= agent.width * env.AGENT_THREAT_RADIUS[agent.threat]:
                return False
    return True


def find_safe_intermediate_waypoint(env, start, end):
    vector = (end[0] - start[0], end[1] - start[1])
    perpendicular = (-vector[1], vector[0])
    magnitude = math.hypot(*perpendicular)
    if magnitude == 0:
        return None
    unit_perpendicular = (perpendicular[0] / magnitude, perpendicular[1] / magnitude)

    for fraction in [0.25, 0.5, 0.75]:
        for distance in [50, 100, 150, -50, -100, -150]:
            intermediate = (
                start[0] + vector[0] * fraction + unit_perpendicular[0] * distance,
                start[1] + vector[1] * fraction + unit_perpendicular[1] * distance
            )
            if is_point_safe(env, intermediate) and is_path_safe(env, start, intermediate) and is_path_safe(env,
                                                                                                            intermediate,
                                                                                                            end):
                return intermediate

    return None


def move_towards_safely(env, start, end):
    vector = (end[0] - start[0], end[1] - start[1])
    distance = math.hypot(*vector)
    if distance == 0:
        return start

    unit_vector = (vector[0] / distance, vector[1] / distance)
    step_size = 10  # pixels

    for i in range(1, int(distance / step_size) + 1):
        point = (start[0] + unit_vector[0] * i * step_size,
                 start[1] + unit_vector[1] * i * step_size)
        if not is_point_safe(env, point):
            # Return the last safe point
            return (start[0] + unit_vector[0] * (i - 1) * step_size,
                    start[1] + unit_vector[1] * (i - 1) * step_size)

    # If the entire path is safe, return the end point
    return end

# Code below works fast but tends to get stick on the edge of a target's threat ring
"""def safe_target_id_policy(env, aircraft_id, quadrant='full', id_type='target'):
    # Get the original waypoint from the existing target_id_policy
    original_waypoint, original_direction = target_id_policy(env, aircraft_id, quadrant, id_type)

    aircraft = env.agents[aircraft_id]
    start = (aircraft.x, aircraft.y)

    # If we're already at the waypoint, no need to adjust
    if start == original_waypoint:
        return original_waypoint, original_direction

    # Check if the direct path to the waypoint is safe
    if is_path_safe(env, start, original_waypoint):
        return original_waypoint, original_direction

    # If not safe, find a safe intermediate waypoint
    safe_waypoint = find_safe_intermediate_waypoint(env, start, original_waypoint)

    if safe_waypoint:
        new_direction = math.atan2(safe_waypoint[1] - aircraft.y, safe_waypoint[0] - aircraft.x)
        return safe_waypoint, new_direction

    # If no safe waypoint found, move towards the original waypoint as far as safely possible
    safe_waypoint = move_towards_safely(env, start, original_waypoint)
    new_direction = math.atan2(safe_waypoint[1] - aircraft.y, safe_waypoint[0] - aircraft.x)
    return safe_waypoint, new_direction


def is_path_safe(env, start, end):
    # Check points along the path
    steps = 10
    for i in range(steps + 1):
        t = i / steps
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        if not is_point_safe(env, (x, y)):
            return False
    return True


def is_point_safe(env, point):
    for agent in env.agents:
        if agent.agent_class == "ship" and agent.threat > 0:
            distance = math.hypot(point[0] - agent.x, point[1] - agent.y)
            if distance <= agent.width * env.AGENT_THREAT_RADIUS[agent.threat]:
                return False
    return True


def find_safe_intermediate_waypoint(env, start, end):
    vector = (end[0] - start[0], end[1] - start[1])
    perpendicular = (-vector[1], vector[0])
    magnitude = math.hypot(*perpendicular)
    if magnitude == 0:
        return None
    unit_perpendicular = (perpendicular[0] / magnitude, perpendicular[1] / magnitude)

    for fraction in [0.25, 0.5, 0.75]:  # Try different points along the path
        for distance in [50, 100, 150, -50, -100, -150]:  # Try different perpendicular distances
            intermediate = (
                start[0] + vector[0] * fraction + unit_perpendicular[0] * distance,
                start[1] + vector[1] * fraction + unit_perpendicular[1] * distance
            )
            if is_point_safe(env, intermediate) and is_path_safe(env, start, intermediate) and is_path_safe(env,
                                                                                                            intermediate,
                                                                                                            end):
                return intermediate

    return None


def move_towards_safely(env, start, end):
    vector = (end[0] - start[0], end[1] - start[1])
    distance = math.hypot(*vector)
    if distance == 0:
        return start

    unit_vector = (vector[0] / distance, vector[1] / distance)
    step_size = 10  # pixels

    for i in range(1, int(distance / step_size) + 1):
        point = (start[0] + unit_vector[0] * i * step_size,
                 start[1] + unit_vector[1] * i * step_size)
        if not is_point_safe(env, point):
            # Return the last safe point
            return (start[0] + unit_vector[0] * (i - 1) * step_size,
                    start[1] + unit_vector[1] * (i - 1) * step_size)

    # If the entire path is safe, return the end point
    return end"""