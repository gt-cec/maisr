import gym
import numpy as np
from numpy.random.mtrand import get_state
import pygame
import random
import agents
from isr_gui import Button, ScoreWindow, HealthWindow, TimeWindow
import datetime
import math


class MAISREnv(gym.Env):
    """Multi-Agent ISR Environment following the Gym format"""

    def __init__(self, config={}, window=None, clock=None, render=False):
        super().__init__()

        self.config = config
        self.window = window
        self.clock = clock

        # constants
        self.AGENT_BASE_DRAW_WIDTH = 10  # an agent scale unit of 1 draws this many pixels wide
        self.AGENT_COLOR_UNOBSERVED = (255, 215, 0)  # gold
        self.AGENT_COLOR_OBSERVED = (128, 0, 128)  # purple
        self.AGENT_COLOR_THREAT = (255, 0, 0)  # red
        self.AGENT_THREAT_RADIUS = [0, 1.4, 2.5, 4]  # threat radius for each level

        self.AIRCRAFT_NOSE_LENGTH = 10  # pixel length of aircraft nose (forward of wings)
        self.AIRCRAFT_TAIL_LENGTH = 25  # pixel length of aircraft tail (behind wings)
        self.AIRCRAFT_TAIL_WIDTH = 7  # pixel width of aircraft tail (perpendicular to body)
        self.AIRCRAFT_WING_LENGTH = 18  # pixel length of aircraft wings (perpendicular to body)
        self.AIRCRAFT_LINE_WIDTH = 5  # pixel width of aircraft lines
        self.AIRCRAFT_ENGAGEMENT_RADIUS = 50 #100  # pixel width of aircraft engagement (to identify WEZ of threats)
        self.AIRCRAFT_ISR_RADIUS = 85 #170  # pixel width of aircraft scanner (to identify hostile vs benign)

        self.GAMEBOARD_NOGO_RED = (255, 200, 200)  # color of the red no-go zone
        self.GAMEBOARD_NOGO_YELLOW = (255, 225, 200)  # color of the yellow no-go zone
        self.FLIGHTPLAN_EDGE_MARGIN = .2  # proportion distance from edge of gameboard to flight plan, e.g., 0.2 = 20% in, meaning a flight plan of (1,1) would go to 80%,80% of the gameboard
        self.AIRCRAFT_COLORS = [(0, 160, 160), (0, 0, 255), (200, 0, 200), (80, 80, 80)]  # colors of aircraft 1, 2, 3, ... add more colors here, additional aircraft will repeat the last color
        # Old agent0 color: (255, 165, 0)

        self.gameboard_offset = 0  # How far from left edge to start drawing gameboard
        self.window_x = self.config["window size"][0]
        self.window_y = self.config["window size"][1]
        self.right_pane_edge = self.config['gameboard size'] + 20  # Left edge of gameplan button windows
        self.comm_pane_edge = self.config['gameboard size'] + 40 + 405
        # self.right_pane_edge = self.gameboard_offset + 20 + self.config['gameboard size']  # Left edge of gameplan button windows

        # Set point quantities for each event
        self.score = 0
        self.all_targets_points = 20  # All targets ID'd
        self.target_points = 10  # Each target ID'd
        self.threat_points = 5  # Each threat ID'd
        self.time_points = 2  # Points given per second remaining
        self.agent_damage_points = -0.1  # Point subtracted per damage point taken TODO currently excluded, need to decide
        self.human_damage_points = -0.2  # Point subtracted per damage point taken TODO currently excluded, need to decide
        self.wingman_dead_points = -30  # Points subtracted for agent wingman dying
        self.human_dead_points = -40  # Points subtracted for human dying

        self.first_step = True  # TODO testing
        self.paused = False
        self.agent0_dead = False # Used at end of loop to check if agent recently deceased.
        self.agent1_dead = False
        self.risk_level = 'LOW'

        # Comm log
        self.comm_messages = []
        self.max_messages = 7
        self.font = pygame.font.SysFont(None,24)
        self.ai_color = self.AIRCRAFT_COLORS[0]
        self.human_color = self.AIRCRAFT_COLORS[1]

        # Target tally
        self.identified_targets = 0
        self.identified_threat_types = 0
        self.tally_font = pygame.font.SysFont(None,24)

        self.display_time = 0 # Time that is used for the on-screen timer. Accounts for pausing.
        self.pause_start_time = 0
        self.total_pause_time = 0
        self.button_latch_dict = {'target_id':False,'wez_id':False,'hold':False,'waypoint':False,'NW':False,'SW':False,'NE':False,'SE':False,'full':False,'autonomous':False,'pause':False} # Hacky way to get the buttons to visually latch even when they're redrawn every frame
        self.render_bool = render
        self.pause_font = pygame.font.SysFont(None, 74)
        self.done = False # TODO was using this to render a game complete screen. Not currently working.

        # Situational-awareness based agent transparency config
        self.show_agent_waypoint = self.config['show agent waypoint']
        self.agent_priorities = 'placeholder'

        # labeled ISR flight plans (Note: (1.0 * margin, 1.0 * margin) is top left, and all paths and scaled to within the region within the FLIGHTPLAN_EDGE_MARGIN)
        self.FLIGHTPLANS = { "square": [(0, 1),(0, 0), (1, 0),(1, 1)],
                             "ladder": [(1, 1),(1, .66),(0, .66),(0, .33),(1, .33),(1, 0),(0, 0)],
                             "hold":[(.4, .4),(.4, .6),(.6, .6),(.6, .4)]}

        # set the random seed
        if "seed" in config:
            random.seed(config["seed"])
        else:
            print("Note: 'seed' is not in the env config, defaulting to 0.")
            random.seed(0)

        # check gameplay color config value
        if "gameplay color" not in self.config:
            print("Note: 'gameplay color' is not in the env config, defaulting to 'white'.")

        # determine the number of ships
        if "targets iteration" not in config:
            print("Note: 'targets iteration' is not in the env config, defaulting to 10 ships.")
            self.num_ships = 10
        else:
            if config["targets iteration"] == "A":
                self.num_ships = 10
            elif config["targets iteration"] == "B":
                self.num_ships = 20
            elif config["targets iteration"] == "C":
                self.num_ships = 30
            elif config["targets iteration"] == "D":
                self.num_ships = 50
            elif config["targets iteration"] == "E":
                self.num_ships = 100
            else:
                print(f"Note: 'targets iteration' had an invalid value ({config['targets iteration']}), defaulting to 10 ships.")
                self.num_ships = 10

        self.total_targets = self.num_ships

        # check default search pattern
        if "search pattern" not in config:
            print("Note: 'search pattern' is not in the env config, defaulting to 'square'.")
            self.config["search pattern"] = "square"

        # for ISR, action space is a waypoint (x,y)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # for ISR, observation space is the gameboard state
        self.observation_space = gym.spaces.Dict({
            "aircraft pos": gym.spaces.Box(low=-1, high=1, shape=(self.config["num aircraft"], 2), dtype=np.float32),
            "threat pos": gym.spaces.Box(low=-1, high=1, shape=(self.num_ships, 2), dtype=np.float32),
            "threat class": gym.spaces.MultiBinary([self.num_ships, 2])
        })

        self.reset()

        if render:
            #self.window = pygame.display.set_mode((self.config["gameboard size"], self.config["gameboard size"]))
            self.window = pygame.display.set_mode((self.window_x,self.window_y))

    def reset(self):
        self.agents = []
        self.aircraft_ids = []  # indexes of the aircraft agents
        self.damage = 0  # total damage from all agents (TODO: Reset each aircraft's damage here too)
        self.num_identified_ships = 0  # number of ships with accessed threat levels, used for determining game end
        self.display_time = 0  # Time that is used for the on-screen timer. Accounts for pausing.
        self.pause_start_time = 0
        self.total_pause_time = 0
        self.score = 0
        self.identified_targets = 0
        self.identified_threat_types = 0
        self.first_step = 0 # TODO testing
        self.done = False # TODO was using this to render a game complete screen. Not currently working.

        # create the ships
        for i in range(self.num_ships):
            agents.Ship(self)  # create the agent, will add itself to the env

        # create the aircraft
        for i in range(self.config["num aircraft"]):
            agents.Aircraft(self, 0, color=self.AIRCRAFT_COLORS[i] if i < len(self.AIRCRAFT_COLORS) else self.AIRCRAFT_COLORS[-1], speed=1, flight_pattern=self.config["search pattern"])
        return get_state()

    def step(self, actions:list):
        # if an action was specified, handle that agent's waypoint

        if actions is not None and actions != []:
            for action in actions:  # handle each action (waypoint override) provided
                agent = action[0]
                waypoint = action[1]
                self.agents[agent].waypoint_override = waypoint
        # move the agents and check for gameplay updates
        for agent in self.agents:
            # move the agents
            if agent.agent_class == "aircraft":
                if agent.alive:
                    agent.move()

            # handle ships
            if agent.agent_class == "ship":
                # check if ship is within range of aircraft
                for aircraft_id in self.aircraft_ids:
                    dist = agent.distance(self.agents[aircraft_id])
                    # if in the aircraft's ISR range, set to observed
                    if not agent.observed and self.agents[aircraft_id].in_isr_range(distance=dist):
                        agent.observed = True
                        self.identified_targets += 1 # Used for the tally box
                        self.score += self.target_points
                        if agent.threat == 0:
                            agent.observed_threat = True
                            self.num_identified_ships += 1
                            self.identified_threat_types += 1
                            self.score += self.threat_points

                        if self.config["verbose"]:
                            print("Ship {} observed by aircraft {}".format(agent.agent_idx, aircraft_id))

                    # if in the aircraft's engagement range, identify threat level
                    if not agent.observed_threat and (self.agents[aircraft_id].in_engagement_range(distance=dist)):# or agent.threat == 0):
                        agent.observed_threat = True
                        self.num_identified_ships += 1
                        self.identified_threat_types += 1 # Used for the tally box
                        self.score += self.threat_points
                        if self.config["verbose"]:
                            print("Ship {} threat level identified by aircraft {}".format(agent.agent_idx, aircraft_id))
                        break
                    # if in the ship's weapon range, damage the aircraft
                    if agent.in_weapon_range(distance=dist) and agent.threat > 0:
                        if self.agents[aircraft_id].alive:
                            self.agents[aircraft_id].damage += .1
                            self.damage += .1
                        # TODO: If agent 0 (AI), subtract 0.1 points per damage. If agent 1 (player), subtract 0.2 points per damage.
                    # add some logic here if you want the no-go zones to damage the aircrafts

        agent0 = self.agents[self.aircraft_ids[0]]
        hostile_targets_nearby = sum(1 for agent in self.agents
                                     if agent.agent_class == "ship"
                                     and agent.threat > 0
                                     and agent.observed_threat
                                     and math.hypot(agent.x - agent0.x, agent.y - agent0.y) <= 30)
        risk_level_function = 10*hostile_targets_nearby + self.agents[self.num_ships].damage # TODO tune this
        self.risk_level = 'LOW' if risk_level_function <= 30 else 'MEDIUM' if risk_level_function <= 60 else 'HIGH' if risk_level_function <= 80 else 'EXTREME'

        # Check if any aircraft are recently deceased (RIP)
        if not self.agents[self.num_ships].alive and not self.agent0_dead:
            self.agent0_dead = True
            self.score += self.wingman_dead_points
            print(f'{self.wingman_dead_points} points for agent wingman destruction')

        # progress update
        if self.config["verbose"]:
            print("   Found:", self.num_identified_ships, "Total:", len(self.agents) - len(self.aircraft_ids), "Damage:", self.damage)
        # exit when all ships are identified
        state = self.get_state()  # you can make this self.observation_space and use that (will require a tiny bit of customization, look into RL tutorials)
        reward = self.get_reward()
        done = self.num_identified_ships >= len(self.agents) - len(self.aircraft_ids) or self.agents[self.num_ships+1].damage > 100 # round is complete when all ships have been identified # TODO: Currently requires you to identify all targets and all WEZs. Consider changing.

        if (self.num_identified_ships >= len(self.agents) - len(self.aircraft_ids)) or (self.agents[self.num_ships+1].damage > 100) or (self.display_time/1000 >= 120):
            done = True
            self.done = True
            print('Done!')

            if self.num_identified_ships >= len(self.agents) - len(self.aircraft_ids):
                self.score += self.all_targets_points
                self.score += (self.display_time/1000)*self.time_points
                #self.score += (self.agent_damage_points)*self.agents[self.num_ships].damage
                #self.score += (self.human_damage_points)*self.agents[self.num_ships+1].damage # TODO Currently not subtracting points for damage. Decide whether to include.


            elif self.agents[self.num_ships+1].damage > 100:
                self.score += self.human_dead_points
                print('Human aircraft destroyed, game over.')
            elif self.display_time/1000 >= 120:
                print('Out of time, game over.')

            print(f'\nTargets identified: {self.identified_targets} / {self.total_targets} ({self.identified_targets * 10} points)')
            print(f'Threat levels identified: {self.identified_threat_types} / {self.total_targets} ({self.identified_threat_types * 5} points)')
            if self.num_identified_ships >= len(self.agents) - len(self.aircraft_ids):
                print(f'All targets identified (+{self.all_targets_points} points)')
                print(f'{(round(self.display_time,1) / 1000)}*{self.time_points} = {(round(self.display_time / 1000,1)) * self.time_points} points added for time remaining')
            # TODO add printout for -30 points if agent destroyed
            print(f'Time remaining: {round(self.display_time/1000,1)} seconds')
            print(f'\nFinal score = {self.score}')
            if self.render_bool: pygame.time.wait(5000)

        return state, reward, done, {}


    def get_reward(self):
        # define a custom reward function here
        # Ryan TODO: Add points for IDing targets (+), IDing all targets (+), taking damage (-), or agent A/C dying (-)
        return -self.damage

    def add_comm_message(self,message,is_ai=True):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        sender = "AGENT" if is_ai else "HUMAN"
        full_message = f"[{timestamp}] {sender}: {message}"
        self.comm_messages.append((full_message, is_ai))
        #self.comm_messages.append(message)
        if len(self.comm_messages) > self.max_messages:
            self.comm_messages.pop(0)

    def render(self, mode='human', close=False):
                # Set window dimensions. Used for placing buttons etc.
        window_width, window_height = self.config['window size'][0], self.config['window size'][0]
        game_width = self.config["gameboard size"]
        ui_width = window_width - game_width

        # gameboard background
        self.window.fill((255, 255, 255))  # white background
        # set up no-go areas
        #if "gameplay color" in self.config: # TODO: Ryan temporarily commented this out because it was stuck rendering yellow even when env_config said white.
            #pygame.draw.rect(self.window, self.GAMEBOARD_NOGO_RED if self.config["gameplay color"] == "yellow" else self.GAMEBOARD_NOGO_YELLOW if self.config["gameplay color"] else (255, 255, 255), (self.config["gameboard size"] * 0.8, 0, self.config["gameboard size"] * 0.8, self.config["gameboard size"]))
        self.__render_box__(1, (0, 0, 0), 3)  # outer box
        self.__render_box__(self.config["gameboard border margin"], (0, 128, 0), 2)  # inner box
        pygame.draw.line(self.window, (0, 0, 0), (self.config["gameboard size"] // 2, 0), (self.config["gameboard size"] // 2, self.config["gameboard size"]), 2)
        pygame.draw.line(self.window, (0, 0, 0), (0, self.config["gameboard size"] // 2), (self.config["gameboard size"], self.config["gameboard size"] // 2), 2)
        pygame.draw.rect(self.window, (100, 100, 100), (game_width+self.gameboard_offset, 0, ui_width, window_height))
        pygame.draw.rect(self.window, (100, 100, 100), (0, game_width, game_width, window_height))  # Fill bottom portion with gray

        # Draw the agents
        for agent in self.agents:
            agent.draw(self.window)

        # Draw Agent Gameplan sub-window
        self.quadrant_button_height = 120
        self.gameplan_button_width = 180
        pygame.draw.rect(self.window, (230,230,230), pygame.Rect(self.right_pane_edge, 10, 405, 555))  # Agent gameplan sub-window box
        #pygame.draw.rect(self.window, (200,200,200), pygame.Rect(self.right_pane_edge, 10, 445, 40))  # Agent gameplan title box
        gameplan_text_surface = pygame.font.SysFont(None, 36).render('Agent Gameplan', True, (0,0,0))
        self.window.blit(gameplan_text_surface, gameplan_text_surface.get_rect(center=(self.right_pane_edge+425 // 2, 10+40 // 2)))
        pygame.draw.line(self.window, (0, 0, 0), (self.right_pane_edge, 10),(self.right_pane_edge + 405, 10), 4)  # Top edge of gameplan panel
        pygame.draw.line(self.window, (0, 0, 0), (self.right_pane_edge, 10), (self.right_pane_edge, 565), 4)
        pygame.draw.line(self.window, (0, 0, 0), (self.right_pane_edge + 405, 10), (self.right_pane_edge + 405, 565), 4)
        pygame.draw.line(self.window, (0, 0, 0), (self.right_pane_edge, 10+555), (self.right_pane_edge + 405, 10+555),4)  # Top edge of gameplan panel

        self.autonomous_button = Button("Autonomous", self.right_pane_edge+15, 20, self.gameplan_button_width*2 + 15, 65)
        self.autonomous_button.is_latched = self.button_latch_dict['autonomous']
        self.autonomous_button.color = (50, 180, 180)
        self.autonomous_button.draw(self.window)

        pygame.draw.line(self.window, (0, 0, 0), (self.right_pane_edge, 20+65+6), (self.right_pane_edge + 405, 20+65+6), 4)

        type_text_surface = pygame.font.SysFont(None, 26).render('SEARCH TYPE', True, (0,0,0))
        self.window.blit(type_text_surface, type_text_surface.get_rect(center=(self.right_pane_edge+425 // 2, 10+40+110 // 2)))

        self.target_id_button = Button("TARGET ID", self.right_pane_edge + 15, 60+55, self.gameplan_button_width, 60)# (255, 120, 80))
        self.target_id_button.is_latched = self.button_latch_dict['target_id']
        self.target_id_button.draw(self.window)

        self.wez_id_button = Button("TARGET + WEZ", self.right_pane_edge + 30 + self.gameplan_button_width, 60+55, self.gameplan_button_width, 60) # 15 pixel gap b/w buttons
        self.wez_id_button.is_latched = self.button_latch_dict['wez_id']
        self.wez_id_button.draw(self.window)

        pygame.draw.line(self.window, (0, 0, 0), (self.right_pane_edge, 130+45+5),(self.right_pane_edge+405,130+45+5),4) # Separating line between target/WEZ ID selection and quadrant select

        search_area_text_surface = pygame.font.SysFont(None, 26).render('SEARCH AREA', True, (0, 0, 0))
        self.window.blit(search_area_text_surface,search_area_text_surface.get_rect(center=(self.right_pane_edge + 425 // 2, 50 + 10 + 40 + 195 // 2)))

        self.NW_quad_button = Button("NW", self.right_pane_edge + 15, 60+80+10+10+50, self.gameplan_button_width, self.quadrant_button_height)
        self.NW_quad_button.is_latched = self.button_latch_dict['NW']
        self.NW_quad_button.draw(self.window)

        self.NE_quad_button = Button("NE", self.right_pane_edge + 30 + self.gameplan_button_width, 60+80+10+10+50, self.gameplan_button_width, self.quadrant_button_height)
        self.NE_quad_button.is_latched = self.button_latch_dict['NE']
        self.NE_quad_button.draw(self.window)

        self.SW_quad_button = Button("SW", self.right_pane_edge + 15, 50+2*(self.quadrant_button_height)+50, self.gameplan_button_width, self.quadrant_button_height)
        self.SW_quad_button.is_latched = self.button_latch_dict['SW']
        self.SW_quad_button.draw(self.window)

        self.SE_quad_button = Button("SE", self.right_pane_edge + 30 + self.gameplan_button_width, 50+2*(self.quadrant_button_height)+50, self.gameplan_button_width, self.quadrant_button_height)
        self.SE_quad_button.is_latched = self.button_latch_dict['SE']
        self.SE_quad_button.draw(self.window)

        self.full_quad_button = Button("Full", self.right_pane_edge+200-35-10, 60+2*(80+10)+20-35+5+50, 100, 100)
        self.full_quad_button.color = (50,180,180)
        self.full_quad_button.is_latched = self.button_latch_dict['full']
        self.full_quad_button.draw(self.window)

        self.waypoint_button = Button("WAYPOINT", self.right_pane_edge + 30 + self.gameplan_button_width, 3*(self.quadrant_button_height) + 115, self.gameplan_button_width, 80)
        #self.waypoint_button.is_latched = self.button_latch_dict['waypoint']
        self.waypoint_button.draw(self.window)

        self.hold_button = Button("HOLD", self.right_pane_edge + 15, 3*(self.quadrant_button_height) + 115, self.gameplan_button_width, 80)
        self.hold_button.is_latched = self.button_latch_dict['hold']
        self.hold_button.draw(self.window)

        pygame.draw.line(self.window, (0, 0, 0), (self.right_pane_edge, 465), (self.right_pane_edge + 405, 465),4)  # Separating line between quadrant select and hold/waypoint

        # Draw Comm Log
        pygame.draw.rect(self.window, (200, 200, 200), pygame.Rect(self.comm_pane_edge, 450-280, 445, 40))  # Comm log title box
        pygame.draw.rect(self.window, (230,230,230), pygame.Rect(self.comm_pane_edge, 485-280, 445, 210))  # Comm Log sub-window box
        comm_text_surface = pygame.font.SysFont(None, 36).render('COMM LOG', True, (0, 0, 0))
        self.window.blit(comm_text_surface, comm_text_surface.get_rect(center=(self.comm_pane_edge + 445 // 2, 450-280 + 40 // 2)))

        # Draw incoming comm log text
        y_offset = 495-280
        for entry in self.comm_messages:
            message = entry[0]
            is_ai = entry[1]
            color = self.ai_color if is_ai else self.human_color
            message_surface = self.font.render(message, True, color)
            self.window.blit(message_surface, (self.comm_pane_edge+10, y_offset))
            y_offset += 30  # Adjust this value to change spacing between messages

        # Draw point tally
        pygame.draw.rect(self.window, (200, 200, 200), pygame.Rect(self.comm_pane_edge, 705-280, 445, 40))  # Target tally title box
        pygame.draw.rect(self.window, (230, 230, 230), pygame.Rect(self.comm_pane_edge, 740-280, 445, 100))  # Target tally sub-window box
        tally_title_surface = pygame.font.SysFont(None, 36).render('TARGET STATUS', True, (0, 0, 0))
        self.window.blit(tally_title_surface, tally_title_surface.get_rect(center=(self.comm_pane_edge + 445 // 2, 705-280 + 40 // 2)))

        id_tally_text = f"Identified Targets: {self.identified_targets} / {self.total_targets}"
        id_tally_surface = self.tally_font.render(id_tally_text, True, (0, 0, 0))
        self.window.blit(id_tally_surface, (self.comm_pane_edge+10, 750-280))

        threat_tally_text = f"Observed Threat Types: {self.identified_threat_types} / {self.total_targets}"
        threat_tally_surface = self.tally_font.render(threat_tally_text, True, (0, 0, 0))
        self.window.blit(threat_tally_surface, (self.comm_pane_edge+10, 780-280))

        # Draw agent status window
        status_window_width = 445
        status_window_height = 150
        status_window_x = self.comm_pane_edge
        status_window_y = 10  # Adjust this value to position the window properly
        pygame.draw.rect(self.window, (230, 230, 230),pygame.Rect(status_window_x, status_window_y, status_window_width, status_window_height))

        pygame.draw.rect(self.window, (200, 200, 200),pygame.Rect(status_window_x, status_window_y, status_window_width, 40))
        status_title_surface = pygame.font.SysFont(None, 36).render('AGENT STATUS', True, (0, 0, 0))
        self.window.blit(status_title_surface, status_title_surface.get_rect(center=(status_window_x + status_window_width // 2, status_window_y + 20)))

        # Show agent autonomous policy priorities
        # TODO make dynamic
        priorities_font = pygame.font.SysFont(None, 36)
        priorities_text = f"PRIORITIES: {self.agent_priorities}"
        priorities_surface = priorities_font.render(priorities_text, True, (0, 0, 0))
        self.window.blit(priorities_surface, (status_window_x + 10, status_window_y + 50))

        risk_font = pygame.font.SysFont(None, 36)
        risk_colors = {"LOW": (0, 190, 0), "MEDIUM": (255, 165, 0),"HIGH": (255, 0, 0),"EXTREME": (0, 0, 0)} # Green, Orange, Red, Black
        risk_text_dict = {"LOW": "LOW","MEDIUM": "MEDIUM","HIGH": "HIGH","EXTREME": "EXTREME"}
        risk_text_color = {"LOW": (0,0,0),"MEDIUM": (0,0,0),"HIGH": (0,0,0),"EXTREME": (255,255,255)}
        risk_square_length = {"LOW": 45*1.5,"MEDIUM": 70*1.5,"HIGH": 50*1.5,"EXTREME": 90*1.5}
        square_size = 30
        square_x = status_window_x + 10
        square_y = status_window_y + 80
        pygame.draw.rect(self.window, risk_colors.get(self.risk_level),(square_x+160, square_y-3+5, risk_square_length.get(self.risk_level), square_size))

        risk_label = f"RISK LEVEL:"
        risk_label_surface = risk_font.render(risk_label, True, (0, 0, 0))
        risk_text = risk_text_dict.get(self.risk_level)
        risk_text_surface = risk_font.render(risk_text, True,risk_text_color.get(self.risk_level))
        self.window.blit(risk_label_surface, (status_window_x + 10, status_window_y + 85))
        self.window.blit(risk_text_surface, (square_x + square_size + 135, square_y+5))

        # Draw health boxes TODO: Add support for >2 aircraft
        agent0_health_window = HealthWindow(self.num_ships,10,game_width+10, 'AGENT',self.AIRCRAFT_COLORS[0])
        agent0_health_window.update(self.agents[self.num_ships].damage)
        agent0_health_window.draw(self.window)

        agent1_health_window = HealthWindow(self.num_ships+1, game_width-150, game_width + 10, 'HUMAN',self.AIRCRAFT_COLORS[1])
        agent1_health_window.update(self.agents[self.num_ships+1].damage)
        agent1_health_window.draw(self.window)

        # Draw score box and update with new score value every tick
        score_button = ScoreWindow(self.score,game_width*0.5 - 320/2, game_width + 10)
        score_button.update(self.score)
        score_button.draw(self.window)

        self.pause_button = Button("PAUSE", game_width*0.5 - 150/2, 790, 150, 50)
        self.pause_button.color = (220,150,40)
        self.pause_button.is_latched = self.button_latch_dict['pause']
        self.pause_button.draw(self.window)

        current_time = pygame.time.get_ticks()
        if not self.paused:
            self.display_time = current_time - self.total_pause_time

        self.time_window = TimeWindow(game_width * 0.5 + 10, game_width + 10, current_time=self.display_time)
        self.time_window.update(self.display_time)
        self.time_window.draw(self.window)

        if self.paused:
            pause_surface = pygame.Surface((self.window.get_width(), self.window.get_height()))
            pause_surface.set_alpha(128)  # 50% transparent
            pause_surface.fill((100, 100, 100))  # Gray color
            self.window.blit(pause_surface, (0, 0))

            # Render "GAME PAUSED" text
            pause_text = self.pause_font.render('GAME PAUSED', True, (255, 255, 255))
            text_rect = pause_text.get_rect(center=(self.window.get_width() // 2, self.window.get_height() // 2))
            self.window.blit(pause_text, text_rect)

        if self.done: # TODO not currently working
            self._render_game_complete()

        # update the display
        pygame.display.update()
        self.clock.tick(60)

    # convert the environment into a state dictionary
    def get_state(self):
        # Ryan TODO: Can define points here maybe
        state = {
            "aircrafts": {},
            "ships": {},
            "damage": self.damage,
            "num identified ships": self.num_identified_ships,
        }
        for agent in self.agents:
            if agent.agent_class == "ship":
                state["ships"][agent.agent_idx] = {
                    "position": (agent.x, agent.y),
                    "direction": agent.direction,
                    "observed": agent.observed,
                    "threat": agent.threat,
                    "observed threat": agent.observed_threat
                }
            elif agent.agent_class == "aircraft":
                state["aircrafts"][agent.agent_idx] = {
                    "position": (agent.x, agent.y),
                    "direction": agent.direction,
                    "damage": agent.damage
                }
        return state

    # utility function for drawing a square box
    def __render_box__(self, distance_from_edge, color=(0,0,0), width=2):
        pygame.draw.line(self.window, color, (distance_from_edge, distance_from_edge), (distance_from_edge, self.config["gameboard size"] - distance_from_edge), width)
        pygame.draw.line(self.window, color, (distance_from_edge, self.config["gameboard size"] - distance_from_edge), (self.config["gameboard size"] - distance_from_edge, self.config["gameboard size"] - distance_from_edge), width)
        pygame.draw.line(self.window, color, (self.config["gameboard size"] - distance_from_edge, self.config["gameboard size"] - distance_from_edge), (self.config["gameboard size"] - distance_from_edge, distance_from_edge), width)
        pygame.draw.line(self.window, color, (self.config["gameboard size"] - distance_from_edge, distance_from_edge), (distance_from_edge, distance_from_edge), width)

    def pause(self,unpause_key):
        print('Game paused')
        self.pause_start_time = pygame.time.get_ticks()
        self.button_latch_dict['pause'] = True
        print('paused at %s (env.display_time = %s)' % (self.pause_start_time, self.display_time))
        self.paused = True
        while self.paused:
            pygame.time.wait(200)
            #pygame.draw.rect(self.window, (220, 150, 4), pygame.Rect(350, 350, 200, 200))
            #paused_surface = pygame.font.SysFont(None, 36).render('GAME PAUSED', True, (0, 0, 0))
            #self.window.blit(paused_surface, paused_surface.get_rect(center=(350 + 200 // 2, 350 + 200 // 2)))
            self.render()
            ev = pygame.event.get()
            for event in ev:
                if event.type == unpause_key:
                    mouse_position = pygame.mouse.get_pos()
                    if unpause_key == pygame.K_SPACE or self.pause_button.is_clicked(mouse_position):
                        self.paused = False
                        self.button_latch_dict['pause'] = False
                        pause_end_time = pygame.time.get_ticks()
                        pause_duration = pause_end_time - self.pause_start_time
                        self.total_pause_time += pause_duration
                        print('Paused for %s' % pause_duration)

    def _render_game_complete(self): # TODO was using this to render a game complete screen. Not currently working.
        """Render the game complete screen with final statistics"""
        # Create semi-transparent overlay
        overlay = pygame.Surface((self.window_x, self.window_y))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(128)
        self.window.blit(overlay, (0, 0))

        # Create stats window
        window_width = 500
        window_height = 400
        window_x = self.window_x // 2 - window_width // 2
        window_y = self.window_y // 2 - window_height // 2

        # Draw stats window background
        pygame.draw.rect(self.window, (230, 230, 230),
                         pygame.Rect(window_x, window_y, window_width, window_height))
        pygame.draw.rect(self.window, (200, 200, 200),
                         pygame.Rect(window_x, window_y, window_width, 60))

        # Initialize fonts
        title_font = pygame.font.SysFont(None, 48)
        stats_font = pygame.font.SysFont(None, 36)

        # Render title
        title_surface = title_font.render('GAME COMPLETE', True, (0, 0, 0))
        self.window.blit(title_surface, title_surface.get_rect(
            center=(window_x + window_width // 2, window_y + 30)))

        # Calculate statistics
        agent_status = "ALIVE" if self.agents[self.num_ships].damage <= 100 else "DESTROYED"
        agent_status_color = (0, 255, 0) if agent_status == "ALIVE" else (255, 0, 0)

        # Create stats text surfaces
        stats_items = [
            f"Final Score: {self.score}",
            f"Targets Identified: {self.identified_targets} / {self.total_targets}",
            f"Threat Levels Observed: {self.identified_threat_types} / {self.total_targets}",
            f"Agent Status: {agent_status}"
        ]

        # Render stats
        y_offset = window_y + 100
        for i, text in enumerate(stats_items):
            if i == len(stats_items) - 1:  # Agent Status line
                text_surface = stats_font.render(text.split(': ')[0] + ': ', True, (0, 0, 0))
                status_surface = stats_font.render(agent_status, True, agent_status_color)

                # Center align the text
                total_width = text_surface.get_width() + status_surface.get_width()
                start_x = window_x + (window_width - total_width) // 2

                self.window.blit(text_surface, (start_x, y_offset))
                self.window.blit(status_surface, (start_x + text_surface.get_width(), y_offset))
            else:
                text_surface = stats_font.render(text, True, (0, 0, 0))
                self.window.blit(text_surface, text_surface.get_rect(
                    center=(window_x + window_width // 2, y_offset)))
            y_offset += 60

        # Add decorative elements
        border_width = 4
        pygame.draw.rect(self.window, (100, 100, 100),
                         pygame.Rect(window_x, window_y, window_width, window_height),
                         border_width)

        # Add "Press any key to continue" message
        continue_font = pygame.font.SysFont(None, 24)
        continue_surface = continue_font.render('Press any key to continue...', True, (100, 100, 100))
        self.window.blit(continue_surface, continue_surface.get_rect(
            center=(window_x + window_width // 2, window_y + window_height - 40)))