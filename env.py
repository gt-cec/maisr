import gym
import numpy as np
#from numpy.random.mtrand import get_state
import pygame
import random
import agents
from gui import Button, ScoreWindow, HealthWindow, TimeWindow, AgentInfoDisplay
import datetime
import math
import webbrowser


class MAISREnv(gym.Env):
    """Multi-Agent ISR Environment following the Gym format"""

    def __init__(self, config={}, window=None, clock=None, render=False,agent_training=False,subject_id='99',user_group='99',round_number='99'):
        super().__init__()

        self.config = config
        self.window = window
        self.clock = clock
        self.init = True # Used to render static windows the first time
        self.start_countdown_time = 5000 # How long in milliseconds to count down at the beginning of the game before it starts

        self.subject_id = subject_id
        self.round_number = round_number
        self.user_group = user_group
        self.training = True if (self.round_number == 0 and user_group != 'test') else False # True for the training round at start of experiment, false for rounds 1-4 (NOTE: This is NOT agent training!)
        self.agent_training = agent_training
        #print(self.round_number)
        #print(self.training)

        #self.config["gameboard size"] = int(self.BASE_GAMEBOARD_SIZE * self.scaling_ratio)
        #self.config["window size"] = (int(self.BASE_WINDOW_WIDTH * self.scaling_ratio),
            #int(self.BASE_WINDOW_HEIGHT * self.scaling_ratio))

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
        self.AIRCRAFT_ENGAGEMENT_RADIUS = 40 #100  # pixel width of aircraft engagement (to identify WEZ of threats)
        self.AIRCRAFT_ISR_RADIUS = 85 #170  # pixel width of aircraft scanner (to identify hostile vs benign)

        self.GAMEBOARD_NOGO_RED = (255, 200, 200)  # color of the red no-go zone
        self.GAMEBOARD_NOGO_YELLOW = (255, 225, 200)  # color of the yellow no-go zone
        self.FLIGHTPLAN_EDGE_MARGIN = .2  # proportion distance from edge of gameboard to flight plan, e.g., 0.2 = 20% in, meaning a flight plan of (1,1) would go to 80%,80% of the gameboard
        self.AIRCRAFT_COLORS = [(0, 160, 160), (0, 0, 255), (200, 0, 200), (80, 80, 80)]  # colors of aircraft 1, 2, 3, ... add more colors here, additional aircraft will repeat the last color

        # Set GUI locations
        self.gameboard_offset = 0  # How far from left edge to start drawing gameboard
        self.window_x = self.config["window size"][0]
        self.window_y = self.config["window size"][1]
        self.right_pane_edge = self.config['gameboard size'] + 20  # Left edge of gameplan button windows
        self.comm_pane_edge = self.right_pane_edge
        self.gameplan_button_width = 180
        self.quadrant_button_height = 120
        self.autonomous_button_y = 590

        self.missiles_enabled = self.config['missiles_enabled']

        # Initialize buttons
        self.gameplan_button_color = (255, 120, 80)
        self.manual_priorities_button = Button("Manual Priorities", self.right_pane_edge + 15, 20,self.gameplan_button_width * 2 + 15, 65)
        self.target_id_button = Button("TARGET", self.right_pane_edge + 15, 60 + 55, self.gameplan_button_width,60)  # (255, 120, 80))
        self.wez_id_button = Button("WEAPON", self.right_pane_edge + 30 + self.gameplan_button_width, 60 + 55,self.gameplan_button_width, 60)  # 15 pixel gap b/w buttons
        self.NW_quad_button = Button("NW", self.right_pane_edge + 15, 60 + 80 + 10 + 10 + 50,self.gameplan_button_width, self.quadrant_button_height)
        self.NE_quad_button = Button("NE", self.right_pane_edge + 30 + self.gameplan_button_width,60 + 80 + 10 + 10 + 50, self.gameplan_button_width, self.quadrant_button_height)
        self.SW_quad_button = Button("SW", self.right_pane_edge + 15, 50 + 2 * (self.quadrant_button_height) + 50,self.gameplan_button_width, self.quadrant_button_height)
        self.SE_quad_button = Button("SE", self.right_pane_edge + 30 + self.gameplan_button_width,50 + 2 * (self.quadrant_button_height) + 50, self.gameplan_button_width,self.quadrant_button_height)
        self.full_quad_button = Button("FULL", self.right_pane_edge + 200 - 35 - 10,60 + 2 * (80 + 10) + 20 - 35 + 5 + 50, 100, 100)
        self.waypoint_button = Button("WAYPOINT", self.right_pane_edge + 30 + self.gameplan_button_width,3 * (self.quadrant_button_height) + 115, self.gameplan_button_width, 80)
        self.hold_button = Button("HOLD", self.right_pane_edge + 15, 3 * (self.quadrant_button_height) + 115,self.gameplan_button_width, 80)

        # Advanced gameplans
        #self.regroup_button = Button("REGROUP", self.right_pane_edge+15, self.autonomous_button_y-10,self.gameplan_button_width, 80)
        #self.tag_team_button = Button("TAG TEAM", self.right_pane_edge + 30 + self.gameplan_button_width, self.autonomous_button_y-10,self.gameplan_button_width, 80)
        #self.fan_out_button = Button("FAN OUT", 1230, 1035,self.gameplan_button_width, 50)

        # Set point quantities for each event
        self.score = 0
        self.all_targets_points = 0  # All targets ID'd
        self.target_points = 10  # Each target ID'd
        self.threat_points = 5  # Each threat ID'd
        self.time_points = 15  # Points given per second remaining
        self.human_hp_remaining_points = 70
        self.wingman_dead_points = -300  # Points subtracted for agent wingman dying
        self.human_dead_points = -400  # Points subtracted for human dying

        self.paused = False
        self.unpause_countdown = False
        self.new_target_id = None # Used to check if a new target has been ID'd so the data logger in main.py can log it
        self.new_weapon_id = None

        self.agent0_dead = False # Used at end of loop to check if agent recently deceased.
        self.agent1_dead = False
        self.risk_level = 'LOW'
        self.agent_waypoint_clicked = False # Flag to determine whether clicking on the map sets the humans' waypoint or the agent's. True when "waypoint" gameplan button set.
        self.regroup_clicked = False
        self.tag_team_commanded = False
        self.fan_out_commanded = False

        self.human_quadrant = None

        # Comm log
        self.comm_messages = []
        self.max_messages = 4
        self.message_font = pygame.font.SysFont(None,30)
        self.ai_color = self.AIRCRAFT_COLORS[0]
        self.human_color = self.AIRCRAFT_COLORS[1]

        # Target tally
        self.identified_targets = 0
        self.identified_threat_types = 0
        self.tally_font = pygame.font.SysFont(None,28)

        self.display_time = 0 # Time that is used for the on-screen timer. Accounts for pausing.
        self.pause_start_time = 0
        self.total_pause_time = 0
        self.button_latch_dict = {'target_id':False,'wez_id':False,'hold':False,'waypoint':False,'NW':False,'SW':False,'NE':False,'SE':False,'full':False,'autonomous':True,'pause':False,'risk_low':False, 'risk_medium':True, 'risk_high':False,'manual_priorities':False,'tag_team':False,'fan_out':False} # Hacky way to get the buttons to visually latch even when they're redrawn every frame
        self.render_bool = render
        self.pause_font = pygame.font.SysFont(None, 74)
        self.pause_subtitle_font = pygame.font.SysFont(None, 40)
        self.done = False
        self.time_limit = self.config['time limit']

        # For visual damage flash
        self.damage_flash_duration = 500  # Duration of flash in milliseconds
        self.damage_flash_start = 0  # When the last damage was taken
        self.damage_flash_alpha = 0  # Current opacity of flash effect
        self.agent_damage_flash_start = 0
        self.agent_damage_flash_alpha = 0
        self.last_health_points = {0: 10, 1: 10}  # Track health points to detect changes

        # Situational-awareness based agent transparency config
        self.show_agent_waypoint = self.config['show agent waypoint']
        self.agent_priorities = 'placeholder'

        # SAGAT survey flags
        self.survey1_launched, self.survey2_launched, self.survey3_launched = False, False, False

        # Calculate required height of agent status info
        self.agent_info_height_req = 0
        if self.config['show_low_level_goals']: self.agent_info_height_req += 1
        if self.config['show_high_level_goals']: self.agent_info_height_req += 1.7
        if self.config['show_tracked_factors']: self.agent_info_height_req += 1.7

        if self.agent_info_height_req > 0: # Only render agent info display if at least one of the info elements is used
            self.agent_info_display = AgentInfoDisplay(self.comm_pane_edge, 10, 445, 40+35*self.agent_info_height_req)

        self.time_window = TimeWindow(self.config["gameboard size"] * 0.43, self.config["gameboard size"]+5,current_time=self.display_time, time_limit=self.time_limit)

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
        self.num_ships = self.config['num ships']
        self.total_targets = self.num_ships

        self.human_idx = self.num_ships + 1 # Agent ID for the human-controlled aircraft. Dynamic so that if human dies in training round, their ID increments 1

        # check default search pattern
        if "search pattern" not in config:
            print("Note: 'search pattern' is not in the env config, defaulting to 'square'.")
            self.config["search pattern"] = "square"

        # for ISR, action space is a waypoint (x,y)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # for ISR, observation space is the gameboard state
        self.observation_space = gym.spaces.Box(low=0,high=1,shape=(505,),dtype=np.float32)
        self.observation = np.zeros(444, dtype=np.float32)
        self.target_data = np.zeros((self.num_ships, 8), dtype=np.float32)  # For target features

        self.reset()

        if render:
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
        self.done = False
        self.init = True

        # create the ships
        for i in range(self.num_ships):
            agents.Ship(self)  # create the agent, will add itself to the env

        # create the aircraft
        agent0_initial_location = self.config['agent start location']
        agent1_initial_location = self.config['human start location']

        # Agent speed was originally set by self.config['agent speed'] but currently overridden with static value
        agents.Aircraft(self, 0,prob_detect=0.0015,max_health=10,color=self.AIRCRAFT_COLORS[0],speed=self.config['game speed']*self.config['human speed'], flight_pattern=self.config["search pattern"]) # Agent
        agents.Aircraft(self, 0,prob_detect=0.04,max_health=10,color=self.AIRCRAFT_COLORS[1],speed=self.config['game speed']*self.config['human speed']*1.1, flight_pattern=self.config["search pattern"]) # Human
        self.agents[self.num_ships].x,self.agents[self.num_ships].y = agent0_initial_location
        self.agents[self.human_idx].x, self.agents[self.human_idx].y = agent1_initial_location

        if self.config['num aircraft'] == 1: self.agents[self.num_ships].is_visible = False # Do not draw the agent during solo training runs

        return self.get_state()

    def step(self, actions:list):
        old_score = self.score


        if self.regroup_clicked: self.agents[self.num_ships].regroup_clicked = True
        else: self.agents[self.num_ships].regroup_clicked = False


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

            if agent.agent_class == 'missile':
                agent.target_point = (self.agents[agent.target_aircraft_id].x, self.agents[agent.target_aircraft_id].y)
                agent.waypoint_override = (self.agents[agent.target_aircraft_id].x, self.agents[agent.target_aircraft_id].y)
                agent.move()

            # handle ships
            if agent.agent_class == "ship":
                # check if ship is within range of aircraft
                for aircraft_id in self.aircraft_ids:
                    dist = agent.distance(self.agents[aircraft_id])
                    # if in the aircraft's ISR range, set to observed
                    if not agent.observed and self.agents[aircraft_id].in_isr_range(distance=dist):
                        agent.observed = True
                        self.new_target_id = ['human' if aircraft_id == self.human_idx else 'AI', 'target_identified',agent.agent_idx]  # Will be used in main.py to log target ID event
                        self.identified_targets += 1 # Used for the tally box
                        self.score += self.target_points
                        #print('new target id: ', self.new_target_id)

                        if agent.threat == 0:
                            agent.observed_threat = True
                            #self.new_weapon_id = ['human' if aircraft_id == self.human_idx else 'AI','weapon_identified', agent.agent_idx]
                            self.num_identified_ships += 1
                            self.identified_threat_types += 1
                            self.score += self.threat_points

                        if self.config["verbose"]: print("Ship {} observed by aircraft {}".format(agent.agent_idx, aircraft_id))

                    # if in the aircraft's engagement range, identify threat level
                    if not agent.observed_threat and (self.agents[aircraft_id].in_engagement_range(distance=dist)):# or agent.threat == 0):
                        agent.observed_threat = True
                        self.num_identified_ships += 1
                        self.identified_threat_types += 1 # Used for the tally box
                        self.score += self.threat_points
                        self.new_weapon_id = ['human' if aircraft_id == self.human_idx else 'AI', 'weapon_identified', agent.agent_idx]
                        #print('new weapon id: ', self.new_weapon_id)
                        if self.config["verbose"]: print("Ship {} threat level identified by aircraft {}".format(agent.agent_idx, aircraft_id))
                        break

                    # if in the ship's weapon range, damage the aircraft
                    if agent.in_weapon_range(distance=dist) and agent.threat > 0:
                        if self.agents[aircraft_id].alive:
                            if random.random() < self.agents[aircraft_id].prob_detect:

                                if self.missiles_enabled:
                                    agents.Missile(self, 0, max_health=10, color=(200, 0, 0),speed=self.config['game speed'] * self.config['human speed']*0.9,flight_pattern=self.config["search pattern"], target_aircraft_id=aircraft_id)
                                    self.agents[-1].x, self.agents[-1].y = agent.x, agent.y
                                    self.agents[-1].waypoint_override = (self.agents[aircraft_id].x, self.agents[aircraft_id].y)

                                elif aircraft_id == self.num_ships: # agent damaged
                                    self.agents[aircraft_id].health_points -= 1
                                    self.agents[aircraft_id].draw_damage()
                                    print(f'Agent {aircraft_id} -1 HP')
                                    self.agents[aircraft_id].damage = ((4 - self.agents[aircraft_id].health_points) / 4) * 100
                                    #self.agent_damage_flash_start = pygame.time.get_ticks()
                                    #self.agent_damage_flash_alpha = 255  # Start fully opaque

                                else: # Human damaged
                                    self.agents[aircraft_id].health_points -= 1
                                    print(f'Human -1 HP')
                                    self.agents[aircraft_id].damage = ((4 - self.agents[aircraft_id].health_points) / 4) * 100
                                    self.damage_flash_start = pygame.time.get_ticks()
                                    self.damage_flash_alpha = 255  # Start fully opaque

                                if self.agents[aircraft_id].health_points <= 0:
                                    self.agents[aircraft_id].damage = 100
                                    if self.config['infinite health'] == False:
                                        self.agents[aircraft_id].alive = False
                                        print(f'Aircraft {aircraft_id} destroyed')
                                        if self.training: # Respawn in training round
                                            agents.Aircraft(self, 0, prob_detect=0.04, max_health=10,color=self.AIRCRAFT_COLORS[1],speed=self.config['game speed'] * self.config['human speed'] * 1.1,flight_pattern=self.config["search pattern"])  # Human
                                            self.human_idx += 1


        agent0 = self.agents[self.aircraft_ids[0]]
        hostile_targets_nearby = sum(1 for agent in self.agents
                                     if agent.agent_class == "ship"
                                     and agent.threat > 0
                                     and agent.observed_threat
                                     and math.hypot(agent.x - agent0.x, agent.y - agent0.y) <= 30)
        risk_level_function = 10*hostile_targets_nearby + (100 - 10*self.agents[self.num_ships].health_points) # TODO tune this

        self.risk_level = 'LOW' if risk_level_function <= 30 else 'MEDIUM' if risk_level_function <= 60 else 'HIGH' if risk_level_function <= 80 else 'EXTREME'

        # Check if any aircraft are recently deceased (RIP)
        if not self.agents[self.num_ships].alive and not self.agent0_dead:
            self.agent0_dead = True
            self.score += self.wingman_dead_points
            print(f'{self.wingman_dead_points} points for agent wingman destruction')

        # progress update
        if self.config["verbose"]: print("   Found:", self.num_identified_ships, "Total:", len(self.agents) - len(self.aircraft_ids), "Damage:", self.damage)

        # Update human's quadrant
        if self.agents[self.human_idx].x < (self.config['gameboard size'] / 2):
            if self.agents[self.human_idx].y < (self.config['gameboard size'] / 2): self.human_quadrant = 'NW'
            else: self.human_quadrant = 'SW'
        else:
            if self.agents[self.human_idx].y < (self.config['gameboard size'] / 2): self.human_quadrant = 'NE'
            else: self.human_quadrant = 'SE'


        # exit when all ships are identified
        state = self.get_state()  # TODO This should eventually be replaced with get_observation()
        reward = self.score - old_score
        done = (self.num_identified_ships >= self.num_ships) or (not self.agents[self.human_idx].alive and not self.config['infinite health'] and not self.training) or (self.display_time/1000 >= self.time_limit)

        if done:
            self.done = True
            print('Done!')

            self.score += self.human_hp_remaining_points * self.agents[self.num_ships+1] # Add points for human HP remaining at end of round (always, not just if round ended early)

            if self.num_identified_ships >= self.num_ships: # Add points for finishing early
                self.score += self.all_targets_points
                self.score += (self.time_limit - self.display_time/1000)*self.time_points

            elif not self.agents[self.human_idx].alive:
                self.score += self.human_dead_points
                print('Human aircraft destroyed, game over.')

            elif self.display_time/1000 >= self.time_limit:
                print('Out of time, game over.')

            print(f'\nTargets identified: {self.identified_targets} / {self.total_targets} ({self.identified_targets * 10} points)')
            print(f'Threat levels identified: {self.identified_threat_types} / {self.total_targets} ({self.identified_threat_types * 5} points)')
            if self.num_identified_ships >= len(self.agents) - len(self.aircraft_ids):
                print(f"All targets identified (+{self.all_targets_points} points)")
                print(f"{round(self.config['time limit'] - self.display_time/1000,1)} * {self.time_points} = {round((self.config['time limit'] - self.display_time / 1000) * self.time_points,0)} points added for time remaining")
            print(f"Time remaining: {round(self.config['time limit'] - self.display_time/1000,1)} seconds")
            print(f"\nFinal score = {round(self.score,0)}")

            if self.render_bool: pygame.time.wait(50)

        return state, reward, done, {}


    def get_observation(self):
        # TODO
        """
        State will include the following features (current count is 444):
            # Agent and basic game info (13 features):
            agent_health,        # (0-1)
            agent_x,             # (0-1) (discretize map into 100x100 grid, then normalize)
            agent_y,             # (0-1)
            agent_alive,         # (0 or 1)
            agent_waypoint_x     # (0-1) x coordinate of agent's current waypoint
            agent_waypoint_y     # (0-1) y coordinate of agent's current waypoint
            teammate_health,     # (0-1)
            teammate_x,          # (0-1)
            teammate_y,          # (0-1)
            teammate_waypoint_x  # (0-1) x coordinate of current waypoint
            teammate_waypoint_y  # (0-1) y coordinate of current waypoint
            teammate_alive,      # (0 or 1)
            time_remaining       # (0-1)

            # Target data (the following are repeated for all 60 targets) (7*60 = 420 features):
            target_x,            # (0-1)
            target_y,            # (0-1)
            id_unknown,          # one-hot (0 or 1)
            id_friendly,         # one-hot (0 or 1)
            id_hostile,          # one-hot (0 or 1)
            threat_known,        # (0 or 1)
            threat_level         # (0.33, 0.66, 1.0) for small/med/large weapon TODO: Figure out if this is too ambiguous for the agent to learn

            # Handcrafted features (11 features):
            unknown_targets_NW      # count normalized (0-1)
            unknown_targets_NE      # count normalized (0-1)
            unknown_targets_SW      # count normalized (0-1)
            unknown_targets_SE      # count normalized (0-1)
            nearest_target_dist     # (0-1)
            nearest_hostile_dist    # (0-1)
            agent_in_weapon_range,     # (0 or 1)
            teammate_in_weapon_range,  # (0 or 1)
            last_25_seconds,        # (0 or 1)
            targets_remaining,      # normalized (0-1)
            threats_remaining       # normalized (0-1)
        """
        pass
        self.observation[0] = self.agents[self.num_ships].health_points / 10.0  # Agent health
        self.observation[1] = self.agents[self.num_ships].x / self.config["gameboard size"] # Agent x
        self.observation[2] = self.agents[self.num_ships].y / self.config["gameboard size"] # Agent y
        self.observation[3] = self.agents[self.num_ships].alive # Agent alive status
        self.observation[4] = self.agents[self.num_ships].target_point[0] # TODO Confirm this is the variable for agent waypoint
        self.observation[5] = self.agents[self.num_ships].target_point[1] # TODO Confirm this is the variable for agent waypoint

        self.observation[6] = self.agents[self.human_idx].health_points / 10.0  # Teammate health
        self.observation[7] = self.agents[self.human_idx].x / self.config["gameboard size"] # Teammate x
        self.observation[8] = self.agents[self.human_idx].y / self.config["gameboard size"] # Teammate y
        self.observation[9] = self.agents[self.human_idx].target_point[0]  # TODO Confirm this is the variable for agent waypoint
        self.observation[10] = self.agents[self.human_idx].target_point[1]  # TODO Confirm this is the variable for agent waypoint
        self.observation[11] = self.agents[self.human_idx].alive
        #self.observation[12] = #TODO Time remaining

        # TODO Everything below is incomplete so far
        # Vectorized target processing
        target_idx = 0
        target_start_idx = 9  # Where target features begin

        # Get all target positions at once
        target_positions = np.array([[agent.x, agent.y] for agent in self.agents if agent.agent_class == "ship"])

        # Process all targets at once where possible
        for agent in (a for a in self.agents if a.agent_class == "ship"):
            idx = target_start_idx + (target_idx * 8)
            self.target_data[target_idx] = [
                agent.x / self.config["gameboard size"],
                agent.y / self.config["gameboard size"],
                1 if not agent.observed else 0,  # unknown
                1 if agent.observed and agent.threat == 0 else 0,  # friendly
                1 if agent.observed and agent.threat > 0 else 0,  # hostile
                1 if not agent.observed_threat else 0,  # threat unknown
                1 if agent.observed_threat else 0,  # threat known
                agent.threat / 3.0 if agent.observed_threat else 0  # normalized threat level
            ]
            target_idx += 1

        self.observation[target_start_idx:target_start_idx + self.num_ships * 8] = self.target_data.flatten()



    def add_comm_message(self,message,is_ai=True):
        #timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        sender = "AGENT" if is_ai else "HUMAN"
        full_message = f"{sender}: {message}"
        self.comm_messages.append((full_message, is_ai))
        if len(self.comm_messages) > self.max_messages:
            self.comm_messages.pop(0)

    def render(self, mode='human', close=False):
        window_width, window_height = self.config['window size'][0], self.config['window size'][0]
        game_width = self.config["gameboard size"]
        ui_width = window_width - game_width

        if self.agent_info_height_req > 0:
            self.comm_pane_height = 220+self.agent_info_height_req
        else: self.comm_pane_height = 10

        # gameboard background
        self.window.fill((255, 255, 255))  # white background
        self.__render_box__(1, (0, 0, 0), 3)  # outer box
        #pygame.draw.line(self.window, (0, 0, 0), (self.config["gameboard size"] // 2, 0), (self.config["gameboard size"] // 2, self.config["gameboard size"]), 2)
        #pygame.draw.line(self.window, (0, 0, 0), (0, self.config["gameboard size"] // 2), (self.config["gameboard size"], self.config["gameboard size"] // 2), 2)
        pygame.draw.rect(self.window, (100, 100, 100), (game_width+self.gameboard_offset, 0, ui_width, window_height))
        pygame.draw.rect(self.window, (100, 100, 100), (0, game_width, game_width, window_height))  # Fill bottom portion with gray

        current_time = pygame.time.get_ticks()

        # Draw the agents
        for agent in self.agents:
            agent.draw(self.window)

        # Draw green lines and black crossbars
        self.__render_box__(self.config["gameboard border margin"], (0, 128, 0), 2)  # inner box
        pygame.draw.line(self.window, (0, 0, 0), (self.config["gameboard size"] // 2, 0),(self.config["gameboard size"] // 2, self.config["gameboard size"]), 2)
        pygame.draw.line(self.window, (0, 0, 0), (0, self.config["gameboard size"] // 2),(self.config["gameboard size"], self.config["gameboard size"] // 2), 2)

        # Draw white rectangles around outside edge
        pygame.draw.rect(self.window, (255,255,255),(0,0,game_width,35)) # Top
        pygame.draw.rect(self.window, (255,255,255), (0, game_width-33, game_width, 33)) # bottom
        pygame.draw.rect(self.window, (255,255,255), (0, 0, 35, game_width))  # Left
        pygame.draw.rect(self.window, (255,255,255), (1000-33, 0, 35, game_width-2))  # Right

        # Handle damage flashes when human is damaged
        if current_time > 1000 and (current_time - self.damage_flash_start < self.damage_flash_duration):
            progress = (current_time - self.damage_flash_start) / self.damage_flash_duration  # Calculate alpha based on time elapsed
            alpha = int(255 * (1 - progress))
            border_surface = pygame.Surface((self.config["gameboard size"], self.config["gameboard size"]),pygame.SRCALPHA)
            border_width = 50
            border_color = (255, 0, 0, alpha)  # Red with calculated alpha
            pygame.draw.rect(border_surface, border_color,(0, 0, self.config["gameboard size"], border_width))  # Top border
            pygame.draw.rect(border_surface, border_color, (0, self.config["gameboard size"] - border_width, self.config["gameboard size"],border_width))  # Bottom border
            pygame.draw.rect(border_surface, border_color,(0, 0, border_width, self.config["gameboard size"]))  # Left border
            pygame.draw.rect(border_surface, border_color, (
            self.config["gameboard size"] - border_width, 0, border_width,
            self.config["gameboard size"]))  # Right border
            self.window.blit(border_surface, (0, 0))  # Blit the border surface onto the main window

        # Handle flash when agent is damaged (TODO: Make this a different graphic)
        if current_time > 1000 and (current_time - self.agent_damage_flash_start < self.damage_flash_duration):
            progress = (current_time - self.agent_damage_flash_start) / self.damage_flash_duration  # Calculate alpha based on time elapsed
            alpha = int(255 * (1 - progress))
            border_surface = pygame.Surface((self.config["gameboard size"], self.config["gameboard size"]),pygame.SRCALPHA)
            border_width = 50
            border_color = (255, 0, 0, alpha)  # Red with calculated alpha
            pygame.draw.rect(border_surface, border_color,(0, 0, self.config["gameboard size"], border_width))  # Top border
            pygame.draw.rect(border_surface, border_color, (0, self.config["gameboard size"] - border_width, self.config["gameboard size"],border_width))  # Bottom border
            pygame.draw.rect(border_surface, border_color,(0, 0, border_width, self.config["gameboard size"]))  # Left border
            pygame.draw.rect(border_surface, border_color, (
            self.config["gameboard size"] - border_width, 0, border_width,
            self.config["gameboard size"]))  # Right border
            self.window.blit(border_surface, (0, 0))  # Blit the border surface onto the main window

        elif self.agents[self.human_idx].health_points <= 3:
            alpha = int(155)
            border_surface = pygame.Surface((self.config["gameboard size"], self.config["gameboard size"]),pygame.SRCALPHA)
            border_width = 35
            border_color = (255, 0, 0, alpha)  # Red with calculated alpha
            pygame.draw.rect(border_surface, border_color,(0, 0, self.config["gameboard size"], border_width))  # Top border
            pygame.draw.rect(border_surface, border_color, (0, self.config["gameboard size"] - border_width, self.config["gameboard size"],border_width))  # Bottom border
            pygame.draw.rect(border_surface, border_color,(0, 0, border_width, self.config["gameboard size"]))  # Left border
            pygame.draw.rect(border_surface, border_color, (self.config["gameboard size"] - border_width, 0, border_width,self.config["gameboard size"]))  # Right border
            self.window.blit(border_surface, (0, 0))  # Blit the border surface onto the main window

        # Draw Agent Gameplan sub-window
        self.quadrant_button_height = 120
        self.gameplan_button_width = 180

        pygame.draw.rect(self.window, (230,230,230), pygame.Rect(self.right_pane_edge, 10, 405, 665))  # Agent gameplan sub-window box
        gameplan_text_surface = pygame.font.SysFont(None, 36).render('Agent Gameplan', True, (0,0,0))
        self.window.blit(gameplan_text_surface, gameplan_text_surface.get_rect(center=(self.right_pane_edge+425 // 2, 10+40 // 2)))
        pygame.draw.line(self.window, (0, 0, 0), (self.right_pane_edge, 10),(self.right_pane_edge + 405, 10), 4)  # Top edge of gameplan panel
        pygame.draw.line(self.window, (0, 0, 0), (self.right_pane_edge, 10), (self.right_pane_edge, 675), 4)
        pygame.draw.line(self.window, (0, 0, 0), (self.right_pane_edge + 405, 10), (self.right_pane_edge + 405, 675), 4)
        pygame.draw.line(self.window, (0, 0, 0), (self.right_pane_edge, 10+665), (self.right_pane_edge + 405, 10+665),4)  # Top edge of gameplan panel

        #self.manual_priorities_button = Button("Manual Priorities", self.right_pane_edge + 15, 20,self.gameplan_button_width * 2 + 15, 65)
        self.manual_priorities_button.is_latched = self.button_latch_dict['manual_priorities']
        self.manual_priorities_button.color = (50, 180, 180)
        self.manual_priorities_button.draw(self.window)

        pygame.draw.line(self.window, (0, 0, 0), (self.right_pane_edge, 20+65+6), (self.right_pane_edge + 405, 20+65+6), 4)

        type_text_surface = pygame.font.SysFont(None, 26).render('SEARCH TYPE', True, (0,0,0))
        self.window.blit(type_text_surface, type_text_surface.get_rect(center=(self.right_pane_edge+425 // 2, 10+40+110 // 2)))

        #self.target_id_button = Button("TARGET", self.right_pane_edge + 15, 60+55, self.gameplan_button_width, 60)# (255, 120, 80))
        self.target_id_button.is_latched = self.button_latch_dict['target_id']
        self.target_id_button.color = self.gameplan_button_color
        self.target_id_button.draw(self.window)

        #self.wez_id_button = Button("WEAPON", self.right_pane_edge + 30 + self.gameplan_button_width, 60+55, self.gameplan_button_width, 60) # 15 pixel gap b/w buttons
        self.wez_id_button.is_latched = self.button_latch_dict['wez_id']
        self.wez_id_button.color = self.gameplan_button_color
        self.wez_id_button.draw(self.window)

        pygame.draw.line(self.window, (0, 0, 0), (self.right_pane_edge, 130+45+5),(self.right_pane_edge+405,130+45+5),4) # Separating line between target/WEZ ID selection and quadrant select

        search_area_text_surface = pygame.font.SysFont(None, 26).render('SEARCH AREA', True, (0, 0, 0))
        self.window.blit(search_area_text_surface,search_area_text_surface.get_rect(center=(self.right_pane_edge + 425 // 2, 50 + 10 + 40 + 195 // 2)))

        #self.NW_quad_button = Button("NW", self.right_pane_edge + 15, 60+80+10+10+50, self.gameplan_button_width, self.quadrant_button_height)
        self.NW_quad_button.is_latched = self.button_latch_dict['NW']
        self.NW_quad_button.color = self.gameplan_button_color
        self.NW_quad_button.draw(self.window)

        #self.NE_quad_button = Button("NE", self.right_pane_edge + 30 + self.gameplan_button_width, 60+80+10+10+50, self.gameplan_button_width, self.quadrant_button_height)
        self.NE_quad_button.is_latched = self.button_latch_dict['NE']
        self.NE_quad_button.color = self.gameplan_button_color
        self.NE_quad_button.draw(self.window)

        #self.SW_quad_button = Button("SW", self.right_pane_edge + 15, 50+2*(self.quadrant_button_height)+50, self.gameplan_button_width, self.quadrant_button_height)
        self.SW_quad_button.is_latched = self.button_latch_dict['SW']
        self.SW_quad_button.color = self.gameplan_button_color
        self.SW_quad_button.draw(self.window)

        #self.SE_quad_button = Button("SE", self.right_pane_edge + 30 + self.gameplan_button_width, 50+2*(self.quadrant_button_height)+50, self.gameplan_button_width, self.quadrant_button_height)
        self.SE_quad_button.is_latched = self.button_latch_dict['SE']
        self.SE_quad_button.color = self.gameplan_button_color
        self.SE_quad_button.draw(self.window)

        #self.full_quad_button = Button("Full", self.right_pane_edge+200-35-10, 60+2*(80+10)+20-35+5+50, 100, 100)
        self.full_quad_button.color = self.gameplan_button_color#(50,180,180)
        self.full_quad_button.is_latched = self.button_latch_dict['full']
        self.full_quad_button.draw(self.window)

        pygame.draw.line(self.window, (0, 0, 0), (self.right_pane_edge, 465), (self.right_pane_edge + 405, 465),4)  # Separating line between quadrant select and hold/waypoint

        #self.waypoint_button = Button("WAYPOINT", self.right_pane_edge + 30 + self.gameplan_button_width, 3*(self.quadrant_button_height) + 115, self.gameplan_button_width, 80)
        self.waypoint_button.is_latched = self.button_latch_dict['waypoint']
        self.waypoint_button.color = self.gameplan_button_color
        self.waypoint_button.draw(self.window)

        #self.hold_button = Button("HOLD", self.right_pane_edge + 15, 3*(self.quadrant_button_height) + 115, self.gameplan_button_width, 80)
        self.hold_button.is_latched = self.button_latch_dict['hold']
        self.hold_button.color = self.gameplan_button_color
        self.hold_button.draw(self.window)

        pygame.draw.line(self.window, (0, 0, 0), (self.right_pane_edge, 3 * (self.quadrant_button_height) + 115 + 90), (self.right_pane_edge + 405, 3 * (self.quadrant_button_height) + 115 + 90),4)  # Separating line between hold/waypoint and regroup/tag team

        self.autonomous_button = Button("Auto Priorities", self.right_pane_edge + 15, 3 * (self.quadrant_button_height) + 115 + 90+20,self.gameplan_button_width * 2 + 15, 65)
        self.autonomous_button.is_latched = self.button_latch_dict['autonomous']
        self.autonomous_button.color = (50, 180, 180)
        self.autonomous_button.draw(self.window)

        # Advanced gameplans currently removed
        # self.regroup_button.is_latched = self.regroup_clicked
        # self.regroup_button.color = self.gameplan_button_color
        # self.regroup_button.draw(self.window)

        # self.tag_team_button.is_latched = self.button_latch_dict['tag_team']
        # self.tag_team_button.color = self.gameplan_button_color
        # self.tag_team_button.draw(self.window)

        # self.fan_out_button.is_latched = self.button_latch_dict['fan_out']
        # self.fan_out_button.color = self.gameplan_button_color
        # self.fan_out_button.draw(self.window)

        # Draw Comm Log
        pygame.draw.rect(self.window, (200, 200, 200), pygame.Rect(self.comm_pane_edge, self.comm_pane_height+680, 400, 40))  # Comm log title box
        pygame.draw.rect(self.window, (230,230,230), pygame.Rect(self.comm_pane_edge, self.comm_pane_height+35+680, 400, 150))  # Comm Log sub-window box
        comm_text_surface = pygame.font.SysFont(None, 28).render('COMM LOG', True, (0, 0, 0))
        self.window.blit(comm_text_surface, comm_text_surface.get_rect(center=(self.comm_pane_edge + 395 // 2, self.comm_pane_height + 40+1320 // 2)))

        # Draw incoming comm log text
        y_offset = self.comm_pane_height+50+680
        for entry in self.comm_messages:
            message = entry[0]
            is_ai = entry[1]
            color = self.ai_color if is_ai else self.human_color
            message_surface = self.message_font.render(message, True, color)
            self.window.blit(message_surface, (self.comm_pane_edge+10, y_offset))
            y_offset += 30  # Adjust this value to change spacing between messages


        #Draw score box and update with new score value every tick
        #pygame.draw.rect(self.window, (230, 230, 230),pygame.Rect(self.comm_pane_edge, self.comm_pane_height + 35 + 850, 400,100))
        #score_button = ScoreWindow(self.score,1300,  1300)
        #score_button.update(self.score)
        #score_button.draw(self.window)

        # # Draw point tally
        # self.target_status_x = self.config['gameboard size'] + 40 + 405
        # if self.agent_info_height_req > 0: self.target_status_y = 500 + self.agent_info_height_req
        # else: self.target_status_y = 280
        #
        # pygame.draw.rect(self.window, (230, 230, 230), pygame.Rect(self.right_pane_edge, self.autonomous_button_y+200, 400, 100))  # Target tally sub-window box
        # pygame.draw.rect(self.window, (200, 200, 200),pygame.Rect(self.right_pane_edge, self.autonomous_button_y+200, 400,40))  # Target tally title box
        # tally_title_surface = pygame.font.SysFont(None, 36).render('SCORE', True, (0, 0, 0))
        # self.window.blit(tally_title_surface, tally_title_surface.get_rect(center=(self.right_pane_edge + 400 // 2, self.autonomous_button_y+240 // 2)))
        #
        # id_tally_text = f"Targets ID\'d (+10 pts):                       {self.identified_targets} / {self.total_targets}"
        # id_tally_surface = self.tally_font.render(id_tally_text, True, (0, 0, 0))
        # self.window.blit(id_tally_surface, (self.right_pane_edge+10, self.autonomous_button_y+250-100))
        #
        # threat_tally_text = f"WEZs ID\'d (+5 pts):                            {self.identified_threat_types} / {self.total_targets}"
        # threat_tally_surface = self.tally_font.render(threat_tally_text, True, (0, 0, 0))
        # self.window.blit(threat_tally_surface, (self.right_pane_edge+10, self.autonomous_button_y+275-100))

        # Draw health boxes
        if self.config['num aircraft'] > 1:
            agent0_health_window = HealthWindow(self.num_ships,10,game_width+5, 'AGENT HP',self.AIRCRAFT_COLORS[0])
            agent0_health_window.update(self.agents[self.num_ships].health_points)
            agent0_health_window.draw(self.window)

        agent1_health_window = HealthWindow(self.human_idx, game_width-150, game_width + 5, 'HUMAN HP',self.AIRCRAFT_COLORS[1])
        agent1_health_window.update(self.agents[self.human_idx].health_points)
        agent1_health_window.draw(self.window)

        current_time = pygame.time.get_ticks()
        if not self.paused:
            self.display_time = current_time - self.total_pause_time

        if current_time > self.start_countdown_time:
            self.time_window.update(self.display_time)
            self.time_window.draw(self.window)

        # Draw agent status window
        if self.agent_info_height_req > 0: self.agent_info_display.draw(self.window)

        corner_round_text = f"ROUND {self.round_number+1}/4" if self.user_group == 'test' else f"ROUND {self.round_number}/4"
        # TODO CHANGE
        corner_round_font = pygame.font.SysFont(None, 36)
        corner_round_text_surface = corner_round_font.render(corner_round_text, True, (255, 255, 255))
        corner_round_rect = corner_round_text_surface.get_rect(
            center=(675, 1030))
        self.window.blit(corner_round_text_surface, corner_round_rect)


        # Countdown from 5 seconds at start of game
        if current_time <= self.start_countdown_time:
            countdown_font = pygame.font.SysFont(None, 120)
            message_font = pygame.font.SysFont(None, 60)
            round_font = pygame.font.SysFont(None, 72)
            countdown_start = 0
            countdown_surface = pygame.Surface((self.window.get_width(), self.window.get_height()))
            countdown_surface.set_alpha(128)  # 50% transparent

            time_left = self.start_countdown_time/1000 - (current_time - countdown_start) / 1000

            # Draw semi-transparent overlay
            countdown_surface.fill((100, 100, 100))
            self.window.blit(countdown_surface, (0, 0))

            # Draw round name
            if self.user_group == 'test':
                round_text = f"ROUND {self.round_number+1}/4"
            else:
                if self.round_number == 0: round_text = "TRAINING ROUND"
                else: round_text = f"ROUND {self.round_number}/4"
            round_text_surface = round_font.render(round_text, True, (255, 255, 255))
            round_rect = round_text_surface.get_rect(center=(self.window.get_width() // 2, self.window.get_height() // 2 - 120))
            self.window.blit(round_text_surface, round_rect)

            # Draw "Get Ready!" message
            ready_text = message_font.render("Get Ready!", True, (255, 255, 255))
            ready_rect = ready_text.get_rect(center=(self.window.get_width() // 2, self.window.get_height() // 2 - 50))
            self.window.blit(ready_text, ready_rect)

            # Draw countdown number
            countdown_text = countdown_font.render(str(max(1, int(time_left + 1))), True, (255, 255, 255))
            text_rect = countdown_text.get_rect(center=(self.window.get_width() // 2, self.window.get_height() // 2 + 20))
            self.window.blit(countdown_text, text_rect)

            pygame.time.wait(50)  # Control update rate

            # Handle any quit events during countdown
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return


        if self.paused and not self.unpause_countdown:
            pause_surface = pygame.Surface((self.window.get_width(), self.window.get_height()))
            pause_surface.set_alpha(128*2)  # 50% transparent
            pause_surface.fill((100, 100, 100))  # Gray color
            self.window.blit(pause_surface, (0, 0))

            pause_text = self.pause_font.render('GAME PAUSED', True, (255, 255, 255))
            pause_subtext = self.pause_subtitle_font.render('[RIGHT CLICK TO UNPAUSE]', True, (255, 255, 255))
            text_rect = pause_text.get_rect(center=(self.window.get_width() // 2, self.window.get_height() // 2))
            pause_sub_rect = pause_subtext.get_rect(center=(self.window.get_width() // 2, (self.window.get_height() // 2) + 45))

            self.window.blit(pause_text, text_rect)
            self.window.blit(pause_subtext, pause_sub_rect)

        if self.done:
            self._render_game_complete()

        pygame.display.update()
        self.clock.tick(60)

    # convert the environment into a state dictionary
    def get_state(self):
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
                    "damage": agent.health_points
                }
        return state

    # utility function for drawing a square box
    def __render_box__(self, distance_from_edge, color=(0,0,0), width=2):
        pygame.draw.line(self.window, color, (distance_from_edge, distance_from_edge), (distance_from_edge, self.config["gameboard size"] - distance_from_edge), width)
        pygame.draw.line(self.window, color, (distance_from_edge, self.config["gameboard size"] - distance_from_edge), (self.config["gameboard size"] - distance_from_edge, self.config["gameboard size"] - distance_from_edge), width)
        pygame.draw.line(self.window, color, (self.config["gameboard size"] - distance_from_edge, self.config["gameboard size"] - distance_from_edge), (self.config["gameboard size"] - distance_from_edge, distance_from_edge), width)
        pygame.draw.line(self.window, color, (self.config["gameboard size"] - distance_from_edge, distance_from_edge), (distance_from_edge, distance_from_edge), width)


    def pause(self, unpause_key):
        print('Game paused')
        self.pause_start_time = pygame.time.get_ticks()
        self.button_latch_dict['pause'] = True
        #print('paused at %s (env.display_time = %s)' % (self.pause_start_time, self.display_time))
        self.paused = True

        countdown_font = pygame.font.SysFont(None, 120)
        countdown_duration = 3  # seconds

        while self.paused:
            pygame.time.wait(50)  # Reduced wait time for smoother rendering
            self.render()

            ev = pygame.event.get()
            for event in ev:
                #if event.type == unpause_key:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 3:
                        # Start countdown
                        countdown_start = pygame.time.get_ticks()
                        countdown_surface = pygame.Surface((self.window.get_width(), self.window.get_height()))
                        countdown_surface.set_alpha(128)  # 50% transparent

                        while (pygame.time.get_ticks() - countdown_start) < countdown_duration * 1000:
                            self.unpause_countdown = True
                            current_time = pygame.time.get_ticks()
                            time_left = countdown_duration - (current_time - countdown_start) / 1000

                            # Regular render
                            self.render()

                            # Draw countdown
                            countdown_text = countdown_font.render(str(max(1, int(time_left + 1))), True, (255, 255, 255))
                            text_rect = countdown_text.get_rect(
                                center=(self.window.get_width() // 2, self.window.get_height() // 2))

                            # Draw semi-transparent overlay
                            countdown_surface.fill((100, 100, 100))
                            self.window.blit(countdown_surface, (0, 0))

                            # Draw countdown number
                            self.window.blit(countdown_text, text_rect)

                            pygame.display.update()
                            pygame.time.wait(50)  # Control update rate

                            # Handle any quit events during countdown
                            for evt in pygame.event.get():
                                if evt.type == pygame.QUIT:
                                    pygame.quit()
                                    return

                        self.paused = False
                        self.unpause_countdown = False
                        self.button_latch_dict['pause'] = False
                        pause_end_time = pygame.time.get_ticks()
                        pause_duration = pause_end_time - self.pause_start_time
                        self.total_pause_time += pause_duration
                        print('Paused for %s' % pause_duration)
                        return  # Exit the pause function

    def SAGAT_survey(self,survey_index):
        if survey_index == 1:
            webbrowser.open_new_tab('https://gatech.co1.qualtrics.com/jfe/form/SV_egiLZSvblF8SVO6?subject_id='+str(self.subject_id)+'&scenario_number='+str(self.round_number)+'&user_group='+str(self.user_group)+'&survey_number=1')
        elif survey_index == 2:
            webbrowser.open_new_tab('https://gatech.co1.qualtrics.com/jfe/form/SV_egiLZSvblF8SVO6?subject_id='+str(self.subject_id)+'&scenario_number='+str(self.round_number)+'&user_group='+str(self.user_group)+'&survey_number=2')
        elif survey_index == 3:
            webbrowser.open_new_tab('https://gatech.co1.qualtrics.com/jfe/form/SV_egiLZSvblF8SVO6?subject_id='+str(self.subject_id)+'&scenario_number='+str(self.round_number)+'&user_group='+str(self.user_group)+'&survey_number=3')

        self.pause(pygame.MOUSEBUTTONDOWN)

    def _render_game_complete(self):
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
        if not self.agents[self.human_idx].alive:
            title_surface = title_font.render('GAME OVER', True, (0, 0, 0))
        elif self.display_time/1000 >= self.time_limit:
            title_surface = title_font.render('GAME COMPLETE: TIME UP', True, (0, 0, 0))
        else:
            title_surface = title_font.render('GAME COMPLETE', True, (0, 0, 0))

        self.window.blit(title_surface, title_surface.get_rect(center=(window_x + window_width // 2, window_y + 30)))

        # Calculate statistics
        agent_status = "ALIVE" if self.agents[self.num_ships].alive else "DESTROYED"
        agent_status_color = (0, 255, 0) if agent_status == "ALIVE" else (255, 0, 0)
        human_status = 'ALIVE' if self.agents[self.human_idx].alive else "DESTROYED"
        human_status_color = (0, 255, 0) if human_status == "ALIVE" else (255, 0, 0)

        # Create stats text surfaces
        stats_items = [
            f"Final Score: {round(self.score,0)}",
            f"Targets Identified: {self.identified_targets} / {self.total_targets}",
            f"Threat Levels Observed: {self.identified_threat_types} / {self.total_targets}",
            f"Human Status: {human_status}",
            f"Agent Status: {agent_status}"]

        # Render stats
        y_offset = window_y + 100
        for i, text in enumerate(stats_items):
            if i == len(stats_items) - 2:  # Human Status line
                text_surface = stats_font.render(text.split(': ')[0] + ': ', True, (0, 0, 0))
                status_surface = stats_font.render(human_status, True, human_status_color)

                # Center align the text
                total_width = text_surface.get_width() + status_surface.get_width()
                start_x = window_x + (window_width - total_width) // 2

                self.window.blit(text_surface, (start_x, y_offset))
                self.window.blit(status_surface, (start_x + text_surface.get_width(), y_offset))

            elif i == len(stats_items) - 1:  # Agent Status line
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
            y_offset += 50

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

        #self.done = False
