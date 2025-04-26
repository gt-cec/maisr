import gymnasium as gym
import numpy as np
import pygame
import random
import agents
from gui import Button, ScoreWindow, HealthWindow, TimeWindow, AgentInfoDisplay
import datetime
import math
import webbrowser


class MAISREnvVec(gym.Env):
    """Multi-Agent ISR Environment following the Gym format"""

    def __init__(self, config={}, window=None, clock=None, render_mode='none',
                 agent_training=False,
                 obs_type = 'vector', action_type = 'continuous', reward_type = 'balanced-sparse',
                 subject_id='99',user_group='99',round_number='99'):
        """
        args:
            time_scale: How much to scale time by
            render_mode: 'none', 'human'
            reward_type:
                balanced-sparse: Points for IDing targets, weapons, and finishing early
                balanced-dense: Variant 1 + penalty for damage + others
                cautious-sparse: BS but very low reward for IDing weapons
                aggressive-sparse: BS but very high reward for IDing weapons

        """
        super().__init__()

        if obs_type not in ['vector', 'cnn']: raise ValueError(f"obs_type must be one of 'vector,'cnn, got '{obs_type}'")
        if action_type not in ['discrete', 'continuous']: raise ValueError(f"action_type must be one of 'discrete,'continuous, got '{action_type}'")
        if reward_type not in ['balanced-sparse']: raise ValueError('reward_type must be normal. Others coming soon')

        self.config = config
        self.verbose = True if self.config['verbose'] == 'true' else False

        self.obs_type = obs_type
        self.action_type = action_type
        self.reward_type = reward_type
        self.agent_training = agent_training

        self.subject_id = subject_id
        self.round_number = round_number
        self.user_group = user_group
        self.human_training = True if (self.round_number == 0 and self.user_group != 'test') else False  # True for the training round at start of experiment, false for rounds 1-4 (NOTE: This is NOT agent training!)

        self.init = True # Used to render static windows the first time
        self.paused = False
        self.unpause_countdown = False
        self.new_target_id = None  # Used to check if a new target has been ID'd so the data logger in main.py can log it
        self.new_weapon_id = None

        self.agent0_dead = False  # Used at end of loop to check if agent recently deceased.
        self.agent1_dead = False

        self.terminated = False
        self.truncated = False

        self.time_limit = self.config['time limit']

        # set the random seed
        if "seed" in config: random.seed(config["seed"])
        else: print("Note: no 'seed' specified in the env config.")

        # determine the number of ships
        self.num_targets = self.config['num ships']
        self.total_targets = self.num_targets



        if self.action_type == 'continuous':
            self.action_space = gym.spaces.Box(
                low=np.array([0, 0, 0], dtype=np.float32),
                high=np.array([1, 1, 1], dtype=np.float32),
                dtype=np.float32)
        else: raise ValueError("Action type discrete not supported yet")

        self.num_obs_features = 13 + 11 + 7 * self.num_targets

        if self.obs_type == 'vector':
            self.observation_space = gym.spaces.Box(
                low=0, high=1,
                shape=(self.num_obs_features,),  # Wrap in tuple to make it iterable
                dtype=np.float32)

            self.observation = np.zeros(self.num_obs_features, dtype=np.float32)
            self.target_data = np.zeros((self.num_targets, 7), dtype=np.float32)  # For target features
        else:
            raise ValueError("Obs type CNN not supported yet")


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
        self.AIRCRAFT_ENGAGEMENT_RADIUS = 40  # 100  # pixel width of aircraft engagement (to identify WEZ of threats)
        self.AIRCRAFT_ISR_RADIUS = 85  # 170  # pixel width of aircraft scanner (to identify hostile vs benign)

        self.GAMEBOARD_NOGO_RED = (255, 200, 200)  # color of the red no-go zone
        self.GAMEBOARD_NOGO_YELLOW = (255, 225, 200)  # color of the yellow no-go zone
        self.FLIGHTPLAN_EDGE_MARGIN = .2  # proportion distance from edge of gameboard to flight plan, e.g., 0.2 = 20% in, meaning a flight plan of (1,1) would go to 80%,80% of the gameboard
        self.AIRCRAFT_COLORS = [(0, 160, 160), (0, 0, 255), (200, 0, 200), (80, 80, 80)]  # colors of aircraft 1, 2, 3, ... add more colors here, additional aircraft will repeat the last color

        self.show_agent_waypoint = self.config['show agent waypoint']

        # Set point quantities for each event
        self.score = 0
        self.all_targets_points = 0  # All targets ID'd
        self.target_points = 10  # Each target ID'd
        self.threat_points = 5  # Each threat ID'd
        self.time_points = 15  # Points given per second remaining
        self.human_hp_remaining_points = 70
        self.wingman_dead_points = -300  # Points subtracted for agent wingman dying
        self.human_dead_points = -400  # Points subtracted for human dying

        if not agent_training:
            self.window = window
            self.clock = clock
            self.start_countdown_time = 5000  # How long in milliseconds to count down at the beginning of the game before it starts

            # Set GUI locations
            self.gameboard_offset = 0  # How far from left edge to start drawing gameboard
            self.window_x = self.config["window size"][0]
            self.window_y = self.config["window size"][1]
            if render_mode == 'human':
                self.window = pygame.display.set_mode((self.window_x, self.window_y))

            self.right_pane_edge = self.config['gameboard size'] + 20  # Left edge of gameplan button windows
            self.comm_pane_edge = self.right_pane_edge
            self.gameplan_button_width = 180
            self.quadrant_button_height = 120
            self.autonomous_button_y = 590

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

            self.agent_waypoint_clicked = False # Flag to determine whether clicking on the map sets the humans' waypoint or the agent's. True when "waypoint" gameplan button set.
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
            self.render_mode = render_mode
            self.pause_font = pygame.font.SysFont(None, 74)
            self.pause_subtitle_font = pygame.font.SysFont(None, 40)

            # For visual damage flash
            self.damage_flash_duration = 500  # Duration of flash in milliseconds
            self.damage_flash_start = 0  # When the last damage was taken
            self.damage_flash_alpha = 0  # Current opacity of flash effect
            self.agent_damage_flash_start = 0
            self.agent_damage_flash_alpha = 0
            self.last_health_points = {0: 10, 1: 10}  # Track health points to detect changes

            # Situational-awareness based agent transparency config
            self.agent_priorities = 'placeholder'

            # Calculate required height of agent status info
            self.agent_info_height_req = 0
            if self.config['show_low_level_goals']: self.agent_info_height_req += 1
            if self.config['show_high_level_goals']: self.agent_info_height_req += 1.7
            if self.config['show_tracked_factors']: self.agent_info_height_req += 1.7
            if self.agent_info_height_req > 0: # Only render agent info display if at least one of the info elements is used
                self.agent_info_display = AgentInfoDisplay(self.comm_pane_edge, 10, 445, 40+35*self.agent_info_height_req)

            self.time_window = TimeWindow(self.config["gameboard size"] * 0.43, self.config["gameboard size"]+5,current_time=self.display_time, time_limit=self.time_limit)

        self.reset()


    def reset(self):
        self.agents = []
        self.aircraft_ids = []  # indexes of the aircraft agents
        self.damage = 0  # total damage from all agents
        self.num_identified_ships = 0  # number of ships with accessed threat levels, used for determining game end
        self.display_time = 0  # Time that is used for the on-screen timer. Accounts for pausing.
        self.pause_start_time = 0
        self.total_pause_time = 0
        self.score = 0
        self.identified_targets = 0
        self.identified_threat_types = 0
        self.init = True

        # Create vectorized ships/targets. Format: [id, value, info_level, x_pos, y_pos]
        self.targets = np.zeros((self.num_targets, 5), dtype=np.float32)
        self.targets[:, 0] = np.arange(self.num_targets) # Assign IDs (column 0)
        self.targets[:, 1] = np.random.choice([0, 1], size=self.num_targets, p=[0.67, 0.33]) # Assign values (column 1) - 67% regular (0), 33% high-value (1)
        self.targets[:, 2] = 0 # Initialize info_level (column 2) to all 0 (unknown)
        self.targets[:, 3] = np.random.uniform(35, self.config["gameboard size"] - 35, size=self.num_targets) # Randomly place targets on gameboard (columns 3-4)
        self.targets[:, 4] = np.random.uniform(35, self.config["gameboard size"] - 35, size=self.num_targets)

        # create the ships
        #for i in range(self.num_targets):
            #agents.Ship(self)  # create the agent, will add itself to the env

        # create the aircraft
        # Agent speed was originally set by self.config['agent speed'] but currently overridden with static value
        print('Num targets: ', self.num_targets)
        print(range(self.config['num aircraft']))
        for i in range(self.config['num aircraft']):
            print(i)
            agents.Aircraft(self, 0,prob_detect=0.0015,max_health=10,color=self.AIRCRAFT_COLORS[0],speed=self.config['game speed']*self.config['human speed'], flight_pattern=self.config["search pattern"]) # Agent

            #self.agents[self.num_targets+i].x, self.agents[self.num_targets+i].y = self.config['agent start location']
            self.agents[self.aircraft_ids[i]].x, self.agents[self.aircraft_ids[i]].y = self.config['agent start location']

        #agents.Aircraft(self, 0,prob_detect=0.04,max_health=10,color=self.AIRCRAFT_COLORS[1],speed=self.config['game speed']*self.config['human speed'], flight_pattern=self.config["search pattern"]) # Human
        #self.agents[self.human_idx].x, self.agents[self.human_idx].y = self.config['human start location']
        #if self.config['num aircraft'] == 1: self.agents[self.agent_idx].is_visible = False # Do not draw the agent during solo training runs

        self.agent_idx = self.aircraft_ids[0]
        self.human_idx = self.aircraft_ids[1]  # Agent ID for the human-controlled aircraft. Dynamic so that if human dies in training round, their ID increments 1

        print('Agents in the environment now:')
        print(self.agents)

        return self.get_observation()


    def step(self, actions:list):
        """
        args:
            actions: List of (agent_id, action) tuples, where action = dict('waypoint': (x,y), 'id_method': 0, 1, or 2')

        returns:
        """

        # TODO implement vectorized update positions and process-interactions methods at bottom
        new_score = 0

        for action in actions:
            #print(f'Action in queue is {action}')
            #print(f'Passing {action[1]} to agent waypoint override')
            self.agents[action[0]].waypoint_override = action[1]

        # move the agents and check for gameplay updates
        #aircraft_agents = [agent for agent in self.agents if agent.agent_class == "aircraft" and agent.alive]
        for aircraft in [agent for agent in self.agents if agent.agent_class == "aircraft" and agent.alive]:

            aircraft.move() # First, move using the waypoint override set above

            aircraft_pos = np.array([aircraft.x, aircraft.y]) # Get aircraft position

            # Calculate distances to all targets
            target_positions = self.targets[:, 3:5]  # x,y coordinates
            distances = np.sqrt(np.sum((target_positions - aircraft_pos) ** 2, axis=1))

            # Find targets within ISR range (for identification)
            isr_range = self.AIRCRAFT_ISR_RADIUS
            in_isr_range = distances <= isr_range

            # Find unidentified targets within ISR range
            unidentified_in_range = in_isr_range & (self.targets[:, 2] == 0)
            newly_identified_indices = np.where(unidentified_in_range)[0]

            # Process newly identified targets
            for idx in newly_identified_indices:
                # Update target info level to 0.5 (identified but threat unknown)
                self.targets[idx, 2] = 0.5
                self.identified_targets += 1
                new_score += self.target_points

                # Log which aircraft identified the target
                self.new_target_id = [
                    'human' if aircraft.agent_idx == self.human_idx else 'AI',
                    'target_identified',
                    int(self.targets[idx, 0])  # Target ID
                ]

                # If target is non-valuable (value=0), automatically ID threat too
                if self.targets[idx, 1] == 0:
                    self.targets[idx, 2] = 1.0  # Fully identified
                    self.num_identified_ships += 1
                    self.identified_threat_types += 1
                    new_score += self.threat_points

            # Find targets within engagement range (for threat identification)
            engagement_range = self.AIRCRAFT_ENGAGEMENT_RADIUS
            in_engagement_range = distances <= engagement_range

            # Find partially identified targets within engagement range
            partially_identified_in_range = in_engagement_range & (self.targets[:, 2] == 0.5)
            newly_threat_identified_indices = np.where(partially_identified_in_range)[0]

            # Process newly threat-identified targets
            for idx in newly_threat_identified_indices:
                # Update target info level to 1.0 (fully identified)
                self.targets[idx, 2] = 1.0
                self.num_identified_ships += 1
                self.identified_threat_types += 1
                new_score += self.threat_points

                # Log which aircraft identified the threat
                self.new_weapon_id = [
                    'human' if aircraft.agent_idx == self.human_idx else 'AI',
                    'weapon_identified',
                    int(self.targets[idx, 0])  # Target ID
                ]

            # Process damage to aircraft from high-value targets
            # Find high-value targets that can damage aircraft
            high_value_targets = self.targets[:, 1] == 1  # Value of 1 indicates high-value/threat

            # Calculate threat range for these targets (using column 1 as threat level)
            # For simplicity, I'm using threat level 2 for all high-value targets
            threat_level = 2
            threat_range = self.AGENT_BASE_DRAW_WIDTH * self.AGENT_THREAT_RADIUS[threat_level]

            # Find high-value targets within threat range
            in_threat_range = distances <= threat_range
            threatening_targets = in_threat_range & high_value_targets

            # Apply damage from threatening targets
            if np.any(threatening_targets) and np.random.random() < aircraft.prob_detect:
                aircraft.health_points -= 1

                if self.render_mode == 'human' and pygame.get_init():
                    self.agent_damage_flash_start = pygame.time.get_ticks()
                    self.agent_damage_flash_alpha = 255

            # Check if aircraft is destroyed
            if aircraft.health_points <= 0 and not self.config['infinite health']:
                #aircraft.damage = 100
                aircraft.alive = False

        # Check if the agent is recently deceased (RIP)
        if not self.agents[self.aircraft_ids[0]].alive and not self.agent0_dead:
            self.agent0_dead = True
            new_score += self.wingman_dead_points
            print(f'{self.wingman_dead_points} points for agent wingman destruction')

        # progress update
        if self.verbose: print("   Found:", self.num_identified_ships, "Total:", len(self.agents) - len(self.aircraft_ids), "Damage:", self.damage)

        # Check termination conditions # TODO make this more configurable. Either AI, human or both need to be alive
        self.terminated = (self.num_identified_ships >= self.num_targets) or (not self.agents[self.aircraft_ids[0]].alive and not self.config['infinite health'] and not self.human_training)
        self.truncated = (self.display_time / 1000 >= self.time_limit)
        if self.terminated or self.truncated:
            if self.num_identified_ships >= self.num_targets: # Add points for finishing early
                new_score += self.all_targets_points
                new_score += (self.time_limit - self.display_time/1000)*self.time_points

            elif not self.agents[self.human_idx].alive:
                new_score += self.human_dead_points

            print(f'\n FINAL SCORE {self.score} | {self.identified_targets} targets | {self.identified_threat_types} threats | {self.agents[self.aircraft_ids[0]].health_points} HP left | {round(self.time_limit-self.display_time/1000,1)} secs left')

            if self.render_mode == 'human':
                pygame.time.wait(50)

        # Calculate reward
        reward = self.get_reward(new_score) # TODO
        self.score += new_score

        observation = self.get_observation() # Get observation
        info = {} # Additional info

        if self.agent_training: self.display_time = self.display_time + (1000/60)

        elif not self.paused:
            current_time = pygame.time.get_ticks()
            self.display_time = current_time - self.total_pause_time

        return observation, reward, self.terminated, self.truncated, info


    def get_reward(self, new_score):
        if self.reward_type == 'balanced-sparse': # Considers all the points
            reward = new_score
        else: raise ValueError('Unknown reward type')

        return reward

    def get_observation(self):
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
            threat_level         # (0.33, 0.66, 1.0) for small/med/large weapon

            # Handcrafted features (11 features): TODO Not currently implemented
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

        # TODO Add handcrafted features
            # Update human's quadrant
            # if self.agents[self.human_idx].x < (self.config['gameboard size'] / 2):
            #     if self.agents[self.human_idx].y < (self.config['gameboard size'] / 2):
            #         self.human_quadrant = 'NW'
            #     else:
            #         self.human_quadrant = 'SW'
            # else:
            #     if self.agents[self.human_idx].y < (self.config['gameboard size'] / 2):
            #         self.human_quadrant = 'NE'
            #     else:
            #         self.human_quadrant = 'SE'

        if self.obs_type == 'vector':
            self.observation = np.zeros(self.num_obs_features, dtype=np.float32)

            self.observation[0] = self.agents[self.aircraft_ids[0]].health_points / 10.0  # Agent health
            self.observation[1] = self.agents[self.aircraft_ids[0]].x / self.config["gameboard size"] # Agent x
            self.observation[2] = self.agents[self.aircraft_ids[0]].y / self.config["gameboard size"] # Agent y
            self.observation[3] = 1.0 if self.agents[self.aircraft_ids[1]].alive else 0.0  # Agent alive status

            if self.agents[self.aircraft_ids[0]].target_point is not None:
                self.observation[4] = self.agents[self.aircraft_ids[0]].target_point[0] / self.config["gameboard size"]
                self.observation[5] = self.agents[self.aircraft_ids[0]].target_point[1] / self.config["gameboard size"]
            else:
                self.observation[4] = self.observation[1]  # Default to current position
                self.observation[5] = self.observation[2]

            self.observation[6] = self.agents[self.aircraft_ids[1]].health_points / 10.0  # Teammate health
            self.observation[7] = self.agents[self.aircraft_ids[1]].x / self.config["gameboard size"]  # Teammate x
            self.observation[8] = self.agents[self.aircraft_ids[1]].y / self.config["gameboard size"]  # Teammate y

            # Teammate waypoint
            if self.agents[self.aircraft_ids[1]].target_point is not None:
                self.observation[9] = self.agents[self.aircraft_ids[1]].target_point[0] / self.config["gameboard size"]
                self.observation[10] = self.agents[self.aircraft_ids[1]].target_point[1] / self.config["gameboard size"]
            else:
                self.observation[9] = self.observation[7]  # Default to current position
                self.observation[10] = self.observation[8]

            self.observation[11] = 1.0 if self.agents[self.aircraft_ids[1]].alive else 0.0  # Teammate alive status
            self.observation[12] = (self.time_limit - self.display_time / 1000) / self.time_limit # Time remaining

            # Process all target data (7 features per target)
            # Create an array to hold target data
            self.target_data = np.zeros((self.num_targets, 7), dtype=np.float32)

            # Normalized positions (columns 0-1 of target_data)
            self.target_data[:, 0] = self.targets[:, 3] / self.config["gameboard size"]  # x position
            self.target_data[:, 1] = self.targets[:, 4] / self.config["gameboard size"]  # y position

            # Get info levels and target values
            info_levels = self.targets[:, 2]
            target_values = self.targets[:, 1]

            # One-hot encoding of target identity (columns 2-4 of target_data)
            self.target_data[:, 2] = (info_levels == 0).astype(np.float32)  # unknown
            self.target_data[:, 3] = ((info_levels > 0) & (target_values == 0)).astype(np.float32)  # friendly
            self.target_data[:, 4] = ((info_levels > 0) & (target_values == 1)).astype(np.float32)  # hostile

            # Threat status (column 5 of target_data)
            self.target_data[:, 5] = (info_levels < 1.0).astype(np.float32)  # threat unknown

            # Normalized threat level (column 6 of target_data)
            # For high-value targets (1), we'll use threat level 2/3
            # For regular targets (0), we'll use threat level 0
            self.target_data[:, 6] = np.where(
                info_levels >= 1.0,
                (2.0 / 3.0) * target_values,
                0.0
            )

            # self.target_data = np.zeros((self.num_targets, 7), dtype=np.float32)
            # target_idx = 0
            # for agent_idx, agent in enumerate(self.agents):
            #     if agent.agent_class == "ship":
            #         # Normalized position
            #         self.target_data[target_idx, 0] = agent.x / self.config["gameboard size"]
            #         self.target_data[target_idx, 1] = agent.y / self.config["gameboard size"]
            #         # One-hot encoding of target identity
            #         self.target_data[target_idx, 2] = 1.0 if not agent.observed else 0.0  # unknown
            #         self.target_data[target_idx, 3] = 1.0 if agent.observed and agent.threat == 0 else 0.0  # friendly
            #         self.target_data[target_idx, 4] = 1.0 if agent.observed and agent.threat > 0 else 0.0  # hostile
            #         # Threat status
            #         self.target_data[target_idx, 5] = 0.0 if agent.observed_threat else 1.0  # threat unknown
            #         # Normalized threat level
            #         self.target_data[target_idx, 6] = agent.threat / 3.0 if agent.observed_threat else 0.0
            #         target_idx += 1

            target_start_idx = 13
            self.observation[target_start_idx:target_start_idx + self.num_targets * 7] = self.target_data.flatten()         # Copy target data into observation vector

        if self.init:
            print("Observation vector explanation:")
            print(f"[0] Agent health: {self.observation[0]}")
            print(f"[1] Agent x position: {self.observation[1]}")
            print(f"[2] Agent y position: {self.observation[2]}")
            print(f"[3] Agent alive status: {self.observation[3]}")
            print(f"[4] Agent waypoint x: {self.observation[4]}")
            print(f"[5] Agent waypoint y: {self.observation[5]}")
            print(f"[6] Teammate health: {self.observation[6]}")
            print(f"[7] Teammate x position: {self.observation[7]}")
            print(f"[8] Teammate y position: {self.observation[8]}")
            print(f"[9] Teammate waypoint x: {self.observation[9]}")
            print(f"[10] Teammate waypoint y: {self.observation[10]}")
            print(f"[11] Teammate alive status: {self.observation[11]}")
            print(f"[12] Time remaining normalized: {self.observation[12]}")

            # Print target data
            for i in range(1):
                base_idx = 13 + i * 7
                print(f"\nTarget {i} data:")
                print(f"[{base_idx}] Target x position: {self.observation[base_idx]}")
                print(f"[{base_idx + 1}] Target y position: {self.observation[base_idx + 1]}")
                print(f"[{base_idx + 2}] Target ID unknown: {self.observation[base_idx + 2]}")
                print(f"[{base_idx + 3}] Target ID friendly: {self.observation[base_idx + 3]}")
                print(f"[{base_idx + 4}] Target ID hostile: {self.observation[base_idx + 4]}")
                print(f"[{base_idx + 5}] Target threat unknown: {self.observation[base_idx + 5]}")
                print(f"[{base_idx + 6}] Target threat level: {self.observation[base_idx + 6]}")

        elif self.obs_type == 'cnn':
            raise ValueError('CNN Obs type not supported yet')

        return self.observation


    def add_comm_message(self,message,is_ai=True):
        #timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        sender = "AGENT" if is_ai else "HUMAN"
        full_message = f"{sender}: {message}"
        self.comm_messages.append((full_message, is_ai))
        if len(self.comm_messages) > self.max_messages:
            self.comm_messages.pop(0)

    def render(self, mode='human', close=False):
        if mode == 'none': # TODO replace this with something cleaner
            return

        window_width, window_height = self.config['window size'][0], self.config['window size'][0]
        game_width = self.config["gameboard size"]
        ui_width = window_width - game_width

        if self.agent_info_height_req > 0:
            self.comm_pane_height = 220+self.agent_info_height_req
        else: self.comm_pane_height = 10

        # gameboard background
        self.window.fill((255, 255, 255))  # white background
        self.__render_box__(1, (0, 0, 0), 3)  # outer box
        pygame.draw.rect(self.window, (100, 100, 100), (game_width+self.gameboard_offset, 0, ui_width, window_height))
        pygame.draw.rect(self.window, (100, 100, 100), (0, game_width, game_width, window_height))  # Fill bottom portion with gray

        current_time = pygame.time.get_ticks()

        # Draw the aircraft
        for agent in self.agents:
            agent.draw(self.window)

        # Draw the targets TODO

        # Create vectorized ships/targets. Format: [id, value, info_level, x_pos, y_pos]
        #         self.targets = np.zeros((self.num_targets, 5), dtype=np.float32)

        color_list = [  # color_list[value][info level]
            ['regular-unknown', 'regular-lowQ', 'regular-highQ'],
            ['highval-unknown', 'highval-lowQ', 'highval-highQ'], ]

        # TODO fix get_observation to use self.targets?

        for target in self.targets:
            target_width = 5 if target[1] == 0 else 10
            target_color = color_list[int(target[1])][int(target[2])]
            print('################')
            print(target[3],target[4])
            pygame.draw.circle(self.window, target_color, (float(target[3]), float(target[4])), target_width) # TODO center msut be a pair of numbers

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

        # Draw health boxes
        if self.config['num aircraft'] > 1:
            agent0_health_window = HealthWindow(self.aircraft_ids[0],10,game_width+5, 'AGENT HP',self.AIRCRAFT_COLORS[0])
            agent0_health_window.update(self.agents[self.aircraft_ids[0]].health_points)
            agent0_health_window.draw(self.window)

        agent1_health_window = HealthWindow(self.human_idx, game_width-150, game_width + 5, 'HUMAN HP',self.AIRCRAFT_COLORS[1])
        agent1_health_window.update(self.agents[self.human_idx].health_points)
        agent1_health_window.draw(self.window)

        current_time = pygame.time.get_ticks()

        if current_time > self.start_countdown_time:
            self.time_window.update(self.display_time)
            self.time_window.draw(self.window)

        # Draw agent status window
        if self.agent_info_height_req > 0: self.agent_info_display.draw(self.window)

        corner_round_text = f"ROUND {self.round_number+1}/4" if self.user_group == 'test' else f"ROUND {self.round_number}/4"
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

        if self.terminated or self.truncated:
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
        agent_status = "ALIVE" if self.agents[self.aircraft_ids[0]].alive else "DESTROYED"
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

    def get_nearby_hostiles(self, aircraft_agent):
        """
        Calculates number of hostile targets close to the agent
        args:
            agent: The aircraft agent to check for. Must access via self.agents[self.aircraft_ids[0]]
        """
        hostile, friendly, unknown = 0,0,0

        for agent in self.agents:
            if agent.agent_class == 'ship' and math.hypot(agent.x - aircraft_agent.x, agent.y - aircraft_agent.y) <= 30:
                if not agent.observed: # TODO make sure agent.observed corresponds to the agent being totally unknown
                    unknown += 1
                elif agent.threat > 0 and agent.observed_threat: # Increment 1 for hostile
                    hostile += 1
                elif agent.threat <= 0 and agent.observed_threat:
                    friendly += 1

        print(f'HOSTILE/unknown/FRIENDLY: {hostile} {unknown} {friendly}')
        return hostile, friendly, unknown