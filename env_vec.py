import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import random
import agents
from gui import Button, ScoreWindow, HealthWindow, TimeWindow, AgentInfoDisplay
import datetime
import math


class MAISREnvVec(gym.Env):
    """Multi-Agent ISR Environment following the Gym format"""

    def __init__(self, config={}, window=None, clock=None, render_mode='headless',
                 obs_type = 'vector', action_type = 'continuous', reward_type = 'balanced-sparse',
                 subject_id='999',user_group='99',round_number='99'):
        """
        args:
            time_scale: How much to scale time by
            render_mode: 'headless' for no-render agent training, 'human' for playing with humans
            reward_type:
                balanced-sparse: Points for IDing targets, weapons, and finishing early
                balanced-dense: Variant 1 + penalty for damage + others
                cautious-sparse: BS but very low reward for IDing weapons
                aggressive-sparse: BS but very high reward for IDing weapons

        """
        super().__init__()

        if obs_type not in ['vector', 'pixel']: raise ValueError(f"obs_type must be one of 'vector,'pixel', got '{obs_type}'")
        if action_type not in ['discrete', 'continuous']: raise ValueError(f"action_type must be one of 'discrete,'continuous, got '{action_type}'")
        if reward_type not in ['balanced-sparse']: raise ValueError('reward_type must be normal. Others coming soon')


        self.config = config
        self.verbose = True if self.config['verbose'] == 'true' else False
        self.render_mode = render_mode
        self.gather_info = self.render_mode == 'human' # Only populate the info dict if playing with humans

        #self.config['num aircraft'] = 1 # TODO temp override

        self.obs_type = obs_type
        self.action_type = action_type
        self.reward_type = reward_type


        self.subject_id = subject_id
        self.round_number = round_number
        self.user_group = user_group
        self.human_training = True if (self.round_number == 0 and self.user_group != 'test') else False  # True for the training round at start of experiment, false for rounds 1-4 (NOTE: This is NOT agent training!)

        self.init = True # Used to render static windows the first time
        self.paused = False
        self.unpause_countdown = False
        #self.new_target_id = None  # Used to check if a new target has been ID'd so the data logger in main.py can log it
        #self.new_weapon_id = None

        self.agent0_dead = False  # Used at end of loop to check if agent recently deceased.
        self.agent1_dead = False

        self.terminated = False
        self.truncated = False

        self.time_limit = self.config['time limit']

        # set the random seed
        if "seed" in config: random.seed(config["seed"])
        else: print("Note: no 'seed' specified in the env config.")

        # determine the number of ships
        self.max_targets = 30
        self.num_targets = min(30, self.config['num ships']) # If more than 30 targets specified, overwrite to 30

        if self.action_type == 'continuous':
            self.action_space = gym.spaces.Box(
                low=np.array([-1, -1, -1], dtype=np.float32),
                high=np.array([1, 1, 1], dtype=np.float32),
                dtype=np.float32)
        elif self.action_type == 'discrete':
            self.action_space = MultiDiscrete([101, 101, 101])
        else:
            raise ValueError("Action type must be continuous or discrete")

        self.obs_size = 9 + 5 * 30

        if self.obs_type == 'vector':
            self.observation_space = gym.spaces.Box(
                low=0, high=1,
                shape=(self.obs_size,),  # Wrap in tuple to make it iterable
                dtype=np.float32)

            self.observation = np.zeros(self.obs_size, dtype=np.float32)

            #self.target_data = np.zeros((self.num_targets, 7), dtype=np.float32)  # For target features
        elif self.obs_type == 'pixel':
            # Define pixel observation space
            pixel_height = self.config["gameboard size"]
            pixel_width = self.config["gameboard size"]
            channels = 3  # RGB

            # You might want to resize for performance (optional)
            self.resize_factor = 2  # e.g., resize to half resolution
            pixel_height = pixel_height // self.resize_factor
            pixel_width = pixel_width // self.resize_factor

            self.observation_space = gym.spaces.Box(
                low=0, high=1,
                shape=(pixel_height, pixel_width, channels),
                dtype=np.float32
            )
        else:
            raise ValueError("Obs type not recognized")


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

        # Set reward quantities for each event (agent only)
        self.lowqual_regulartarget_reward = 0.25  # Reward earned for gathering low quality info about a regular target
        self.lowqual_highvaltarget_reward = 0.5  # Reward earned for gathering low quality info about a high value target
        self.highqual_regulartarget_reward = 0.5 # Reward earned for gathering high quality info about a regular value target
        self.highqual_highvaltarget_reward = 1.0 # Reward earned for gathering high quality info about a high value target
        self.detections_reward = -1.0
        self.time_reward = 0.1  # Reward earned for every second early. 0.1 translates to 1.0 per 10 seconds

        # Set point quantities for each event (human only)
        self.score = 0
        self.all_targets_points = 0  # All targets ID'd
        self.low_qual_points = 10  # Points earned for gathering low quality info about a target
        self.high_qual_points = 10  # Points earned for gathering high quality info about a target
        self.time_points = 15  # Points given per second remaining
        self.human_hp_remaining_points = 70
        self.wingman_dead_points = -300  # Points subtracted for agent wingman dying
        self.human_dead_points = -400  # Points subtracted for human dying

        self.steps_for_lowqual_info = 3*60 # TODO tune this
        self.steps_for_highqual_info = 7*60 # TODO tune this. Currently 7 seconds
        self.prob_detect = 0.00333333333 # Probability of being detected on each step. Probability per second = prob_detect * 60 (TODO tune)

        if render_mode == 'human':
            self.window = window
            self.clock = clock
            self.start_countdown_time = 5000  # How long in milliseconds to count down at the beginning of the game before it starts

            # Set GUI locations
            self.gameboard_offset = 0  # How far from left edge to start drawing gameboard
            self.window_x = self.config["window size"][0]
            self.window_y = self.config["window size"][1]
            self.window = pygame.display.set_mode((self.window_x, self.window_y))

            self.right_pane_edge = self.config['gameboard size'] + 20  # Left edge of gameplan button windows
            self.comm_pane_edge = self.right_pane_edge
            self.gameplan_button_width = 180
            self.quadrant_button_height = 120
            self.autonomous_button_y = 590

            # Initialize buttons
            self.gameplan_button_color = (255, 120, 80)
            self.manual_priorities_button = Button("Manual Priorities", self.right_pane_edge + 15, 20,self.gameplan_button_width * 2 + 15, 65)
            self.target_id_button = Button("TARGET", self.right_pane_edge + 15, 60 + 55, self.gameplan_button_width,60)
            self.wez_id_button = Button("WEAPON", self.right_pane_edge + 30 + self.gameplan_button_width, 60 + 55,self.gameplan_button_width, 60)
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

            self.display_time = 0 # Time that is used for the on-screen timer. Accounts for pausing.
            self.pause_start_time = 0
            self.total_pause_time = 0
            self.button_latch_dict = {'target_id':False,'wez_id':False,'hold':False,'waypoint':False,'NW':False,'SW':False,'NE':False,'SE':False,'full':False,'autonomous':True,'pause':False,'risk_low':False, 'risk_medium':True, 'risk_high':False,'manual_priorities':False,'tag_team':False,'fan_out':False} # Hacky way to get the buttons to visually latch even when they're redrawn every frame
            self.pause_font = pygame.font.SysFont(None, 74)
            self.pause_subtitle_font = pygame.font.SysFont(None, 40)

            # For visual damage flash
            self.damage_flash_duration = 500  # Duration of flash in milliseconds
            self.damage_flash_start = 0  # When the last damage was taken
            self.damage_flash_alpha = 0  # Current opacity of flash effect
            self.agent_damage_flash_start = 0
            self.agent_damage_flash_alpha = 0
            self.last_health_points = {0: 10, 1: 10}  # Track health points to detect changes

            # Calculate required height of agent status info
            self.agent_info_height_req = 0
            if self.config['show_low_level_goals']: self.agent_info_height_req += 1
            if self.config['show_high_level_goals']: self.agent_info_height_req += 1.7
            if self.config['show_tracked_factors']: self.agent_info_height_req += 1.7
            if self.agent_info_height_req > 0: # Only render agent info display if at least one of the info elements is used
                self.agent_info_display = AgentInfoDisplay(self.comm_pane_edge, 10, 445, 40+35*self.agent_info_height_req)

            self.time_window = TimeWindow(self.config["gameboard size"] * 0.43, self.config["gameboard size"]+5,current_time=self.display_time, time_limit=self.time_limit)

        self.reset()


    def reset(self, seed = None):
        self.agents = []
        self.aircraft_ids = []  # indexes of the aircraft agents
        self.damage = 0  # total damage from all agents
        #self.num_identified_ships = 0  # number of ships with accessed threat levels, used for determining game end
        self.display_time = 0  # Time that is used for the on-screen timer. Accounts for pausing.
        self.pause_start_time = 0
        self.total_pause_time = 0
        self.score = 0
        #self.identified_targets = 0
        #self.identified_threat_types = 0
        self.num_lowq_gathered = 0
        self.num_highq_gathered = 0
        self.init = True

        # Create vectorized ships/targets. Format: [id, value, info_level, x_pos, y_pos]
        self.targets = np.zeros((self.num_targets, 5), dtype=np.float32)
        self.targets[:, 0] = np.arange(self.num_targets) # Assign IDs (column 0)
        self.targets[:, 1] = np.random.choice([0, 1], size=self.num_targets, p=[0.67, 0.33]) # Assign values (column 1) - 67% regular (0), 33% high-value (1)
        self.targets[:, 2] = 0 # Initialize info_level (column 2) to all 0 (unknown)
        self.targets[:, 3] = np.random.uniform(35, self.config["gameboard size"] - 35, size=self.num_targets) # Randomly place targets on gameboard (columns 3-4)
        self.targets[:, 4] = np.random.uniform(35, self.config["gameboard size"] - 35, size=self.num_targets)

        self.target_timers = np.zeros(self.num_targets, dtype=np.int32)  # How long each target has been sensed for
        self.detections = 0 # Number of times a target has detected us. Results in a score penalty


        # create the aircraft
        for i in range(self.config['num aircraft']):
            agents.Aircraft(self, 0,prob_detect=self.prob_detect,max_health=10,color=self.AIRCRAFT_COLORS[i],speed=self.config['game speed']*self.config['human speed'], flight_pattern=self.config["search pattern"])
            self.agents[self.aircraft_ids[i]].x, self.agents[self.aircraft_ids[i]].y = self.config['agent start location']

        self.agent_idx = self.aircraft_ids[0]
        if self.config['num aircraft'] == 2:
            self.human_idx = self.aircraft_ids[1]  # Agent ID for the human-controlled aircraft. Dynamic so that if human dies in training round, their ID increments 1

        self.observation = self.get_observation()
        info = {}
        return self.observation, info


    def step(self, actions:list):
        """
        args:
            actions: List of (agent_id, action) tuples, where action = dict('waypoint': (x,y), 'id_method': 0, 1, or 2')

        returns:
        """

        # Track events that give reward. Will be passed to get_reward at end of step
        new_reward = {'detections': 0,
                      'low qual regular': 0,
                      'high qual regular': 0,
                      'low qual high value': 0,
                      'high qual high value': 0,
                      'early finish': 0}

        new_score = 0 # For tracking human-understandable reward

        if self.gather_info:
            info = {
                "new_identifications": [],  # List to track newly identified targets/threats
                "detections": self.detections,  # Current detection count
                "score_breakdown": {
                    "target_points": 0,
                    "threat_points": 0,
                    "time_points": 0,
                    "completion_points": 0,
                    "penalty_points": 0}
            }
        else:
            info = {}

        if isinstance(actions, list): # Action is passed in as a list of (agent_id, action) tuples
            print('Action is a [id, action] list, executing normal mode')
            for action in actions:
                agent_id, action_value  = action
                waypoint, id_method = self.denormalize_action(action_value)
                self.agents[agent_id].waypoint_override = waypoint

        elif isinstance(actions, np.ndarray): # Single agent, action passed in directly as an array instead of list(arrays)
            action_value = actions
            waypoint, id_method = self.denormalize_action(action_value)
            self.agents[0].waypoint_override = waypoint

        else: raise ValueError('Actions input is an unknown type')


        # move the agents and check for gameplay updates
        for aircraft in [agent for agent in self.agents if agent.agent_class == "aircraft" and agent.alive]:
            aircraft.move() # First, move using the waypoint override set above

            # Calculate distances to all targets
            aircraft_pos = np.array([aircraft.x, aircraft.y])  # Get aircraft position
            target_positions = self.targets[:, 3:5]  # x,y coordinates
            distances = np.sqrt(np.sum((target_positions - aircraft_pos) ** 2, axis=1))

            # Find targets within ISR range (for identification)
            isr_range = self.AIRCRAFT_ENGAGEMENT_RADIUS
            in_isr_range = distances <= isr_range

            # Process newly identified targets (TODO below should be a fix but untested)
            for target_idx in range(self.num_targets):
                if in_isr_range[target_idx] and self.targets[target_idx, 2] < 1.0: # If target is in range and not fully identified
                    self.target_timers[target_idx] += 1  # Increment timer for this target

                    if self.target_timers[target_idx] >= self.steps_for_lowqual_info and self.targets[target_idx, 2] < 0.5:
                        self.targets[target_idx, 2] = 0.5
                        new_score += self.low_qual_points
                        if self.targets[target_idx, 1] == 0.0: new_reward['low qual regular'] += 1
                        elif self.targets[target_idx, 1] == 1.0: new_reward['low qual high value'] += 1

                        if self.gather_info:
                            info["score_breakdown"]["target_points"] += self.low_qual_points
                            info["new_identifications"].append({
                                "type": "low quality info gathered",
                                "target_id": int(self.targets[target_idx, 0]),
                                "aircraft": aircraft.agent_idx,
                                "time": self.display_time
                            })

                    elif self.target_timers[target_idx] >= self.steps_for_highqual_info and self.targets[target_idx, 2] < 1.0:
                        self.targets[target_idx, 2] = 1.0
                        new_score += self.high_qual_points
                        if self.targets[target_idx, 1] == 0.0: new_reward['high qual regular'] += 1
                        elif self.targets[target_idx, 1] == 1.0: new_reward['high qual high value'] += 1

                        if self.gather_info:
                            info["new_identifications"].append({
                                "type": "high quality info gathered",
                                "target_id": int(self.targets[target_idx, 0]),
                                "aircraft": aircraft.agent_idx,
                                "time": self.display_time
                            })

                    # Check if this is a high-value target that can detect us
                    if self.targets[target_idx, 1] == 1.0 and np.random.random() < aircraft.prob_detect:
                        self.detections += 1 # High value target detected us
                        new_reward['detections'] += 1

                        if self.gather_info:
                            info["detections"] = self.detections

                        if self.render_mode == 'human' and pygame.get_init():
                            self.agent_damage_flash_start = pygame.time.get_ticks()
                            self.agent_damage_flash_alpha = 255


        # Check termination conditions # TODO make this more configurable. Either AI, human or both need to be alive
        self.all_targets_identified = np.all(self.targets[:, 2] == 1.0)
        self.low_quality_identified = np.sum(self.targets[:, 2] == 0.5) # Count targets with at least low-quality ID (value > 0.5)
        self.high_quality_identified = np.sum(self.targets[:, 2] == 1.0) # Count targets with high-quality ID (value = 1.0)
        print(f'Low Q {self.low_quality_identified} | high q {self.high_quality_identified}')

        if self.verbose:
            print("Targets with low-quality info: ", self.low_quality_identified, " Targets with high-quality info: ", self.high_quality_identified, "Detections: ", self.detections)

        self.terminated = self.all_targets_identified or self.detections >= 5 or self.display_time / 1000 >= self.time_limit
        # self.truncated = (TODO: No current use for truncated
        if self.terminated or self.truncated:
            if self.all_targets_identified: # Add points for finishing early
                new_score += self.all_targets_points # Left this but it doesn't go into reward
                new_score += (self.time_limit - self.display_time/1000)*self.time_points
                new_reward['early finish'] = (self.time_limit - self.display_time/1000)


            print(f'\n FINAL SCORE {self.score} | {self.low_quality_identified} low quality | {self.high_quality_identified} high quality | {self.agents[self.aircraft_ids[0]].health_points} AI HP left | {round(self.time_limit-self.display_time/1000,1)} secs left')

            if self.render_mode == 'human':
                pygame.time.wait(50)

        # Calculate reward
        reward = self.get_reward(new_reward) # For agent
        self.score += new_score # For human
        observation = self.get_observation() # Get observation

        # Advance time
        if self.render_mode == 'headless':
            self.display_time = self.display_time + (1000/60) # If agent training, each step is 1/60th of a second

        elif not self.paused:
            self.display_time = pygame.time.get_ticks() - self.total_pause_time

        if self.init: self.init = False

        return observation, reward, self.terminated, self.truncated, info


    def get_reward(self, new_reward):

        if self.reward_type == 'balanced-sparse': # Default reward function
            reward = (new_reward['low qual regular'] * self.lowqual_regulartarget_reward) + \
                     (new_reward['high qual regular'] * self.highqual_regulartarget_reward) + \
                     (new_reward['low qual high value'] * self.lowqual_highvaltarget_reward) + \
                     (new_reward['high qual high value'] * self.highqual_highvaltarget_reward) + \
                     (new_reward['early finish'] * self.time_reward)

        else:
            raise ValueError('Unknown reward type')

        return reward

    def get_observation(self):
        """
        State will include the following features (current count is 309):
            # Agent and basic game info (8 features):
            0 agent_x,             # (0-1) (discretize map into 100x100 grid, then normalize)
            1 agent_y,             # (0-1)

            2 teammate_exists      # 0 (does not exist), 1 (does exist)
            3 teammate_x,          # (0-1)
            4 teammate_y,          # (0-1)
            5 teammate_waypoint_x  # (0-1) x coordinate of current waypoint
            6 teammate_waypoint_y  # (0-1) y coordinate of current waypoint

            7 time_remaining       # (0-1)
            8 num detections       # (0 to 1) Normalized using self.num_detections / 5. When num_detections = 5, this hits 1 and the episode is terminated

            # Target data (the following are repeated for all targets, for a max of 5*30 = 150 features)
            9+i target_exists        # Used to allow configurable numbers of targets. 0 if the target does not exist (for num_targets < 60), 1 if it does
            10+i target_value         # 0 for regular, 1 for high value
            11+i info_level           # 0 for no info, 0.5 for low quality info, 1.0 for full info
            12+i target_x,            # (0-1)
            13+i target_y,            # (0-1)


            # Handcrafted features, TBD (TODO)
        """

        if self.obs_type == 'vector':
            self.observation = np.zeros(self.obs_size, dtype=np.float32)

            self.observation[0] = self.agents[self.aircraft_ids[0]].x / self.config["gameboard size"] # Agent x
            self.observation[1] = self.agents[self.aircraft_ids[0]].y / self.config["gameboard size"] # Agent y

            if self.config['num aircraft'] == 2:
                self.observation[2] = 1 # Teammate_exists
                self.observation[3] = self.agents[self.aircraft_ids[1]].x / self.config["gameboard size"]  # Teammate x
                self.observation[4] = self.agents[self.aircraft_ids[1]].y / self.config["gameboard size"]  # Teammate y

                if self.agents[self.aircraft_ids[1]].target_point is not None: # Teammate waypoint
                    self.observation[5] = self.agents[self.aircraft_ids[1]].target_point[0] / self.config["gameboard size"] # Teammate waypoint x
                    self.observation[6] = self.agents[self.aircraft_ids[1]].target_point[1] / self.config["gameboard size"] # Teammate waypoint y
                else:
                    self.observation[5] = self.observation[4]  # Default to current position
                    self.observation[6] = self.observation[5]

            else: # Teammate does not exist
                self.observation[2:7] = 0

            self.observation[7] = (self.time_limit - self.display_time / 1000) / self.time_limit  # Time remaining
            self.observation[8] = self.detections

            # Process target data
            #max_targets = 60 self.max_targets
            targets_per_entry = 5  # Each target has 5 features in the observation

            target_features = np.zeros((self.max_targets, targets_per_entry), dtype=np.float32)
            target_features[:self.num_targets, 0] = 1.0 # Set the target_exists flag to 1 for all actual targets

            if self.num_targets > 0: # For actual targets, copy the relevant data
                target_features[:self.num_targets, 1] = self.targets[:, 1] # Copy target values (0=regular, 1=high-value)
                target_features[:self.num_targets, 2] = self.targets[:, 2] # Copy info levels (0=unknown, 0.5=low quality, 1.0=high quality)
                target_features[:self.num_targets, 3] = self.targets[:, 3] / self.config["gameboard size"]  # x position
                target_features[:self.num_targets, 4] = self.targets[:, 4] / self.config["gameboard size"]  # y position

            target_start_idx = 9 # Insert all target features into the observation vector
            self.observation[target_start_idx:target_start_idx + self.max_targets * targets_per_entry] = target_features.flatten()


            # if self.init:
            #     print("Observation vector explanation:")
            #     print(f"[0] Agent x position: {self.observation[0]}")
            #     print(f"[1] Agent y position: {self.observation[1]}")
            #     print(f"[2] Teammate exists: {self.observation[2]}")
            #     print(f"[3] Teammate x position: {self.observation[3]}")
            #     print(f"[4] Teammate y position: {self.observation[4]}")
            #     print(f"[5] Teammate waypoint x: {self.observation[5]}")
            #     print(f"[6] Teammate waypoint y: {self.observation[6]}")
            #
            #     print(f"[7] Time remaining normalized: {self.observation[7]}")
            #     print(f"[8] Normalized detections: {self.observation[8]}")
            #
            #     for i in range(min(3, self.num_targets)): # Print some target data examples
            #         base_idx = 9 + i * targets_per_entry
            #         print(f"\nTarget {i} data:")
            #         print(f"[{base_idx}] Target exists: {self.observation[base_idx]}")
            #         print(f"[{base_idx + 1}] Target value: {self.observation[base_idx + 1]}")
            #         print(f"[{base_idx + 2}] Info level: {self.observation[base_idx + 2]}")
            #         print(f"[{base_idx + 3}] Target x position: {self.observation[base_idx + 3]}")
            #         print(f"[{base_idx + 4}] Target y position: {self.observation[base_idx + 4]}")

        elif self.obs_type == 'pixel':
            frame = self.render()

            if hasattr(self, 'resize_factor') and self.resize_factor > 1:
                import cv2
                frame = cv2.resize(
                    frame,
                    (self.config["gameboard size"] // self.resize_factor,
                     self.config["gameboard size"] // self.resize_factor),
                    interpolation=cv2.INTER_AREA)

            self.observation = frame

        else: raise ValueError('Unknown obs type')

        return self.observation


    def add_comm_message(self,message,is_ai=True):
        #timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        sender = "AGENT" if is_ai else "HUMAN"
        full_message = f"{sender}: {message}"
        self.comm_messages.append((full_message, is_ai))
        if len(self.comm_messages) > self.max_messages:
            self.comm_messages.pop(0)

    def render(self):
        # TODO modify to return a pixel frame if obs_type is pixel
        if (self.render_mode == 'headless'): # and (not self.obs_type == 'pixel'): # Do not render if in headless mode
            pass


        window_width, window_height = self.config['window size'][0], self.config['window size'][0]
        game_width = self.config["gameboard size"]
        ui_width = window_width - game_width

        if self.agent_info_height_req > 0: self.comm_pane_height = 220+self.agent_info_height_req
        else: self.comm_pane_height = 10

        # gameboard background
        self.window.fill((255, 255, 255))  # white background
        self.__render_box__(1, (0, 0, 0), 3)  # outer box
        pygame.draw.rect(self.window, (100, 100, 100), (game_width+self.gameboard_offset, 0, ui_width, window_height))
        pygame.draw.rect(self.window, (100, 100, 100), (0, game_width, game_width, window_height))  # Fill bottom portion with gray

        current_time = pygame.time.get_ticks()

        # Draw the aircraft
        for agent in self.agents: agent.draw(self.window)

        # Draw the targets
        #SHIP_HIGHVAL_UNOBSERVED = (225, 185, 0)  # gold
        SHIP_REGULAR_UNOBSERVED = (255, 215, 0)
        SHIP_REGULAR_LOWQ = (130, 0, 210)
        #SHIP_HIGHVAL_LOWQ = (150, 0, 255)
        SHIP_REGULAR_HIGHQ = (0, 255, 210)
        #SHIP_HIGHVAL_HIGHQ = (0, 240, 210)

        for target in self.targets:
            target_width = 7 if target[1] == 0 else 10
            target_color = SHIP_REGULAR_HIGHQ if target[2] == 1.0 else SHIP_REGULAR_LOWQ if target[2] == 0.5 else SHIP_REGULAR_UNOBSERVED
            #target_color = color_list[int(target[1])][int(target[2])]
            pygame.draw.circle(self.window, target_color, (float(target[3]), float(target[4])), target_width)

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

        elif self.config['num aircraft'] > 1:
            if self.agents[self.human_idx].health_points <= 3:
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

        self.NW_quad_button.is_latched = self.button_latch_dict['NW']
        self.NW_quad_button.color = self.gameplan_button_color
        self.NW_quad_button.draw(self.window)

        self.NE_quad_button.is_latched = self.button_latch_dict['NE']
        self.NE_quad_button.color = self.gameplan_button_color
        self.NE_quad_button.draw(self.window)

        self.SW_quad_button.is_latched = self.button_latch_dict['SW']
        self.SW_quad_button.color = self.gameplan_button_color
        self.SW_quad_button.draw(self.window)

        self.SE_quad_button.is_latched = self.button_latch_dict['SE']
        self.SE_quad_button.color = self.gameplan_button_color
        self.SE_quad_button.draw(self.window)

        self.full_quad_button.color = self.gameplan_button_color#(50,180,180)
        self.full_quad_button.is_latched = self.button_latch_dict['full']
        self.full_quad_button.draw(self.window)

        pygame.draw.line(self.window, (0, 0, 0), (self.right_pane_edge, 465), (self.right_pane_edge + 405, 465),4)  # Separating line between quadrant select and hold/waypoint

        self.waypoint_button.is_latched = self.button_latch_dict['waypoint']
        self.waypoint_button.color = self.gameplan_button_color
        self.waypoint_button.draw(self.window)

        self.hold_button.is_latched = self.button_latch_dict['hold']
        self.hold_button.color = self.gameplan_button_color
        self.hold_button.draw(self.window)

        pygame.draw.line(self.window, (0, 0, 0), (self.right_pane_edge, 3 * (self.quadrant_button_height) + 115 + 90), (self.right_pane_edge + 405, 3 * (self.quadrant_button_height) + 115 + 90),4)  # Separating line between hold/waypoint and regroup/tag team

        self.autonomous_button = Button("Auto Priorities", self.right_pane_edge + 15, 3 * (self.quadrant_button_height) + 115 + 90+20,self.gameplan_button_width * 2 + 15, 65)
        self.autonomous_button.is_latched = self.button_latch_dict['autonomous']
        self.autonomous_button.color = (50, 180, 180)
        self.autonomous_button.draw(self.window)

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

        agent0_health_window = HealthWindow(self.aircraft_ids[0],10,game_width+5, 'AGENT HP',self.AIRCRAFT_COLORS[0])
        agent0_health_window.update(self.agents[self.aircraft_ids[0]].health_points)
        agent0_health_window.draw(self.window)

        if self.config['num aircraft'] > 1:
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

    def close(self):
        if self.render_mode == 'human' and pygame.get_init():
            pygame.quit()

    # convert the environment into a state dictionary
    def get_state(self):
        state = {
            "aircrafts": {},
            "ships": {},
            "damage": self.damage,
            "num lowQ": self.low_quality_identified,
            "num highQ": self.high_quality_identified
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
    def __render_box__(self, distance_from_edge, color=(0, 0, 0), width=2, surface=None):
        """Utility function for drawing a square box"""
        surface = surface if surface is not None else self.window
        pygame.draw.line(surface, color, (distance_from_edge, distance_from_edge), (distance_from_edge, self.config["gameboard size"] - distance_from_edge), width)
        pygame.draw.line(surface, color, (distance_from_edge, self.config["gameboard size"] - distance_from_edge), (self.config["gameboard size"] - distance_from_edge, self.config["gameboard size"] - distance_from_edge), width)
        pygame.draw.line(surface, color, (self.config["gameboard size"] - distance_from_edge, self.config["gameboard size"] - distance_from_edge), (self.config["gameboard size"] - distance_from_edge, distance_from_edge), width)
        pygame.draw.line(surface, color, (self.config["gameboard size"] - distance_from_edge, distance_from_edge), (distance_from_edge, distance_from_edge), width)

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
        if self.config['num aircraft'] > 1:
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
        if self.config['num aircraft'] > 1:
            human_status = 'ALIVE' if self.agents[self.human_idx].alive else "DESTROYED"
            human_status_color = (0, 255, 0) if human_status == "ALIVE" else (255, 0, 0)

        # Create stats text surfaces
        stats_items = [
            f"Final Score: {round(self.score,0)}",
            f"Targets Identified: {self.identified_targets} / {self.num_targets}",
            f"Threat Levels Observed: {self.identified_threat_types} / {self.num_targets}",
            #f"Human Status: {human_status}",
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

    def denormalize_action(self, action):
        """

        Args:
            action (ndarray, size 3): Agent action to normalize. Should be in the form ndarray(waypoint_x, waypoint_y, id_method), all with range [-1, +1]

        Returns:
            waypoint (tuple, size 2): (x,y) waypoint with range [0, 1000]
            id_method (float): Value in range [0, 1]

        """

        if self.action_type == 'discrete':  # Convert first two values from 0-100 range to x,y coordinates on gameboard
            x_coord = float(action[0]) * (self.config["gameboard size"] / 100)
            y_coord = float(action[1]) * (self.config["gameboard size"] / 100)
            waypoint = (x_coord, y_coord)
            id_method = min([0, 50, 100], key=lambda x: abs(x - action[2]))

        elif self.action_type == 'continuous':
            normalized_x = (action[0] + 1) / 2  # Convert to 0,1 range
            normalized_y = (action[1] + 1) / 2  # Convert to 0,1 range
            x_coord = normalized_x * self.config["gameboard size"]
            y_coord = normalized_y * self.config["gameboard size"]
            waypoint = (x_coord, y_coord)

            #normalized_id = (action_value[2] + 1) / 2  # Convert from -1,1 to 0,1
            #id_method = normalized_id  # Or map to your desired range
            id_method = action[2]

        else:
            raise ValueError('Error in denormalize action: action type not recognized')

        return waypoint, id_method