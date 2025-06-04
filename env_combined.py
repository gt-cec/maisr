import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import random
import agents

import json
import os

from gui import Button, ScoreWindow, HealthWindow, TimeWindow, AgentInfoDisplay
import datetime
import math

"""
Combined env that implements:
    * Curriculum learning (via difficulty setting)
    * Frame stacking (via frame_skip setting)
"""

class MAISREnvVec(gym.Env):
    """Multi-Agent ISR Environment following the Gym format"""

    def __init__(self, config={}, window=None, clock=None, render_mode='headless',
                 num_agents=1,
                 tag='none',
                 run_name='no name',
                 seed=None,
                 difficulty=0,  # Used for curriculum learning
                 subject_id='999', user_group='99', round_number='99'):

        super().__init__()

        self.config = config
        self.run_name = run_name

        self.use_buttons = False # TODO make configurable in config

        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        self.use_beginner_levels = self.config['use_beginner_levels']  # If true, the agent only sees 5 beginner levels to make early training easier
        self.difficulty = difficulty

        self.highval_target_ratio = 0 # TODO make configurable in config

        self.tag = tag
        self.verbose = True if self.config['verbose'] == 'true' else False
        self.render_mode = render_mode
        self.gather_info = True  # self.render_mode == 'human' # Only populate the info dict if playing with humans

        self.obs_type = self.config['obs_type']
        self.action_type = self.config['action_type']
        # reward_type = self.config['reward type']

        self.shaping_decay_rate = self.config['shaping_decay_rate']
        self.shaping_coeff_wtn = self.config['shaping_coeff_wtn']
        self.shaping_coeff_prox = self.config['shaping_coeff_prox']
        self.shaping_coeff_earlyfinish = self.config['shaping_coeff_earlyfinish']
        self.shaping_time_penalty = self.config['shaping_time_penalty']

        if self.obs_type not in ['absolute', 'relative']: raise ValueError(f"obs_type invalid, got '{self.obs_type}'")
        if self.action_type not in ['discrete-downsampled', 'continuous-normalized','direct-control']: raise ValueError(f"action_type invalid, got '{self.action_type}'")
        # if reward_type not in ['proximity and target', 'waypoint-to-nearest', 'proximity and waypoint-to-nearest']: raise ValueError('reward_type must be normal. Others coming soon')
        if render_mode not in ['headless', 'human', 'rgb_array']: raise ValueError('Render mode must be headless, rgb_array, human')

        print(f'Env initialized: Tag={tag}, obs_type={self.obs_type}, action_type={self.action_type}')

        self.num_agents = num_agents

        self.tag = tag
        self.verbose = True if self.config['verbose'] == 'true' else False
        self.render_mode = render_mode
        self.gather_info = True #self.render_mode == 'human' # Only populate the info dict if playing with humans

        #self.time_limit = self.config['time limit'] # MOVED
        self.max_targets = 30
        #self.num_targets = min(30, self.config['num targets']) # If more than 30 targets specified, overwrite to 30 # MOVED

        ######################################### OBSERVATION AND ACTION SPACES ########################################
        if self.action_type == 'continuous-normalized':
            self.action_space = gym.spaces.Box(
                low=np.array([-1, -1], dtype=np.float32),
                high=np.array([1, 1], dtype=np.float32),
                dtype=np.float32)
        # elif self.action_type == 'discrete-downsampled':
        #     self.grid_size = 20 # Agent has grid_size * grid_size possible actions
        #     self.action_space = gym.spaces.MultiDiscrete([self.grid_size, self.grid_size])
        # elif self.action_type == 'direct-control':
        #     self.action_space = gym.spaces.Discrete(8) # 8 directions: up, up-right, right, down-right, down, down-left, left, up-left
        else: 
            raise ValueError("Action type must be continuous or discrete")

        self.obs_size = 2 + 3 * self.max_targets
        if self.obs_type == 'absolute':
            self.observation_space = gym.spaces.Box(
                low=-1, high=1,
                shape=(self.obs_size,),  # Wrap in tuple to make it iterable
                dtype=np.float32)
            self.observation = np.zeros(self.obs_size, dtype=np.float32)

        # elif self.obs_type == 'relative':
        #     self.observation_space = gym.spaces.Box(
        #         low=-1, high=1,
        #         shape=(self.obs_size,),  # Same size as the absolute observation
        #         dtype=np.float32)
        #     self.observation = np.zeros(self.obs_size, dtype=np.float32)

        else:
            raise ValueError("Obs type not recognized")

        ############################################## TUNABLE PARAMETERS ##############################################
        self.time_limit = self.config['time_limit']

        self.steps_for_lowqual_info = 3 * 60  # TODO tune this
        self.steps_for_highqual_info = 7 * 60  # TODO tune this. Currently 7 seconds
        self.prob_detect = self.config['prob_detect']  # 0.00133333333 # Probability of being detected on each step. Probability per second = prob_detect * 60 (TODO tune)

        # Set reward quantities for each event (agent only)
        #self.lowqual_regulartarget_reward = 0.25  # Reward earned for gathering low quality info about a regular target
        #self.lowqual_highvaltarget_reward = 0.5  # Reward earned for gathering low quality info about a high value target
        self.highqual_regulartarget_reward = self.config['highqual_regulartarget_reward'] # Reward earned for gathering high quality info about a regular value target
        self.highqual_highvaltarget_reward = self.config['highqual_highvaltarget_reward'] # Reward earned for gathering high quality info about a high value target
        #self.target_id_reward = 1.0

        self.detections_reward = 0 # -1.0 (# TODO temporarily removed for simplified env
        self.time_reward = 0.3  # Reward earned for every second early. 0.1 translates to 1.0 per 10 seconds


        ################################################# HUMAN THINGS #################################################
        self.subject_id = subject_id
        self.round_number = round_number
        self.user_group = user_group
        self.human_training = True if (self.round_number == 0 and self.user_group != 'test') else False  # True for the training round at start of experiment, false for rounds 1-4 (NOTE: This is NOT agent training!)

        self.paused = False
        self.unpause_countdown = False

        # Track score points (for human eyes only)
        self.score = 0
        self.all_targets_points = 0  # All targets ID'd
        self.low_qual_points = 10  # Points earned for gathering low quality info about a target
        self.high_qual_points = 10  # Points earned for gathering high quality info about a target
        self.time_points = 15  # Points given per second remaining
        self.human_hp_remaining_points = 70
        self.wingman_dead_points = -300  # Points subtracted for agent wingman dying
        self.human_dead_points = -400  # Points subtracted for human dying

        self.show_agent_waypoint = True #self.config['show_agent_waypoint']

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
        self.AIRCRAFT_ENGAGEMENT_RADIUS = 30  # TODO temporarily made smaller  # 100  # pixel width of aircraft engagement (to identify WEZ of threats)
        self.AIRCRAFT_ISR_RADIUS = 85  # 170  # pixel width of aircraft scanner (to identify hostile vs benign)

        self.GAMEBOARD_NOGO_RED = (255, 200, 200)  # color of the red no-go zone
        self.GAMEBOARD_NOGO_YELLOW = (255, 225, 200)  # color of the yellow no-go zone
        self.FLIGHTPLAN_EDGE_MARGIN = .2  # proportion distance from edge of gameboard to flight plan, e.g., 0.2 = 20% in, meaning a flight plan of (1,1) would go to 80%,80% of the gameboard
        self.AIRCRAFT_COLORS = [(0, 160, 160), (0, 0, 255), (200, 0, 200), (80, 80,80)]  # colors of aircraft 1, 2, 3, ... add more colors here, additional aircraft will repeat the last color

        if render_mode in ['rgb_array', 'human']:
            self.window = window
            self.clock = clock
            self.start_countdown_time = 5000  # How long in milliseconds to count down at the beginning of the game before it starts

            # Set GUI locations
            self.gameboard_offset = 0  # How far from left edge to start drawing gameboard
            self.window_x = self.config["window_size"][0]
            self.window_y = self.config["window_size"][1]
            self.window = pygame.display.set_mode((self.window_x, self.window_y))

            self.right_pane_edge = self.config['gameboard_size'] + 20  # Left edge of gameplan button windows
            self.comm_pane_edge = self.right_pane_edge
            self.gameplan_button_width = 180
            self.quadrant_button_height = 120
            self.autonomous_button_y = 590

            if render_mode == 'human':
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
                #self.pause_start_time = 0
                #self.total_pause_time = 0
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

                self.time_window = TimeWindow(self.config["gameboard_size"] * 0.43, self.config["gameboard_size"]+5,current_time=self.display_time, time_limit=self.time_limit)

        self.episode_counter = 0

        self.reset()


    def reset(self, seed=None):

        if self.config['use_curriculum'] == True:
            self.load_difficulty()

        if self.tag == 'pti_test':
            np.random.seed(42)
            random.seed(42)

        if self.use_beginner_levels:
            seed_list = [42, 123, 456, 789, 101] # List of seeds to cycle through
            current_seed_index = self.episode_counter % len(seed_list)
            current_seed = seed_list[current_seed_index]
            np.random.seed(current_seed)
            random.seed(current_seed)

        self.episode_counter += 1
        self.agents = [] # List of names of all current agents. Typically integers
        self.possible_agents = [0, 1] # PettingZoo format. List of possible agents
        self.max_num_agents = 2
        self.aircraft_ids = []  # Indices of the aircraft agents

        self.score = 0
        self.num_lowq_gathered = 0
        self.num_highq_gathered = 0

        self.display_time = 0  # Time that is used for the on-screen timer. Accounts for pausing.
        self.pause_start_time = 0
        self.total_pause_time = 0
        self.init = True

        # For plotting
        self.action_history = []
        self.agent_location_history = []
        self.direct_action_history = [] # Direct control only

        # self.damage = 0  # total damage from all agents
        # self.num_identified_ships = 0  # number of ships with accessed threat levels, used for determining game end
        #print(f'gameboard_size is {self.config["gameboard_size"]}')



        # Create vectorized ships/targets. Format: [id, value, info_level, x_pos, y_pos]
        self.num_targets = min(self.max_targets, self.config['num_targets'])  # If more than 30 targets specified, overwrite to 30

        self.targets = np.zeros((self.num_targets, 5), dtype=np.float32)
        self.targets[:, 0] = np.arange(self.num_targets) # Assign IDs (column 0) (Note, this does not go into the observation vector. It is just for reference)
        self.targets[:, 1] = np.random.choice([0, 1], size=self.num_targets, p=[1 - self.highval_target_ratio, self.highval_target_ratio]) # Assign target values (column 1) - regular (0) or high-value (1)
        self.targets[:, 2] = 0 # Initialize info_level (column 2) to all 0 (unknown)

        # Convert to [-150, +150] coordinate system
        map_half_size = self.config["gameboard_size"] / 2  # 150 for a 300x300 map
        margin = map_half_size * 0.03  # 3% margin from edges
        self.targets[:, 3] = np.random.uniform(-map_half_size + margin, map_half_size - margin, size=self.num_targets)
        self.targets[:, 4] = np.random.uniform(-map_half_size + margin, map_half_size - margin, size=self.num_targets)

        #self.targets[:, 3] = np.random.uniform(-1, 1, size=self.num_targets) # Randomly place targets on gameboard (columns 3-4)
        #self.targets[:, 4] = np.random.uniform(-1, 1, size=self.num_targets)

        self.target_timers = np.zeros(self.num_targets, dtype=np.int32)  # How long each target has been sensed for
        self.detections = 0 # Number of times a target has detected us. Results in a score penalty
        self.targets_identified = 0

        # Adjust shaping reward magnitudes
        self.shaping_coeff_wtn = self.shaping_coeff_wtn * self.shaping_decay_rate
        self.shaping_coeff_prox = self.shaping_coeff_prox * self.shaping_decay_rate


        if self.config['agent_start_location'] == "random":
            map_half_size = self.config["gameboard_size"] / 2
            agent_x = np.random.uniform(-map_half_size + 10, map_half_size - 10)
            agent_y = np.random.uniform(-map_half_size + 10, map_half_size - 10)
        else:
            # If specific coordinates are provided, convert them to centered system
            agent_x, agent_y = self.config['agent_start_location']

        # create the aircraft
        for i in range(self.num_agents):
            agents.Aircraft(self, 0, max_health=10,color=self.AIRCRAFT_COLORS[i],speed=self.config['game_speed']*self.config['human_speed'], flight_pattern=self.config["search pattern"])
            self.agents[self.aircraft_ids[i]].x, self.agents[self.aircraft_ids[i]].y = agent_x, agent_y

        #self.agent_idx = self.aircraft_ids[0]
        #print(f'Aircraft IDs: {self.aircraft_ids}')
        if self.num_agents == 2: # TODO Delete
            self.human_idx = self.aircraft_ids[1]  # Agent ID for the human-controlled aircraft. Dynamic so that if human dies in training round, their ID increments 1

        self.step_count_inner = 0 
        self.ep_reward = 0
        self.step_count_outer = 0 # Counts the outer steps (4 inner steps)

        self.all_targets_identified = False
        self.terminated = False
        self.truncated = False

        self.observation = self.get_observation()

        info = {}
        return self.observation, info


    def step(self, actions:dict):
        total_reward = 0
        info = None

        # Skip frames by repeating the action multiple times
        for frame in range(self.config['frame_skip']):
            observation, reward, self.terminated, self.truncated, info = self._single_step(actions)
            total_reward += reward

            # Break early if the episode is done to avoid unnecessary computation
            if self.terminated or self.truncated: break

        self.step_count_outer += 1

        #print(self.episode_counter)

        if self.tag == 'oar_test' and self.episode_counter in [0, 1, 2, 3, 50, 100, 500, 1000]:
            self.save_oar(observation, actions, total_reward)

        return observation, total_reward, self.terminated, self.truncated, info


    def _single_step(self, actions:dict):
        """
        args:
            actions: (Option 1) Dictionary of {agent_id: action(ndarray)}.
                     (Option 2) A single ndarray
        """

        self.step_count_inner += 1
        new_reward = {'high val target id': 0, 'regular val target id': 0,
                      'proximity':0, 'early finish': 0, 'waypoint-to-nearest':0} # Track events that give reward. Will be passed to get_reward at end of step
        new_score = 0 # For tracking human-understandable reward
        info = {
            "new_identifications": [],  # List to track newly identified targets/threats
            "reward_components": {},
            "detections": self.detections,  # Current detection count
            "target_ids": 0,
            'episode': {'r': 0, 'l': self.step_count_inner},
            "score_breakdown": {"target_points": 0, "threat_points": 0, "time_points": 0, "completion_points": 0, "penalty_points": 0}}

        # Process actions
        if isinstance(actions, dict): # Action is passed in as a dict {agent_id: action}
            for agent_id, action_value in actions.items():
                waypoint = self.process_action(action_value)
                self.agents[agent_id].waypoint_override = waypoint

        elif isinstance(actions, np.ndarray): # Single agent, action passed in directly as an array instead of list(arrays)

            waypoint = self.process_action(actions)
            #print(f'Waypoint is {waypoint}')
            self.agents[0].waypoint_override = (float(waypoint[0]), float(waypoint[1]))

        elif isinstance(actions, np.int64) and self.action_type == 'direct-control':
            #print(f'Agent selected action {actions}')
            waypoint = self.process_action(actions)
            self.agents[0].x = waypoint[0]
            self.agents[0].y = waypoint[1]

            if hasattr(self.agents[0], 'previous_x'):  # Update direction based on movement
                dx = self.agents[0].x - self.agents[0].previous_x
                dy = self.agents[0].y - self.agents[0].previous_y
                if dx != 0 or dy != 0:
                    self.agents[0].direction = math.atan2(dy, dx)

            # Store current position for next step
            self.agents[0].previous_x = self.agents[0].x
            self.agents[0].previous_y = self.agents[0].y

        else:
            print(f'Action type: {type(actions)}')
            raise ValueError('Actions input is an unknown type')

        # Log actions to action_history plot
        #if self.step_count_inner % 60 == 0:
        if self.action_type == 'direct-control':
            current_position = (self.agents[self.aircraft_ids[0]].x, self.agents[self.aircraft_ids[0]].y)
            self.action_history.append(current_position)  # Store current position before movement
            self.direct_action_history.append(actions)
            self.agent_location_history.append(current_position)

        else:
            #print(f'Agent took action {self.agents[0].waypoint_override}. Location history appended {(self.agents[self.aircraft_ids[0]].x, self.agents[self.aircraft_ids[0]].y)}')
            self.action_history.append(self.agents[0].waypoint_override)
            self.agent_location_history.append((self.agents[self.aircraft_ids[0]].x, self.agents[self.aircraft_ids[0]].y))

        # move the agents and check for gameplay updates
        for aircraft in [agent for agent in self.agents if agent.agent_class == "aircraft" and agent.alive]:

            if not self.action_type == 'direct-control':
                #print('Moving aircraft')
                #print(f'Aircraft target_point is {aircraft.target_point}, location is {aircraft.x, aircraft.y}')
                #print(f'Aircraft waypoint_override is {aircraft.waypoint_override}')
                aircraft.move() # First, move using the waypoint override set above

            # Calculate distances to all targets
            aircraft_pos = np.array([aircraft.x, aircraft.y])  # Get aircraft position
            target_positions = self.targets[:, 3:5]  # x,y coordinates
            distances = np.sqrt(np.sum((target_positions - aircraft_pos) ** 2, axis=1))

            # Create a mask for unidentified targets (info_level < 1.0)
            #proximity_reward = 0
            if aircraft.agent_idx == 0:
                unidentified_mask = self.targets[:, 2] < 1.0
                if np.any(unidentified_mask):
                    unidentified_distances = distances[unidentified_mask]
                    nearest_unidentified_distance = np.min(unidentified_distances)

                    if not hasattr(self, 'previous_nearest_distance'):
                        self.previous_nearest_distance = nearest_unidentified_distance

                    unidentified_indices = np.where(unidentified_mask)[0]
                    nearest_unidentified_idx = unidentified_indices[np.argmin(unidentified_distances)]

                    current_waypoint = self.agents[0].waypoint_override
                    if current_waypoint is not None:
                        # Check if waypoint is within 30 px of nearest unknown target
                        nearest_target_location = target_positions[nearest_unidentified_idx]

                        waypoint_to_target_distance = np.sqrt(np.sum((nearest_target_location - current_waypoint) ** 2))
                        if waypoint_to_target_distance <= 40:
                            new_reward['waypoint-to-nearest'] = 1.0
                        else:
                            new_reward['waypoint-to-nearest'] = 0.0

                    distance_improvement = self.previous_nearest_distance - nearest_unidentified_distance
                    if distance_improvement > 0:
                        new_reward['proximity'] = distance_improvement

                    self.previous_nearest_distance = nearest_unidentified_distance

            # Find targets within ISR range (for identification)
            in_isr_range = distances <= self.AIRCRAFT_ENGAGEMENT_RADIUS


            # Process newly identified targets
            for target_idx in range(self.num_targets):
                if in_isr_range[target_idx] and self.targets[target_idx, 2] < 1.0: # If target is in range and not fully identified
                    #self.target_timers[target_idx] += 1  # Increment timer for this target
                    self.targets_identified += 1
                    self.targets[target_idx, 2] = 1.0

                    # Add reward (for agent) and score (for human).
                    if self.targets[target_idx, 1] == 0.0:
                        new_score += self.highqual_regulartarget_reward
                        new_reward['regular val target id'] += 1
                    else:
                        new_score += self.highqual_highvaltarget_reward
                        new_reward['high val target id'] += 1

                    # Update info dictionary
                    info["score_breakdown"]["target_points"] += self.highqual_regulartarget_reward if self.targets[target_idx, 1] == 0.0 else self.highqual_highvaltarget_reward
                    info["new_identifications"].append({
                        "type": "low quality info gathered",
                        "target_id": int(self.targets[target_idx, 0]),
                        "aircraft": aircraft.agent_idx,
                        "time": self.display_time
                    })

                # Handle aircraft being detected by high value targets
                if self.prob_detect > 0.0: # If prob detect is zero, skip
                    if in_isr_range[target_idx] and self.targets[target_idx, 1] == 1.0: # Only if we're in range of high value targets
                        if np.random.random() < self.prob_detect: # Roll RNG to see if we're detected
                            self.detections += 1
                            new_reward['detections'] += 1
                            info["detections"] = self.detections


        self.all_targets_identified = np.all(self.targets[:, 2] == 1.0)
        #if self.all_targets_identified:
            #print(self.all_targets_identified)
            #print(f'DEBUG: self.all_targets_identified = {self.all_targets_identified}; self.targets_identified = {self.targets_identified}')

        #if self.verbose: print("Targets with low-quality info: ", self.low_quality_identified, " Targets with high-quality info: ", self.high_quality_identified, "Detections: ", self.detections)

        if self.all_targets_identified:
            print('TERMINATED: All targets identified')
            self.terminated = True
            new_score += self.all_targets_points  # Left this but it doesn't go into reward
            new_score += (self.time_limit - self.display_time / 1000) * self.time_points
            new_reward['early finish'] = (self.time_limit - self.display_time / 1000)

        if self.step_count_outer >= 490 or self.display_time / 1000 >= self.time_limit: # TODO: Temporarily hard-coding 490 steps
            #print('TERMINATED: TIME UP')
            self.terminated = True

        # Calculate reward
        reward = self.get_reward(new_reward) # For agent
        self.score += new_score # For human
        self.observation = self.get_observation() # Get observation
        self.ep_reward += reward

        info['episode'] = {'r': self.ep_reward, 'l': self.step_count_inner, }
        info['reward_components'] = new_reward
        info['detections'] = self.detections
        info["target_ids"] =self.targets_identified


        if (self.terminated or self.truncated):
            print(f'Round complete, reward {round(info['episode']['r'],3)}, outer steps {self.step_count_outer}, inner timesteps {info['episode']['l']}, score {self.score} | {self.targets_identified} low quality | {self.detections} detections | {round(self.time_limit-self.display_time/1000,1)} secs left')
            if self.action_type == 'direct-control':
                print(f'Action history: {self.direct_action_history}')

            if self.tag == 'pti_test':
                self.save_action_history_plot()

            if self.tag == 'oar_test':
                if self.episode_counter in [0, 1, 2, 3, 50, 100, 500, 1000]:
                    self.save_action_history_plot()

            if self.tag in ['eval', 'train_mp0']:
                if self.episode_counter in [0, 1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500, 800, 1000, 1200, 1400, 1700, 2000, 2300, 2400, 2600, 2800, 3000, 4000, 5000, 6000, 7000]:
                        self.save_action_history_plot()
                elif self.episode_counter % 500 == 0:
                    self.save_action_history_plot()

            if self.render_mode == 'human': pygame.time.wait(50)

        # Advance time
        if self.render_mode == 'headless': self.display_time = self.display_time + (1000/60) # If agent training, each step is 1/60th of a second
        elif not self.paused: self.display_time = pygame.time.get_ticks() - self.total_pause_time
        if self.init: self.init = False

        return self.observation, reward, self.terminated, self.truncated, info

    def get_reward(self, new_reward):

        reward = (new_reward['high val target id'] * self.highqual_regulartarget_reward) + \
                 (new_reward['regular val target id'] * self.highqual_highvaltarget_reward) + \
                 (new_reward['waypoint-to-nearest'] * self.shaping_coeff_wtn) + \
                 (new_reward['proximity'] * self.shaping_coeff_prox) + \
                 (new_reward['early finish'] * self.time_reward) + \
                 (self.shaping_time_penalty)

        return reward

    def get_observation(self):
        """
        State will include the following features:
        # TODO OUT OF DATE. Coord frame modified to be -1,+1
            Absolute mode:
                0 agent_x,             # (0-1) normalized position
                1 agent_y,             # (0-1) normalized position
                2+i*3 info_level       # 0 for no info, 0.5 for low quality info, 1.0 for full info
                3+i*3 target_x,        # (0-1) normalized position
                4+i*3 target_y,        # (0-1) normalized position

            Relative mode:
                0 agent_x,             # (0-1) normalized position (for reference)
                1 agent_y,             # (0-1) normalized position (for reference)
                2+i*3 info_level       # 0 for no info, 0.5 for low quality info, 1.0 for full info
                3+i*3 rel_target_x,    # (-1 to 1) relative position from agent to target
                4+i*3 rel_target_y,    # (-1 to 1) relative position from agent to target
        """

        if self.obs_type == 'absolute':
            self.observation = np.zeros(self.obs_size, dtype=np.float32)

            # New agent position (-1, +1)
            map_half_size = self.config["gameboard_size"] / 2
            self.observation[0] = (self.agents[self.aircraft_ids[0]].x) / map_half_size
            self.observation[1] = (self.agents[self.aircraft_ids[0]].y) / map_half_size

            # Agent position (normalized to [0,1])
            #self.observation[0] = self.agents[self.aircraft_ids[0]].x / self.config["gameboard_size"]
            #self.observation[1] = self.agents[self.aircraft_ids[0]].y / self.config["gameboard_size"]

            # Process target data
            targets_per_entry = 3  # Each target has 3 features in the observation
            target_features = np.zeros((self.max_targets, targets_per_entry), dtype=np.float32)

            target_features[:self.num_targets, 0] = self.targets[:, 2]  # info levels
            target_features[:self.num_targets, 1] = (self.targets[:, 3]) / map_half_size
            target_features[:self.num_targets, 2] = (self.targets[:, 4]) / map_half_size

            #target_features[:self.num_targets, 1] = self.targets[:, 3] / self.config["gameboard_size"]  # x position (normalized)
            #target_features[:self.num_targets, 2] = self.targets[:, 4] / self.config["gameboard_size"]  # y position (normalized)

            target_start_idx = 2
            self.observation[target_start_idx:target_start_idx + self.max_targets * targets_per_entry] = target_features.flatten()

        # elif self.obs_type == 'relative':
        #     self.observation = np.zeros(self.obs_size, dtype=np.float32)
        #
        #     # Get agent position (normalized)
        #     agent_x_norm = self.agents[self.aircraft_ids[0]].x / self.config["gameboard_size"]
        #     agent_y_norm = self.agents[self.aircraft_ids[0]].y / self.config["gameboard_size"]
        #
        #     # Store agent's absolute position for reference (still useful for the agent)
        #     self.observation[0] = agent_x_norm
        #     self.observation[1] = agent_y_norm
        #
        #     # Process target data with relative positions
        #     targets_per_entry = 3
        #     target_features = np.zeros((self.max_targets, targets_per_entry), dtype=np.float32)
        #
        #     # Copy info levels
        #     target_features[:self.num_targets, 0] = self.targets[:, 2]
        #
        #     # Calculate relative positions for all targets at once (vectorized)
        #     if self.num_targets > 0:
        #         # Get all target normalized positions
        #         target_x_norm = self.targets[:self.num_targets, 3] / self.config["gameboard_size"]
        #         target_y_norm = self.targets[:self.num_targets, 4] / self.config["gameboard_size"]
        #
        #         # Calculate relative positions (normalized difference)
        #         # This represents the vector from agent to target in normalized space
        #         rel_x = target_x_norm - agent_x_norm  # Range: [-1, 1]
        #         rel_y = target_y_norm - agent_y_norm  # Range: [-1, 1]
        #
        #         target_features[:self.num_targets, 1] = rel_x
        #         target_features[:self.num_targets, 2] = rel_y
        #
        #     target_start_idx = 2
        #     self.observation[
        #     target_start_idx:target_start_idx + self.max_targets * targets_per_entry] = target_features.flatten()

        else:
            raise ValueError('Unknown obs type')

        return self.observation


    def add_comm_message(self,message,is_ai=True):
        #timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        sender = "AGENT" if is_ai else "HUMAN"
        full_message = f"{sender}: {message}"
        self.comm_messages.append((full_message, is_ai))
        if len(self.comm_messages) > self.max_messages:
            self.comm_messages.pop(0)

    def render(self):

        if (self.render_mode == 'headless'): # and (not self.obs_type == 'pixel'): # Do not render if in headless mode
            pass

        window_width, window_height = self.config['window_size'][0], self.config['window_size'][0]
        game_width = self.config["gameboard_size"]
        ui_width = window_width - game_width

        if self.render_mode == 'human':
            if self.agent_info_height_req > 0: self.comm_pane_height = 220+self.agent_info_height_req
            else: self.comm_pane_height = 10

        # gameboard background
        self.window.fill((255, 255, 255))  # white background
        self.__render_box__(1, (0, 0, 0), 3)  # outer box
        pygame.draw.rect(self.window, (100, 100, 100), (game_width+self.gameboard_offset, 0, ui_width, window_height))
        pygame.draw.rect(self.window, (100, 100, 100), (0, game_width, game_width, window_height))  # Fill bottom portion with gray

        current_time = pygame.time.get_ticks()

        # Draw the aircraft (# TODO UPDATE FOR -1,+1 grid)
        for agent in self.agents:
            agent.draw(self.window)

        # Draw the targets
        SHIP_REGULAR_UNOBSERVED = (255, 215, 0)
        SHIP_REGULAR_LOWQ = (130, 0, 210)
        SHIP_REGULAR_HIGHQ = (0, 255, 210)

        for target in self.targets:
            target_width = 7 if target[1] == 0 else 10
            target_color = SHIP_REGULAR_HIGHQ if target[2] == 1.0 else SHIP_REGULAR_LOWQ if target[2] == 0.5 else SHIP_REGULAR_UNOBSERVED
            #target_color = color_list[int(target[1])][int(target[2])]
            #pygame.draw.circle(self.window, target_color, (float(target[3]), float(target[4])), target_width)

            map_half_size = self.config["gameboard_size"] / 2
            screen_x = target[3] + map_half_size
            screen_y = target[4] + map_half_size
            pygame.draw.circle(self.window, target_color, (float(screen_x), float(screen_y)), target_width)

        # Draw green lines and black crossbars
        self.__render_box__(35, (0, 128, 0), 2)  # inner box
        pygame.draw.line(self.window, (0, 0, 0), (self.config["gameboard_size"] // 2, 0),(self.config["gameboard_size"] // 2, self.config["gameboard_size"]), 2)
        pygame.draw.line(self.window, (0, 0, 0), (0, self.config["gameboard_size"] // 2),(self.config["gameboard_size"], self.config["gameboard_size"] // 2), 2)

        # Draw white rectangles around outside edge
        # pygame.draw.rect(self.window, (255,255,255),(0,0,game_width,35)) # Top
        # pygame.draw.rect(self.window, (255,255,255), (0, game_width-33, game_width, 33)) # bottom
        # pygame.draw.rect(self.window, (255,255,255), (0, 0, 35, game_width))  # Left
        # pygame.draw.rect(self.window, (255,255,255), (1000-33, 0, 35, game_width-2))  # Right

        # Handle damage flashes when human is damaged
        if self.render_mode == 'human':
            if current_time > 1000 and (current_time - self.damage_flash_start < self.damage_flash_duration):
                progress = (current_time - self.damage_flash_start) / self.damage_flash_duration  # Calculate alpha based on time elapsed
                alpha = int(255 * (1 - progress))
                border_surface = pygame.Surface((self.config["gameboard_size"], self.config["gameboard_size"]),pygame.SRCALPHA)
                border_width = 50
                border_color = (255, 0, 0, alpha)  # Red with calculated alpha
                pygame.draw.rect(border_surface, border_color,(0, 0, self.config["gameboard_size"], border_width))  # Top border
                pygame.draw.rect(border_surface, border_color, (0, self.config["gameboard_size"] - border_width, self.config["gameboard_size"],border_width))  # Bottom border
                pygame.draw.rect(border_surface, border_color,(0, 0, border_width, self.config["gameboard_size"]))  # Left border
                pygame.draw.rect(border_surface, border_color, (
                self.config["gameboard_size"] - border_width, 0, border_width,
                self.config["gameboard_size"]))  # Right border
                self.window.blit(border_surface, (0, 0))  # Blit the border surface onto the main window

            # Handle flash when agent is damaged (TODO: Make this a different graphic)
            if current_time > 1000 and (current_time - self.agent_damage_flash_start < self.damage_flash_duration):
                progress = (current_time - self.agent_damage_flash_start) / self.damage_flash_duration  # Calculate alpha based on time elapsed
                alpha = int(255 * (1 - progress))
                border_surface = pygame.Surface((self.config["gameboard_size"], self.config["gameboard_size"]),pygame.SRCALPHA)
                border_width = 50
                border_color = (255, 0, 0, alpha)  # Red with calculated alpha
                pygame.draw.rect(border_surface, border_color,(0, 0, self.config["gameboard_size"], border_width))  # Top border
                pygame.draw.rect(border_surface, border_color, (0, self.config["gameboard_size"] - border_width, self.config["gameboard_size"],border_width))  # Bottom border
                pygame.draw.rect(border_surface, border_color,(0, 0, border_width, self.config["gameboard_size"]))  # Left border
                pygame.draw.rect(border_surface, border_color, (
                self.config["gameboard_size"] - border_width, 0, border_width,
                self.config["gameboard_size"]))  # Right border
                self.window.blit(border_surface, (0, 0))  # Blit the border surface onto the main window

            if self.use_buttons:
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

        # if self.config['num aircraft'] > 1:
        #     agent1_health_window = HealthWindow(self.human_idx, game_width-150, game_width + 5, 'HUMAN HP',self.AIRCRAFT_COLORS[1])
        #     agent1_health_window.update(self.agents[self.human_idx].health_points)
        #     agent1_health_window.draw(self.window)

        # current_time = pygame.time.get_ticks()
        #
        # if current_time > self.start_countdown_time:
        #     self.time_window.update(self.display_time)
        #     self.time_window.draw(self.window)

        # Draw agent status window
        #if self.agent_info_height_req > 0: self.agent_info_display.draw(self.window)

        if self.render_mode == 'human':

            corner_round_text = f"ROUND {self.round_number + 1}/4" if self.user_group == 'test' else f"ROUND {self.round_number}/4"
            corner_round_font = pygame.font.SysFont(None, 36)
            corner_round_text_surface = corner_round_font.render(corner_round_text, True, (255, 255, 255))
            corner_round_rect = corner_round_text_surface.get_rect(
                center=(675, 1030))
            self.window.blit(corner_round_text_surface, corner_round_rect)

            # # Countdown from 5 seconds at start of game
            # (TODO TEMP REMOVED)
            # if current_time <= self.start_countdown_time:
            #     countdown_font = pygame.font.SysFont(None, 120)
            #     message_font = pygame.font.SysFont(None, 60)
            #     round_font = pygame.font.SysFont(None, 72)
            #     countdown_start = 0
            #     countdown_surface = pygame.Surface((self.window.get_width(), self.window.get_height()))
            #     countdown_surface.set_alpha(128)  # 50% transparent
            #
            #     time_left = self.start_countdown_time/1000 - (current_time - countdown_start) / 1000
            #
            #     # Draw semi-transparent overlay
            #     countdown_surface.fill((100, 100, 100))
            #     self.window.blit(countdown_surface, (0, 0))
            #
            #     # Draw round name
            #     if self.user_group == 'test':
            #         round_text = f"ROUND {self.round_number+1}/4"
            #     else:
            #         if self.round_number == 0: round_text = "TRAINING ROUND"
            #         else: round_text = f"ROUND {self.round_number}/4"
            #     round_text_surface = round_font.render(round_text, True, (255, 255, 255))
            #     round_rect = round_text_surface.get_rect(center=(self.window.get_width() // 2, self.window.get_height() // 2 - 120))
            #     self.window.blit(round_text_surface, round_rect)
            #
            #     # Draw "Get Ready!" message
            #     ready_text = message_font.render("Get Ready!", True, (255, 255, 255))
            #     ready_rect = ready_text.get_rect(center=(self.window.get_width() // 2, self.window.get_height() // 2 - 50))
            #     self.window.blit(ready_text, ready_rect)
            #
            #     # Draw countdown number
            #     countdown_text = countdown_font.render(str(max(1, int(time_left + 1))), True, (255, 255, 255))
            #     text_rect = countdown_text.get_rect(center=(self.window.get_width() // 2, self.window.get_height() // 2 + 20))
            #     self.window.blit(countdown_text, text_rect)
            #
            #     pygame.time.wait(50)  # Control update rate
            #
            #     # Handle any quit events during countdown
            #     for event in pygame.event.get():
            #         if event.type == pygame.QUIT:
            #             pygame.quit()
            #             return

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

        #if self.terminated or self.truncated:
            #self._render_game_complete() TODO temp removed

        pygame.display.update()
        if self.render_mode == 'human': self.clock.tick(60)

    def close(self):
        if self.render_mode == 'human' and pygame.get_init():
            pygame.quit()

    # convert the environment into a state dictionary
    def get_state(self):
        state = {
            "aircrafts": {},
            "ships": {},
            "detections": self.detections,
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
        pygame.draw.line(surface, color, (distance_from_edge, distance_from_edge), (distance_from_edge, self.config["gameboard_size"] - distance_from_edge), width)
        pygame.draw.line(surface, color, (distance_from_edge, self.config["gameboard_size"] - distance_from_edge), (self.config["gameboard_size"] - distance_from_edge, self.config["gameboard_size"] - distance_from_edge), width)
        pygame.draw.line(surface, color, (self.config["gameboard_size"] - distance_from_edge, self.config["gameboard_size"] - distance_from_edge), (self.config["gameboard_size"] - distance_from_edge, distance_from_edge), width)
        pygame.draw.line(surface, color, (self.config["gameboard_size"] - distance_from_edge, distance_from_edge), (distance_from_edge, distance_from_edge), width)

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
            if self.detections >= 5:
                title_surface = title_font.render('GAME OVER (>5 DETECTIONS)', True, (0, 0, 0))
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
            f"Targets Identified: {self.targets_identified} / {self.num_targets}",
            #f"Threat Levels Observed: {self.identified_threat_types} / {self.num_targets}",
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

    
    def process_action(self, action):
        """
        If the action is continuous-normalized, this denormalizes it from -1, +1 to [0, gameboard_size] in both axes
        If the action is discrete-downsampled, this converts it to the gameboard continuous grid, placing the waypoint in the center of the selected grid square

        Args:
            action (ndarray, size 2): Agent action to normalize. Should be in the form ndarray(waypoint_x, waypoint_y), all with range [-1, +1]

        Returns:
            waypoint (tuple, size 2): (x,y) waypoint with range [0, gameboard_size]
        """

        # if self.action_type == 'discrete-downsampled':
        #     # Convert discrete grid coordinates to actual gameboard coordinates
        #     # For a grid_size x grid_size grid on a gameboard_size x gameboard_size board
        #     # We place the waypoint at the center of the grid cell
        #
        #     # Ensure action values are within grid bounds
        #     x_grid = min(max(int(action[0]), 0), self.grid_size - 1)
        #     y_grid = min(max(int(action[1]), 0), self.grid_size - 1)
        #
        #     # Convert to gameboard coordinates (center of grid cell)
        #     cell_size = self.config["gameboard_size"] / self.grid_size
        #     x_coord = (x_grid + 0.5) * cell_size  # +0.5 to get to center of cell
        #     y_coord = (y_grid + 0.5) * cell_size
        #
        #     waypoint = (float(x_coord), float(y_coord))

        if self.action_type == 'continuous-normalized':
            # Validate action range
            if action[0] > 1.1 or action[1] > 1.1 or action[0] < -1.1 or action[1] < -1.1:
                raise ValueError(f'ERROR: Actions are not normalized to [-1, +1]. Got: {action}')

            if self.obs_type == 'absolute':
                # Convert from [-1,+1] to [-150,+150]
                map_half_size = self.config["gameboard_size"] / 2
                x_coord = action[0] * map_half_size  # [-1,+1] -> [-150,+150]
                y_coord = action[1] * map_half_size

            # elif self.obs_type == 'relative':
            #     # Get current agent position
            #     current_x = self.agents[self.aircraft_ids[0]].x
            #     current_y = self.agents[self.aircraft_ids[0]].y
            #
            #     # Define maximum movement distance per action
            #     # This should be tuned based on your environment's needs
            #     # Option 1: Fixed percentage of map size (more predictable)
            #     max_move_distance = self.config["gameboard_size"] * 0.3  # 30% of map size
            #
            #     # Option 2: Adaptive based on position (ensures agent can reach edges)
            #     # max_move_distance = min(
            #     #     self.config["gameboard_size"] * 0.5,  # Cap at 50% of map
            #     #     max(current_x, current_y,
            #     #         self.config["gameboard_size"] - current_x,
            #     #         self.config["gameboard_size"] - current_y)
            #     # )
            #
            #     # Calculate displacement from current position
            #     dx = action[0] * max_move_distance  # action[0] in [-1,+1] -> dx in [-max_move, +max_move]
            #     dy = action[1] * max_move_distance
            #
            #     # Calculate new position and clip to boundaries
            #     x_coord = np.clip(current_x + dx, 0, self.config["gameboard_size"])
            #     y_coord = np.clip(current_y + dy, 0, self.config["gameboard_size"])

        else:
            raise ValueError(f'Error in process_action: action type "{self.action_type}" not recognized')

        waypoint = (float(x_coord), float(y_coord))

        return waypoint

    def save_action_history_plot(self, note=''):
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import datetime
            import os

            # Create directory if it doesn't exist
            os.makedirs(f'logs/action_histories_new/{self.run_name}', exist_ok=True)

            # Calculate map bounds for centered coordinate system
            map_half_size = self.config["gameboard_size"] / 2  # 150 for a 300x300 map

            # Extract agent location history (already in centered coordinates)
            agent_x_coords = [pos[0] for pos in self.agent_location_history]
            agent_y_coords = [pos[1] for pos in self.agent_location_history]  # No flipping needed

            # Create a new figure
            plt.figure(figsize=(10, 10))

            # Set up the plot with centered coordinate limits
            plt.xlim(-map_half_size, map_half_size)
            plt.ylim(-map_half_size, map_half_size)

            # Plot targets (already in centered coordinates)
            for i in range(self.num_targets):
                target_x = self.targets[i, 3]  # Already in centered coordinates
                target_y = self.targets[i, 4]  # Already in centered coordinates

                size_factor = 1000 / self.config["gameboard_size"]  # Assuming 1000 was the original reference size
                marker_size = (100 * size_factor) if self.targets[i, 1] == 1 else (50 * size_factor)

                if i == 0:
                    color = 'blue' if self.targets[i, 2] == 1.0 else 'chocolate'
                else:
                    #color = 'red' if self.targets[i, 2] == 1.0 else 'orange'  # Identified are red, unidentified are orange
                    color = 'cyan' if self.targets[i, 2] == 1.0 else 'orange'

                plt.scatter(target_x, target_y, s=marker_size, color=color, alpha=0.7, marker='o')

            # Plot agent trajectory (actual location history) as a line with points
            if agent_x_coords and agent_y_coords:
                plt.plot(agent_x_coords, agent_y_coords, 'g-', alpha=0.7, linewidth=2)
                plt.scatter(agent_x_coords, agent_y_coords, s=20, c=range(len(agent_x_coords)),
                            cmap='Greens', alpha=0.7, marker='o', label='Agent Path')

            # Only plot waypoint history for waypoint-based action types
            if self.action_type != 'direct-control':
                # Extract x and y coordinates from action history (waypoints, already in centered coordinates)
                x_coords = [action[0] for action in self.action_history]
                y_coords = [action[1] for action in self.action_history]

                # Plot waypoint history (action history) as a line with points
                if x_coords and y_coords:
                    plt.plot(x_coords, y_coords, 'b-', alpha=0.2, linewidth=1)
                    plt.scatter(x_coords, y_coords, s=30, c=range(len(x_coords)),
                                cmap='cool', alpha=0.8, marker='x', label='Agent Waypoints')

                    # Add starting and ending points with different markers
                    plt.scatter(x_coords[0], y_coords[0], s=120, color='blue', marker='*', label='Start Waypoint')
                    plt.scatter(x_coords[-1], y_coords[-1], s=120, color='cyan', marker='*', label='End Waypoint')

            if agent_x_coords and agent_y_coords:
                plt.scatter(agent_x_coords[0], agent_y_coords[0], s=120, color='darkgreen', marker='*',
                            label='Start Position')
                plt.scatter(agent_x_coords[-1], agent_y_coords[-1], s=120, color='lime', marker='*',
                            label='End Position')

            # Add a colorbar to show time progression
            cbar = plt.colorbar()
            cbar.set_label('Timestep')

            # Add grid lines centered at origin
            plt.grid(True, alpha=0.3)
            plt.gca().set_xticks(range(int(-map_half_size), int(map_half_size) + 1, 25))
            plt.gca().set_yticks(range(int(-map_half_size), int(map_half_size) + 1, 25))

            # Add labels and title
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            action_type_label = 'direct-control' if self.action_type == 'direct-control' else self.action_type
            plot_title = f'{self.tag} - episode {self.episode_counter} - Agent Movement (Reward: {self.ep_reward:.2f}, Steps: {self.step_count_outer})'
            plt.title(plot_title)

            # Add a legend
            plt.legend(loc='upper right')

            # Add centered quadrant lines (origin at center)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)  # Horizontal line at y=0
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=2)  # Vertical line at x=0

            # Optional: Add boundary lines to show map edges
            plt.axhline(y=map_half_size, color='red', linestyle='--', alpha=0.3, label='Map Boundary')
            plt.axhline(y=-map_half_size, color='red', linestyle='--', alpha=0.3)
            plt.axvline(x=map_half_size, color='red', linestyle='--', alpha=0.3)
            plt.axvline(x=-map_half_size, color='red', linestyle='--', alpha=0.3)

            # Add coordinate system info to the plot
            plt.text(-map_half_size + 10, map_half_size - 20,
                     f'Coordinate System: [{-map_half_size:.0f}, {map_half_size:.0f}]',
                     fontsize=4, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            # Save the figure with a timestamp
            filename = f'logs/action_histories_new/{self.run_name}/{note}{self.tag}_ep{self.episode_counter}_{self.run_name}.png'
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            plt.close()

            print(f"Action history plot saved to {filename}")
        except ImportError as e:
            print(f"Could not save action history plot: {e}")
        except Exception as e:
            print(f"Error saving action history plot: {e}")


    def set_difficulty(self, difficulty):
        """Method to change difficulty level from an external method (i.e. a training loop"""
        self.difficulty = difficulty
        print(f'env.set_difficulty: Difficulty is now {self.difficulty}')

    def load_difficulty(self):
        # TODO make config use num targets
        """Method to update env parameters using the current difficulty setting"""
        if self.difficulty == 0:
            #self.config['gameboard_size'] = 300
            #self.config['num targets'] = 1
            self.prob_detect = 0
            self.reward_type = 'proximity and waypoint-to-nearest'
            self.highval_target_ratio = 0
            #self.shaping_time_penalty = 0 TODO
            self.shaping_time_penalty = self.config['shaping_time_penalty']

        if self.difficulty == 1:
            #self.config['gameboard_size'] = 300
            self.config['num targets'] = 10
            self.prob_detect = 0
            self.reward_type = 'proximity and waypoint-to-nearest'
            self.highval_target_ratio = 0
            self.shaping_time_penalty = self.config['shaping_time_penalty']


        if self.difficulty == 2:
            self.config['gameboard_size'] = self.config['gameboard_size'] * (400/300)
            self.config['num targets'] = 10
            self.prob_detect = 0
            self.reward_type = 'proximity and waypoint-to-nearest'
            self.highval_target_ratio = 0
            self.shaping_time_penalty = self.config['shaping_time_penalty']


        if self.difficulty == 3:
            #self.config['gameboard_size'] = 400
            self.config['num targets'] = 10
            self.prob_detect = 0.00167
            self.reward_type = 'proximity and waypoint-to-nearest'
            self.highval_target_ratio = 0
            self.shaping_time_penalty = self.config['shaping_time_penalty']


        if self.difficulty == 4:
            self.config['gameboard_size'] = 600
            self.config['num targets'] = 10
            self.prob_detect = 0.00167
            self.reward_type = 'proximity and waypoint-to-nearest'
            self.highval_target_ratio = 0
            self.shaping_time_penalty = self.config['shaping_time_penalty']


        if self.difficulty == 5:
            self.config['gameboard_size'] = 600
            self.config['num targets'] = 10
            self.prob_detect = 0.00167
            self.reward_type = 'sparse'
            self.highval_target_ratio = 0
            self.shaping_time_penalty = self.config['shaping_time_penalty']


        if self.difficulty == 6:
            self.config['gameboard_size'] = 600
            self.config['num targets'] = 10
            self.prob_detect = 0.00167
            self.reward_type = 'sparse'
            self.highval_target_ratio = 0.3
            self.shaping_time_penalty = self.config['shaping_time_penalty']

        #print(f'env.load_difficulty: DIFFICULTY {self.difficulty}: board size {self.config["gameboard_size"]}, targets {self.config['num targets']}')

    def save_oar(self, observation, actions, reward):
        """Save O, A, and R to a json at each timestep"""

        # Create directory if it doesn't exist
        os.makedirs('logs/oar_logs', exist_ok=True)

        # Create filename based on run_name and episode
        filename = f'logs/oar_logs/oar_{self.run_name}_ep{self.episode_counter}.json'

        # Create the data entry for this timestep

        raw_action = actions.tolist() if hasattr(actions, 'tolist') else actions
        processed_action = self.process_action(raw_action)

        timestep_data = {
            'timestep': self.step_count_outer,
            'observation': observation.tolist() if hasattr(observation, 'tolist') else observation,
            'raw actions': raw_action,
            'processed actions': processed_action,
            'reward': float(reward)
        }

        # Load existing data or create new structure
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
        else:
            #print('Initializing json')
            data = {'episode': self.episode_counter, 'timesteps': []}

        # Add new timestep data
        data['timesteps'].append(timestep_data)

        # Save back to file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)