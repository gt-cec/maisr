import time

import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import random
import utility.agents as agents

import json
import os
#import cv2

from utility.gui import Button, ScoreWindow, HealthWindow, TimeWindow, AgentInfoDisplay
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
                 subject_id='999', user_group='99', round_number='99'):

        super().__init__()

        self.config = config # Loaded from .json into a dictionary


        self.run_name = run_name # For logging

        self.use_buttons = False # TODO make configurable in config

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.start_location_list = [(-0.040025224653889024, -0.9625879446233576), (-0.7540669154968491, -0.7678726873009407),
                                    (0.9345335196185689, -0.39997004285253945), (-0.5163174741656262, 0.5376409004137273),
                                    (0.6583087133274226, 0.5564162525991561), (0.7272350284428963, 0.18485499597619803),
                                    (0.30589404581350754, -0.290256375901367), (0.79881559364427, 0.9301793065831461),
                                    (0.8434517570450266, 0.2828723363229593), (0.0450388363149643, 0.6458992596871613)]
        #print(self.start_location_list)
        self.level_seeds = [42, 123, 465, 299, 928, 1, 22, 7, 81, 0, 1337, 2023, 9876, 5432, 8888, 1234, 7777, 3141,
                            2718, 9999, 1111, 6666, 4444, 8080, 3333, 7890, 1029, 5678, 9012, 2468, 1357, 8642, 9753,
                            1470, 2581, 3692, 7410, 8520, 9630, 1590, 7531, 4682, 9173, 2640, 5791, 8462, 3951, 6284,
                            7395, 1683, 4729, 5064, 8317, 9428, 2756, 6049, 3870, 7152, 4681]

        self.difficulty = self.config['starting_difficulty'] # Curriculum learning level (starts at 0)
        self.config['gameboard_size'] = self.config["gameboard_size_per_lesson"][str(self.difficulty)]

        self.max_steps = self.config['max_steps']#150*30#14703 # Max step count of the episode

        self.highval_target_ratio = 0 # The ratio of targets that are high value (more points for IDing, but also have chance of detecting the player). TODO make configurable in config

        self.tag = tag # Name for differentiating envs for training, eval, software testing etc.
        self.render_mode = render_mode

        self.generate_plot_list()

        self.check_valid_config()

        #print(f'Env initialized: Tag={tag}, obs_type={self.config['obs_type']}, action_type={self.config['action_type']}')

        self.num_agents = num_agents
        self.max_targets = 30

        ######################################### OBSERVATION AND ACTION SPACES ########################################
        if self.config['action_type'] == 'Discrete8':
            self.action_space = gym.spaces.Discrete(8)  # 8 directions

        elif self.config['action_type'] == 'Discrete16':
            self.action_space = gym.spaces.Discrete(16)  # 8 directions

        elif self.config['action_type'] == 'continuous-normalized':
            self.action_space = gym.spaces.Box(
                low=np.array([-1, -1], dtype=np.float32),
                high=np.array([1, 1], dtype=np.float32),
                dtype=np.float32)
        else: raise ValueError("Invalid action type")

        if self.config['obs_type'] == 'pixel':  # CNN observation space - grayscale 84x84
            self.observation_space = gym.spaces.Box(
                low=0, high=255,
                shape=(84, 84, 1),  # Height, Width, Channels (grayscale)
                dtype=np.uint8  # Change from np.float32 to np.uint8
            )
        elif self.config['obs_type'] == 'absolute':
            self.obs_size = 2 + 3 * self.max_targets
            self.observation_space = gym.spaces.Box(
                low=-1, high=1,
                shape=(self.obs_size,),
                dtype=np.float32)

        elif self.config['obs_type'] == 'nearest':
            self.obs_size = 2 * self.config['num_observed_targets'] + 2*self.config['num_observed_threats'] # x,y components of unit vector
            self.observation_space = gym.spaces.Box(
                low=-1, high=1,
                shape=(self.obs_size,),
                dtype=np.float32)
            if self.tag == 'train_mp0':
                print(f'Using obs space size {self.obs_size}')

        # elif self.config['obs_type'] == 'absolute-1target':
        #     self.obs_size = 4  # Changed from 2 + 3 * self.max_targets to 4
        #     self.observation_space = gym.spaces.Box(
        #         low=-1, high=1,
        #         shape=(self.obs_size,),
        #         dtype=np.float32)
        else: raise ValueError("Obs type not recognized")

        ############################################## TUNABLE PARAMETERS ##############################################
        # Set reward quantities for each event (agent only)
        
        #self.steps_for_lowqual_info = 3 * 60  # TODO currently not used
        #self.steps_for_highqual_info = 7 * 60  # TODO currently not used
        # self.detections_reward = 0 # -1.0 # TODO temporarily removed for simplified env
        #self.highqual_regulartarget_reward = self.config['highqual_regulartarget_reward'] # Reward earned for gathering high quality info about a regular value target
        #self.config['highqual_highvaltarget_reward'] = self.config['self.config['highqual_highvaltarget_reward']'] # Reward earned for gathering high quality info about a high value target

        ################################################# HUMAN THINGS #################################################
        self.subject_id = subject_id
        self.round_number = round_number
        self.user_group = user_group
        self.human_training = True if (self.round_number == 0 and self.user_group != 'test') else False  # True for the training round at start of experiment, false for rounds 1-4 (NOTE: This is NOT agent training!)

        self.paused = False
        self.unpause_countdown = False

        # Track score points (for human eyes only)
        self.score = 0
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
                self.time_window = TimeWindow(self.config["gameboard_size"] * 0.43, self.config["gameboard_size"]+5,current_time=self.display_time, time_limit=self.config['time_limit'])

        if self.config['obs_type'] == 'pixel': # Create offscreen surface for pixel observations
            pygame.init()
            self.pixel_surface = pygame.Surface((self.config["gameboard_size"], self.config["gameboard_size"]))

        self.episode_counter = 0
        self.reset()


    def reset(self, seed=None, options=None):

        # Load settings based on difficulty level
        self.config['gameboard_size'] = self.config["gameboard_size_per_lesson"][str(self.difficulty)]
        self.num_levels = self.config["levels_per_lesson"][str(self.difficulty)]
        #print(f'using {self.num_levels} levels for difficulty {self.difficulty}')

        if self.config['use_curriculum']:
            self.generate_plot_list()  # Generate list of episodes to plot using save_action_history_plot()

        # Set seed for this level
        seed_list = self.level_seeds[0:self.num_levels]
        if self.tag in ['eval','test_suite']:
            current_seed_index = self.episode_counter % len(seed_list)
        else:
            current_seed_index = (self.episode_counter+int(self.tag[-1])) % len(seed_list) # Shuffling seeds for each subprocess env to avoid overfitting
        current_seed = seed_list[current_seed_index]
        #print(f'Seed index = {current_seed_index}, seed = {current_seed}')
        np.random.seed(current_seed)
        random.seed(current_seed)

        self.agents = [] # List of names of all current agents. Typically integers
        self.possible_agents = [0, 1] # PettingZoo format. List of possible agents
        self.max_num_agents = 2
        self.aircraft_ids = []  # Indices of the aircraft agents

        self.score = 0
        self.display_time = 0  # Time that is used for the on-screen timer. Accounts for pausing.
        self.pause_start_time = 0
        self.total_pause_time = 0
        self.init = True

        # For plotting
        self.action_history = []
        self.agent_location_history = []
        self.direct_action_history = [] # Direct control only

        # Initialize potential for reward shaping
        self.potential = None

        ##################### Create vectorized ships/targets. Format: [info_level, x_pos, y_pos] ######################
        self.num_targets = min(self.max_targets, self.config['num_targets'])  # If more than 30 targets specified, overwrite to 30

        self.targets = np.zeros((self.num_targets, 5), dtype=np.float32)
        self.targets[:, 0] = np.arange(self.num_targets) # Assign IDs (column 0) (Note, this does not go into the observation vector. It is just for reference)
        self.targets[:, 1] = np.random.choice([0, 1], size=self.num_targets, p=[1 - self.highval_target_ratio, self.highval_target_ratio]) # Assign target values (column 1) - regular (0) or high-value (1)
        self.targets[:, 2] = 0 # Initialize info_level (column 2) to all 0 (unknown)

        map_half_size = self.config["gameboard_size"] / 2  # Convert to [-150, +150] coordinate system
        margin = map_half_size * 0.03  # 3% margin from edges
        self.targets[:, 3] = np.random.uniform(-map_half_size + margin, map_half_size - margin, size=self.num_targets)
        self.targets[:, 4] = np.random.uniform(-map_half_size + margin, map_half_size - margin, size=self.num_targets)
        #print(f'Target xs: {self.targets[:, 3]}')

        self.target_timers = np.zeros(self.num_targets, dtype=np.int32)  # How long each target has been sensed for
        self.detections = 0 # Number of times a target has detected us. Results in a score penalty
        self.targets_identified = 0

        # Create a single threat at random location
        #self.threat = np.zeros(3, dtype=np.float32)  # [x_pos, y_pos, radius]
        self.threat = np.zeros(2, dtype=np.float32)  # [x_pos, y_pos, radius]
        margin = map_half_size * 0.03  # 3% margin from edges
        self.threat[0] = np.random.uniform(-map_half_size + margin, map_half_size - margin)  # x position
        self.threat[1] = np.random.uniform(-map_half_size + margin, map_half_size - margin)  # y position
        #self.threat[2] = 50.0  # radius in pixels

        # Decay shaping rewards
        self.config['shaping_coeff_prox'] = self.config['shaping_coeff_prox'] * self.config['shaping_decay_rate']

        # Set agent start location
        map_half_size = self.config["gameboard_size"] / 2
        if self.config["agent_start_locations_per_lesson"][str(self.difficulty)] == 99:
            agent_x, agent_y = np.random.uniform(-1,1) * map_half_size, np.random.uniform(-1,1) * map_half_size
        else:
            self.start_locations = self.start_location_list[0:self.config["agent_start_locations_per_lesson"][str(self.difficulty)]]

            if self.tag in ['eval','test_suite']: start_loc_index = self.episode_counter % len(self.start_locations)
            else: start_loc_index = (self.episode_counter+int(self.tag[-1])) % len(self.start_locations)

            #map_half_size = self.config["gameboard_size"] / 2
            agent_x, agent_y = self.start_locations[start_loc_index][0] * map_half_size, self.start_locations[start_loc_index][1] * map_half_size
            #print(f'Agent spawned at {agent_x}, {agent_y}')


        ############################################# Create the aircraft ##############################################
        for i in range(self.num_agents):
            agents.Aircraft(self, 0, max_health=10,color=self.AIRCRAFT_COLORS[i],speed=self.config['game_speed']*self.config['agent_speed'])
            self.agents[self.aircraft_ids[i]].x, self.agents[self.aircraft_ids[i]].y = agent_x, agent_y

        if self.num_agents == 2: # TODO delete
            self.human_idx = self.aircraft_ids[1]  # Agent ID for the human-controlled aircraft. Dynamic so that if human dies in training round, their ID increments 1

        # Reset step, episode, and reward counters
        self.step_count_inner = 0
        self.step_count_outer = 0
        self.ep_reward = 0
        self.episode_counter += 1

        self.all_targets_identified = False
        self.terminated = False
        self.truncated = False

        self.observation = self.get_observation()

        info = {}
        return self.observation, info


    def step(self, actions:dict):
        """ Skip frames by repeating the action multiple times """

        total_reward, info, total_potential_gain = 0, None, 0

        for frame in range(self.config['frame_skip']):
            observation, reward, self.terminated, self.truncated, info = self._single_step(actions)
            total_reward += reward
            total_potential_gain += info["potential_gain"]

            # Break early if the episode is done to avoid unnecessary computation
            if self.terminated or self.truncated:
                break

        #print(f'Rew|shaping_rew = {round(total_reward,1)} | {total_potential_gain*self.config['shaping_coeff_prox']}')
        self.step_count_outer += 1
        info["outerstep_potential_gain"] = total_potential_gain

        return observation, total_reward, self.terminated, self.truncated, info


    def _single_step(self, actions:dict):
        """
        args:
            actions: (Option 1) Dictionary of {agent_id: action(ndarray)}.
                     (Option 2) A single ndarray
        """

        self.step_count_inner += 1
        if self.potential:
            last_potential = self.potential
        else:
            last_potential = 0

        new_reward = {'high val target id': 0, 'regular val target id': 0, 'early finish': 0} # Track events that give reward. Will be passed to get_reward at end of step
        new_score = 0 # For tracking human-understandable reward
        info = {
            "new_identifications": [],  # List to track newly identified targets/threats
            "reward_components": {},
            "detections": self.detections,  # Current detection count
            "target_ids": 0,
            'episode': {'r': 0, 'l': self.step_count_inner},
            "score_breakdown": {"target_points": 0, "threat_points": 0, "time_points": 0, "completion_points": 0, "penalty_points": 0}}


        ############################################### Process actions ################################################
        if self.config['action_type'] in ['Discrete8', 'Discrete16'] and isinstance(actions, (np.int64, np.float32, int)):
            waypoint = self.process_action(actions)
            self.agents[0].waypoint_override = waypoint

        elif self.config['action_type'] == 'continuous-normalized':
            waypoint = self.process_action(actions)
            self.agents[0].waypoint_override = waypoint

        elif isinstance(actions, dict): # Action is passed in as a dict {agent_id: action}
            for agent_id, action_value in actions.items():
                waypoint = self.process_action(action_value)
                self.agents[agent_id].waypoint_override = waypoint

        elif isinstance(actions, np.ndarray): # Single agent, action passed in directly as an array instead of list(arrays)
            waypoint = self.process_action(actions)
            self.agents[0].waypoint_override = (float(waypoint[0]), float(waypoint[1]))

        else:
            print(f'Action type: {type(actions)}')
            raise ValueError('Actions input is an unknown type')

        # Log actions to action_history plot
        self.action_history.append(self.agents[0].waypoint_override)
        self.agent_location_history.append((self.agents[self.aircraft_ids[0]].x, self.agents[self.aircraft_ids[0]].y))

        ################################ Move the agents and check for gameplay updates ################################
        for aircraft in [agent for agent in self.agents if agent.agent_class == "aircraft" and agent.alive]:
            aircraft.move() # First, move using the waypoint override set above

            # # Calculate distances to all targets
            aircraft_pos = np.array([aircraft.x, aircraft.y])  # Get aircraft position
            target_positions = self.targets[:, 3:5]  # x,y coordinates
            distances = np.sqrt(np.sum((target_positions - aircraft_pos) ** 2, axis=1))

            # Find targets within ISR range (for identification)
            in_isr_range = distances <= self.AIRCRAFT_ENGAGEMENT_RADIUS

            # Process newly identified targets
            for target_idx in range(self.num_targets):
                if in_isr_range[target_idx] and self.targets[target_idx, 2] < 1.0: # If target is in range and not fully identified
                    self.targets_identified += 1
                    self.targets[target_idx, 2] = 1.0

                    # Add reward (for agent) and score (for human).
                    if self.targets[target_idx, 1] == 0.0:
                        new_score += self.config['highqual_regulartarget_reward']
                        new_reward['regular val target id'] += 1
                    else:
                        new_score += self.config['highqual_highvaltarget_reward']
                        new_reward['high val target id'] += 1

                    # Update info dictionary
                    info["score_breakdown"]["target_points"] += self.config['highqual_regulartarget_reward'] if self.targets[target_idx, 1] == 0.0 else self.config['highqual_highvaltarget_reward']
                    info["new_identifications"].append({
                        "type": "low quality info gathered",
                        "target_id": int(self.targets[target_idx, 0]),
                        "aircraft": aircraft.agent_idx,
                        "time": self.display_time
                    })

                # # Handle aircraft being detected by high value targets
                # if self.config['prob_detect'] > 0.0: # If prob detect is zero, skip
                #     if in_isr_range[target_idx] and self.targets[target_idx, 1] == 1.0: # Only if we're in range of high value targets
                #         if np.random.random() < self.config['prob_detect']: # Roll RNG to see if we're detected
                #             self.detections += 1
                #             new_reward['detections'] += 1
                #             info["detections"] = self.detections

        self.all_targets_identified = np.all(self.targets[:, 2] == 1.0)

        if self.all_targets_identified:
            self.terminated = True
            new_score += (self.config['time_limit'] - self.display_time / 1000) * self.time_points
            new_reward['early finish'] = self.max_steps - self.step_count_inner # Number of steps finished early (will be multiplied by reward coeff in get_reward

        if self.step_count_inner >= self.max_steps: # TODO: Temporarily hard-coding 490 steps
            self.terminated = True

        # Advance time (only relevant for human play)
        if self.render_mode == 'headless': self.display_time = self.display_time + (1000/60) # If agent training, each step is 1/60th of a second
        elif not self.paused: self.display_time = pygame.time.get_ticks() - self.total_pause_time
        if self.init: self.init = False

        self.observation = self.get_observation()  # Get observation

        # Calculate potential (distance improvement to target)
        self.potential = self.get_potential(self.observation)
        potential_gain = max(-0.1, min(0.1, self.potential - last_potential))  # Cap between -10 and +10
        #print(round(potential_gain,2))

        # Calculate reward
        reward = self.get_reward(new_reward, potential_gain)  # For agent
        self.ep_reward += reward
        self.score += new_score  # For human

        # Populate info dict
        info['episode'] = {'r': self.ep_reward, 'l': self.step_count_inner, }
        info['reward_components'] = new_reward
        info['detections'] = self.detections
        info["target_ids"] = self.targets_identified
        info["potential_gain"] = potential_gain

        if self.terminated or self.truncated:
            print(f'ROUND {self.episode_counter} COMPLETE ({self.targets_identified} IDs), reward {round(info['episode']['r'], 1)}, {self.step_count_outer}({info['episode']['l']}) steps, | {self.detections} detections | {round(self.max_steps/self.config['frame_skip'] - self.step_count_outer,0)} outer steps early')
            if self.tag in ['eval', 'train_mp0', 'bc'] and self.episode_counter in self.episodes_to_plot:
                self.save_action_history_plot()
            if self.render_mode == 'human':
                pygame.time.wait(50)

        return self.observation, reward, self.terminated, self.truncated, info


    def get_reward(self, new_reward, potential_gain):

        # Check if agent is inside threat radius and apply penalty # TODO make this per agent
        # apply_threat_penalty = False
        # if hasattr(self, 'threat'):
        #     for aircraft in [agent for agent in self.agents if agent.agent_class == "aircraft" and agent.alive]:
        #         aircraft_pos = np.array([aircraft.x, aircraft.y])
        #         threat_pos = np.array([self.threat[0], self.threat[1]])
        #         distance_to_threat = np.sqrt(np.sum((threat_pos - aircraft_pos) ** 2))
        #
        #         if distance_to_threat <= self.config['threat_radius']:
        #             apply_threat_penalty = True

        # Calculate gradual threat penalty based on distance
        threat_penalty = 0
        if hasattr(self, 'threat'):
            for aircraft in [agent for agent in self.agents if agent.agent_class == "aircraft" and agent.alive]:
                aircraft_pos = np.array([aircraft.x, aircraft.y])
                threat_pos = np.array([self.threat[0], self.threat[1]])
                distance_to_threat = np.sqrt(np.sum((threat_pos - aircraft_pos) ** 2))

                threat_radius = self.config['threat_radius']
                warning_radius = threat_radius * 2  # 50% larger than threat radius

                if distance_to_threat <= threat_radius:
                    # Maximum penalty when at center, decreasing linearly to zero at radius edge
                    normalized_distance = distance_to_threat / threat_radius  # 0 at center, 1 at edge
                    penalty_multiplier = 1.0 - normalized_distance  # 1 at center, 0 at edge
                    threat_penalty += self.config['inside_threat_penalty'] * penalty_multiplier

                elif distance_to_threat <= warning_radius:
                    # Warning zone - penalty decreases from 50% to 0% as distance increases
                    normalized_distance = (distance_to_threat - threat_radius) / (warning_radius - threat_radius)
                    penalty_multiplier = 0.5 * (1.0 - normalized_distance)  # 0.5 at threat edge, 0 at warning edge
                    threat_penalty += self.config['inside_threat_penalty'] * penalty_multiplier

        reward = (new_reward['high val target id'] * self.config['highqual_highvaltarget_reward']) + \
                 (new_reward['regular val target id'] * self.config['highqual_regulartarget_reward']) + \
                 (new_reward['early finish'] * self.config['shaping_coeff_earlyfinish']) + \
                 (potential_gain * self.config['shaping_coeff_prox'] * (300/self.config['gameboard_size'])) + \
                 (self.config['shaping_time_penalty']) - \
                 threat_penalty

        # if apply_threat_penalty:
        #     reward -= self.config['inside_threat_penalty']
        #     #print(f'Penalty for being inside threat range')

        return reward

    def get_potential(self, observation):
        """
        Calculate potential as negative distance to nearest unknown target.
        Returns a higher (less negative) value when closer to unknown targets.
        """

        # Get agent position from observation (first 2 elements, normalized)
        map_half_size = self.config["gameboard_size"] / 2

        #agent_x = (self.agents[self.aircraft_ids[0]].x) / map_half_size #observation[0] * map_half_size
        #agent_y = (self.agents[self.aircraft_ids[0]].y) / map_half_size #observation[1] * map_half_size
        #agent_pos = np.array([agent_x, agent_y])
        agent_x = self.agents[self.aircraft_ids[0]].x
        agent_y = self.agents[self.aircraft_ids[0]].y
        agent_pos = np.array([agent_x, agent_y])

        # Get target positions and info levels
        target_positions = self.targets[:, 3:5]  # x,y coordinates
        target_info_levels = self.targets[:, 2]  # info levels

        # Create mask for unidentified targets (info_level < 1.0)
        unidentified_mask = target_info_levels < 1.0

        if not np.any(unidentified_mask): # No unidentified targets remaining
            return 0.0

        # Calculate distances to unidentified targets only
        unidentified_positions = target_positions[unidentified_mask]
        distances = np.sqrt(np.sum((unidentified_positions - agent_pos) ** 2, axis=1))
        nearest_distance = np.min(distances)

        # Progressive multiplier - higher when fewer targets remain
        targets_remaining = np.sum(unidentified_mask)
        total_targets = len(target_info_levels)
        progress_multiplier = 1.0 + (total_targets - targets_remaining) * 0.3

        return -nearest_distance #* progress_multiplier

    def get_observation(self):
        """Main function to return the observation vector. Calls specific observation functions depending on obs type. """
        if self.config['obs_type'] == 'absolute':
            self.observation = self.get_observation_alltargets()

        elif self.config['obs_type'] == 'nearest':
            self.observation = self.get_observation_nearest_n()

        elif self.config['obs_type'] == 'absolute-1target':
            self.observation = self.get_observation_1target()

        elif self.config['obs_type'] == 'pixel':
            self.observation = self.get_pixel_observation()

        return self.observation

    def get_observation_nearest(self):
        """
        State will include the following features:
            0 unit_vector_x,           # (-1 to +1) x component of unit vector to nearest unknown target
            1 unit_vector_y,           # (-1 to +1) y component of unit vector to nearest unknown target
        """

        self.observation = np.zeros(2, dtype=np.float32)

        agent_pos = np.array([self.agents[self.aircraft_ids[0]].x, self.agents[self.aircraft_ids[0]].y])

        # Get target positions and info levels
        target_positions = self.targets[:self.num_targets, 3:5]  # x,y coordinates
        target_info_levels = self.targets[:self.num_targets, 2]  # info levels

        unknown_mask = target_info_levels < 1.0 # Create mask for unknown targets (info_level < 1.0)

        if np.any(unknown_mask):
            unknown_positions = target_positions[unknown_mask]
            distances = np.sqrt(np.sum((unknown_positions - agent_pos) ** 2, axis=1))
            nearest_idx = np.argmin(distances)

            nearest_target_pos = unknown_positions[nearest_idx]
            vector_to_target = nearest_target_pos - agent_pos

            distance = np.linalg.norm(vector_to_target)

            if distance > 0:
                unit_vector = vector_to_target / distance
                self.observation[0] = unit_vector[0]
                self.observation[1] = unit_vector[1]
            else: # Agent is exactly at target position

                self.observation[0] = 0.0
                self.observation[1] = 0.0
        else: # No unknown targets remaining, return zero vector

            self.observation[0] = 0.0
            self.observation[1] = 0.0

        #print(self.observation)
        return self.observation

    def get_observation_nearest_n(self):
        """
        State will include the following features:
            For each of the N nearest unknown targets:
                unit_vector_x,           # (-1 to +1) x component of unit vector to target
                unit_vector_y,           # (-1 to +1) y component of unit vector to target
        """

        # Get N from config
        N = self.config['num_observed_targets']
        M = self.config['num_observed_threats']

        # Initialize observation array (2 * N for x,y components of N targets)
        self.observation = np.zeros(2 * (N+M), dtype=np.float32)

        agent_pos = np.array([self.agents[self.aircraft_ids[0]].x, self.agents[self.aircraft_ids[0]].y])

        # Get target positions and info levels
        target_positions = self.targets[:self.num_targets, 3:5]  # x,y coordinates
        target_info_levels = self.targets[:self.num_targets, 2]  # info levels

        unknown_mask = target_info_levels < 1.0  # Create mask for unknown targets (info_level < 1.0)

        if np.any(unknown_mask):
            unknown_positions = target_positions[unknown_mask]
            distances = np.sqrt(np.sum((unknown_positions - agent_pos) ** 2, axis=1))

            # Get indices of N nearest targets (or all if fewer than N)
            num_targets_to_use = min(N, len(distances))
            nearest_indices = np.argsort(distances)[:num_targets_to_use]

            # Fill observation with unit vectors to nearest N targets
            for i in range(num_targets_to_use):
                target_idx = nearest_indices[i]
                target_pos = unknown_positions[target_idx]
                vector_to_target = target_pos - agent_pos

                distance = np.linalg.norm(vector_to_target)

                if distance > 0:
                    unit_vector = vector_to_target
                    self.observation[i * 2] = unit_vector[0]  # x component
                    self.observation[i * 2 + 1] = unit_vector[1]  # y component
                else:
                    # Agent is exactly at target position
                    self.observation[i * 2] = 0.0
                    self.observation[i * 2 + 1] = 0.0

            # dx, dy vector to threat as last two elements of the observation
            threat_pos = np.array([self.threat[0], self.threat[1]])
            vector_to_threat = threat_pos - agent_pos
            self.observation[-2] = vector_to_threat[0]  # x component
            self.observation[-1] = vector_to_threat[1]  # y component

        # If no unknown targets remaining, observation stays all zeros
        return self.observation

    def get_observation_1target(self):
        """
        State will include the following features:
            Absolute mode:
                0 agent_x,                 # (-1 to +1) normalized position
                1 agent_y,                 # (-1 to +1) normalized position
                2 nearest_target_x,        # (-1 to +1) normalized position
                3 nearest_target_y,        # (-1 to +1) normalized position
        """

        # New observation size: agent (x,y) + nearest target (x,y) = 4 elements
        self.observation = np.zeros(4, dtype=np.float32)

        map_half_size = self.config["gameboard_size"] / 2

        # Agent position (normalized)
        self.observation[0] = (self.agents[self.aircraft_ids[0]].x) / map_half_size
        self.observation[1] = (self.agents[self.aircraft_ids[0]].y) / map_half_size

        # Find nearest target
        agent_pos = np.array([self.agents[self.aircraft_ids[0]].x, self.agents[self.aircraft_ids[0]].y])
        target_positions = self.targets[:self.num_targets, 3:5]  # x,y coordinates of existing targets

        if self.num_targets > 0:
            # Calculate distances to all targets
            distances = np.sqrt(np.sum((target_positions - agent_pos) ** 2, axis=1))
            nearest_idx = np.argmin(distances)

            # Nearest target position (normalized)
            self.observation[2] = self.targets[nearest_idx, 3] / map_half_size
            self.observation[3] = self.targets[nearest_idx, 4] / map_half_size
        else:
            # No targets exist, set to agent position
            self.observation[2] = self.observation[0]
            self.observation[3] = self.observation[1]

        return self.observation

    def get_observation_alltargets(self):
        """
        State will include the following features:
            Absolute mode:
                0 agent_x,                 # (-1 to +1) normalized position
                1 agent_y,                 # (-1 to +1) normalized position

                for target i:
                2+i*3 target_info_level    # 0 if unknown, 1 if known
                3+i*3 target_x,            # (-1 to +1) normalized position
                4+i*3 target_y,            # (-1 to +1) normalized position

        """

        self.observation = np.zeros(self.obs_size, dtype=np.float32)

        map_half_size = self.config["gameboard_size"] / 2
        self.observation[0] = (self.agents[self.aircraft_ids[0]].x) / map_half_size
        self.observation[1] = (self.agents[self.aircraft_ids[0]].y) / map_half_size

        # Process target data
        targets_per_entry = 3  # Each target has 3 features in the observation
        target_features = np.zeros((self.max_targets, targets_per_entry), dtype=np.float32)

        target_features[:self.num_targets, 0] = self.targets[:, 2]  # info levels
        target_features[:self.num_targets, 1] = (self.targets[:, 3]) / map_half_size
        target_features[:self.num_targets, 2] = (self.targets[:, 4]) / map_half_size

        target_start_idx = 2
        self.observation[target_start_idx:target_start_idx + self.max_targets * targets_per_entry] = target_features.flatten()

        return self.observation

    def get_pixel_observation(self):
        """Render the game state to an 84x84 grayscale pixel array for CNN input with clear target differentiation"""

        # Use existing render surface if in human mode, otherwise create temporary surface
        if self.render_mode == 'human' and hasattr(self, 'window'):
            # Capture the main game area from the existing window
            game_rect = pygame.Rect(0, 0, self.config["gameboard_size"], self.config["gameboard_size"])
            pixel_array = pygame.surfarray.array3d(self.window.subsurface(game_rect))
        else:
            # Create offscreen surface and render to it with enhanced target visibility
            self.pixel_surface.fill((255, 255, 255))  # White background
            self._render_game_to_surface_enhanced(self.pixel_surface)
            pixel_array = pygame.surfarray.array3d(self.pixel_surface)

        # Pygame arrays are (width, height, channels), we need (height, width, channels)
        pixel_array = np.transpose(pixel_array, (1, 0, 2))

        # Convert to grayscale using weighted average for better contrast
        if len(pixel_array.shape) == 3:
            # Use standard RGB to grayscale conversion
            pixel_array = np.dot(pixel_array[..., :3], [0.2989, 0.5870, 0.1140])

        # Resize to 84x84 using OpenCV for consistency
        pixel_array_resized = cv2.resize(pixel_array, (84, 84), interpolation=cv2.INTER_AREA)

        # Add channel dimension for grayscale
        pixel_array_resized = np.expand_dims(pixel_array_resized, axis=-1)

        return pixel_array_resized.astype(np.uint8)

    def _render_game_to_surface_enhanced(self, surface):
        """
        Render game elements to a pygame surface with enhanced target visibility for pixel observations.
        Uses distinct grayscale values and shapes to ensure unknown vs known targets are clearly differentiable.
        """

        # Draw outer box (same as human render)
        self.__render_box_to_surface__(surface, 1, (0, 0, 0), 3)  # outer box

        # Draw aircraft (same as human render)
        for agent in self.agents:
            agent.draw(surface)

        # Enhanced target rendering with clear grayscale differentiation
        map_half_size = self.config["gameboard_size"] / 2

        for target in self.targets:
            screen_x = target[3] + map_half_size
            screen_y = target[4] + map_half_size

            # Base size for targets
            base_size = 10 if target[1] == 1 else 7  # High-value vs regular targets

            if target[2] == 1.0:  # Known/identified targets
                # Use very dark gray (almost black) for known targets - will be ~51 in grayscale
                target_color = (11, 11, 11)
                # Draw filled circle for known targets
                pygame.draw.circle(surface, target_color, (int(screen_x), int(screen_y)), base_size)

                # Add a white center dot to make them even more distinct
                pygame.draw.circle(surface, (255, 255, 255), (int(screen_x), int(screen_y)), max(2, base_size // 3))

            else:  # Unknown targets (info_level < 1.0)
                # Use light gray for unknown targets - will be ~204 in grayscale
                target_color = (204, 204, 204)
                # Draw filled circle for unknown targets
                pygame.draw.circle(surface, target_color, (int(screen_x), int(screen_y)), base_size)

                # Add black border to make them stand out against white background
                pygame.draw.circle(surface, (0, 0, 0), (int(screen_x), int(screen_y)), base_size, 2)

        # Draw the threat for pixel observations
        if hasattr(self, 'threat'):
            threat_screen_x = int(self.threat[0] + map_half_size)
            threat_screen_y = int(self.threat[1] + map_half_size)
            threat_radius = self.config['threat_radius']

            # Draw circle (lighter for pixel obs)
            pygame.draw.circle(surface, (200, 200, 0), (threat_screen_x, threat_screen_y), threat_radius, 2)

            # Draw upside-down triangle
            triangle_size = 12
            triangle_points = [
                (threat_screen_x, threat_screen_y + triangle_size),
                (threat_screen_x - triangle_size, threat_screen_y - triangle_size),
                (threat_screen_x + triangle_size, threat_screen_y - triangle_size)
            ]
            pygame.draw.polygon(surface, (200, 200, 0), triangle_points)

    def __render_box_to_surface__(self, surface, distance_from_edge, color=(0, 0, 0), width=2):
        """Utility function for drawing a square box to a specific surface"""
        pygame.draw.line(surface, color, (distance_from_edge, distance_from_edge),
                         (distance_from_edge, self.config["gameboard_size"] - distance_from_edge), width)
        pygame.draw.line(surface, color, (distance_from_edge, self.config["gameboard_size"] - distance_from_edge),
                         (self.config["gameboard_size"] - distance_from_edge,
                          self.config["gameboard_size"] - distance_from_edge), width)
        pygame.draw.line(surface, color, (
        self.config["gameboard_size"] - distance_from_edge, self.config["gameboard_size"] - distance_from_edge),
                         (self.config["gameboard_size"] - distance_from_edge, distance_from_edge), width)
        pygame.draw.line(surface, color, (self.config["gameboard_size"] - distance_from_edge, distance_from_edge),
                         (distance_from_edge, distance_from_edge), width)


    def render(self):

        window_width, window_height = self.config['window_size'][0], self.config['window_size'][0]
        game_width = self.config["gameboard_size"]
        ui_width = window_width - game_width

        if self.render_mode == 'human' and self.use_buttons:
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

            map_half_size = self.config["gameboard_size"] / 2
            screen_x = target[3] + map_half_size
            screen_y = target[4] + map_half_size
            pygame.draw.circle(self.window, target_color, (float(screen_x), float(screen_y)), target_width)

        # Draw the threat (gold upside-down triangle with circle)
        if hasattr(self, 'threat'):
            map_half_size = self.config["gameboard_size"] / 2
            threat_screen_x = int(self.threat[0] + map_half_size)
            threat_screen_y = int(self.threat[1] + map_half_size)
            threat_radius = self.config['threat_radius']

            # Draw the circle around the threat
            pygame.draw.circle(self.window, (255, 215, 0), (threat_screen_x, threat_screen_y), threat_radius,
                               3)  # Gold circle outline

            # Draw upside-down triangle (pointing down)
            triangle_size = 15
            triangle_points = [
                (threat_screen_x, threat_screen_y + triangle_size),  # Bottom point
                (threat_screen_x - triangle_size, threat_screen_y - triangle_size),  # Top left
                (threat_screen_x + triangle_size, threat_screen_y - triangle_size)  # Top right
            ]
            pygame.draw.polygon(self.window, (255, 215, 0), triangle_points)  # Gold triangle

        # Draw green lines and black crossbars
        self.__render_box__(35, (0, 128, 0), 2)  # inner box
        pygame.draw.line(self.window, (0, 0, 0), (self.config["gameboard_size"] // 2, 0),(self.config["gameboard_size"] // 2, self.config["gameboard_size"]), 2)
        pygame.draw.line(self.window, (0, 0, 0), (0, self.config["gameboard_size"] // 2),(self.config["gameboard_size"], self.config["gameboard_size"] // 2), 2)

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

        if self.render_mode == 'human':

            corner_round_text = f'STEP {self.step_count_outer}'#f"ROUND {self.round_number + 1}/4" if self.user_group == 'test' else f"ROUND {self.round_number}/4"
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
        if self.render_mode == 'human': self.clock.tick(self.config['tick_rate'])

    def close(self):
        if self.render_mode == 'human' and pygame.get_init():
            pygame.quit()

    # utility function for drawing a square box
    def __render_box__(self, distance_from_edge, color=(0, 0, 0), width=2, surface=None):
        """Utility function for drawing a square box"""
        surface = surface if surface is not None else self.window
        pygame.draw.line(surface, color, (distance_from_edge, distance_from_edge), (distance_from_edge, self.config["gameboard_size"] - distance_from_edge), width)
        pygame.draw.line(surface, color, (distance_from_edge, self.config["gameboard_size"] - distance_from_edge), (self.config["gameboard_size"] - distance_from_edge, self.config["gameboard_size"] - distance_from_edge), width)
        pygame.draw.line(surface, color, (self.config["gameboard_size"] - distance_from_edge, self.config["gameboard_size"] - distance_from_edge), (self.config["gameboard_size"] - distance_from_edge, distance_from_edge), width)
        pygame.draw.line(surface, color, (self.config["gameboard_size"] - distance_from_edge, distance_from_edge), (distance_from_edge, distance_from_edge), width)

    def check_valid_config(self):
        valid_obs_types = ['absolute', 'pixel', 'absolute-1target', 'nearest']
        valid_action_types = ['Discrete8', 'Discrete16', 'continuous-normalized']  # 'continuous_normalized
        valid_render_modes = ['headless', 'human', 'rgb_array']

        if self.config['obs_type'] not in valid_obs_types:
            raise ValueError(f"obs_type invalid, got '{self.config['obs_type']}'")
        if self.config['action_type'] not in valid_action_types:
            raise ValueError(f"action_type invalid, got '{self.config['action_type']}'")
        if self.render_mode not in valid_render_modes:
            raise ValueError('Render mode must be headless, rgb_array, human')

    def pause(self, unpause_key):
        print('Game paused')
        self.pause_start_time = pygame.time.get_ticks()
        self.button_latch_dict['pause'] = True
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
            if self.detections >= 3:
                title_surface = title_font.render('GAME OVER (>5 DETECTIONS)', True, (0, 0, 0))
        elif self.display_time/1000 >= self.config['time_limit']:
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
        If the action type is Discrete8, this converts the discrete action chosen into an x,y in the appropriate direction

        Args:
            action (ndarray, size 1): Agent discrete action to convert to waypoint coords

        Returns:
            waypoint (tuple, size 2): (x,y) waypoint with range [0, gameboard_size]
        """

        if isinstance(action, np.ndarray):
            if action.ndim > 1:
                action = action.flatten()
            if action.ndim == 0 or (action.ndim == 1 and action.size == 1):
                action = action.item()  # Use .item() instead of indexing

        if self.config['action_type'] == 'Discrete8':
            # Define 8 directions: 0=up, 1=up-right, 2=right, 3=down-right, 4=down, 5=down-left, 6=left, 7=up-left
            direction_map = {
                0: (0, 1),  # up
                1: (1, 1),  # up-right
                2: (1, 0),  # right
                3: (1, -1),  # down-right
                4: (0, -1),  # down
                5: (-1, -1),  # down-left
                6: (-1, 0),  # left
                7: (-1, 1)  # up-left
            }

            # Get current agent position
            current_x = self.agents[self.aircraft_ids[0]].x
            current_y = self.agents[self.aircraft_ids[0]].y
            if isinstance(action, np.ndarray):
                try: action = int(action)
                except: action = int(action[0])

            dx, dy = direction_map[action]
            length = math.sqrt(dx * dx + dy * dy)  # Normalize diagonal directions
            dx_norm = dx / length
            dy_norm = dy / length

            # Calculate waypoint 50 pixels away in chosen direction
            waypoint_distance = 50
            x_coord = current_x + (dx_norm * waypoint_distance)
            y_coord = current_y + (dy_norm * waypoint_distance)

            # Clip to map boundaries
            map_half_size = self.config["gameboard_size"] / 2
            x_coord = np.clip(x_coord, -map_half_size, map_half_size)
            y_coord = np.clip(y_coord, -map_half_size, map_half_size)

        if self.config['action_type'] == 'Discrete16':
            direction_map = {
                0: (0, 1),  # North (0°)
                1: (0.383, 0.924),  # NNE (22.5°)
                2: (0.707, 0.707),  # NE (45°)
                3: (0.924, 0.383),  # ENE (67.5°)
                4: (1, 0),  # East (90°)
                5: (0.924, -0.383),  # ESE (112.5°)
                6: (0.707, -0.707),  # SE (135°)
                7: (0.383, -0.924),  # SSE (157.5°)
                8: (0, -1),  # South (180°)
                9: (-0.383, -0.924),  # SSW (202.5°)
                10: (-0.707, -0.707),  # SW (225°)
                11: (-0.924, -0.383),  # WSW (247.5°)
                12: (-1, 0),  # West (270°)
                13: (-0.924, 0.383),  # WNW (292.5°)
                14: (-0.707, 0.707),  # NW (315°)
                15: (-0.383, 0.924)  # NNW (337.5°)
            }

            current_x = self.agents[self.aircraft_ids[0]].x
            current_y = self.agents[self.aircraft_ids[0]].y

            # Get normalized direction vector
            if isinstance(action, np.ndarray):
                try: action = int(action)
                except: action = int(action[0])

            # Handle actions outside the 16-direction range
            #action = action % 16

            dx_norm, dy_norm = direction_map[action]

            # Calculate waypoint at fixed distance in chosen direction
            waypoint_distance = 50
            x_coord = current_x + (dx_norm * waypoint_distance)
            y_coord = current_y + (dy_norm * waypoint_distance)

            # Clip to map boundaries
            map_half_size = self.config["gameboard_size"] / 2
            x_coord = np.clip(x_coord, -map_half_size, map_half_size)
            y_coord = np.clip(y_coord, -map_half_size, map_half_size)

        elif self.config['action_type'] == 'continuous-normalized':
            if action[0] > 1.1 or action[1] > 1.1 or action[0] < -1.1 or action[1] < -1.1:
                raise ValueError('ERROR: Actions are not normalized to -1, +1')

            map_half_size = self.config["gameboard_size"] / 2
            x_coord = action[0] * map_half_size
            y_coord = action[1] * map_half_size

        else: raise ValueError(f'Error in process_action: action type "{self.config['action_type']}" not recognized')

        waypoint = (float(x_coord), float(y_coord))

        return waypoint

    def save_action_history_plot(self, note=''):
        """ Save plot of the agent's trajectory, actions, and targets for the entire episode. """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import datetime
            import os

            # Create directory if it doesn't exist
            #os.makedirs(f'logs/action_histories/{self.run_name}', exist_ok=True)
            full_dir_path = f'logs/action_histories/{self.run_name}'
            os.makedirs(full_dir_path, exist_ok=True)

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

                if i == 0: # Color target 0 differently for debugging
                    color = 'purple' if self.targets[i, 2] == 1.0 else 'chocolate'
                else:
                    color = 'red' if self.targets[i, 2] == 1.0 else 'orange'

                plt.scatter(target_x, target_y, s=marker_size, color=color, alpha=0.7, marker='o')

            # Plot the threat if it exists
            if hasattr(self, 'threat'):
                threat_x = self.threat[0]
                threat_y = self.threat[1]
                threat_radius = self.config['threat_radius'] * (1000 / self.config["gameboard_size"])  # Scale for plot

                # Draw threat circle
                circle = plt.Circle((threat_x, threat_y), threat_radius, fill=False, color='gold', linewidth=2,
                                    alpha=0.7)
                plt.gca().add_patch(circle)

                # Draw upside-down triangle marker
                plt.scatter(threat_x, threat_y, s=200, color='gold', marker='v', alpha=0.8, label='Threat',
                            edgecolors='black')

            # Plot agent trajectory (actual location history) as a line with points
            if agent_x_coords and agent_y_coords:
                plt.plot(agent_x_coords, agent_y_coords, 'g-', alpha=0.7, linewidth=2)
                plt.scatter(agent_x_coords, agent_y_coords, s=20, c=range(len(agent_x_coords)),
                            cmap='Greens', alpha=0.7, marker='o', label='')

            # Only plot waypoint history for waypoint-based action types
            if self.config['action_type'] != 'direct-control':
                # Extract x and y coordinates from action history (waypoints, already in centered coordinates)
                x_coords = [action[0] for action in self.action_history]
                y_coords = [action[1] for action in self.action_history]

                # Plot waypoint history (action history) as a line with points
                if x_coords and y_coords:
                    plt.plot(x_coords, y_coords, 'b-', alpha=0.15, linewidth=1)

                    # Plot only every fourth waypoint
                    x_coords_subset = x_coords[::self.config['frame_skip']]  # Plot one action per outer step instead of every action
                    y_coords_subset = y_coords[::self.config['frame_skip']]
                    subset_indices = list(range(0, len(x_coords), self.config['frame_skip']))  # Corresponding indices for colormap

                    plt.scatter(x_coords_subset, y_coords_subset, s=15, c=subset_indices,
                                cmap='cool', alpha=0.7, marker='x', label='Agent Waypoints')

                    # plt.scatter(x_coords, y_coords, s=15, c=range(len(x_coords)),
                    #             cmap='cool', alpha=0.7, marker='x', label='Agent Waypoints')

                    # Add starting and ending points with different markers
                    plt.scatter(x_coords[0], y_coords[0], s=120, color='blue', marker='*', label='Start Waypoint')
                    plt.scatter(x_coords[-1], y_coords[-1], s=120, color='cyan', marker='*', label='End Waypoint')

            if agent_x_coords and agent_y_coords:
                plt.scatter(agent_x_coords[0], agent_y_coords[0], s=120, color='lime', marker='*',
                            label='Start Position')
                plt.scatter(agent_x_coords[-1], agent_y_coords[-1], s=120, color='darkgreen', marker='*',
                            label='End Position')

            # Add a colorbar to show time progression
            cbar = plt.colorbar(aspect=70)
            cbar.set_label('Episode Progress')

            # Add grid lines centered at origin
            plt.grid(True, alpha=0.3)
            plt.gca().set_xticks(range(int(-map_half_size), int(map_half_size) + 1, 100))
            plt.gca().set_yticks(range(int(-map_half_size), int(map_half_size) + 1, 100))

            # Add labels and title
            #plt.xlabel('X Coordinate')
            #plt.ylabel('Y Coordinate')
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            action_type_label = 'direct-control' if self.config['action_type'] == 'direct-control' else self.config['action_type']
            plot_title = f'{self.tag} - Episode {self.episode_counter} (Reward: {self.ep_reward:.2f}, {self.targets_identified} targets, steps: {self.step_count_outer})'
            plt.title(plot_title)

            # Add a legend
            #plt.legend(loc='upper right', fontsize='x-small')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize='x-small')

            # Add centered quadrant lines (origin at center)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1.5)  # Horizontal line at y=0
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1.5)  # Vertical line at x=0

            # Optional: Add boundary lines to show map edges
            #plt.axhline(y=map_half_size, color='red', linestyle='--', alpha=0.3, label='Map Boundary')
            #plt.axhline(y=-map_half_size, color='red', linestyle='--', alpha=0.3)
            #plt.axvline(x=map_half_size, color='red', linestyle='--', alpha=0.3)
            #plt.axvline(x=-map_half_size, color='red', linestyle='--', alpha=0.3)
            
            # Save the figure with a timestamp
            filename = f'logs/action_histories/{self.run_name}/{note}{self.tag}_ep{self.episode_counter}.png'
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            plt.close()

            print(f"Action history plot saved to {filename}")
        except ImportError as e:
            print(f"Could not save action history plot: {e}")
        except Exception as e:
            print(f"Error saving action history plot: {e}")


    def set_difficulty(self, difficulty):
        """Method to change difficulty level from an external method (i.e. a training loop)"""
        self.difficulty = difficulty
        print(f'env.set_difficulty: Difficulty is now {self.difficulty}')


    def generate_plot_list(self):

        # TODO Temp workaround
        self.num_levels = self.config["levels_per_lesson"][str(self.difficulty)]
        #print(f'num_levels: {self.num_levels}')
        self.episodes_to_plot = []
        for j in range(min(self.num_levels,5)):
            self.episodes_to_plot.extend([1+j, 2+j, 5+j, 10+j, 20+j, 40+j, 50+j, 80+j, 100+j, 150+j, 200+j])
            self.episodes_to_plot.extend([(50 * i) + j for i in range(80)])
        self.episodes_to_plot = list(set(self.episodes_to_plot))
        self.episodes_to_plot.sort()
        #print(f'episodes to plot: {self.episodes_to_plot}')
        return

        """Generate list of env episodes to plot using save_action_history_plot"""

        base_episodes = []
        if self.tag == 'bc':
            self.episodes_to_plot = [10*i for i in range(20)]
        else:
            for i in range(self.num_levels):
                base_episodes.extend(
                    [0 + i, 2 + i, 5 + i, 10 + i, 50 + i, 100 + i, 200 + i, 300 + i, 400 + i, 500 + i, 800 + i,
                     1000 + i, 1200 + i, 1400 + i, 1700 + i, 2000 + i, 2300 + i, 2400 + i, 2600 + i, 2800 + i, 3000 + i,
                     4000 + i, 5000 + i, 6000 + i, 7000 + i])
            for j in range(self.num_levels):
                base_episodes.extend([(500 + j) * i for i in range(80)])
            base_episodes.sort()
            self.episodes_to_plot = list(set(base_episodes))
            self.episodes_to_plot.sort()
