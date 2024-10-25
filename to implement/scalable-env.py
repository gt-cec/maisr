import gym
import numpy as np
from numpy.random.mtrand import get_state
import pygame
import random
import agents
from isr_gui import Button, ScoreWindow, HealthWindow, TimeWindow, AgentInfoDisplay
import datetime
import math

class MAISREnv(gym.Env):
    def __init__(self, config={}, window=None, clock=None, render=False):
        super().__init__()
        
        self.config = config
        self.window = window
        self.clock = clock
        
        # Get scaling ratio from config or default to 1.0
        self.scaling_ratio = self.config.get("scaling_ratio", 1.0)
        
        # Base sizes that will be scaled
        self.BASE_GAMEBOARD_SIZE = 700
        self.BASE_WINDOW_WIDTH = 1800
        self.BASE_WINDOW_HEIGHT = 850
        
        # Scale the window and gameboard sizes
        self.config["gameboard size"] = int(self.BASE_GAMEBOARD_SIZE * self.scaling_ratio)
        self.config["window size"] = (
            int(self.BASE_WINDOW_WIDTH * self.scaling_ratio),
            int(self.BASE_WINDOW_HEIGHT * self.scaling_ratio)
        )

        # Scale all constants
        self.AGENT_BASE_DRAW_WIDTH = int(10 * self.scaling_ratio)
        self.AIRCRAFT_NOSE_LENGTH = int(10 * self.scaling_ratio)
        self.AIRCRAFT_TAIL_LENGTH = int(25 * self.scaling_ratio)
        self.AIRCRAFT_TAIL_WIDTH = int(7 * self.scaling_ratio)
        self.AIRCRAFT_WING_LENGTH = int(18 * self.scaling_ratio)
        self.AIRCRAFT_LINE_WIDTH = int(5 * self.scaling_ratio)
        self.AIRCRAFT_ENGAGEMENT_RADIUS = int(40 * self.scaling_ratio)
        self.AIRCRAFT_ISR_RADIUS = int(85 * self.scaling_ratio)

        # Colors remain unchanged
        self.AGENT_COLOR_UNOBSERVED = (255, 215, 0)
        self.AGENT_COLOR_OBSERVED = (128, 0, 128)
        self.AGENT_COLOR_THREAT = (255, 0, 0)
        self.AGENT_THREAT_RADIUS = [0, 1.4, 2.5, 4]
        self.GAMEBOARD_NOGO_RED = (255, 200, 200)
        self.GAMEBOARD_NOGO_YELLOW = (255, 225, 200)
        self.AIRCRAFT_COLORS = [(0, 160, 160), (0, 0, 255), (200, 0, 200), (80, 80, 80)]

        # Scale UI positions and dimensions
        self.gameboard_offset = 0
        self.window_x = self.config["window size"][0]
        self.window_y = self.config["window size"][1]
        self.right_pane_edge = self.config['gameboard size'] + int(20 * self.scaling_ratio)
        self.comm_pane_edge = self.config['gameboard size'] + int(40 * self.scaling_ratio) + int(405 * self.scaling_ratio)

        # Scoring constants don't need scaling
        self.score = 0
        self.all_targets_points = 20
        self.target_points = 10
        self.threat_points = 5
        self.time_points = 2
        self.agent_damage_points = -0.1
        self.human_damage_points = -0.2
        self.wingman_dead_points = -30
        self.human_dead_points = -40

        # Scale fonts
        self.base_font_size = int(36 * self.scaling_ratio)
        self.message_font = pygame.font.SysFont(None, self.base_font_size)
        self.tally_font = pygame.font.SysFont(None, int(24 * self.scaling_ratio))
        self.pause_font = pygame.font.SysFont(None, int(74 * self.scaling_ratio))

        # UI State variables
        self.first_step = True
        self.paused = False
        self.agent0_dead = False
        self.agent1_dead = False
        self.risk_level = 'LOW'
        self.agent_waypoint_clicked = False
        
        # Comm log setup
        self.comm_messages = []
        self.max_messages = 7
        self.ai_color = self.AIRCRAFT_COLORS[0]
        self.human_color = self.AIRCRAFT_COLORS[1]

        # Game state tracking
        self.identified_targets = 0
        self.identified_threat_types = 0
        self.display_time = 0
        self.pause_start_time = 0
        self.total_pause_time = 0
        
        # Button states
        self.button_latch_dict = {
            'target_id': False, 'wez_id': False, 'hold': False, 'waypoint': False,
            'NW': False, 'SW': False, 'NE': False, 'SE': False, 'full': False,
            'autonomous': False, 'pause': False, 'risk_low': False,
            'risk_medium': True, 'risk_high': False
        }

        # Visual effects
        self.damage_flash_duration = 500
        self.damage_flash_start = 0
        self.damage_flash_alpha = 0
        self.last_health_points = {0: 4, 1: 4}

        # Agent display settings
        self.show_agent_waypoint = self.config['show agent waypoint']
        self.agent_priorities = 'placeholder'
        
        # Create scaled agent info display
        self.agent_info_display = AgentInfoDisplay(
            self.comm_pane_edge, 
            int(10 * self.scaling_ratio),
            int(470 * self.scaling_ratio), 
            int(260 * self.scaling_ratio)
        )

        # Flight plans remain proportional, no scaling needed
        self.FLIGHTPLANS = {
            "square": [(0, 1), (0, 0), (1, 0), (1, 1)],
            "ladder": [(1, 1), (1, .66), (0, .66), (0, .33), (1, .33), (1, 0), (0, 0)],
            "hold": [(.4, .4), (.4, .6), (.6, .6), (.6, .4)]
        }
        self.FLIGHTPLAN_EDGE_MARGIN = .2

        # Set random seed if provided
        if "seed" in config:
            random.seed(config["seed"])
        else:
            print("Note: 'seed' is not in the env config, defaulting to 0.")
            random.seed(0)

        # Initialize targets
        if "targets iteration" not in config:
            print("Note: 'targets iteration' is not in the env config, defaulting to 10 ships.")
            self.num_ships = 10
        else:
            self.num_ships = {
                "A": 10, "B": 20, "C": 30, "D": 50, "E": 100
            }.get(config["targets iteration"], 10)
        
        self.total_targets = self.num_ships

        # Set default search pattern if not provided
        if "search pattern" not in config:
            print("Note: 'search pattern' is not in the env config, defaulting to 'square'.")
            self.config["search pattern"] = "square"

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "aircraft pos": gym.spaces.Box(low=-1, high=1, shape=(self.config["num aircraft"], 2), dtype=np.float32),
            "threat pos": gym.spaces.Box(low=-1, high=1, shape=(self.num_ships, 2), dtype=np.float32),
            "threat class": gym.spaces.MultiBinary([self.num_ships, 2])
        })

        self.reset()

        if render:
            self.window = pygame.display.set_mode(self.config["window size"])

    # ... [Rest of the original methods, updated to use scaled values] ...
