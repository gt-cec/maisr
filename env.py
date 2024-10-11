import gym
import numpy as np
from numpy.random.mtrand import get_state
import pygame
import random
import agents

class MAISREnv(gym.Env):
    """Multi-Agent ISR Environment following the Gym format"""

    def __init__(self, config={}, window=None, clock=None, render=False):
        super().__init__()

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
        self.AIRCRAFT_ENGAGEMENT_RADIUS = 100  # pixel width of aircraft engagement (to identify WEZ of threats)
        self.AIRCRAFT_ISR_RADIUS = 170  # pixel width of aircraft scanner (to identify hostile vs benign)

        self.GAMEBOARD_NOGO_RED = (255, 200, 200)  # color of the red no-go zone
        self.GAMEBOARD_NOGO_YELLOW = (255, 225, 200)  # color of the yellow no-go zone
        self.FLIGHTPLAN_EDGE_MARGIN = .2  # proportion distance from edge of gameboard to flight plan, e.g., 0.2 = 20% in, meaning a flight plan of (1,1) would go to 80%,80% of the gameboard
        self.AIRCRAFT_COLORS = [(255, 165, 0), (0, 0, 255), (200, 0, 200), (80, 80, 80)]  # colors of aircraft 1, 2, 3, ... add more colors here, additional aircraft will repeat the last color


        # labeled ISR flight plans
        self.FLIGHTPLANS = {
            # Note: (1.0 * margin, 1.0 * margin) is top left, and all paths and scaled to within the region within the FLIGHTPLAN_EDGE_MARGIN
            "square":
                [
                    (0, 1),  # top left
                    (0, 0),  # bottom left
                    (1, 0),  # bottom right
                    (1, 1)  # top right
                ],
            "ladder":
                [
                    (1, 1),  # top right
                    (1, .66),
                    (0, .66),
                    (0, .33),
                    (1, .33),
                    (1, 0),  # bottom right
                    (0, 0)  # bottom left
                ],
            "hold":
                [
                    (.4, .4),
                    (.4, .6),
                    (.6, .6),
                    (.6, .4)
                ]
        }

        self.config = config
        self.window = window
        self.clock = clock

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
            self.window = pygame.display.set_mode((self.config["gameboard size"], self.config["gameboard size"]))


    def reset(self):
        self.agents = []
        self.aircraft_ids = []  # indexes of the aircraft agents
        self.damage = 0  # total damage from all agents
        self.num_identified_ships = 0  # number of ships with accessed threat levels, used for determining game end

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
            agent.move() # Ryan TODO: How does this line interact with the waypoint override above?

            # handle ships
            if agent.agent_class == "ship":
                # check if ship is within range of aircraft
                for aircraft_id in self.aircraft_ids:
                    dist = agent.distance(self.agents[aircraft_id])
                    # if in the aircraft's ISR range, set to observed
                    if not agent.observed and self.agents[aircraft_id].in_isr_range(distance=dist):
                        agent.observed = True
                        # Ryan TODO: Add points for ship ID here (self.agents[aircraft_id].damage += 1)
                        if self.config["verbose"]:
                            print("Ship {} observed by aircraft {}".format(agent.agent_idx, aircraft_id))

                    # if in the aircraft's engagement range, identify threat level
                    if not agent.observed_threat and (self.agents[aircraft_id].in_engagement_range(distance=dist) or agent.threat == 0):
                        agent.observed_threat = True
                        # Ryan TODO: Add points for WEZ ID here (self.agents[aircraft_id].damage += 1)
                        self.num_identified_ships += 1
                        if self.config["verbose"]:
                            print("Ship {} threat level identified by aircraft {}".format(agent.agent_idx, aircraft_id))
                        break
                    # if in the ship's weapon range, damage the aircraft
                    if agent.in_weapon_range(distance=dist):
                        self.agents[aircraft_id].damage += 1
                        self.damage += 1
                        # Ryan TODO: Add -points for damage here
                    # add some logic here if you want the no-go zones to damage the aircrafts
                    # Ryan TODO: Check if AI agent damage

        # progress update
        if self.config["verbose"]:
            print("   Found:", self.num_identified_ships, "Total:", len(self.agents) - len(self.aircraft_ids), "Damage:", self.damage)
        # exit when all ships are identified
        state = self.get_state()  # you can make this self.observation_space and use that (will require a tiny bit of customization, look into RL tutorials)
        reward = self.get_reward()
        done = self.num_identified_ships >= len(self.agents) - len(self.aircraft_ids)  # round is complete when all ships have been identified
        # Ryan TODO: Add if human damage >= 100, done
        # Ryan TODO: Add bonus points if all identified
        return state, reward, done, {}


    def get_reward(self):
        # define a custom reward function here
        # Ryan TODO: Add points for IDing targets (+), IDing all targets (+), taking damage (-), or agent A/C dying (-)
        return -self.damage


    def render(self, mode='human', close=False):
        # gameboard background
        self.window.fill((255, 255, 255))  # white background
        # set up no-go areas
        if "gameplay color" in self.config:
            pygame.draw.rect(self.window, self.GAMEBOARD_NOGO_RED if self.config["gameplay color"] == "yellow" else self.GAMEBOARD_NOGO_YELLOW if self.config["gameplay color"] else (255, 255, 255), (self.config["gameboard size"] * 0.8, 0, self.config["gameboard size"] * 0.8, self.config["gameboard size"]))
        self.__render_box__(1, (0, 0, 0), 3)  # outer box
        #self.__render_box__(self.config["gameboard size"] * 0.1, (0, 128, 0), 2)  # inner box
        self.__render_box__(self.config["gameboard border margin"], (0, 128, 0), 2)  # inner box
        pygame.draw.line(self.window, (0, 0, 0), (self.config["gameboard size"] // 2, 0), (self.config["gameboard size"] // 2, self.config["gameboard size"]), 2)
        pygame.draw.line(self.window, (0, 0, 0), (0, self.config["gameboard size"] // 2), (self.config["gameboard size"], self.config["gameboard size"] // 2), 2)
        # draw the agents
        for agent in self.agents:
            agent.draw(self.window)
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
