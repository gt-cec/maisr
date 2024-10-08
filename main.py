import pygame
import math
from env import MAISREnv
import agents
import random
import sys

# environment configuration, use this for the gameplay parameters
env_config = {
    "gameboard size": 700,
    "num aircraft": 2,  # supports any number of aircrafts, colors are set in env.py:AIRCRAFT_COLORS
    "gameplay color": "red",
    "targets iteration": "A",
    "motion iteration": "G",
    "search pattern": "ladder",
    "verbose": False
}

if __name__ == "__main__":
    print("Starting MAISR environment")
    render = "headless" not in sys.argv

    if render:
        print("Starting in PyGame mode")
        pygame.init()  # init pygame
        clock = pygame.time.Clock()
        window = pygame.display.set_mode((env_config["gameboard size"], env_config["gameboard size"]))
        env = MAISREnv(env_config, window, clock=clock, render=True)
    else:
        print("Starting in headless mode")
        env = MAISREnv(env_config, None, render=False)

    game_count = 0
    while True:
        game_count += 1
        state = env.reset()  # reset the environment
        done = False  # flag for when the run is complete
        while not done:  # game loop
            actions = [] # use agent policies to get actions as a list of tuple [(agent index, waypoint)], None will use the default search behaviors
            state, reward, done, _ = env.step(actions)  # step through the environment
            # update agent policy here if desired, note that you can use env.observation_space and env.action_space instead of the dictionary format
            if render:  # if in PyGame mode, render the environment
                env.render()
        print("Game complete:", game_count)

    if render:
        pygame.quit()

    print("DONE!")
