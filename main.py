import pygame
import math

from agents import target_id_policy
from env import MAISREnv
import agents
import random
import sys

# environment configuration, use this for the gameplay parameters
env_config = {
    "gameboard size": 700,
    "num aircraft": 2,  # supports any number of aircrafts, colors are set in env.py:AIRCRAFT_COLORS
    "gameplay color": "white",
    "targets iteration": "C",
    "motion iteration": "F",
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
            # Ryan TODO: create policies for agent 1 (rule based) and agent 2 (player mouse/keyboard control) and use them to set action[0] and action[1]

            # Code below is an incomplete implementation of an agent policy that flies toward the nearest unknown ship.
            # TODO: Aircraft visual direction not updating like it should.
            target_id_action, env.agents[env.aircraft_ids[0]].direction = target_id_policy(env,env.aircraft_ids[0],quadrant='full')
            print(target_id_action)
            actions = [(env.aircraft_ids[0],target_id_action)]

            state, reward, done, _ = env.step(actions)  # step through the environment

            # update agent policy here if desired, note that you can use env.observation_space and env.action_space instead of the dictionary format
            if render:  # if in PyGame mode, render the environment
                env.render()
        print("Game complete:", game_count)

    if render:
        pygame.quit()

    print("DONE!")
