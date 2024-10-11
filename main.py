# TODO: Aircraft visual direction not updating when new waypoint is set.

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
    "gameboard border margin": 35, # Ryan added, to make green bounds configurable. Default is 10% of gameboard size
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
    agent0_id = env.num_ships # TODO: Delete later, added as a hack to get agent IDs dynamically
    agent1_id = env.num_ships + 1
    while True:
        game_count += 1
        state = env.reset()  # reset the environment
        done = False  # flag for when the run is complete
        while not done:  # game loop
            actions = [] # use agent policies to get actions as a list of tuple [(agent index, waypoint)], None will use the default search behaviors

            # Agent 1: target_id policy (fly towards nearest unknown target)
            agent0_action, env.agents[env.aircraft_ids[0]].direction = target_id_policy(env,env.aircraft_ids[0],quadrant='full') # TODO: Incomplete
            print('Test: Agent0 ID is %s' % env.aircraft_ids[0])
            actions.append((env.aircraft_ids[0], agent0_action))

            # Agent 2: Mouse click waypoint control
            ev = pygame.event.get()
            for event in ev:
                if event.type == pygame.MOUSEBUTTONUP:
                    mouse_waypoint = pygame.mouse.get_pos()
                    print('mouse click detected at position %s' % (mouse_waypoint,))
                    if env.config['gameboard border margin'] < mouse_waypoint[0] < env.config['gameboard size']-env.config['gameboard border margin'] and env.config['gameboard border margin'] < mouse_waypoint[1] < env.config['gameboard size']-env.config['gameboard border margin']:
                        print('mouse click is inside legal bounds')
                        agent1_action = mouse_waypoint
                        actions.append((env.aircraft_ids[1], agent1_action))
                        #env.agents[agent1_id] = math.atan2(mouse_waypoint[1] - env.agents[agent1_id].y,mouse_waypoint[0] - env.agents[agent1_id].x) # TODO: Fix, causes game to crash right now

            state, reward, done, _ = env.step(actions)  # step through the environment
            # update agent policy here if desired, note that you can use env.observation_space and env.action_space instead of the dictionary format

            if render:  # if in PyGame mode, render the environment
                env.render()
        print("Game complete:", game_count)

    if render:
        pygame.quit()

    print("DONE!")
