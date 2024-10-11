# TODO:
#  * Make WEZ vs target ID a toggleable option within the same policy. One button that switches between target and WEZ, another that chooses quadrant.
#  * Make buttons interactive.
#  * Finish score system (started in env.py
#  * Add point tally on screen (# of targets IDd, # of WEZs, etc)
#  * Add timer to screen
#  * Add logic to end game when certain conditions met
#     * End game when human damage > 100 (failure)
#     * All targets + WEZs ID'd (success)
#     * Timer ran out (success if all targets ID'd, failure if not?)
#  * Kill agent when damage > 100

# Current bugs:
#  * Fix drawn orange circle around unknown WEZ for neutral targets (inside env.py:shipagent class:draw)
#  * Fix score counting (agents start with around ~40 score but should be 0)
#  * Done condition doesn't trigger when all ships ID'd
#  * Buttons don't change color when clicked

# Possible optimizations
#  * Don't re-render every GUI element every tick. Just the updates

import pygame
import math
from agents import *
from env import MAISREnv
from isr_gui import *
#import agents
import random
import sys

# environment configuration, use this for the gameplay parameters
env_config = {
    "gameboard size": 700, # NOTE: The rest of the GUI doesn't dynamically scale with different gameboard sizes. Stick to 700 for now
    "num aircraft": 2,  # supports any number of aircraft, colors are set in env.py:AIRCRAFT_COLORS (NOTE: Many aspects of the game currently only support 2 aircraft
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
        window_width, window_height = 1300, 850 #700  # Note: If you change this, you also have to change the render line in env.py:MAISREnv init function
        window = pygame.display.set_mode((window_width, window_height))
        env = MAISREnv(env_config, window, clock=clock, render=True)

    else:
        print("Starting in headless mode")
        env = MAISREnv(env_config, None, render=False)

    game_count = 0
    agent0_id = env.num_ships # TODO: Delete later, added as a hack to dynamically get agent IDs
    agent1_id = env.num_ships + 1

    agent0_policy, kwargs = target_id_policy, {} # Initialize agent 0's policy (will change when gameplan buttons are clicked

    # TODO: Testing how to make buttons visually latch
    buttons_clicked = {'target_id_button':False,'wez_id_button':False,'hold_button':False,'NW_quad_button':False, 'NE_quad_button':False, 'SW_quad_button':False, 'SE_quad_button':False}

    while True:
        game_count += 1
        state = env.reset()  # reset the environment
        done = False  # flag for when the run is complete
        while not done:  # game loop
            actions = [] # use agent policies to get actions as a list of tuple [(agent index, waypoint)], None will use the default search behaviors

            # Agent 0: Act based on currently selected gameplan
            agent0_action, _ = agent0_policy(env,env.aircraft_ids[0], **kwargs)
            actions.append((env.aircraft_ids[0], agent0_action))

            ev = pygame.event.get()
            for event in ev:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_position = pygame.mouse.get_pos()

                    # Agent 2: Mouse click waypoint control
                    if env.config['gameboard border margin'] < mouse_position[0] < env.config['gameboard size']-env.config['gameboard border margin'] and env.config['gameboard border margin'] < mouse_position[1] < env.config['gameboard size']-env.config['gameboard border margin']:
                        print('Waypoint set to %s' % (mouse_position,))
                        agent1_action = mouse_position
                        actions.append((env.aircraft_ids[1], agent1_action))

                    # Agent gameplan buttons
                    elif env.target_id_button.is_clicked(mouse_position):
                        comm = 'Agent 0 WILCO target ID'
                        agent0_policy = target_id_policy
                        kwargs = {'quadrant':'full','id_type':'target'}
                        print(comm)

                    elif env.wez_id_button.is_clicked(mouse_position): # TODO: Allow specifying quadrants
                        comm = 'Agent 0 WILCO target+WEZ ID'
                        agent0_policy = target_id_policy
                        kwargs = {'quadrant':'full','id_type':'wez'}
                        print(comm)

                    elif env.hold_button.is_clicked(mouse_position):
                        comm = 'Agent 0 WILCO hold'
                        agent0_policy = hold_policy
                        kwargs = {}
                        print(comm)
                    elif env.waypoint_button.is_clicked(mouse_position):
                        comm = 'Waypoint gameplan not implemented'
                        # agent0_policy = (TODO: Add)
                        print(comm)
                    elif env.NW_quad_button.is_clicked(mouse_position):
                        comm = 'Agent 0 WILCO target ID in NW quadrant'
                        agent0_policy = target_id_policy
                        kwargs = {'quadrant':'NW','id_type':'target'}
                        print(comm)
                    elif env.NE_quad_button.is_clicked(mouse_position):
                        comm = 'Agent 0 WILCO target ID in NE quadrant'
                        agent0_policy = target_id_policy
                        kwargs = {'quadrant':'NE','id_type':'target'}
                        print(comm)
                    elif env.SW_quad_button.is_clicked(mouse_position):
                        comm = 'Agent 0 WILCO target ID in SW quadrant'
                        agent0_policy = target_id_policy
                        kwargs = {'quadrant':'SW','id_type':'target'}
                        print(comm)
                    elif env.SE_quad_button.is_clicked(mouse_position):
                        comm = 'Agent 0 WILCO target ID in SE quadrant'
                        agent0_policy = target_id_policy
                        kwargs = {'quadrant':'SE','id_type':'target'}
                        print(comm)


            state, reward, done, _ = env.step(actions)  # step through the environment
            # update agent policy here if desired, note that you can use env.observation_space and env.action_space instead of the dictionary format
            if render:  # if in PyGame mode, render the environment
                env.render()
        print("Game complete:", game_count)

    if render:
        pygame.quit()

    print("DONE!")