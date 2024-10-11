# TODO:
#  Agent policies
#    * Target ID should have WEZ avoidance
#    * Add waypoint command
#    * Hold policy should fly circles, not freeze
#    * Move "full" button somewhere else (should probably just double check "target ID" to reset quadrants)
#    * Autonomous policy: Very inefficient, shouldn't change quadrants until that quadrant is empty
#  Dynamic button appearance
#    * Target ID and Target+WEZ ID latch when clicked (and are mutually exclusive)
#    * Quadrant buttons latch when clicked
#    * "Hold" button latches when clicked
#  Comm log
#    * Center text properly
#    * Show more than one message
#    * Add timestamp
#    * Add text when human commands a gameplan too
#  Game end logic
#    * End game when human damage > 100 (failure)
#    * All targets + WEZs ID'd (success)
#    * Timer ran out (success if all targets ID'd, failure if not?)
#  Point system
#    * Add +20 points when all targets ID'd
#    * Subtract points for damage
#    * Add point tally on screen (# of targets IDd, # of WEZs, etc)
#    * Need to think through game termination criteria. If all targets ID'd, should the game end and give player bonus points for time remaining? Or continue and let them ID WEZs too?

# Current bugs:
#  * Taking damage from all hostiles as if their WEZ = max size
#  * Game clock doesn't stop when game is paused
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

    agent0_policy, kwargs = target_id_policy, {'quadrant':'full','id_type':'target'} # Initialize agent 0's policy (will change when gameplan buttons are clicked

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

                    # Agent 1: Mouse click waypoint control
                    if env.config['gameboard border margin'] < mouse_position[0] < env.config['gameboard size']-env.config['gameboard border margin'] and env.config['gameboard border margin'] < mouse_position[1] < env.config['gameboard size']-env.config['gameboard border margin']:
                        print('Waypoint set to %s' % (mouse_position,))
                        agent1_action = mouse_position
                        actions.append((env.aircraft_ids[1], agent1_action))

                    # Agent 0 gameplan buttons TODO: Comm text not accurate, make dynamic to say target or target+WEZ
                    elif env.target_id_button.is_clicked(mouse_position):
                        agent0_policy = target_id_policy
                        kwargs['id_type'] = 'target'
                        env.comm_text = 'Agent 0 WILCO target ID'
                        print(env.comm_text + ' (Gameplan: ' + str(kwargs) + ')')

                    elif env.wez_id_button.is_clicked(mouse_position):
                        agent0_policy = target_id_policy
                        kwargs['id_type'] = 'wez'
                        env.comm_text = 'Agent 0 WILCO target+WEZ ID'
                        print(env.comm_text + '(Gameplan: ' + str(kwargs) + ')')

                    elif env.hold_button.is_clicked(mouse_position):
                        agent0_policy = hold_policy
                        #kwargs = {}
                        env.comm_text = 'Agent 0 WILCO hold'
                        print(env.comm_text + '(Gameplan: ' + str(kwargs) + ')')

                    elif env.waypoint_button.is_clicked(mouse_position): # TODO: In progress
                        env.comm_text = 'Waypoint gameplan not implemented'
                        print(env.comm_text)

                    elif env.NW_quad_button.is_clicked(mouse_position):
                        agent0_policy = target_id_policy
                        #kwargs = {'quadrant':'NW','id_type':'target'}
                        kwargs['quadrant'] = 'NW'
                        env.comm_text = 'Agent 0 WILCO target ID in NW quadrant'
                        print(env.comm_text + '(Gameplan: ' + str(kwargs) + ')')
                    elif env.NE_quad_button.is_clicked(mouse_position):
                        agent0_policy = target_id_policy
                        kwargs['quadrant'] = 'NE'
                        env.comm_text = 'Agent 0 WILCO target ID in NE quadrant'
                        print(env.comm_text + '(Gameplan: ' + str(kwargs) + ')')
                    elif env.SW_quad_button.is_clicked(mouse_position):
                        agent0_policy = target_id_policy
                        kwargs['quadrant'] = 'SW'
                        env.comm_text = 'Agent 0 WILCO target ID in SW quadrant'
                        print(env.comm_text + '(Gameplan: ' + str(kwargs) + ')')
                    elif env.SE_quad_button.is_clicked(mouse_position):
                        agent0_policy = target_id_policy
                        kwargs['quadrant'] = 'SE'
                        env.comm_text = 'Agent 0 WILCO target ID in SE quadrant'
                        print(env.comm_text + '(Gameplan: ' + str(kwargs) + ')')
                    elif env.full_quad_button.is_clicked(mouse_position):
                        agent0_policy = target_id_policy
                        kwargs['quadrant'] = 'full'
                        env.comm_text = 'Agent 0 WILCO, full map'
                        print(env.comm_text + '(Gameplan: ' + str(kwargs) + ')')
                    elif env.autonomous_button.is_clicked(mouse_position):
                        agent0_policy = autonomous_policy
                        env.comm_text = 'Agent 0 WILCO, autonomous'
                        print(env.comm_text)

                    elif env.pause_button.is_clicked(mouse_position):
                        print('Game paused')
                        paused = True
                        unpaused = False
                        while paused:
                            env.paused = True
                            pygame.time.wait(200)
                            ev = pygame.event.get()
                            for event in ev:
                                if event.type == pygame.MOUSEBUTTONDOWN:
                                    mouse_position = pygame.mouse.get_pos()
                                    if env.pause_button.is_clicked(mouse_position):
                                        paused = False
                                        env.paused = False

            state, reward, done, _ = env.step(actions)  # step through the environment
            # update agent policy here if desired, note that you can use env.observation_space and env.action_space instead of the dictionary format
            if render:  # if in PyGame mode, render the environment
                env.render()
        print("Game complete:", game_count)

    if render:
        pygame.quit()

    print("DONE!")