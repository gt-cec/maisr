# TODO:
#  Priority bugs to fix
#    * Game clock doesn't stop when game is paused
#    * WEZ damage is triggering if we're within WEZ ID range, not the ship's actual WEZ range.
#  Agent policies
#    * (Priority) Rewrite basic target ID policy using A* search to calculate path to each unknown target (avoiding WEZs),
#      selecting the one with the shortest path.
#    * Add waypoint command
#    * Autonomous policy code is very inefficient, shouldn't change quadrants until that quadrant is empty.
#    * Implement holding patterns for hold policy and human when no waypoint set (currently just freezes in place)
#  Comm log
#    * Center text properly
#    * Show more than one message
#    * Add timestamp
#    * Add text when human commands a gameplan too
#  Point system
#    * Subtract points for damage
#    * When game ends, show popup on screen with point tally (# of targets IDd, # of WEZs, etc)
#  Other bugs:
#    * Fix drawn orange circle around unknown WEZ for neutral targets (inside env.py:shipagent class:draw)
#    * Fix score counting (agents start with around ~40 score but should be 0)

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
    agent0_id = env.num_ships # Hack to dynamically get agent IDs
    agent1_id = env.num_ships + 1

    agent0_policy, kwargs = target_id_policy, {'quadrant':'full','id_type':'target'} # Initialize agent 0's policy (will change when gameplan buttons are clicked

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
                        env.button_latch_dict['target_id'] = True #not env.button_latch_dict['target_id']
                        if env.button_latch_dict['target_id']: env.button_latch_dict['wez_id'], env.button_latch_dict['autonomous'] = False, False # target id and wez id policies are mutually exclusive

                        kwargs['id_type'] = 'target'
                        env.comm_text = '[Agent 0] WILCO target ID'
                        print(env.comm_text + ' (Gameplan: ' + str(kwargs) + ')')

                    elif env.wez_id_button.is_clicked(mouse_position):
                        agent0_policy = target_id_policy
                        env.button_latch_dict['wez_id'] = True #not env.button_latch_dict['wez_id']
                        if env.button_latch_dict['wez_id']: env.button_latch_dict['target_id'], env.button_latch_dict['autonomous'] = False, False  # target id and wez id policies are mutually exclusive
                        kwargs['id_type'] = 'wez'
                        env.comm_text = '[Agent 0] WILCO target+WEZ ID'
                        print(env.comm_text + '(Gameplan: ' + str(kwargs) + ')')

                    elif env.hold_button.is_clicked(mouse_position):  # TODO: Supposed to toggle, but currently implemented to simply revert back to target_id when clicked a second time. Should go back to previous policy.
                        if not agent0_policy == hold_policy:
                            agent0_policy = hold_policy
                            env.button_latch_dict['hold'] = True #not env.button_latch_dict['hold']
                        else:
                            agent0_policy = target_id_policy
                            env.button_latch_dict['hold'] = False  # not env.button_latch_dict['hold']
                            env.button_latch_dict['target_id'] = True  # not env.button_latch_dict['hold']
                            env.button_latch_dict['wez_id'] = False  # not env.button_latch_dict['hold']
                        #kwargs = {}
                        env.comm_text = '[Agent 0] WILCO hold'
                        print(env.comm_text + ' (Gameplan: ' + str(kwargs) + ')')

                    elif env.waypoint_button.is_clicked(mouse_position): # TODO: In progress
                        env.comm_text = 'Waypoint gameplan not implemented'
                        print(env.comm_text)

                    elif env.NW_quad_button.is_clicked(mouse_position) and not env.full_quad_button.is_clicked(mouse_position):
                        agent0_policy = target_id_policy
                        env.button_latch_dict['NW'] = True #not env.button_latch_dict['NW']
                        if env.button_latch_dict['NW']: env.button_latch_dict['NE'], env.button_latch_dict['SE'], env.button_latch_dict['SW'], env.button_latch_dict['full'],env.button_latch_dict['autonomous'] = False, False, False, False, False  # mutually exclusive
                        kwargs['quadrant'] = 'NW'
                        env.comm_text = '[Agent 0] WILCO target ID in NW quadrant'
                        print(env.comm_text + '(Gameplan: ' + str(kwargs) + ')')
                    elif env.NE_quad_button.is_clicked(mouse_position) and not env.full_quad_button.is_clicked(mouse_position):
                        agent0_policy = target_id_policy
                        env.button_latch_dict['NE'] = True #not env.button_latch_dict['NE']
                        if env.button_latch_dict['NE']: env.button_latch_dict['NW'], env.button_latch_dict['SE'], env.button_latch_dict['SW'], env.button_latch_dict['full'], env.button_latch_dict['autonomous'] = False, False, False, False, False  # mutually exclusive
                        kwargs['quadrant'] = 'NE'
                        env.comm_text = '[Agent 0] WILCO target ID in NE quadrant'
                        print(env.comm_text + '(Gameplan: ' + str(kwargs) + ')')
                    elif env.SW_quad_button.is_clicked(mouse_position) and not env.full_quad_button.is_clicked(mouse_position):
                        agent0_policy = target_id_policy
                        env.button_latch_dict['SW'] = True #not env.button_latch_dict['SW']
                        if env.button_latch_dict['SW']: env.button_latch_dict['NE'], env.button_latch_dict['SE'], env.button_latch_dict['NW'], env.button_latch_dict['full'], env.button_latch_dict['autonomous'] = False, False, False, False, False  # mutually exclusive
                        kwargs['quadrant'] = 'SW'
                        env.comm_text = '[Agent 0] WILCO target ID in SW quadrant'
                        print(env.comm_text + '(Gameplan: ' + str(kwargs) + ')')
                    elif env.SE_quad_button.is_clicked(mouse_position) and not env.full_quad_button.is_clicked(mouse_position):
                        agent0_policy = target_id_policy
                        env.button_latch_dict['SE'] = True #not env.button_latch_dict['SE']
                        if env.button_latch_dict['SE']: env.button_latch_dict['NE'], env.button_latch_dict['SW'], env.button_latch_dict['NW'], env.button_latch_dict['full'], env.button_latch_dict['autonomous'] = False, False, False, False, False  # mutually exclusive
                        kwargs['quadrant'] = 'SE'
                        env.comm_text = '[Agent] 0 WILCO target ID in SE quadrant'
                        print(env.comm_text + '(Gameplan: ' + str(kwargs) + ')')
                    elif env.full_quad_button.is_clicked(mouse_position):
                        agent0_policy = target_id_policy
                        env.button_latch_dict['full'] = True #not env.button_latch_dict['full']
                        if env.button_latch_dict['full']: env.button_latch_dict['NE'], env.button_latch_dict['SW'], env.button_latch_dict['NW'], env.button_latch_dict['SE'], env.button_latch_dict['autonomous'] = False, False, False, False, False  # mutually exclusive
                        kwargs['quadrant'] = 'full'
                        env.comm_text = '[Agent 0] WILCO, full map'
                        print(env.comm_text + '(Gameplan: ' + str(kwargs) + ')')
                    elif env.autonomous_button.is_clicked(mouse_position):
                        agent0_policy = autonomous_policy
                        env.button_latch_dict['autonomous'] = True #not env.button_latch_dict['autonomous']
                        env.comm_text = '[Agent 0] WILCO, autonomous'
                        print(env.comm_text)

                    elif env.pause_button.is_clicked(mouse_position):
                        env.pause(pygame.MOUSEBUTTONDOWN)
                if event.type == pygame.KEYDOWN: # TODO: Doesn't work yet
                    if event.key == pygame.K_SPACE:
                        env.pause(pygame.K_SPACE)

            state, reward, done, _ = env.step(actions)  # step through the environment
            # update agent policy here if desired, note that you can use env.observation_space and env.action_space instead of the dictionary format
            if render:  # if in PyGame mode, render the environment
                env.render()
        print("Game complete:", game_count)

    if render:
        pygame.quit()

    print("DONE!")