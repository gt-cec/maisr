# TODO:
#  Priority 1:
#    * Intermittent location reporting
#    * Implement agent status window info in autonomous policy too
#    * Show agent waypoint: 0 shows none, 1 shows next one, 2 shows next two, 3 shows next 3 (to be implemented inside agents.py
#    * Populate agent_priorities (pull from autonomous policy)
#    * Massively clean up agent policies. Make one default policy that avoids WEZs well but prioritizes badly.
#    * Append subject ID (configurable here) to the log filename
#  Agent policies
#    * (Priority) Target_id_policy: Currently a working but slow and flawed A* search policy is implemented
#       (safe_target_id_policy). Have partially updated code in this script to replace target_id_policy but need to clean up.
#    * Autonomous policy code shouldn't change quadrants until that quadrant is empty.
#  Lower priority
#    * BUG: Game time not resetting when time done condition hit, so only the first game runs.
#    * Implement holding patterns for hold policy and human when no waypoint set (currently just freezes in place)
#    * Add waypoint command
#  Code optimization/cleanup
#    * Move a lot of the button handling code out of main.py and into isr_gui.py
#  Possible optimizations
#  * Don't re-render every GUI element every tick. Just the updates

from agents import *
from env import MAISREnv
from isr_gui import *
import sys
from data_logging import GameLogger
#from msfs_integration import MSFSConnector

from agent_policies.advanced_policies import target_id_policy, autonomous_policy
from agent_policies.basic_policies import hold_policy, mouse_waypoint_policy

# environment configuration, use this for the gameplay parameters
log_data = False  # Set to false if you don't want to save run data to a json file
use_msfs = False

env_config = {
    "gameboard size": 700, # NOTE: The rest of the GUI doesn't dynamically scale with different gameboard sizes. Stick to 700 for now
    "num aircraft": 2,  # supports any number of aircraft, colors are set in env.py:AIRCRAFT_COLORS (NOTE: Many aspects of the game currently only support 2 aircraft
    "gameplay color": "white",
    "gameboard border margin": 35,
    "targets iteration": "D",
    "motion iteration": "F",
    "search pattern": "ladder",
    "verbose": False,
    "window size": (1800,850), # width,height
    'show agent waypoint':2, # For SA-based agent transparency study TODO change to 0, 1, 2, 3
    'show agent location':'persistent', # For SA-based agent transparency. 'persistent', 'spotty', 'none' TODO not implemented
    'show_current_action': True, # for SA study, testing
    'show_risk_info': True, # for SA study, testing
    'show_decision_rationale': True, # for SA study, testing
}

# To update config for different experiment setups
"""env_config.update({
    'show_current_action': True,
    'show_risk_info': True, 
    'show_decision_rationale': True,
    # Disable any of these for control conditions
})"""


if __name__ == "__main__":
    print("Starting MAISR environment")
    render = "headless" not in sys.argv
    #if use_msfs:
     #   msfs = MSFSConnector()
      #  if not msfs.spawn_ai_aircraft():
       #     print("Failed to initialize MSFS integration")

    if render:
        print("Starting in PyGame mode")
        pygame.init()  # init pygame
        clock = pygame.time.Clock()
        window_width, window_height = env_config['window size'][0], env_config['window size'][1]
        window = pygame.display.set_mode((window_width, window_height))
        env = MAISREnv(env_config, window, clock=clock, render=True)

    else:
        print("Starting in headless mode")
        env = MAISREnv(env_config, None, render=False)

    game_count = 0
    agent0_id = env.num_ships # Hack to dynamically get agent IDs
    agent1_id = env.num_ships + 1

    # Set agent0's default policy (game will cycle back to this after ending a holding pattern etc
    default_agent0_policy, kwargs = target_id_policy, {'quadrant':'full','id_type':'target'} # Initialize agent 0's policy (will change when gameplan buttons are clicked
    #default_agent0_policy, kwargs = safe_target_id_policy, {'quadrant': 'full','id_type': 'target'}  # Initialize agent 0's policy (will change when gameplan buttons are clicked

    agent0_policy = default_agent0_policy

    if log_data: game_logger = GameLogger()

    while True:
        game_count += 1
        state = env.reset()  # reset the environment
        done = False  # flag for when the run is complete

        while not done:  # game loop
            if log_data:
                game_logger.log_state(env, pygame.time.get_ticks())
            actions = [] # use agent policies to get actions as a list of tuple [(agent index, waypoint)], None will use the default search behaviors

            # Agent 0: Act based on currently selected gameplan
            agent0_action, _ = agent0_policy(env,env.aircraft_ids[0], **kwargs)
            actions.append((env.aircraft_ids[0], agent0_action))

            # Update AI aircraft in MSFS # TODO this is new, still testing
            """if agent0_action is not None:
                msfs.update_ai_aircraft(
                    agent0_action[0],  # x coordinate
                    agent0_action[1],  # y coordinate
                    env.agents[agent0_id].direction * (180 / math.pi),  # convert radians to degrees
                    env_config["gameboard size"])

            # TODO this is new, still testing
            player_x, player_y, player_heading = msfs.get_player_position(env_config["gameboard size"])
            if player_x is not None and player_y is not None:
                agent1_action = (player_x, player_y)
                actions.append((env.aircraft_ids[1], agent1_action))"""

            ev = pygame.event.get()
            for event in ev:
                if event.type == pygame.QUIT:
                    #msfs.cleanup() # TODO this is new, testing
                    pygame.quit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_position = pygame.mouse.get_pos()

                    if log_data: game_logger.log_mouse_event(mouse_position,"click",pygame.time.get_ticks())

                    # Agent 1: Mouse click waypoint control
                    if env.config['gameboard border margin'] < mouse_position[0] < env.config['gameboard size']-env.config['gameboard border margin'] and env.config['gameboard border margin'] < mouse_position[1] < env.config['gameboard size']-env.config['gameboard border margin']:
                        print('Waypoint set to %s' % (mouse_position,))
                        agent1_action = mouse_position
                        actions.append((env.aircraft_ids[1], agent1_action))

                    # Agent 0 gameplan buttons TODO: Comm text not accurate, make dynamic to say target or target+WEZ
                    elif env.target_id_button.is_clicked(mouse_position):
                        agent0_policy = default_agent0_policy
                        env.button_latch_dict['target_id'] = True #not env.button_latch_dict['target_id']
                        if env.button_latch_dict['target_id']: env.button_latch_dict['wez_id'], env.button_latch_dict['autonomous'], env.button_latch_dict['hold'] = False, False, False # target id and wez id policies are mutually exclusive

                        kwargs['id_type'] = 'target'
                        env.comm_text = 'Beginning target ID'
                        print(env.comm_text + ' (Gameplan: ' + str(kwargs) + ')')
                        env.add_comm_message(env.comm_text)

                    elif env.wez_id_button.is_clicked(mouse_position):
                        #agent0_policy = target_id_policy
                        agent0_policy = default_agent0_policy
                        env.button_latch_dict['wez_id'] = True #not env.button_latch_dict['wez_id']
                        if env.button_latch_dict['wez_id']: env.button_latch_dict['target_id'], env.button_latch_dict['autonomous'], env.button_latch_dict['hold']= False, False, False  # target id and wez id policies are mutually exclusive
                        kwargs['id_type'] = 'wez'
                        env.comm_text = 'Beginning target+WEZ ID'
                        print(env.comm_text + '(Gameplan: ' + str(kwargs) + ')')
                        env.add_comm_message(env.comm_text)

                    elif env.hold_button.is_clicked(mouse_position):  # TODO: Supposed to toggle, but currently implemented to simply revert back to target_id when clicked a second time. Should go back to previous policy.
                        if not agent0_policy == hold_policy:
                            agent0_policy = hold_policy
                            env.button_latch_dict['hold'] = True #not env.button_latch_dict['hold']
                            env.comm_text = 'Holding'
                        else:
                            #agent0_policy = target_id_policy
                            agent0_policy = default_agent0_policy
                            env.button_latch_dict['hold'] = False  # not env.button_latch_dict['hold']
                            env.button_latch_dict['target_id'] = True  # not env.button_latch_dict['hold']
                            env.button_latch_dict['wez_id'] = False  # not env.button_latch_dict['hold']
                            env.comm_text = 'Resuming search'
                        if env.button_latch_dict['hold']: env.button_latch_dict['autonomous'] = False
                        #print(env.comm_text + ' (Gameplan: ' + str(kwargs) + ')')
                        env.add_comm_message(env.comm_text,is_ai=True)

                    elif env.waypoint_button.is_clicked(mouse_position): # TODO: In progress
                        env.comm_text = 'Waypoint gameplan not implemented'
                        print(env.comm_text)

                    elif env.NW_quad_button.is_clicked(mouse_position) and not env.full_quad_button.is_clicked(mouse_position):
                        #agent0_policy = target_id_policy
                        agent0_policy = default_agent0_policy
                        env.button_latch_dict['NW'] = True #not env.button_latch_dict['NW']
                        if env.button_latch_dict['NW']: env.button_latch_dict['NE'], env.button_latch_dict['SE'], env.button_latch_dict['SW'], env.button_latch_dict['full'],env.button_latch_dict['autonomous'],env.button_latch_dict['hold'] = False, False, False, False, False, False  # mutually exclusive
                        kwargs['quadrant'] = 'NW'
                        env.comm_text = 'Beginning target ID search in NW'
                        print(env.comm_text + '(Gameplan: ' + str(kwargs) + ')')
                        env.add_comm_message(env.comm_text,is_ai=True)
                    elif env.NE_quad_button.is_clicked(mouse_position) and not env.full_quad_button.is_clicked(mouse_position):
                        #agent0_policy = target_id_policy
                        agent0_policy = default_agent0_policy
                        env.button_latch_dict['NE'] = True #not env.button_latch_dict['NE']
                        if env.button_latch_dict['NE']: env.button_latch_dict['NW'], env.button_latch_dict['SE'], env.button_latch_dict['SW'], env.button_latch_dict['full'], env.button_latch_dict['autonomous'],env.button_latch_dict['hold'] = False,False, False, False, False, False  # mutually exclusive
                        kwargs['quadrant'] = 'NE'
                        env.comm_text = 'Beginning target ID search in NE'
                        print(env.comm_text + '(Gameplan: ' + str(kwargs) + ')')
                        env.add_comm_message(env.comm_text,is_ai=True)
                    elif env.SW_quad_button.is_clicked(mouse_position) and not env.full_quad_button.is_clicked(mouse_position):
                        #agent0_policy = target_id_policy
                        agent0_policy = default_agent0_policy
                        env.button_latch_dict['SW'] = True #not env.button_latch_dict['SW']
                        if env.button_latch_dict['SW']: env.button_latch_dict['NE'], env.button_latch_dict['SE'], env.button_latch_dict['NW'], env.button_latch_dict['full'], env.button_latch_dict['autonomous'],env.button_latch_dict['hold'] = False, False, False, False, False,False  # mutually exclusive
                        kwargs['quadrant'] = 'SW'
                        env.comm_text = 'Beginning target ID search in SW'
                        print(env.comm_text + '(Gameplan: ' + str(kwargs) + ')')
                        env.add_comm_message(env.comm_text,is_ai=True)
                    elif env.SE_quad_button.is_clicked(mouse_position) and not env.full_quad_button.is_clicked(mouse_position):
                        #agent0_policy = target_id_policy
                        agent0_policy = default_agent0_policy
                        env.button_latch_dict['SE'] = True #not env.button_latch_dict['SE']
                        if env.button_latch_dict['SE']: env.button_latch_dict['NE'], env.button_latch_dict['SW'], env.button_latch_dict['NW'], env.button_latch_dict['full'], env.button_latch_dict['autonomous'],env.button_latch_dict['hold'] = False,False, False, False, False, False  # mutually exclusive
                        kwargs['quadrant'] = 'SE'
                        env.comm_text = 'Beginning target ID search in SE'
                        print(env.comm_text + '(Gameplan: ' + str(kwargs) + ')')
                        env.add_comm_message(env.comm_text,is_ai=True)
                    elif env.full_quad_button.is_clicked(mouse_position):
                        #agent0_policy = target_id_policy
                        agent0_policy = default_agent0_policy
                        env.button_latch_dict['full'] = True #not env.button_latch_dict['full']
                        if env.button_latch_dict['full']: env.button_latch_dict['NE'], env.button_latch_dict['SW'], env.button_latch_dict['NW'], env.button_latch_dict['SE'], env.button_latch_dict['autonomous'],env.button_latch_dict['hold'] = False, False, False, False, False,False  # mutually exclusive
                        kwargs['quadrant'] = 'full'
                        env.comm_text = 'Beginning full map search'
                        print(env.comm_text + '(Gameplan: ' + str(kwargs) + ')')
                        env.add_comm_message(env.comm_text,is_ai=True)
                    elif env.autonomous_button.is_clicked(mouse_position):
                        agent0_policy = autonomous_policy
                        env.button_latch_dict['autonomous'] = True #not env.button_latch_dict['autonomous']
                        if env.button_latch_dict['autonomous']: env.button_latch_dict['NE'], env.button_latch_dict['SW'], \
                        env.button_latch_dict['NW'], env.button_latch_dict['full'], env.button_latch_dict[
                            'hold'],env.button_latch_dict['target_id'],env.button_latch_dict['target_wez_id'],env.button_latch_dict['hold'] = False, False, False, False, False, False, False,False  # mutually exclusive
                        env.comm_text = 'Beginning autonomous search'
                        print(env.comm_text)
                        env.add_comm_message(env.comm_text,is_ai=True)

                    elif env.pause_button.is_clicked(mouse_position):
                        env.pause(pygame.MOUSEBUTTONDOWN)
                if event.type == pygame.KEYDOWN: # TODO: Doesn't work yet
                    if event.key == pygame.K_SPACE:
                        env.pause(pygame.K_SPACE)

            state, reward, done, _ = env.step(actions)  # step through the environment
            # update agent policy here if desired, note that you can use env.observation_space and env.action_space instead of the dictionary format
            if render:  # if in PyGame mode, render the environment
                env.render()
            clock.tick(60)  # TODO this is new for MSFS

        if done:
            waiting_for_key = True
            while waiting_for_key:
                env.render()  # Keep rendering while waiting
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                        waiting_for_key = False
                        break
                    elif event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
        print("Game complete:", game_count)

    if render:
        #msfs.cleanup()
        pygame.quit()

    print("DONE!")