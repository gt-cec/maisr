from agents import *
import sys
import ctypes

from env import MAISREnv
from gui import *
from utility.data_logging import GameLogger, load_env_config
from config import subject_id, log_data, config_filename
from autonomous_policy import AutonomousPolicy
#from rl_policy import RLPolicy


if __name__ == "__main__":

    config_file_path = './config_files/' + config_filename
    env_config = load_env_config(config_file_path)

    print("Starting MAISR environment")
    render = "headless" not in sys.argv

    if log_data:
        game_logger = GameLogger(subject_id,config_filename)
        game_logger.initial_log()

    game_count = 0
    total_games = 20 # Number of games to run

    while game_count < total_games:
        if render:
            print("Starting in PyGame mode")
            pygame.init()
            clock = pygame.time.Clock()
            ctypes.windll.user32.SetProcessDPIAware()  # Disables display scaling so the game fits on small, high-res monitors
            window_width, window_height = env_config['window size'][0], env_config['window size'][1]
            window = pygame.display.set_mode((window_width, window_height))

            env = MAISREnv(env_config, window, clock=clock, render=True)

        else:
            print("Starting in headless mode")
            pygame.init()  # init pygame
            clock = pygame.time.Clock()
            pygame.font.init()
            env = MAISREnv(env_config, None, render=False)

        agent0_id = env.num_ships  # Hack to dynamically get agent IDs
        agent0_policy = AutonomousPolicy(env, agent0_id)
        agent0_policy.show_low_level_goals = env.config['show_low_level_goals']
        agent0_policy.show_high_level_goals = env.config['show_high_level_goals']
        agent0_policy.show_high_level_rationale = env.config['show_high_level_rationale']
        agent0_policy.show_tracked_factors = env.config['show_tracked_factors']
        agent0_policy.collision_ok = True
        agent0_policy.use_thread_the_needle = False  # Currently debugging thread_the_needle, disabled for now
        agent1_id = env.num_ships + 1

        game_count += 1
        state = env.reset()  # reset the environment
        done = False  # flag for when the run is complete

        while not done:  # game loop
            if log_data:
                game_logger.log_state(env, pygame.time.get_ticks())
            actions = [] # use agent policies to get actions as a list of tuple [(agent index, waypoint)], None will use the default search behaviors
            time_sec = float(env.display_time)/1000


            # Agent 0: Act based on currently selected gameplan
            agent0_policy.act()
            actions.append((env.aircraft_ids[0], agent0_policy.target_point))
            agent0_policy.update_agent_info()

            # Handle SAGAT surveys
            if 59.95 < time_sec < 60.05 and not env.survey1_launched:
                env.survey1_launched = True
                env.SAGAT_survey(1)

            if 119.95 < time_sec < 120.05 and not env.survey2_launched:
                env.survey2_launched = True
                env.SAGAT_survey(2)

            if 179.95 < time_sec < 180.05 and not env.survey3_launched:
                env.survey3_launched = True
                env.SAGAT_survey(3)

            ev = pygame.event.get()
            for event in ev:
                if event.type == pygame.QUIT:
                    pygame.quit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_position = pygame.mouse.get_pos()

                    if log_data: game_logger.log_mouse_event(mouse_position,"click",pygame.time.get_ticks())

                    # Agent 1: Mouse click waypoint control
                    if env.config['gameboard border margin'] < mouse_position[0] < env.config['gameboard size']-env.config['gameboard border margin'] and env.config['gameboard border margin'] < mouse_position[1] < env.config['gameboard size']-env.config['gameboard border margin']:
                        if env.agent_waypoint_clicked:
                            print('Agent waypoint set to %s' % (mouse_position,))
                            agent0_action = mouse_position
                            agent0_policy.waypoint_override = mouse_position
                            actions.append((env.aircraft_ids[0], agent0_action))
                            env.agent_waypoint_clicked = False
                            env.button_latch_dict['waypoint'] = False
                        else:
                            print('Human waypoint set to %s' % (mouse_position,))
                            agent1_action = mouse_position
                            actions.append((env.aircraft_ids[1], agent1_action))

                    # Agent 0 gameplan buttons
                    elif env.target_id_button.is_clicked(mouse_position):
                        agent0_policy.search_type_override = 'target'
                        env.button_latch_dict['target_id'] = True #not env.button_latch_dict['target_id']
                        if env.button_latch_dict['target_id']:
                            env.button_latch_dict['wez_id'], env.button_latch_dict['autonomous'], env.button_latch_dict['hold'] = False, False, False # target id and wez id policies are mutually exclusive
                            env.button_latch_dict['manual_priorities'] = True
                        env.comm_text = 'Beginning target ID'
                        print(env.comm_text)
                        env.add_comm_message(env.comm_text)

                    elif env.wez_id_button.is_clicked(mouse_position):
                        agent0_policy.search_type_override = 'wez'
                        env.button_latch_dict['wez_id'] = True #not env.button_latch_dict['wez_id']
                        if env.button_latch_dict['wez_id']:
                            env.button_latch_dict['target_id'], env.button_latch_dict['autonomous'], env.button_latch_dict['hold']= False, False, False  # target id and wez id policies are mutually exclusive
                            env.button_latch_dict['manual_priorities'] = True
                        env.comm_text = 'Beginning target+WEZ ID'
                        print(env.comm_text)
                        env.add_comm_message(env.comm_text)

                    elif env.hold_button.is_clicked(mouse_position):
                        if not agent0_policy.hold_commanded:
                            agent0_policy.hold_commanded = True
                            env.button_latch_dict['hold'] = True  # not env.button_latch_dict['hold']
                            env.button_latch_dict['autonomous'] = False
                            env.comm_text = 'Holding'

                        else:
                            agent0_policy.hold_commanded = False
                            env.button_latch_dict['hold'] = False  # not env.button_latch_dict['hold']
                            env.comm_text = 'Resuming search'
                            env.button_latch_dict['autonomous'] = True
                        env.add_comm_message(env.comm_text,is_ai=True)

                    elif env.waypoint_button.is_clicked(mouse_position):
                        env.comm_text = 'Moving to waypoint'
                        env.button_latch_dict['waypoint'] = True
                        env.agent_waypoint_clicked = True
                        env.add_comm_message(env.comm_text, is_ai=True)

                    elif env.NW_quad_button.is_clicked(mouse_position) and not env.full_quad_button.is_clicked(mouse_position):
                        agent0_policy.search_quadrant_override = 'NW'
                        env.button_latch_dict['NW'] = True #not env.button_latch_dict['NW']
                        if env.button_latch_dict['NW']:
                            env.button_latch_dict['NE'], env.button_latch_dict['SE'], env.button_latch_dict['SW'], env.button_latch_dict['full'],env.button_latch_dict['autonomous'],env.button_latch_dict['hold'] = False, False, False, False, False, False  # mutually exclusive
                            env.button_latch_dict['manual_priorities'] = True
                        env.comm_text = 'Beginning target ID search in NW'
                        print(env.comm_text + '(Gameplan:')
                        env.add_comm_message(env.comm_text,is_ai=True)

                    elif env.NE_quad_button.is_clicked(mouse_position) and not env.full_quad_button.is_clicked(mouse_position):
                        agent0_policy.search_quadrant_override = 'NE'
                        env.button_latch_dict['NE'] = True #not env.button_latch_dict['NE']
                        if env.button_latch_dict['NE']:
                            env.button_latch_dict['NW'], env.button_latch_dict['SE'], env.button_latch_dict['SW'], env.button_latch_dict['full'], env.button_latch_dict['autonomous'],env.button_latch_dict['hold'] = False,False, False, False, False, False  # mutually exclusive
                            env.button_latch_dict['manual_priorities'] = True
                        env.comm_text = 'Beginning target ID search in NE'
                        print(env.comm_text + '(Gameplan:')
                        env.add_comm_message(env.comm_text,is_ai=True)

                    elif env.SW_quad_button.is_clicked(mouse_position) and not env.full_quad_button.is_clicked(mouse_position):
                        agent0_policy.search_quadrant_override = 'SW'
                        env.button_latch_dict['SW'] = True #not env.button_latch_dict['SW']
                        if env.button_latch_dict['SW']:
                            env.button_latch_dict['NE'], env.button_latch_dict['SE'], env.button_latch_dict['NW'], env.button_latch_dict['full'], env.button_latch_dict['autonomous'],env.button_latch_dict['hold'] = False, False, False, False, False,False  # mutually exclusive
                            env.button_latch_dict['manual_priorities'] = True
                        env.comm_text = 'Beginning target ID search in SW'
                        print(env.comm_text)
                        env.add_comm_message(env.comm_text,is_ai=True)

                    elif env.SE_quad_button.is_clicked(mouse_position) and not env.full_quad_button.is_clicked(mouse_position):
                        agent0_policy.search_quadrant_override = 'SE'
                        env.button_latch_dict['SE'] = True #not env.button_latch_dict['SE']
                        if env.button_latch_dict['SE']:
                            env.button_latch_dict['NE'], env.button_latch_dict['SW'], env.button_latch_dict['NW'], env.button_latch_dict['full'], env.button_latch_dict['autonomous'],env.button_latch_dict['hold'] = False,False, False, False, False, False  # mutually exclusive
                            env.button_latch_dict['manual_priorities'] = True
                        env.comm_text = 'Beginning target ID search in SE'
                        print(env.comm_text)
                        env.add_comm_message(env.comm_text,is_ai=True)
                    elif env.full_quad_button.is_clicked(mouse_position):
                        agent0_policy.search_quadrant_override = 'full'
                        env.button_latch_dict['full'] = True #not env.button_latch_dict['full']
                        if env.button_latch_dict['full']:
                            env.button_latch_dict['NE'], env.button_latch_dict['SW'], env.button_latch_dict['NW'], env.button_latch_dict['SE'], env.button_latch_dict['autonomous'],env.button_latch_dict['hold'] = False, False, False, False, False,False  # mutually exclusive
                            env.button_latch_dict['manual_priorities'] = True
                        env.comm_text = 'Beginning full map search'
                        print(env.comm_text)
                        env.add_comm_message(env.comm_text,is_ai=True)


                    elif env.manual_priorities_button.is_clicked(mouse_position):
                        agent0_policy.search_quadrant_override = 'none'
                        agent0_policy.search_type_override = 'none'
                        env.button_latch_dict['manual_priorities'] = True #not env.button_latch_dict['autonomous']
                        if env.button_latch_dict['manual_priorities']: env.button_latch_dict['autonomous'] = False
                        #env.comm_text = 'Beginning autonomous search'
                        #print(env.comm_text)
                        #env.add_comm_message(env.comm_text,is_ai=True)

                    elif env.autonomous_button.is_clicked(mouse_position):
                        agent0_policy.search_quadrant_override = 'none'
                        agent0_policy.search_type_override = 'none'
                        env.button_latch_dict['autonomous'] = True #not env.button_latch_dict['autonomous']
                        if env.button_latch_dict['autonomous']:
                            env.button_latch_dict['NE'], env.button_latch_dict['SW'], env.button_latch_dict['NW'], env.button_latch_dict['SE'], env.button_latch_dict['full'], env.button_latch_dict['hold'],env.button_latch_dict['target_id'],env.button_latch_dict['wez_id'],env.button_latch_dict['hold'],env.button_latch_dict['manual_priorities'] = False, False, False, False, False, False, False,False,False,False  # mutually exclusive
                        env.comm_text = 'Beginning autonomous search'
                        print(env.comm_text)
                        env.add_comm_message(env.comm_text,is_ai=True)

                    elif env.risk_low_button.is_clicked(mouse_position):
                        agent0_policy.risk_tolerance = 'low'
                        env.button_latch_dict['risk_low'] = True
                        env.button_latch_dict['risk_medium'] = False
                        env.button_latch_dict['risk_high'] = False
                        env.comm_text = 'Setting risk tolerance to LOW'
                        env.add_comm_message(env.comm_text, is_ai=True)

                    elif env.risk_medium_button.is_clicked(mouse_position):
                        agent0_policy.risk_tolerance = 'medium'
                        env.button_latch_dict['risk_low'] = False
                        env.button_latch_dict['risk_medium'] = True
                        env.button_latch_dict['risk_high'] = False
                        env.comm_text = 'Setting risk tolerance to MEDIUM'
                        env.add_comm_message(env.comm_text, is_ai=True)

                    elif env.risk_high_button.is_clicked(mouse_position):
                        agent0_policy.risk_tolerance = 'high'
                        env.button_latch_dict['risk_low'] = False
                        env.button_latch_dict['risk_medium'] = False
                        env.button_latch_dict['risk_high'] = True
                        env.comm_text = 'Setting risk tolerance to HIGH'
                        env.add_comm_message(env.comm_text, is_ai=True)

                    elif env.pause_button.is_clicked(mouse_position):
                        env.pause(pygame.MOUSEBUTTONDOWN)

                    elif env.quit_button.is_clicked(mouse_position):
                        pygame.quit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        env.pause(pygame.K_SPACE)

            state, reward, done, _ = env.step(actions)  # step through the environment
            # update agent policy here if desired, note that you can use env.observation_space and env.action_space instead of the dictionary format
            if render:  # if in PyGame mode, render the environment
                env.render()

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
            pygame.quit()

    print("DONE!")