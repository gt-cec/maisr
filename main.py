from agents import *
import sys
import os
import ctypes
import sys

from env import MAISREnv
from gui import *
from utility.data_logging import GameLogger, load_env_config
#from config import subject_id, user_group, log_data, x, y, config_dict, run_order
#from config import x, y, config_dict, run_order
import config
from autonomous_policy import AutonomousPolicy
import webbrowser


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Missing args, run as: python main.py subject_id user_group run_order starting_round_number log_data")
        sys.exit()

    subject_id = sys.argv[1]
    if not subject_id.isdigit():
        print("Invalid subject ID: >" + subject_id + "<")
        sys.exit()

    user_group = sys.argv[2]
    if user_group not in ["test", "card", "control", "in-situ", "transparency"]:
        print("Invalid user group: >" + user_group + "<")
        sys.exit()

    run_order = sys.argv[3]
    if run_order not in ["1", "2", "3", "4"] or not run_order.isdigit():
        print("Invalid run order: " + ">" + run_order + "<")
        sys.exit()
    run_order = int(run_order)

    round_number = sys.argv[4]
    if round_number not in ["0", "1", "2", "3", "4"] or not round_number.isdigit():
        print("Invalid round number: " + ">" + round_number + "<")
        sys.exit()
    round_number = int(round_number)

    log_data = sys.argv[5]
    if log_data == 'y':
        log_data = True
    elif log_data == 'n':
        log_data = False
    else:
        print("Invalid input for log data")
        sys.exit()

    print(f"\nStarting MAISR environment (subject_id = {subject_id}, group = {user_group}, run_order = {run_order}, data logging = {log_data})")
    render = "headless" not in sys.argv

    #update run order in config.py and create local copies of other variables in config.py
    config.run_order = run_order
    x = config.x
    y = config.y
    config_dict = config.get_config_dict()
    config_list = config_dict[user_group]
    #print("For run order "+ str(run_order) + " config list "+ str(config_list))
    total_games = 5 # Number of games to run
    game_count = 0 # Used to track how many games have been completed so far

    while round_number < total_games:
        config = config_list[round_number]
        env_config = load_env_config(config)
        gameplan_command_history = []  # For data logging

        if log_data:
            game_logger = GameLogger(subject_id, config,user_group,round_number,run_order)
            game_logger.initial_log()

        if render:
            print("Starting in PyGame mode")
            pygame.init()
            clock = pygame.time.Clock()
            ctypes.windll.user32.SetProcessDPIAware()  # Disables display scaling so the game fits on small, high-res monitors
            window_width, window_height = env_config['window size'][0], env_config['window size'][1]
            os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"
            window = pygame.display.set_mode((window_width, window_height),flags=pygame.NOFRAME)
            env = MAISREnv(env_config, window, clock=clock, render=True,subject_id=subject_id,user_group=user_group,round_number=round_number, run_order=run_order)

        else:
            print("Starting in headless mode")
            pygame.init()  # init pygame
            clock = pygame.time.Clock()
            pygame.font.init()
            env = MAISREnv(env_config, None, render=False)

        agent0_id = env.num_ships  # Hack to dynamically get agent IDs
        agent0_policy = AutonomousPolicy(env, agent0_id)
        agent0_policy.show_low_level_goals,agent0_policy.show_high_level_goals, agent0_policy.show_high_level_rationale,agent0_policy.show_tracked_factors = env.config['show_low_level_goals'], env.config['show_high_level_goals'], env.config['show_high_level_rationale'], env.config['show_tracked_factors']

        game_count += 1
        state = env.reset()  # reset the environment
        done = False  # flag for when the run is complete
        agent0_waypoint = (0,0)
        agent1_waypoint = (0, 0)

        agent_log_info = {
            'waypoint': 'None',
            'priority mode': 'None',
            'search type': 'None',
            'search area': 'None'
        }

        while not done:  # main game loop
            agent_log_info = {
                'waypoint':agent0_waypoint,
                'priority mode': 'hold' if agent0_policy.hold_commanded else 'waypoint override' if agent0_policy.waypoint_override else 'manual' if env.button_latch_dict['manual_priorities'] else 'auto',
                'search type': agent0_policy.search_type,
                'search area': agent0_policy.search_quadrant,
            }

            if log_data:
                game_logger.log_state(env, env.display_time,agent1_waypoint,agent_log_info)
                if env.new_target_id is not None:
                    game_logger.log_target_id(env.new_target_id[0],env.new_target_id[1],env.new_target_id[2],env.display_time)
                    env.new_target_id = None

                if env.new_weapon_id is not None:
                    game_logger.log_target_id(env.new_weapon_id[0], env.new_weapon_id[1], env.new_weapon_id[2],env.display_time)
                    env.new_weapon_id = None

            actions = [] # use agent policies to get actions as a list of tuple [(agent index, waypoint)]
            time_sec = float(env.display_time)/1000

            if env.regroup_clicked:
                agent_human_distance = math.hypot(env.agents[env.num_ships].x - env.agents[env.human_idx].x,env.agents[env.num_ships].y - env.agents[env.human_idx].y)
                if agent_human_distance <= 50: env.regroup_clicked = False

            # Agent 0: Act based on currently selected gameplan
            if env.regroup_clicked:
                actions.append((env.
                                _ids[0], (env.agents[env.human_idx].x,env.agents[env.human_idx].y)))
            else:
                agent0_policy.act()
                agent0_waypoint = agent0_policy.target_point
                actions.append((env.aircraft_ids[0], agent0_policy.target_point))
                agent0_policy.update_agent_info()

            # Handle SAGAT surveys
            if 64.00 < time_sec < 65.00 and not env.survey1_launched and env.config['surveys_enabled']:
                env.survey1_launched = True
                if log_data: game_logger.log_state(env, env.display_time,agent1_waypoint,agent_log_info)
                env.SAGAT_survey(1)
            if 119.00+10 < time_sec < 120.00+10 and not env.survey2_launched and env.config['surveys_enabled']:
                env.survey2_launched = True
                if log_data: game_logger.log_state(env, env.display_time,agent1_waypoint,agent_log_info)
                env.SAGAT_survey(2)
            if 179.0-5 < time_sec < 180.0-5 and not env.survey3_launched and env.config['surveys_enabled']:
                env.survey3_launched = True
                if log_data: game_logger.log_state(env, env.display_time,agent1_waypoint,agent_log_info)
                env.SAGAT_survey(3)
            # if 239.0+5 < time_sec < 240.0+5 and not env.survey4_launched and env.config['surveys_enabled']:
            #     env.survey4_launched = True
            #     if log_data: game_logger.log_state(env, env.display_time,agent1_waypoint,agent_log_info)
            #     env.SAGAT_survey(4)

            # Handle mouse clicks
            ev = pygame.event.get()
            for event in ev:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F1:
                        if log_data: game_logger.final_log(gameplan_command_history, env)
                        pygame.quit()
                    if event.key == pygame.K_SPACE: env.pause(pygame.MOUSEBUTTONDOWN)

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_position = pygame.mouse.get_pos()

                    if env.config['gameboard border margin'] < mouse_position[0] < env.config['gameboard size']-env.config['gameboard border margin'] and env.config['gameboard border margin'] < mouse_position[1] < env.config['gameboard size']-env.config['gameboard border margin']:
                        if env.agent_waypoint_clicked:
                            if log_data:
                                game_logger.log_mouse_event(mouse_position,"waypoint override",env.display_time)
                                gameplan_command_history.append([time_sec,'waypoint override',mouse_position])
                            #print('Agent waypoint set to %s' % (mouse_position,))
                            env.comm_text = 'Moving to waypoint'
                            env.add_comm_message(env.comm_text, is_ai=True)
                            agent0_action = mouse_position
                            agent0_policy.waypoint_override = mouse_position
                            actions.append((env.aircraft_ids[0], agent0_action))
                            if agent0_policy.hold_commanded:
                                agent0_policy.hold_commanded = False
                                env.button_latch_dict['hold'] = False
                            env.agent_waypoint_clicked = False
                            env.button_latch_dict['waypoint'] = False

                        else: # Set human waypoint
                            if log_data: game_logger.log_mouse_event(mouse_position,"human waypoint",env.display_time)
                            actions.append((env.aircraft_ids[-1], mouse_position))
                            agent1_waypoint = mouse_position

                    # Agent 0 gameplan buttons
                    elif env.target_id_button.is_clicked(mouse_position):
                        if log_data: game_logger.log_mouse_event(mouse_position,"target id",env.display_time)
                        agent0_policy.search_type_override = 'target'
                        agent0_policy.waypoint_override = False
                        #if env.button_latch_dict['target_id'] == False:
                        gameplan_command_history.append([time_sec, 'target_id'])

                        env.button_latch_dict['target_id'] = True
                        if env.button_latch_dict['target_id']:
                            env.button_latch_dict['wez_id'], env.button_latch_dict['auto_type'], env.button_latch_dict['hold'] = False, False, False # target id and wez id policies are mutually exclusive
                            env.button_latch_dict['manual_priorities'] = True
                            if env.show_high_level_goals:
                                agent0_policy.update_show_agent_search_type(show_agent_search_type=True)

                        if agent0_policy.hold_commanded:
                            agent0_policy.hold_commanded = False
                            env.button_latch_dict['hold'] = False

                        env.comm_text = 'Beginning target ID'
                        #print(env.comm_text)
                        env.add_comm_message(env.comm_text)

                    elif env.wez_id_button.is_clicked(mouse_position):
                        if log_data: game_logger.log_mouse_event(mouse_position,"wez id",env.display_time)
                        agent0_policy.search_type_override = 'wez'
                        agent0_policy.waypoint_override = False
                        #if env.button_latch_dict['wez_id'] == False:
                        gameplan_command_history.append([time_sec, 'wez_id'])

                        env.button_latch_dict['wez_id'] = True
                        if env.button_latch_dict['wez_id']:
                            env.button_latch_dict['target_id'], env.button_latch_dict['auto_type'], env.button_latch_dict['hold']= False, False, False  # target id and wez id policies are mutually exclusive
                            env.button_latch_dict['manual_priorities'] = True
                            if env.show_high_level_goals:
                                agent0_policy.update_show_agent_search_type(show_agent_search_type=True)

                        if agent0_policy.hold_commanded:
                            agent0_policy.hold_commanded = False
                            env.button_latch_dict['hold'] = False

                        env.comm_text = 'Beginning WEZ+target ID'
                        env.add_comm_message(env.comm_text)

                    elif env.auto_type_button.is_clicked(mouse_position):
                        if log_data: game_logger.log_mouse_event(mouse_position,"auto type id",env.display_time)
                        agent0_policy.search_type_override = 'none'
                        agent0_policy.waypoint_override = False
                        #if env.button_latch_dict['wez_id'] == False:
                        gameplan_command_history.append([time_sec, 'auto_type'])

                        env.button_latch_dict['auto_type'] = True 
                        if env.button_latch_dict['auto_type']:
                            env.button_latch_dict['target_id'], env.button_latch_dict['wez_id'], env.button_latch_dict['hold']= False, False, False  # target id and wez id policies are mutually exclusive
                            env.button_latch_dict['manual_priorities'] = False
                            # if show high level goals
                            if not env.show_high_level_goals:
                                agent0_policy.update_show_agent_search_type(show_agent_search_type=False)

                        if agent0_policy.hold_commanded:
                            agent0_policy.hold_commanded = False
                            env.button_latch_dict['hold'] = False

                        env.comm_text = 'Beginning auto type ID'
                        env.add_comm_message(env.comm_text)

                    # elif env.regroup_button.is_clicked(mouse_position):
                    #     if not env.regroup_clicked:
                    #         gameplan_command_history.append([time_sec, 'regroup'])  # For data logging
                    #         agent0_policy.waypoint_override = False
                    #         env.regroup_clicked = True
                    #     else: env.regroup_clicked = False
                    #
                    # elif env.tag_team_button.is_clicked(mouse_position):
                    #     if not env.tag_team_commanded:
                    #         env.tag_team_commanded = True
                    #         agent0_policy.search_type_override = 'tag team'
                    #         agent0_policy.waypoint_override = False
                    #         env.button_latch_dict['tag_team'] = True
                    #         env.button_latch_dict['autonomous'],env.button_latch_dict['manual_priorities'] = False, False
                    #     else:
                    #         agent0_policy.search_type_override = 'none'
                    #         env.tag_team_commanded = False
                    #         env.button_latch_dict['tag_team'] = False
                    #         env.button_latch_dict['autonomous'] = True

                    # elif env.fan_out_button.is_clicked(mouse_position):
                    #     if not env.fan_out_commanded:
                    #         env.fan_out_commanded = True
                    #         agent0_policy.waypoint_override = False
                    #         agent0_policy.search_type_override = 'fan out'
                    #         env.button_latch_dict['fan_out'] = True
                    #         env.button_latch_dict['autonomous'],env.button_latch_dict['manual_priorities'] = False, False
                    #     else:
                    #         agent0_policy.search_type_override = 'none'
                    #         env.fan_out_commanded = False
                    #         env.button_latch_dict['fan_out'] = False
                    #         env.button_latch_dict['autonomous'] = True

                    # elif env.hold_button.is_clicked(mouse_position):
                    #     if not agent0_policy.hold_commanded:
                    #         if log_data: game_logger.log_mouse_event(mouse_position,"hold",env.display_time)
                    #         agent0_policy.hold_commanded = True
                    #         agent0_policy.waypoint_override = False
                    #         gameplan_command_history.append([time_sec, 'hold'])
                    #         env.button_latch_dict['hold'] = True  # not env.button_latch_dict['hold']
                    #         env.button_latch_dict['autonomous'] = False
                    #         env.comm_text = 'Holding'

                    #     else:
                    #         agent0_policy.hold_commanded = False
                    #         env.button_latch_dict['hold'] = False  # not env.button_latch_dict['hold']
                    #         env.comm_text = 'Resuming search'
                    #         env.button_latch_dict['autonomous'] = True
                    #     env.add_comm_message(env.comm_text,is_ai=True)

                    # elif env.waypoint_button.is_clicked(mouse_position):
                    #     env.button_latch_dict['waypoint'] = True
                    #     env.agent_waypoint_clicked = True
                    #     #gameplan_command_history.append([time_sec, 'waypoint'])

                    elif env.NW_quad_button.is_clicked(mouse_position): #and not env.full_quad_button.is_clicked(mouse_position):
                        if log_data: game_logger.log_mouse_event(mouse_position,"quadrant - NW",env.display_time)
                        agent0_policy.search_quadrant_override = 'NW'
                        agent0_policy.waypoint_override = False
                        #if not env.button_latch_dict['NW']:
                        gameplan_command_history.append([time_sec, 'NW']) # For data logging

                        env.button_latch_dict['NW'] = True #not env.button_latch_dict['NW']
                        if env.button_latch_dict['NW']:
                            env.button_latch_dict['NE'], env.button_latch_dict['SE'], env.button_latch_dict['SW'], env.button_latch_dict['full'],env.button_latch_dict['auto_area'],env.button_latch_dict['hold'] = False, False, False, False, False, False  # mutually exclusive
                            env.button_latch_dict['manual_priorities'] = True
                        if agent0_policy.hold_commanded:
                            agent0_policy.hold_commanded = False
                            env.button_latch_dict['hold'] = False
                        env.comm_text = 'Prioritizing NW quadrant'
                        env.add_comm_message(env.comm_text,is_ai=True)

                    elif env.NE_quad_button.is_clicked(mouse_position): #and not env.full_quad_button.is_clicked(mouse_position):
                        if log_data: game_logger.log_mouse_event(mouse_position,"quadrant - NE",env.display_time)
                        agent0_policy.search_quadrant_override = 'NE'
                        agent0_policy.waypoint_override = False
                        #if not env.button_latch_dict['NE']:
                        gameplan_command_history.append([time_sec, 'NE'])  # For data logging

                        env.button_latch_dict['NE'] = True #not env.button_latch_dict['NE']
                        if env.button_latch_dict['NE']:
                            env.button_latch_dict['NW'], env.button_latch_dict['SE'], env.button_latch_dict['SW'], env.button_latch_dict['full'], env.button_latch_dict['auto_area'],env.button_latch_dict['hold'] = False,False, False, False, False, False  # mutually exclusive
                            env.button_latch_dict['manual_priorities'] = True
                        if agent0_policy.hold_commanded:
                            agent0_policy.hold_commanded = False
                            env.button_latch_dict['hold'] = False
                        env.comm_text = 'Prioritizing NE quadrant'
                        env.add_comm_message(env.comm_text,is_ai=True)

                    elif env.SW_quad_button.is_clicked(mouse_position): #and not env.full_quad_button.is_clicked(mouse_position):
                        if log_data: game_logger.log_mouse_event(mouse_position,"quadrant - SW",env.display_time)
                        agent0_policy.search_quadrant_override = 'SW'
                        agent0_policy.waypoint_override = False
                        #if not env.button_latch_dict['SW']:
                        gameplan_command_history.append([time_sec, 'SW'])  # For data logging

                        env.button_latch_dict['SW'] = True #not env.button_latch_dict['SW']
                        if env.button_latch_dict['SW']:
                            env.button_latch_dict['NE'], env.button_latch_dict['SE'], env.button_latch_dict['NW'], env.button_latch_dict['full'], env.button_latch_dict['auto_area'],env.button_latch_dict['hold'] = False, False, False, False, False,False  # mutually exclusive
                            env.button_latch_dict['manual_priorities'] = True
                        if agent0_policy.hold_commanded:
                            agent0_policy.hold_commanded = False
                            env.button_latch_dict['hold'] = False
                        env.comm_text = 'Prioritizing SW quadrant'
                        env.add_comm_message(env.comm_text,is_ai=True)

                    elif env.SE_quad_button.is_clicked(mouse_position): #and not env.full_quad_button.is_clicked(mouse_position):
                        if log_data: game_logger.log_mouse_event(mouse_position,"quadrant - SE",env.display_time)
                        agent0_policy.search_quadrant_override = 'SE'
                        agent0_policy.waypoint_override = False
                        #if not env.button_latch_dict['SE']:
                        gameplan_command_history.append([time_sec, 'SE'])  # For data logging

                        env.button_latch_dict['SE'] = True #not env.button_latch_dict['SE']
                        if env.button_latch_dict['SE']:
                            env.button_latch_dict['NE'], env.button_latch_dict['SW'], env.button_latch_dict['NW'], env.button_latch_dict['full'], env.button_latch_dict['auto_area'],env.button_latch_dict['hold'] = False,False, False, False, False, False  # mutually exclusive
                            env.button_latch_dict['manual_priorities'] = True
                        if agent0_policy.hold_commanded:
                            agent0_policy.hold_commanded = False
                            env.button_latch_dict['hold'] = False
                        env.comm_text = 'Prioritizing SE quadrant'
                        env.add_comm_message(env.comm_text,is_ai=True)

                    # elif env.full_quad_button.is_clicked(mouse_position):
                    #     if log_data: game_logger.log_mouse_event(mouse_position,"quadrant - full",env.display_time)
                    #     agent0_policy.search_quadrant_override = 'full'
                    #     agent0_policy.waypoint_override = False
                    #     #if not env.button_latch_dict['full']:
                    #     gameplan_command_history.append([time_sec, 'full'])  # For data logging
                    #     env.button_latch_dict['full'] = True
                    #     if env.button_latch_dict['full']:
                    #         env.button_latch_dict['NE'], env.button_latch_dict['SW'], env.button_latch_dict['NW'], env.button_latch_dict['SE'], env.button_latch_dict['autonomous'],env.button_latch_dict['hold'] = False, False, False, False, False,False  # mutually exclusive
                    #         env.button_latch_dict['manual_priorities'] = True
                    #     if agent0_policy.hold_commanded:
                    #         agent0_policy.hold_commanded = False
                    #         env.button_latch_dict['hold'] = False
                    #     env.comm_text = 'Prioritizing full map'
                    #     #print(env.comm_text)
                    #     env.add_comm_message(env.comm_text,is_ai=True)


                    # elif env.manual_priorities_button.is_clicked(mouse_position):
                    #     if log_data: game_logger.log_mouse_event(mouse_position,"manual priorities",env.display_time)
                    #     agent0_policy.search_quadrant_override = 'none'
                    #     agent0_policy.waypoint_override = False
                    #     #if not env.button_latch_dict['manual_priorities']:
                    #     gameplan_command_history.append([time_sec, 'manual_priorities']) # For data logging
                    #     agent0_policy.search_type_override = 'none'
                    #     env.button_latch_dict['manual_priorities'] = True #not env.button_latch_dict['autonomous']
                    #     if env.button_latch_dict['manual_priorities']: env.button_latch_dict['autonomous'] = False
                    #     if agent0_policy.hold_commanded:
                    #         agent0_policy.hold_commanded = False
                    #         env.button_latch_dict['hold'] = False
                    #     #env.comm_text = 'Beginning manual search'
                    #     #print(env.comm_text)
                    #     #env.add_comm_message(env.comm_text,is_ai=True)

                    elif env.auto_area_button.is_clicked(mouse_position):
                        if log_data: game_logger.log_mouse_event(mouse_position,"autonomous area",env.display_time)
                        agent0_policy.search_quadrant_override = 'none'
                        agent0_policy.search_type_override = 'none'
                        agent0_policy.waypoint_override = False
                        #if not env.button_latch_dict['autonomous']:
                        gameplan_command_history.append([time_sec, 'autonomous area'])  # For data logging
                        env.button_latch_dict['auto_area'] = True #not env.button_latch_dict['autonomous']
                        if env.button_latch_dict['auto_area']:
                            env.button_latch_dict['NE'], env.button_latch_dict['SW'], env.button_latch_dict['NW'], env.button_latch_dict['SE'], env.button_latch_dict['full'], env.button_latch_dict['hold'],env.button_latch_dict['target_id'],env.button_latch_dict['wez_id'],env.button_latch_dict['hold'],env.button_latch_dict['manual_priorities'] = False, False, False, False, False, False, False,False,False,False  # mutually exclusive
                        if agent0_policy.hold_commanded:
                            agent0_policy.hold_commanded = False
                            env.button_latch_dict['hold'] = False
                        env.comm_text = 'Beginning autonomous area'
                        #print(env.comm_text)
                        env.add_comm_message(env.comm_text,is_ai=True)


            if env.init or pygame.time.get_ticks() > env.start_countdown_time:
                state, reward, done, _ = env.step(actions)  # step through the environment
            if env.init: env.init = False

            # update agent policy here if desired, note that you can use env.observation_space and env.action_space instead of the dictionary format
            if render:  # if in PyGame mode, render the environment
                env.render(quadrant=agent0_policy.search_quadrant, type=agent0_policy.search_type)
                env.agents[agent0_id].waypoint_type = agent0_policy.search_type
                

        if done:
            done_time = pygame.time.get_ticks()
            if log_data:
                game_logger.log_state(env, env.display_time,agent1_waypoint,agent_log_info)
                game_logger.final_log(gameplan_command_history, env)

            waiting_for_key = True
            while waiting_for_key:
                env.render()  # Keep rendering while waiting
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                        if (pygame.time.get_ticks()-done_time)/1000 > 1.5: # Cooldown to prevent accidentally clicking continue
                            waiting_for_key = False
                            break
                    elif event.type == pygame.QUIT:
                        if log_data: game_logger.final_log(gameplan_command_history, env)
                        pygame.quit()
                        sys.exit()

        round_number += 1
        print("Game complete:", game_count)
        if render:
            pygame.quit()

    print("ALL GAMES COMPLETE")
    webbrowser.open_new_tab('https://gatech.co1.qualtrics.com/jfe/form/SV_01m24RxdV29Y9pk?subject_id='+str(subject_id)+'&user_group='+str(user_group)+'&run_order='+str(run_order)+'&round_number='+str(round_number))