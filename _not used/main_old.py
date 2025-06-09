import os
import ctypes

from env import MAISREnv
from utility.gui import *
from utility.data_logging import GameLogger, load_env_config
from config import x, y, config_dict, run_order
from autonomous_policy import AutonomousPolicy

if __name__ == "__main__":

    if len(sys.argv) == 1:
        print('No args specified, loading parameters from config.py')
        from config import subject_id, user_group, log_data, x, y, round_number

    elif len(sys.argv) < 5:
        print("Missing args, run as: python main.py subject_id user_group starting_round_number log_data")
        sys.exit()

    else:
        subject_id = sys.argv[1]
        user_group = sys.argv[2]
        round_number = sys.argv[3]
        log_data = sys.argv[4]

    if not subject_id.isdigit():
        print("Invalid subject ID: >" + subject_id + "<")
        sys.exit()

    if user_group not in ["test", "card", "control", "in-situ"]:
        print("Invalid user group: >" + user_group + "<")
        sys.exit()

    if round_number not in ["0", "1", "2", "3", "4"] or not round_number.isdigit():
        print("Invalid round number: " + ">" + round_number + "<")
        sys.exit()
    round_number = int(round_number)

    if log_data == 'y':
        log_data = True
    elif log_data == 'n':
        log_data = False
    else:
        print("Invalid input for log data")
        sys.exit()

    print(f"\nStarting MAISR environment (subject_id = {subject_id}, group = {user_group}, data logging = {log_data})")
    render = "headless" not in sys.argv

    config_list = config_dict[user_group]
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
            env = MAISREnv(env_config, window, clock=clock, render=True,subject_id=subject_id,user_group=user_group,round_number=round_number)

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

        agent_log_info = {'waypoint': 'None', 'priority mode': 'None', 'search type': 'None', 'search area': 'None'}

        while not done:  # main game loop
            agent_log_info = {
                'waypoint':agent0_waypoint, 'search type': agent0_policy.search_type, 'search area': agent0_policy.search_quadrant,
                'priority mode': 'hold' if agent0_policy.hold_commanded else 'waypoint override' if agent0_policy.waypoint_override else 'manual' if env.button_latch_dict['manual_priorities'] else 'auto',}

            if log_data:
                game_logger.log_state(env, env.display_time,agent1_waypoint,agent_log_info)
                if env.new_target_id:
                    game_logger.log_target_id(env.new_target_id[0],env.new_target_id[1],env.new_target_id[2],env.display_time)
                    env.new_target_id = None

                if env.new_weapon_id:
                    game_logger.log_target_id(env.new_weapon_id[0], env.new_weapon_id[1], env.new_weapon_id[2],env.display_time)
                    env.new_weapon_id = None

            # Handle SAGAT surveys
            # if 64.00 < time_sec < 65.00 and not env.survey1_launched and env.config['surveys_enabled']:
            #     env.survey1_launched = True
            #     if log_data: game_logger.log_state(env, env.display_time,agent1_waypoint,agent_log_info)
            #     env.SAGAT_survey(1)
            # if 119.00+5 < time_sec < 120.00+5 and not env.survey2_launched and env.config['surveys_enabled']:
            #     env.survey2_launched = True
            #     if log_data: game_logger.log_state(env, env.display_time,agent1_waypoint,agent_log_info)
            #     env.SAGAT_survey(2)
            # if 179.0+5 < time_sec < 180.0+5 and not env.survey3_launched and env.config['surveys_enabled']:
            #     env.survey3_launched = True
            #     if log_data: game_logger.log_state(env, env.display_time,agent1_waypoint,agent_log_info)
            #     env.SAGAT_survey(3)

            # Handle agent actions (from autonomous_policy)
            actions = []  # use agent policies to get actions as a list of tuple [(agent index, waypoint)]

            agent0_policy.act() # Calculate agent's target waypoint
            agent0_waypoint = agent0_policy.target_point
            actions.append((env.aircraft_ids[0], agent0_policy.target_point))
            agent0_policy.update_agent_info()

            # Handle human actions (mouse clicks)
            time_sec = float(env.display_time) / 1000
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
                        gameplan_command_history.append([time_sec, 'target_id'])

                        env.button_latch_dict['target_id'] = True
                        if env.button_latch_dict['target_id']:
                            env.button_latch_dict['wez_id'], env.button_latch_dict['autonomous'], env.button_latch_dict['hold'] = False, False, False # target id and wez id policies are mutually exclusive
                            env.button_latch_dict['manual_priorities'] = True

                        if agent0_policy.hold_commanded:
                            agent0_policy.hold_commanded = False
                            env.button_latch_dict['hold'] = False

                        env.comm_text = 'Beginning target ID'
                        env.add_comm_message(env.comm_text)

                    elif env.wez_id_button.is_clicked(mouse_position):
                        if log_data: game_logger.log_mouse_event(mouse_position,"wez id",env.display_time)
                        agent0_policy.search_type_override = 'wez'
                        agent0_policy.waypoint_override = False
                        gameplan_command_history.append([time_sec, 'wez_id'])

                        env.button_latch_dict['wez_id'] = True #not env.button_latch_dict['wez_id']
                        if env.button_latch_dict['wez_id']:
                            env.button_latch_dict['target_id'], env.button_latch_dict['autonomous'], env.button_latch_dict['hold']= False, False, False  # target id and wez id policies are mutually exclusive
                            env.button_latch_dict['manual_priorities'] = True

                        if agent0_policy.hold_commanded:
                            agent0_policy.hold_commanded = False
                            env.button_latch_dict['hold'] = False

                        env.comm_text = 'Beginning target+WEZ ID'
                        env.add_comm_message(env.comm_text)

                    elif env.hold_button.is_clicked(mouse_position):
                        if not agent0_policy.hold_commanded:
                            if log_data: game_logger.log_mouse_event(mouse_position,"hold",env.display_time)
                            agent0_policy.hold_commanded = True
                            agent0_policy.waypoint_override = False
                            gameplan_command_history.append([time_sec, 'hold'])
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
                        env.button_latch_dict['waypoint'] = True
                        env.agent_waypoint_clicked = True

                    elif env.NW_quad_button.is_clicked(mouse_position) and not env.full_quad_button.is_clicked(mouse_position):
                        if log_data: game_logger.log_mouse_event(mouse_position,"quadrant - NW",env.display_time)
                        agent0_policy.search_quadrant_override = 'NW'
                        agent0_policy.waypoint_override = False
                        gameplan_command_history.append([time_sec, 'NW']) # For data logging

                        env.button_latch_dict['NW'] = True #not env.button_latch_dict['NW']
                        if env.button_latch_dict['NW']:
                            env.button_latch_dict['NE'], env.button_latch_dict['SE'], env.button_latch_dict['SW'], env.button_latch_dict['full'],env.button_latch_dict['autonomous'],env.button_latch_dict['hold'] = False, False, False, False, False, False  # mutually exclusive
                            env.button_latch_dict['manual_priorities'] = True
                        if agent0_policy.hold_commanded:
                            agent0_policy.hold_commanded = False
                            env.button_latch_dict['hold'] = False
                        env.comm_text = 'Prioritizing NW quadrant'
                        env.add_comm_message(env.comm_text,is_ai=True)

                    elif env.NE_quad_button.is_clicked(mouse_position) and not env.full_quad_button.is_clicked(mouse_position):
                        if log_data: game_logger.log_mouse_event(mouse_position,"quadrant - NE",env.display_time)
                        agent0_policy.search_quadrant_override = 'NE'
                        agent0_policy.waypoint_override = False
                        gameplan_command_history.append([time_sec, 'NE'])  # For data logging

                        env.button_latch_dict['NE'] = True #not env.button_latch_dict['NE']
                        if env.button_latch_dict['NE']:
                            env.button_latch_dict['NW'], env.button_latch_dict['SE'], env.button_latch_dict['SW'], env.button_latch_dict['full'], env.button_latch_dict['autonomous'],env.button_latch_dict['hold'] = False,False, False, False, False, False  # mutually exclusive
                            env.button_latch_dict['manual_priorities'] = True
                        if agent0_policy.hold_commanded:
                            agent0_policy.hold_commanded = False
                            env.button_latch_dict['hold'] = False
                        env.comm_text = 'Prioritizing NE quadrant'
                        env.add_comm_message(env.comm_text,is_ai=True)

                    elif env.SW_quad_button.is_clicked(mouse_position) and not env.full_quad_button.is_clicked(mouse_position):
                        if log_data: game_logger.log_mouse_event(mouse_position,"quadrant - SW",env.display_time)
                        agent0_policy.search_quadrant_override = 'SW'
                        agent0_policy.waypoint_override = False
                        gameplan_command_history.append([time_sec, 'SW'])  # For data logging

                        env.button_latch_dict['SW'] = True #not env.button_latch_dict['SW']
                        if env.button_latch_dict['SW']:
                            env.button_latch_dict['NE'], env.button_latch_dict['SE'], env.button_latch_dict['NW'], env.button_latch_dict['full'], env.button_latch_dict['autonomous'],env.button_latch_dict['hold'] = False, False, False, False, False,False  # mutually exclusive
                            env.button_latch_dict['manual_priorities'] = True
                        if agent0_policy.hold_commanded:
                            agent0_policy.hold_commanded = False
                            env.button_latch_dict['hold'] = False
                        env.comm_text = 'Prioritizing SW quadrant'
                        env.add_comm_message(env.comm_text,is_ai=True)

                    elif env.SE_quad_button.is_clicked(mouse_position) and not env.full_quad_button.is_clicked(mouse_position):
                        if log_data: game_logger.log_mouse_event(mouse_position,"quadrant - SE",env.display_time)
                        agent0_policy.search_quadrant_override = 'SE'
                        agent0_policy.waypoint_override = False
                        gameplan_command_history.append([time_sec, 'SE'])  # For data logging

                        env.button_latch_dict['SE'] = True #not env.button_latch_dict['SE']
                        if env.button_latch_dict['SE']:
                            env.button_latch_dict['NE'], env.button_latch_dict['SW'], env.button_latch_dict['NW'], env.button_latch_dict['full'], env.button_latch_dict['autonomous'],env.button_latch_dict['hold'] = False,False, False, False, False, False  # mutually exclusive
                            env.button_latch_dict['manual_priorities'] = True
                        if agent0_policy.hold_commanded:
                            agent0_policy.hold_commanded = False
                            env.button_latch_dict['hold'] = False
                        env.comm_text = 'Prioritizing SE quadrant'
                        env.add_comm_message(env.comm_text,is_ai=True)

                    elif env.full_quad_button.is_clicked(mouse_position):
                        if log_data: game_logger.log_mouse_event(mouse_position,"quadrant - full",env.display_time)
                        agent0_policy.search_quadrant_override = 'full'
                        agent0_policy.waypoint_override = False
                        gameplan_command_history.append([time_sec, 'full'])  # For data logging
                        env.button_latch_dict['full'] = True
                        if env.button_latch_dict['full']:
                            env.button_latch_dict['NE'], env.button_latch_dict['SW'], env.button_latch_dict['NW'], env.button_latch_dict['SE'], env.button_latch_dict['autonomous'],env.button_latch_dict['hold'] = False, False, False, False, False,False  # mutually exclusive
                            env.button_latch_dict['manual_priorities'] = True
                        if agent0_policy.hold_commanded:
                            agent0_policy.hold_commanded = False
                            env.button_latch_dict['hold'] = False
                        env.comm_text = 'Prioritizing full map'
                        env.add_comm_message(env.comm_text,is_ai=True)


                    elif env.manual_priorities_button.is_clicked(mouse_position):
                        if log_data: game_logger.log_mouse_event(mouse_position,"manual priorities",env.display_time)
                        agent0_policy.search_quadrant_override = 'none'
                        agent0_policy.waypoint_override = False
                        #if not env.button_latch_dict['manual_priorities']:
                        gameplan_command_history.append([time_sec, 'manual_priorities']) # For data logging
                        agent0_policy.search_type_override = 'none'
                        env.button_latch_dict['manual_priorities'] = True #not env.button_latch_dict['autonomous']
                        if env.button_latch_dict['manual_priorities']: env.button_latch_dict['autonomous'] = False
                        if agent0_policy.hold_commanded:
                            agent0_policy.hold_commanded = False
                            env.button_latch_dict['hold'] = False

                    elif env.autonomous_button.is_clicked(mouse_position):
                        if log_data: game_logger.log_mouse_event(mouse_position,"autonomous",env.display_time)
                        agent0_policy.search_quadrant_override = 'none'
                        agent0_policy.search_type_override = 'none'
                        agent0_policy.waypoint_override = False
                        gameplan_command_history.append([time_sec, 'autonomous'])  # For data logging
                        env.button_latch_dict['autonomous'] = True
                        if env.button_latch_dict['autonomous']:
                            env.button_latch_dict['NE'], env.button_latch_dict['SW'], env.button_latch_dict['NW'], env.button_latch_dict['SE'], env.button_latch_dict['full'], env.button_latch_dict['hold'],env.button_latch_dict['target_id'],env.button_latch_dict['wez_id'],env.button_latch_dict['hold'],env.button_latch_dict['manual_priorities'] = False, False, False, False, False, False, False,False,False,False  # mutually exclusive
                        if agent0_policy.hold_commanded:
                            agent0_policy.hold_commanded = False
                            env.button_latch_dict['hold'] = False
                        env.comm_text = 'Beginning autonomous search'
                        env.add_comm_message(env.comm_text,is_ai=True)


            if env.init or pygame.time.get_ticks() > env.start_countdown_time:
                state, reward, done, _ = env.step(actions)  # step through the environment

            if env.init: env.init = False
            if render: env.render()

        if done:
            done_time = pygame.time.get_ticks()
            if log_data:
                game_logger.log_state(env, env.display_time,agent1_waypoint,agent_log_info)
                game_logger.final_log(gameplan_command_history, env)

            if render:
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
    #webbrowser.open_new_tab('https://gatech.co1.qualtrics.com/jfe/form/SV_3ITZnNRRBqioKR8?subject_id='+str(subject_id)+'&user_group='+str(user_group))