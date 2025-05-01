from agents import *
import sys
import os
import ctypes

from env_vec import MAISREnvVec
from gui import *
from utility.data_logging import GameLogger, load_env_config
from config import x, y, config_dict, run_order, surveys_enabled, times
import webbrowser

from stable_baselines3 import PPO

if __name__ == "__main__":

    if len(sys.argv) == 1:
        print('No args specified, loading parameters from config.py')
        from config import subject_id, user_group, log_data, x, y, round_number


    print(f"\nStarting MAISR environment (subject_id = {subject_id}, group = {user_group}, data logging = {log_data})")

    render = "headless" not in sys.argv
    reward_type = 'balanced-sparse' # TODO move into a config file

    config_list = config_dict[user_group]
    total_games = 5 # Number of games to run
    game_count = 0 # Used to track how many games have been completed so far

    while round_number < total_games:
        config = config_list[round_number]
        env_config = load_env_config(config)

        print("Starting in PyGame mode")
        pygame.init()
        clock = pygame.time.Clock()
        ctypes.windll.user32.SetProcessDPIAware()  # Disables display scaling so the game fits on small, high-res monitors
        window_width, window_height = env_config['window size'][0], env_config['window size'][1]
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"
        window = pygame.display.set_mode((window_width, window_height),flags=pygame.NOFRAME)
        env = MAISREnvVec(env_config, window, clock=clock, render_mode='human',
                       reward_type=reward_type, obs_type='vector', action_type='continuous',
                       subject_id=subject_id,user_group=user_group,round_number=round_number)

        model = PPO.load('./trained_models/test_agent.pth', env=env)

        agent0_id = env.aircraft_ids[0]  # Hack to dynamically get agent IDs
        agent0_policy = None # TODO

        button_handler = ButtonHandler(env, agent0_policy, log_data=False)

        game_count += 1
        terminated, truncated = False, False  # flag for when the run is complete
        agent1_waypoint = (0, 0)

        observation = env.reset()  # reset the environment

        while not (terminated or truncated):  # main game loop

            actions = []  # use agent policies to get actions as a list of tuple [(agent index, waypoint)]

            # Handle human actions (mouse clicks)
            ev = pygame.event.get()
            for event in ev:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F1:
                        #if log_data: game_logger.final_log(button_handler.gameplan_command_history, env) (TODO Add back)
                        env.close()
                    if event.key == pygame.K_SPACE:
                        env.pause(pygame.MOUSEBUTTONDOWN)

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_position = pygame.mouse.get_pos()
                    time_sec = float(env.display_time) / 1000

                    agent0_action_override, human_waypoint = button_handler.handle_mouse_click(mouse_position, time_sec)
                    target_point_unscaled = (human_waypoint[0] / env.config['gameboard size'], human_waypoint[1] / env.config['gameboard size'])
                    human_waypoint  = ((2 * target_point_unscaled[0]) - 1, (2 * target_point_unscaled[1]) - 1)

                    if human_waypoint is not None:
                        human_action = (human_waypoint[0], human_waypoint[1], 0)
                        actions.append((env.human_idx, human_action))
                        agent1_waypoint = human_action



            # actions: List of (agent_id, action) tuples, where action = dict('waypoint': (x,y), 'id_method': 0, 1, or 2')
            agent_action = model(observation)
            actions.append((env.aircraft_ids[0], agent_action))

            if env.render_mode == 'headless' or env.init or pygame.time.get_ticks() > env.start_countdown_time:
                observation, reward, terminated, truncated, info = env.step(actions)  # step through the environment

            if env.render_mode == 'human': env.render()

        if terminated or truncated:
            done_time = pygame.time.get_ticks()

            if env.render_mode == 'human':
                waiting_for_key = True
                while waiting_for_key:
                    env.render()  # Keep rendering while waiting
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                            if (pygame.time.get_ticks()-done_time)/1000 > 1.5: # Cooldown to prevent accidentally clicking continue
                                waiting_for_key = False
                                break
                        elif event.type == pygame.QUIT:
                            env.close()
                            sys.exit()

        round_number += 1
        if env.render_mode == 'human':
            print("Game complete:", game_count)
            env.close()

    print("ALL GAMES COMPLETE")