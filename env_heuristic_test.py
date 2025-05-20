from agents import *
import sys
import os
import ctypes
import numpy as np

from env_combined import MAISREnvVec
from autonomous_policy import AutonomousPolicy
from gui import *
from utility.data_logging import GameLogger, load_env_config
from config import x, y, config_dict, run_order, surveys_enabled, times
import webbrowser
from stable_baselines3 import PPO


if __name__ == "__main__":
    from config import subject_id, user_group, log_data, x, y
    round_number = 0

    print(f"\nStarting MAISR environment (subject_id = {subject_id}, group = {user_group}, data logging = {log_data})")

    render = "headless" not in sys.argv
    total_games = 5 # Number of games to run
    game_count = 0 # Used to track how many games have been completed so far
    config = './config_files/rl_training_default.json'
    env_config = load_env_config(config)

    print("Starting in PyGame mode")
    pygame.init()
    clock = pygame.time.Clock()
    ctypes.windll.user32.SetProcessDPIAware()  # Disables display scaling so the game fits on small, high-res monitors
    window_width, window_height = env_config['window size'][0], env_config['window size'][1]
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"
    window = pygame.display.set_mode((window_width, window_height), flags=pygame.NOFRAME)

    env = MAISREnvVec(
        config=env_config,
        clock=clock,
        window=window,
        render_mode='human',
        run_name='heuristic_policy_test',
        tag=f'heuristic_test'
    )

    agent0_id = env.aircraft_ids[0]  # Hack to dynamically get agent IDs
    agent0_policy = AutonomousPolicy(env, agent0_id)

    while round_number < total_games:
        game_count += 1
        terminated, truncated = False, False
        reward_list = []
        actions = {0: None}  # use agent policies to get actions as a dict {agent_id: action}

        observation, info = env.reset()

        while not (terminated or truncated):  # main game loop
            agent_action = agent0_policy.act()
            print(f'Agent action: {agent_action}')
            actions[0] = agent_action
            #actions = np.array(agent_action)

            if env.render_mode == 'headless' or env.init or pygame.time.get_ticks() > env.start_countdown_time:
                observation, reward, terminated, truncated, info = env.step(actions)  # step through the environment
                reward_list.append(reward)

            if env.render_mode == 'human':
                env.render()

        round_reward = sum(reward_list)
        print(f'Earned reward: {round_reward}')
        round_number += 1

        if env.render_mode == 'human':
            print("Game complete:", game_count)
            env.close()

    print("ALL GAMES COMPLETE")