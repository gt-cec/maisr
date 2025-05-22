from agents import *
import os
import ctypes
import numpy as np
import json
import glob

from env_combined import MAISREnvVec
from new_heuristic_policy import AutonomousPolicy
from gui import *
from utility.data_logging import GameLogger, load_env_config
from config import x, y, config_dict, run_order, surveys_enabled, times


if __name__ == "__main__":
    from config import subject_id, user_group, log_data, x, y
    round_number = 0

    print(f"\nStarting MAISR environment (subject_id = {subject_id}, group = {user_group}, data logging = {log_data})")

    config = './config_files/pti_env_test.json'
    env_config = load_env_config(config)

    #print("Starting in PyGame mode")
    #pygame.init()
    # clock = pygame.time.Clock()
    # ctypes.windll.user32.SetProcessDPIAware()  # Disables display scaling so the game fits on small, high-res monitors
    # window_width, window_height = env_config['window size'][0], env_config['window size'][1]
    # os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"
    # window = pygame.display.set_mode((window_width, window_height), flags=pygame.NOFRAME)

    env = MAISREnvVec(
        config=env_config,
        #clock=clock,
        #window=window,
        render_mode='headless',
        run_name='pti_test',
        tag=f'pti_test',
        seed=42
    )

    agent0_id = env.aircraft_ids[0]  # Hack to dynamically get agent IDs
    agent0_policy = AutonomousPolicy(env, agent0_id)

    # Load pti jsons
    pti_list = []
    pti_folder = './ptis/'

    for json_file in glob.glob(os.path.join(pti_folder, '*.json')):
        try:
            with open(json_file, 'r') as file:
                action_sequence = json.load(file)
                pti_list.append(action_sequence)
                print(f"Loaded {len(action_sequence)} actions from {os.path.basename(json_file)}")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading {json_file}: {e}")
    print(f"Total PTI sequences loaded: {len(pti_list)}")

    run_number = 0
    for pti in pti_list:
        run_number += 1
        terminated, truncated = False, False
        reward_list = []

        current_sequence = pti.copy()

        observation, info = env.reset()

        while not (terminated or truncated) and len(current_sequence) > 0:  # main game loop
            agent_action = current_sequence[0]#.pop(0)
            actions = np.array(agent_action)

            #if env.render_mode == 'headless' or env.init or pygame.time.get_ticks() > env.start_countdown_time:
            observation, reward, terminated, truncated, info = env.step(actions)  # step through the environment
            reward_list.append(reward)

            if env.render_mode == 'human':
                env.render()

        if len(current_sequence) == 0 and not (terminated or truncated):
            print(f"Warning: Ran out of actions in sequence")

        round_reward = sum(reward_list)
        print(f'Earned reward: {round_reward}')
        round_number += 1

    print("ALL GAMES COMPLETE")