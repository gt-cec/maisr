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
        run_name='pti_test',
        tag=f'pti_test',
        seed=42
    )

    #agent0_id = env.aircraft_ids[0]  # Hack to dynamically get agent IDs
    #agent0_policy = AutonomousPolicy(env, agent0_id)

    # Load pti jsons into dictionary
    pti_dict = {}
    pti_folder = './ptis/'

    for json_file in glob.glob(os.path.join(pti_folder, '*.json')):
        try:
            with open(json_file, 'r') as file:
                action_sequence = json.load(file)
                # Use the filename (without extension) as the key
                filename = os.path.splitext(os.path.basename(json_file))[0]
                pti_dict[filename] = action_sequence
                print(f"Loaded {len(action_sequence)} actions from {filename}")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading {json_file}: {e}")
    print(f"Total PTI sequences loaded: {len(pti_dict)}")

    run_number = 0
    for pti_name, pti_sequence in pti_dict.items():

        #pti_name = 'pti_circle' # TODO temp
        #pti_sequence = pti_dict[pti_name]

        run_number += 1
        terminated, truncated = False, False
        reward_list = []

        current_sequence = pti_sequence.copy()
        observation, info = env.reset()
        env.render()

        print(f"Running PTI sequence: {pti_name}")

        init = False
        while not (terminated or truncated) and len(current_sequence) > 0:  # main game loop
            if pti_name in ['pti_greedy', 'pti_circle']:
                actions = np.array(current_sequence.pop(0))
            elif pti_name == 'pti_random':
                actions = env.action_space.sample() # TODO should be action or actions?
            else:
                print(f'Got pti_name {pti_name}')
                raise ValueError('Invalid pti_name')

            #if env.render_mode == 'headless' or env.init or pygame.time.get_ticks() > env.start_countdown_time:
            observation, reward, terminated, truncated, info = env.step(actions)  # step through the environment
            reward_list.append(reward)
            if init:
                print(f'observation is shape {observation.shape}')
                print(observation)
                print(f'Agent x,y = {observation[0], observation[1]}')
                print(f'Target 0 at {observation[3], observation[4]}, info status {observation[2]}')
                print(f'Target 1 at {observation[6], observation[7]}, info status {observation[6]}')

            if env.render_mode == 'human':
                env.render()

            if init: init = False

        if len(current_sequence) == 0 and not (terminated or truncated):
            print(f"Warning: Ran out of actions in sequence for {pti_name}")
            print(
                f'Out of actions, reward {round(info['episode']['r'], 3)}, outer steps {env.step_count_outer}, inner timesteps {info['episode']['l']}, score {env.score} | {env.targets_identified} low quality | {env.detections} detections | {round(env.time_limit - env.display_time / 1000, 1)} secs left')

        env.save_action_history_plot(note=pti_name)
        round_reward = sum(reward_list)
        print(f'PTI {pti_name} earned reward: {round_reward}')
        round_number += 1