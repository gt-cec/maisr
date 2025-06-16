import ctypes

import pygame
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from env_combined import MAISREnvVec
from utility.data_logging import load_env_config


def evaluate_bc_policy(
        env_config,
        load_path,
        num_episodes=100,
        render=True):

    if render:
        pygame.display.init()
        pygame.font.init()
        clock = pygame.time.Clock()
        ctypes.windll.user32.SetProcessDPIAware()
        window_width, window_height = env_config['window_size'][0], env_config['window_size'][1]
        window = pygame.display.set_mode((window_width, window_height), flags=pygame.NOFRAME)

        pygame.display.set_caption("MAISR Human Interface")

        eval_env = MAISREnvVec(
            config=env_config,
            clock=clock,
            window=window,
            render_mode='human',
            run_name='bc_eval_name',
            tag='bc_eval_0'
        )
    else:
        eval_env = MAISREnvVec(
            env_config,
            None,
            render_mode='human' if render else 'headless',
            tag='bc_eval_0',
            run_name='bc_eval_name',
        )

    eval_env = Monitor(eval_env)
    print('Instantiated eval_env')

    trained_policy = PPO(
        "MlpPolicy",
        eval_env,
        verbose=1,
        device='cpu',
    )
    trained_policy = trained_policy.__class__.load(load_path, env=eval_env)
    print('instantiated policy')

    # # Create environment
    # env = MAISREnvVec(
    #     config=env_config,
    #     render_mode='headless',
    #     tag='test_suite'
    # )

    reward_history = []
    for episode in range(num_episodes):
        obs, info = eval_env.reset()
        episode_reward = 0

        terminated, truncated = False, False

        while not (terminated or truncated):
            pygame.event.get()
            action = trained_policy.predict(obs)[0]
            obs, reward, terminated, truncated, info = eval_env.step(action)
            if render:
                eval_env.render()
            episode_reward += reward

        reward_history.append(episode_reward)

    eval_env.close()
    print(f'Evaluated {num_episodes} episodes. Average reward: {np.mean(reward_history)}')

    return reward_history, np.mean(reward_history), np.std(reward_history)


if __name__ == '__main__':

    load_path = './bc_models/bcpolicy_10keps_12rewardavg.zip'
    config_filename = '../configs/bigmap.json'
    env_config = load_env_config(config_filename)

    evaluate_bc_policy(env_config, load_path, render=True)
    print('done')