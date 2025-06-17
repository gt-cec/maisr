import numpy as np
import ctypes
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from wandb.cli.cli import local

from env_multi_new import MAISREnvVec
from training_wrappers.localsearch_training_wrapper import MaisrLocalSearchWrapper
from training_wrappers.modeselector_training_wrapper import MaisrModeSelectorWrapper
#from policies.greedy_heuristic_improved import greedy_heuristic_nearest_n
#from policies.sub_policies import SubPolicy, LocalSearch, ChangeRegions, GoToNearestThreat
from utility.league_management import LocalSearch, SubPolicy, ChangeRegions, GoToNearestThreat
from utility.data_logging import load_env_config



if __name__ == "__main__":

    use_normalize = True
    config = load_env_config('configs/june15.json')

    pygame.display.init()
    pygame.font.init()
    clock = pygame.time.Clock()
    ctypes.windll.user32.SetProcessDPIAware()
    window_width, window_height = config['window_size'][0], config['window_size'][1]
    config['tick_rate'] = 30
    window = pygame.display.set_mode((window_width, window_height), flags=pygame.NOFRAME)
    pygame.display.set_caption("MAISR Human Interface")

    base_env = MAISREnvVec(
        config=config,
        clock=clock,
        window=window,
        render_mode='human',
        run_name='hrl_test',
        tag=f'test0',
    )

    # Instantiate subpolicies
    #local_search_policy = LocalSearch(model=None)
    localsearch_model_path = 'trained_models/local_search_700000.0timesteps_0.05obs_noise_0617_0002_6envs_maisr_trained_model.zip'
    localsearch_normstats_path = 'trained_models/local_search_700000.0timesteps_0.05obs_noise_0617_0002_6envslocal_search_norm_stats.npy'
    local_search_policy = LocalSearch(
        model_path = localsearch_model_path,
        norm_stats_filepath = localsearch_normstats_path
    )

    env = MaisrLocalSearchWrapper(base_env, obs_noise_std=0)
    if use_normalize:
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load('./trained_models/local_search_700000.0timesteps_0.05obs_noise_0617_0002_6envslocal_search_vecnormalize.pkl', env)
        env.training = False
        env.norm_Reward = False
        #env = VecNormalize(env, norm_reward=False, training=False)


    ###################################################################################################################

    key_to_action = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2}
    all_observations, episode_rewards,all_actions = [],[],[]


    for episode in range(3):
        if use_normalize:
            obs = env.reset()
        else:
            obs, info = env.reset()

        episode_reward = 0
        episode_observations, episode_actions, potential_gain_history = [], [], []

        done = False
        step_count = 0
        action = 0  # Default action (up)

        print(f"\nStarting human episode {episode + 1}/3")

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        done = True
                        break
            if done:
                break

            action, _ = local_search_policy.act(obs)
            print(f'[Play] Action = {action} (type {type(action)})')
            #action = np.ndarray([action])

            # Store data
            episode_observations.append(obs.copy())
            episode_actions.append(action)

            # Take step
            if use_normalize:
                print(f'action: {action}')
                obses, rewards, dones, infos = env.step([action])
                #print(f'obses: {obses}')
                obs, reward, done, info = obses[0], rewards[0], dones[0], infos[0]
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                done = terminated or truncated
                step_count += 1
            env.render()

    env.close()