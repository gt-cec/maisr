import ctypes
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from env_multi_new import MAISREnvVec
from training_wrappers.modeselector_training_wrapper import MaisrModeSelectorWrapper
#from policies.greedy_heuristic_improved import greedy_heuristic_nearest_n
from policies.sub_policies import SubPolicy, LocalSearch, ChangeRegions, GoToNearestThreat
from utility.data_logging import load_env_config


if __name__ == "__main__":

    config_filename = 'configs/june15.json'


    config = load_env_config(config_filename)
    print(f'LOADED CONFIG {config_filename}')
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
    # TODO after 0615 runs: Modify here and test
    localsearch_model = PPO.load('trained_models/local_search_2000000.0timesteps_0.1threatpenalty_0615_1541_6envs_maisr_trained_model.zip')
    local_search_policy = LocalSearch(model=localsearch_model, norm_stats_filepath = 'trained_models/local_search_2000000.0timesteps_0.1threatpenalty_0615_1541_6envslocal_search_norm_stats.npy')

    go_to_highvalue_policy = GoToNearestThreat(model=None)
    change_region_subpolicy = ChangeRegions(model=None)

    env = MaisrModeSelectorWrapper(
        base_env,
        local_search_policy,
        go_to_highvalue_policy,
        change_region_subpolicy
    )
    #env = DummyVecEnv([lambda: env])
    #env = VecNormalize(env, norm_reward=False, training=False)


    ###################################################################################################################

    key_to_action = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2}
    all_observations = []
    episode_rewards = []
    all_actions = []

    for episode in range(3):
        obs = env.reset()[0]
        episode_reward = 0
        episode_observations, episode_actions, potential_gain_history = [], [], []

        done = False
        step_count = 0
        action = 0  # Default action (up)

        print(f"\nStarting human episode {episode + 1}/3")

        while not done:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        done = True
                        break
                    elif event.key in key_to_action:
                        action = key_to_action[event.key]
                        print(f"Selected subpolicy {action}")
            if done:
                break

            # Store data
            episode_observations.append(obs.copy())
            episode_actions.append(action)

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            # obses, rewards, dones, infos = env.step(action)
            # obs = obses[0]
            # reward = rewards[0]
            # info = infos[0]
            # done = dones[0]
            episode_reward += reward

            step_count += 1
            env.render()

    env.close()