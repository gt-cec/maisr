import ctypes
import pygame
from stable_baselines3 import PPO
from env_multi_new import MAISREnvVec
from training_wrappers.modeselector_training_wrapper import MaisrModeSelectorWrapper
#from policies.greedy_heuristic_improved import greedy_heuristic_nearest_n
#from policies.sub_policies import SubPolicy, LocalSearch, ChangeRegions, GoToNearestThreat
from utility.data_logging import load_env_config
from utility.league_management import LocalSearch, ChangeRegions, GoToNearestThreat, \
    EvadeDetection, TeammateManager

if __name__ == "__main__":

    config_filename = 'configs/june23_poc1_2ship.json'

    league_type = 'strategy_diverse'
    balance_method = 'uniform'
    num_episodes = 20
    tick_rate = 20


    config = load_env_config(config_filename)
    print(f'LOADED CONFIG {config_filename}')
    pygame.display.init()
    pygame.font.init()
    clock = pygame.time.Clock()
    ctypes.windll.user32.SetProcessDPIAware()
    window_width, window_height = config['window_size'][0], config['window_size'][1]
    config['tick_rate'] = tick_rate
    window = pygame.display.set_mode((window_width, window_height), flags=pygame.NOFRAME)
    pygame.display.set_caption("MAISR Human Interface")

    ####################################################################################################################
    localsearch_model_path = None  # 'trained_models/local_search_2000000.0timesteps_0.1threatpenalty_0615_1541_6envs_maisr_trained_model.zip'
    localsearch_normstats_path = 'trained_models/local_search_2000000.0timesteps_0.1threatpenalty_0615_1541_6envslocal_search_norm_stats.npy'

    subpolicies = {
        'local_search': LocalSearch(model_path=None),  # Using heuristic
        'change_region': ChangeRegions(model_path=None),  # Using heuristic
        'go_to_threat': GoToNearestThreat(model_path=None)  # Using heuristic
    }

    ####################################################################################################################
    base_env = MAISREnvVec(
        config=config,
        clock=clock,
        window=window,
        render_mode='human',
        run_name='hrl_test',
        tag=f'test0',
    )

    env = MaisrModeSelectorWrapper(
        base_env,
        local_search_policy=LocalSearch(
            model_path=localsearch_model_path,
            norm_stats_filepath=localsearch_normstats_path),
        go_to_highvalue_policy=GoToNearestThreat(model_path=None),
        change_region_subpolicy=ChangeRegions(model_path=None),
        evade_policy=EvadeDetection(model_path=None),
        teammate_manager=TeammateManager(league_type, balance_method, subpolicies=subpolicies)
    )

    ####################################################################################################################
    # Load model
    model_path = './trained_models/modeselector_poc1_2ship_0.0005lr_1024bs_0623_1424_16envs/maisr_checkpoint_modeselector_poc1_2ship_0.0005lr_1024bs_0623_1424_16envs_149760_steps.zip'
    model = PPO.load(model_path)
    print(f"Loaded PPO model from {model_path}")


    ###################################################################################################################

    key_to_action = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2}
    all_observations = []
    episode_rewards = []
    all_actions = []

    for episode in range(num_episodes):
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
                    # elif event.key in key_to_action:
                    #     action = key_to_action[event.key]
                    #     print(f"Selected subpolicy {action}")
            if done:
                break

            if not done:
                action, _ = model.predict(obs, deterministic=True)
                print(f"PPO selected subpolicy {action}")

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
            done = terminated or truncated

            step_count += 1

            # Render subpolicy icons
            agent0_subpolicy_id, agent0_subpolicy_name = env.get_current_subpolicy_info()
            if config['num_aircraft'] == 2:
                agent1_subpolicy_id, agent1_subpolicy_name = env.get_teammate_subpolicy_info()
            else:
                agent1_subpolicy_id, agent1_subpolicy_name = 0, 'N/A'
            env.env.render_subpolicy_indicators(agent0_subpolicy_id, agent0_subpolicy_name, agent1_subpolicy_id,agent1_subpolicy_name)
            pygame.display.flip()

    env.close()