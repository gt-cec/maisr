import ctypes
import glob
import os

import json
from datetime import datetime

import pygame
import re

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from base_env import MAISREnvVec
from utility.localsearch_training_wrapper import MaisrLocalSearchWrapper
from utility.config_management import load_env_config
from utility.league_management import (RLTeammatePolicy)

import matplotlib.pyplot as plt
import pandas as pd


def populate_agent_list(agent_dir, label='nolabel'):
    model_patterns = [
        os.path.join(agent_dir, "*_model.zip"),
        os.path.join(agent_dir, "**/*_model.zip")
    ]

    normstats_patterns = [
        os.path.join(agent_dir, "*_vecnormalize.pkl"),
        os.path.join(agent_dir, "**/*_vecnormalize.pkl")
    ]

    all_checkpoints = []
    for pattern in model_patterns:
        all_checkpoints.extend(glob.glob(pattern, recursive=True))

    all_normstats = []
    for pattern in normstats_patterns:
        all_normstats.extend(glob.glob(pattern, recursive=True))

    all_checkpoints = list(set(all_checkpoints))  # Remove duplicates and sort by modification time (newest first)
    all_checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    all_normstats.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    agent_list = []
    for agent_model_filename in all_checkpoints:
        model = PPO.load(agent_model_filename)
        name = label + agent_model_filename

        # Extract the prefix by removing '_model.zip' suffix
        prefix = agent_model_filename.replace('_model.zip', '')
        expected_vecnorm_filename = f"{prefix}_vecnormalize.pkl"

        norm_stats_path = None
        if os.path.exists(expected_vecnorm_filename):
            norm_stats_path = expected_vecnorm_filename

        agent_list.append((model, norm_stats_path, name))

    print(f'Populated {len(agent_list)} {label} agents from directory {agent_dir}')
    return agent_list


def load_vecnormalize_wrapper(vecnorm_path, env):
    """Load saved VecNormalize wrapper with stats from training and apply it to the new environment."""
    print(f"Loading VecNormalize stats from: {vecnorm_path}")

    vec_normalize = VecNormalize.load(vecnorm_path, venv=env)
    vec_normalize.training = False  # Disable further normalization updates
    vec_normalize.norm_reward = False
    return vec_normalize


def make_wrapped_env(config, clock, window, teammate_policy, run_name='no_name'):
    def _init():
        base_env = MAISREnvVec(
            config=config,
            clock=clock,
            window=window,
            render_mode='human',
            run_name=f'user_study_subj{subject_id}',
            tag=f'offlinestudy0',
            #agent_appearance='black',
            running_experiment=False
        )

        wrapped_env = MaisrLocalSearchWrapper(
            base_env,
            config['obs_noise_std_localsearch'],
            local_search_policy=None,  # subpolicies['local_search'],
            go_to_highvalue_policy=None,  # subpolicies['go_to_threat'],
            change_region_subpolicy=None,  # subpolicies['change_region'],
            evade_policy=None,  # EvadeDetection(model_path=None),
            teammate_policy=None,
        )

        #wrapped_env = Monitor(wrapped_env)
        wrapped_env.reset()
        return wrapped_env

    return _init


def run_single_rl_eval(env, agent_model, render):

    base_env = env.envs[0].env
    base_env.config['force_specific_level'] = 99

    obs = env.reset()
    done = False
    episode_reward, num_steps, target_ids, threat_ids = 0, 0, 0, 0

    while not done:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
                    break

        agent_action, _ = agent_model.predict(obs, deterministic=True)

        obses, rewards, dones, infos = env.step([agent_action])

        obs = obses[0]
        reward = rewards[0]
        info = infos[0]
        done = dones[0]

        target_ids = int(info.get('target_ids', -1))
        threat_ids = int(info.get('threat_ids', -1))

        if render:
            base_env.render()
            base_env.clock.tick(60)

        episode_reward += reward
        num_steps += 1

    return episode_reward, target_ids, threat_ids, num_steps


def run_single_human_eval(env, agent_model, human_trajectory_file, level, render):

    base_env = env.envs[0].env

    base_env.level_idx = level
    base_env.config['force_specific_level'] = level

    obs = env.reset()
    done = False
    episode_reward, num_steps, target_ids, threat_ids = 0, 0, 0, 0
    step_idx = 0

    # Load trajectory
    with open(human_trajectory_file, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict) and "timesteps" in data:
        timesteps = data["timesteps"]
    elif isinstance(data, list):
        timesteps = data
    waypoints = [entry["human_custom_waypoint"] for entry in timesteps]
    waypoints = waypoints[::10]  # Timescale correction
    current_pos = base_env.agents[base_env.aircraft_ids[1]].x, base_env.agents[base_env.aircraft_ids[1]].y
    waypoints = [wp if wp is not None else current_pos for wp in waypoints]

    while not done:
        if step_idx < len(waypoints):
            base_env.agents[base_env.aircraft_ids[1]].waypoint_override = tuple(waypoints[step_idx])
        else:
            base_env.agents[base_env.aircraft_ids[1]].waypoint_override = tuple(waypoints[-1])  # hold last

        agent_action, _ = agent_model.predict(obs, deterministic=True)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
                    break

        obses, rewards, dones, infos = env.step([agent_action])

        obs = obses[0]
        reward = rewards[0]
        info = infos[0]
        done = dones[0]

        if render:
            base_env.render()
            base_env.clock.tick(30)

        target_ids = int(info.get('target_ids', -1))
        threat_ids = int(info.get('threat_ids', -1))

        episode_reward += reward
        num_steps += 1
        step_idx += 1

    return episode_reward, target_ids, threat_ids, num_steps


def main():

    config_filename = '../../configs/Monolith_index_August.json'
    config = load_env_config(config_filename)

    #config['tick_rate'] = tick_rate
    #config['game_speed'] /= time_factor
    #config['max_steps'] *= (1700 / 1500) * time_factor
    config['use_stuck_detection'] = False
    config['prob_detect'] = 0  # 0.0003
    config['action_type'] = 'Discrete16'

    num_repeats = 1 # How many times to run each level.
    render = False

    ####################################     Pygame setup     ####################################
    if render:
        if hasattr(ctypes, 'windll') and hasattr(ctypes.windll, 'user32'): ctypes.windll.user32.SetProcessDPIAware()
        pygame.display.init()
        pygame.font.init()
        clock = pygame.time.Clock()

        window_width, window_height = config['window_size'][0], config['window_size'][1]
        window = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption(f"MAISR User Study - Subject {subject_id}")
    else:
        pygame.font.init()
        window = None
        clock = None

    ####################################     Instantiate testing agents     ####################################
    testing_agent_dir = 'offline_study_testing_agents'
    heldout_agent_dir = './heldout_agents'
    human_trajectory_dir = '../userstudy_logs' # human_trajectories_for_training
    # TODO populate

    testing_agents = populate_agent_list(testing_agent_dir, label='testagent')
    heldout_agents = populate_agent_list(heldout_agent_dir, label='heldout')

    print('Contents of testing_agents:')
    print(testing_agents)
    print('Contents of heldout_agents:')
    print(heldout_agents)

    # Now I have a list of (PPO model, vecnorm stat filename, name) tuples for my testing agents


    #################################### Load human trajectories ####################################
    dual_traj_pattern = f"{human_trajectory_dir}/subject_*/timestep_data/timesteps_[ABC][13457]_*.json"
    solo_traj_pattern = f"{human_trajectory_dir}/subject_*/timestep_data/timesteps_[PS][13457]_*.json"
    dual_trajectory_files = glob.glob(dual_traj_pattern)
    solo_trajectory_files = glob.glob(solo_traj_pattern)
    if not dual_trajectory_files:
        print(f"[Human trajectory loading] No dual trajectories found")
        raise ValueError
    if not solo_trajectory_files:
        print(f"[Human trajectory loading] No solo trajectories found")
        raise ValueError
    print(f'[Human trajectories] Loaded {len(dual_trajectory_files)} dual trajectories and {len(solo_trajectory_files)} solo trajectories')


    rl_results = {}
    human_results = {}
    for agent_tuple in testing_agents:
        agent_model = agent_tuple[0]
        agent_vecnorm = agent_tuple[1]
        agent_name = agent_tuple[2]

        # TODO temp hack
        temp_teammate_policy = RLTeammatePolicy(agent_model, None, None, None, None, norm_stats_path=agent_vecnorm)

        env_fns = [make_wrapped_env(config, clock, window, temp_teammate_policy) for _ in range(1)]
        env = DummyVecEnv(env_fns)
        env = load_vecnormalize_wrapper(agent_vecnorm, env)
        print(f'\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        print(f'&&&&&& NEW TESTING AGENT, loaded new vecnorm stats from {agent_vecnorm} &&&&&')
        print(f'&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n')
        base_env = env.envs[0].env
        wrapper_env = env.envs[0]
        print(f'BASE ENV IS {base_env}')
        base_env.teammate_active = True
        wrapper_env.teammate_active = True

        #################################### Run RL agent evals ####################################
        print(f'\n#########################################################')
        print(f'Evaluating testing agent {agent_name}')
        print(f'#########################################################\n')
        #for teammate_tuple in heldout_agents + testing_agents:
        for teammate_tuple in testing_agents:

            print(f'%%%%%% Running 3 episodes with teammate {teammate_tuple[2]}\n')

            for run in range(100): # 20
                teammate_model = teammate_tuple[0]
                teammate_vecnorm = teammate_tuple[1]
                teammate_name = teammate_tuple[2]


                teammate = RLTeammatePolicy(teammate_model, env, None, None, None, norm_stats_path=teammate_vecnorm)
                print(f'teammate is using model {teammate_model} and norm stats {teammate_vecnorm}')

                wrapper_env.teammate_policy = teammate
                wrapper_env.current_teammate = teammate

                #print(f'\nRunning single RL eval with agent {agent_name} (model {agent_model}, vecnorm stats {agent_vecnorm}). Teammate is {teammate_name} (model {teammate_model}, vecnorm {teammate_vecnorm})')
                reward, target_ids, threat_ids, num_steps = run_single_rl_eval(env, agent_model, render)
                rl_results[(agent_name, teammate_name, run)] = reward, target_ids, threat_ids, num_steps


        #################################### Run human trajectory evals ####################################
        all_human_trajectories = dual_trajectory_files + solo_trajectory_files
        for trajectory_file in all_human_trajectories[0:250]: # 160
            for run in range(1):
                level = int(trajectory_file.split('_')[4][1])

                print(f'Running human eval with level {level} and trajectory {trajectory_file}')
                teammate_name = trajectory_file
                reward, target_ids, threat_ids, num_steps = run_single_human_eval(env, agent_model, trajectory_file, level, render)
                human_results[(agent_name, teammate_name, run)] = reward, target_ids, threat_ids, num_steps

    #print(f'====== ====== FINAL RESULTS ====== ====== ')
    #print(results)

    def extract_agent_info(agent_name):
        """Extract agent type and seed from agent name."""
        # Remove path components
        name = agent_name.split('/')[-1] if '/' in agent_name else agent_name

        # Extract agent type
        if 'mixed75' in name.lower():
            agent_type = 'mixed75'
        elif 'selfplay' in name.lower():
            agent_type = 'selfplay'
        elif 'fcp' in name.lower():
            agent_type = 'fcp'
        elif 'strategyfinetuned' in name.lower():
            agent_type = 'strat-finetuned'
        else:
            agent_type = 'unknown'

        # Extract seed
        seed_match = re.search(r'seed(\d+)', name.lower())
        if seed_match:
            seed = f"seed{seed_match.group(1)}"
        else:
            seed = 'unknown'

        return f"{agent_type}"

    # Separate RL and human results
    rl_data_rows = []
    human_data_rows = []

    for (agent_name, teammate_name, run), (reward, target_ids, threat_ids, num_steps) in rl_results.items():
        #agent_id = agent_name.split('/')[-1] if '/' in agent_name else agent_name
        agent_id = extract_agent_info(agent_name)
        rl_data_rows.append({
            'agent': agent_id,
            'teammate': teammate_name,
            'run': run,
            'reward': reward,
            'target_ids': target_ids,
            'threat_ids': threat_ids,
            'num_steps': num_steps
        })

    for (agent_name, teammate_name, run), (reward, target_ids, threat_ids, num_steps) in human_results.items():
        #agent_id = agent_name.split('/')[-1] if '/' in agent_name else agent_name
        agent_id = extract_agent_info(agent_name)
        human_data_rows.append({
            'agent': agent_id,
            'teammate': teammate_name,
            'run': run,
            'reward': reward,
            'target_ids': target_ids,
            'threat_ids': threat_ids,
            'num_steps': num_steps
        })

    # Create DataFrames
    rl_df = pd.DataFrame(rl_data_rows)
    human_df = pd.DataFrame(human_data_rows)


    print("RL DataFrame sample:")
    print(rl_df[rl_df['agent'] == 'fcp'])  # Check all FCP results
    print("\nUnique agent-teammate combinations for FCP:")
    print(rl_df[rl_df['agent'] == 'fcp'][['agent', 'teammate', 'reward']].head(40))
    print('\n')

    print(rl_df[rl_df['agent'] == 'mixed75'])  # Check all FCP results
    print("\nUnique agent-teammate combinations for mixed75:")
    print(rl_df[rl_df['agent'] == 'mixed75'][['agent', 'teammate', 'reward']].head(40))
    print('\n')

    fcp_rewards = [row['reward'] for row in rl_data_rows if 'fcp' in row['agent']]
    print(f"FCP rewards: {fcp_rewards}")
    print(f"FCP unique rewards: {set(fcp_rewards)}")
    print('\n')

    mixed75_rewards = [row['reward'] for row in rl_data_rows if 'mixed75' in row['agent']]
    print(f"mixed75 rewards: {mixed75_rewards}")
    print(f"mixed75 unique rewards: {set(mixed75_rewards)}")
    print('\n')

    # Calculate statistics
    rl_stats = rl_df.groupby('agent')['reward'].agg(['mean', 'std', 'count']).reset_index()
    human_stats = human_df.groupby('agent')['reward'].agg(['mean', 'std', 'count']).reset_index()

    # ================ Save results to JSON ================

    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare data structure for JSON export
    results_data = {
        'metadata': {
            'timestamp': timestamp,
            'config_file': config_filename,
            'num_testing_agents': len(testing_agents),
            'num_heldout_agents': len(heldout_agents),
            'num_dual_trajectories': len(dual_trajectory_files),
            'num_solo_trajectories': len(solo_trajectory_files),
            'num_repeats_rl': 20,  # As specified in your RL evaluation loop
            'num_repeats_human': 1  # As specified in your human evaluation loop
        },
        'raw_results': {
            'rl_results': rl_data_rows,
            'human_results': human_data_rows
        },
        'summary_stats': {
            'rl_stats': rl_stats.to_dict('records'),
            'human_stats': human_stats.to_dict('records')
        }
    }

    # Save to JSON file
    output_filename = f'offline_evaluation_results_{timestamp}.json'
    with open(output_filename, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)

    print(f'\n====== RESULTS SAVED TO: {output_filename} ======')

    # Create 2-subplot figure
    # Replace the plotting section (starting from "# Create 2-subplot figure") with this code:

    # Define consistent colors and label mapping for agent types
    agent_colors = {
        'fcp': '#2E86AB',  # Blue
        'mixed75': '#A23B72',  # Purple
        'selfplay': '#F18F01',  # Orange
        'strat-finetuned': '#C73E1D'  # Red
    }

    agent_labels = {
        'fcp': 'FCP',
        'mixed75': 'Strat-FCP',
        'selfplay': 'SP',
        'strat-finetuned': 'Strat-SP'
    }

    # Create 2-subplot figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10))

    # Top subplot: RL results
    rl_colors = [agent_colors.get(agent, '#808080') for agent in rl_stats['agent']]
    bars1 = ax1.bar(range(len(rl_stats)), rl_stats['mean'],
                    yerr=rl_stats['std'], capsize=5, alpha=0.9, color=rl_colors)
    ax1.set_xlabel('Agent', fontsize=12)
    ax1.set_ylabel('Average Reward', fontsize=12)
    ax1.set_title('Performance with Held-Out RL Teammates', fontsize=20)
    ax1.set_xticks(range(len(rl_stats)))
    ax1.set_xticklabels([agent_labels.get(agent, agent) for agent in rl_stats['agent']],rotation=45, ha='right')
    ax1.set_ylim(0, 45)
    ax1.grid(axis='y', alpha=0.3)

    # Bottom subplot: Human results
    human_colors = [agent_colors.get(agent, '#808080') for agent in human_stats['agent']]
    bars2 = ax2.bar(range(len(human_stats)), human_stats['mean'],
                    yerr=human_stats['std'], capsize=5, alpha=0.9, color=human_colors)
    ax2.set_xlabel('Agent', fontsize=12)
    ax2.set_ylabel('Average Reward', fontsize=12)
    ax2.set_title('Performance with Recorded Human Teammates', fontsize=20)
    ax2.set_xticks(range(len(human_stats)))
    ax2.set_xticklabels([agent_labels.get(agent, agent) for agent in human_stats['agent']],rotation=45, ha='right')
    ax1.set_ylim(0, 45)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f'\n====== RL AGENT PERFORMANCE SUMMARY ======')
    for _, row in rl_stats.iterrows():
        agent_label = agent_labels.get(row["agent"], row["agent"])
        print(f'{agent_label}: {row["mean"]:.2f} ± {row["std"]:.2f} (n={row["count"]})')

    print(f'\n====== HUMAN TRAJECTORY PERFORMANCE SUMMARY ======')
    for _, row in human_stats.iterrows():
        agent_label = agent_labels.get(row["agent"], row["agent"])
        print(f'{agent_label}: {row["mean"]:.2f} ± {row["std"]:.2f} (n={row["count"]})')





if __name__ == '__main__':
    main()