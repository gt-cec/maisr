import ctypes
import multiprocessing
import os
import pygame
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sb3_trainagent import train
from behavior_cloning.generate_heuristic_traj import heuristic_policy, heuristic_process_single_observation_vectorized, badheuristic_policy, badheuristic_process_single_observation_vectorized, reset_badheuristic_state
from env_combined import MAISREnvVec
from utility.data_logging import load_env_config


def create_test_directory():
    """Create timestamped directory for test results"""
    timestamp = datetime.now().strftime("%m%d%H%M")
    test_dir = f"logs/env_tests/{timestamp}test"
    os.makedirs(test_dir, exist_ok=True)
    return test_dir


# def save_histograms(observations, rewards, actions, test_dir, test_name):
#     """Save observation, reward, and action histograms"""
#
#     # Convert lists to numpy arrays for easier handling
#     obs_array = np.array(observations)
#     reward_array = np.array(rewards)
#     action_array = np.array(actions)
#
#     # Create observation histograms (17 subplots)
#     fig, axes = plt.subplots(4, 5, figsize=(20, 16))
#     axes = axes.flatten()
#
#     obs_labels = ['Agent X', 'Agent Y'] + [f'Target {i // 3 + 1} {"Info" if i % 3 == 0 else "X" if i % 3 == 1 else "Y"}'
#                                            for i in range(15)]
#
#     for i in range(17):
#         if i < obs_array.shape[1]:
#             axes[i].hist(obs_array[:, i], bins=30, alpha=0.7, edgecolor='black')
#             axes[i].set_title(f'{obs_labels[i]}')
#             axes[i].set_xlabel('Value')
#             axes[i].set_ylabel('Frequency')
#             axes[i].grid(True, alpha=0.3)
#
#     # Hide unused subplots
#     for i in range(17, len(axes)):
#         axes[i].set_visible(False)
#
#     plt.tight_layout()
#     plt.savefig(f"{test_dir}/obs_histogram_{test_name}.png", dpi=300, bbox_inches='tight')
#     plt.close()
#
#     # Create reward histogram
#     fig, ax = plt.subplots(1, 1, figsize=(10, 6))
#     ax.hist(reward_array, bins=20, alpha=0.7, edgecolor='black')
#     ax.set_title(f'Episode Rewards - {test_name}')
#     ax.set_xlabel('Episode Reward')
#     ax.set_ylabel('Frequency')
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(f"{test_dir}/rew_histogram_{test_name}.png", dpi=300, bbox_inches='tight')
#     plt.close()
#
#     # Create action histogram (8 bins for actions 0-7)
#     fig, ax = plt.subplots(1, 1, figsize=(10, 6))
#     bins = np.arange(-0.5, 8.5, 1)  # Bins centered on integers 0-7
#     ax.hist(action_array, bins=bins, alpha=0.7, edgecolor='black')
#     ax.set_title(f'Action Distribution - {test_name}')
#     ax.set_xlabel('Action (0-7)')
#     ax.set_ylabel('Frequency')
#     ax.set_xticks(range(8))
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(f"{test_dir}/act_histogram_{test_name}.png", dpi=300, bbox_inches='tight')
#     plt.close()
#
#     print(f"Saved histograms for {test_name} in {test_dir}")

def save_histograms(observations, rewards, actions, test_dir, test_name):
    """Save observation, reward, and action histograms"""

    # Convert lists to numpy arrays for easier handling
    obs_array = np.array(observations)
    reward_array = np.array(rewards)
    action_array = np.array(actions)

    # Create observation histograms with 2D heatmaps for coordinate pairs
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    plot_idx = 0

    # Agent position heatmap (indices 0, 1)
    if obs_array.shape[1] >= 2:
        agent_x = obs_array[:, 0]
        agent_y = obs_array[:, 1]
        h = axes[plot_idx].hist2d(agent_x, agent_y, bins=30, cmap='Blues')
        axes[plot_idx].set_title('Agent Position Heatmap')
        axes[plot_idx].set_xlabel('Agent X')
        axes[plot_idx].set_ylabel('Agent Y')
        plt.colorbar(h[3], ax=axes[plot_idx])
        plot_idx += 1

    # Target info levels (indices 2, 5, 8, 11, 14)
    info_indices = [2, 5, 8, 11, 14]
    info_data = []
    for idx in info_indices:
        if idx < obs_array.shape[1]:
            info_data.append(obs_array[:, idx])

    if info_data:
        info_array = np.array(info_data).T  # Transpose for proper orientation
        axes[plot_idx].hist(info_array, bins=20, alpha=0.7,
                            label=[f'Target {i + 1}' for i in range(len(info_data))],
                            stacked=False)
        axes[plot_idx].set_title('Target Info Levels')
        axes[plot_idx].set_xlabel('Info Level')
        axes[plot_idx].set_ylabel('Frequency')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

    # Target position heatmaps (5 targets, indices 3-4, 6-7, 9-10, 12-13, 15-16)
    target_coord_pairs = [(3, 4), (6, 7), (9, 10), (12, 13), (15, 16)]

    for i, (x_idx, y_idx) in enumerate(target_coord_pairs):
        if plot_idx < len(axes) and x_idx < obs_array.shape[1] and y_idx < obs_array.shape[1]:
            target_x = obs_array[:, x_idx]
            target_y = obs_array[:, y_idx]

            # Only plot if there's variation in the data (targets exist)
            if np.std(target_x) > 0.001 or np.std(target_y) > 0.001:
                h = axes[plot_idx].hist2d(target_x, target_y, bins=20, cmap='Reds', alpha=0.8)
                axes[plot_idx].set_title(f'Target {i + 1} Position Heatmap')
                axes[plot_idx].set_xlabel(f'Target {i + 1} X')
                axes[plot_idx].set_ylabel(f'Target {i + 1} Y')
                plt.colorbar(h[3], ax=axes[plot_idx])
            else:
                # If target doesn't exist or has no variation, show empty plot
                axes[plot_idx].text(0.5, 0.5, f'Target {i + 1}\n(No Data)',
                                    ha='center', va='center', transform=axes[plot_idx].transAxes)
                axes[plot_idx].set_title(f'Target {i + 1} Position')
            plot_idx += 1

    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{test_dir}/obs_histogram_{test_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Create reward histogram
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(reward_array, bins=30, alpha=0.7, edgecolor='black', range=(-15.0, 15.0))
    ax.set_title(f'Episode Rewards - {test_name}')
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{test_dir}/rew_histogram_{test_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Create action histogram (8 bins for actions 0-7)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bins = np.arange(-0.5, 8.5, 1)  # Bins centered on integers 0-7
    ax.hist(action_array, bins=bins, alpha=0.7, edgecolor='black')
    ax.set_title(f'Action Distribution - {test_name}')
    ax.set_xlabel('Action (0-7)')
    ax.set_ylabel('Frequency')
    ax.set_xticks(range(8))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{test_dir}/act_histogram_{test_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved histograms for {test_name} in {test_dir}")


def test_oar_normalized():
    # TODO: Access observations and actions at various points in the env running and training cycle. Assert that observations are between -1 and +1
    # Save OAR
    print("test_oar_normalized: Skipping for now as requested")
    pass


def test_env_heuristic(heuristic, config, test_dir=None):
    """Run env for 20 episodes using the provided heuristic function"""
    print("Starting heuristic test...")

    # Load config

    #test_dir = create_test_directory()

    # Create environment
    env = MAISREnvVec(
        config=config,
        render_mode='headless',
        tag='test_suite'
    )

    all_observations = []
    episode_rewards = []
    all_actions = []

    for episode in range(20):
        obs, info = env.reset()
        episode_reward = 0
        episode_observations = []
        episode_actions = []

        done = False
        step_count = 0
        max_steps = 1000  # Safety limit

        while not done and step_count < max_steps:
            # Get action from heuristic
            action = heuristic(obs, None, False)
            episode_actions.append(action)

            # Store observation
            episode_observations.append(obs.copy())

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward * (config['gamma'] ** step_count)
            done = terminated or truncated
            step_count += 1

        # Store episode data
        all_observations.extend(episode_observations)
        episode_rewards.append(episode_reward)
        all_actions.extend(episode_actions)

        print(f"Heuristic episode {episode + 1}/20 completed. Reward: {episode_reward:.2f}, Steps: {step_count}")

    # Generate and save histograms
    save_histograms(all_observations, episode_rewards, all_actions, test_dir, "heuristic")

    env.close()
    print(f"Heuristic test completed. Results saved to {test_dir}")

def test_env_badheuristic(badheuristic, config, test_dir=None):
    """Run env for 20 episodes using the provided heuristic function"""
    print("Starting badheuristic test...")

    # Load config
    #config = load_env_config('config_files/testsuite_config.json')
    #test_dir = create_test_directory()

    # Create environment
    env = MAISREnvVec(
        config=config,
        render_mode='headless',
        tag='test_suite'
    )

    all_observations = []
    episode_rewards = []
    all_actions = []

    for episode in range(20):
        obs, info = env.reset()
        reset_badheuristic_state()

        episode_reward = 0
        episode_observations = []
        episode_actions = []

        done = False
        step_count = 0
        max_steps = 1000  # Safety limit

        while not done and step_count < max_steps:
            # Get action from heuristic
            action = badheuristic(obs, None, False)
            episode_actions.append(action)

            # Store observation
            episode_observations.append(obs.copy())

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward * (config['gamma'] ** step_count)
            done = terminated or truncated
            step_count += 1

        # Store episode data
        all_observations.extend(episode_observations)
        episode_rewards.append(episode_reward)
        all_actions.extend(episode_actions)

        print(f"Badheuristic episode {episode + 1}/20 completed. Reward: {episode_reward:.2f}, Steps: {step_count}")

    # Generate and save histograms
    save_histograms(all_observations, episode_rewards, all_actions, test_dir, "badheuristic")

    env.close()
    print(f"Bad heuristic test completed. Results saved to {test_dir}")

def test_env_humanplaytest(config, test_dir=None):
    """Run env for 3 episodes allowing human to take actions using numpad keys"""
    print("Starting human playtest...")
    print("Controls: Numpad 8=Up, 2=Down, 4=Left, 6=Right")
    print("Diagonals: 7=Up-Left, 9=Up-Right, 1=Down-Left, 3=Down-Right")
    print("Press ESC to quit early")

    # Load config
    #config = load_env_config('config_files/testsuite_config.json')
    #test_dir = create_test_directory()

    pygame.init()
    clock = pygame.time.Clock()

    # Disable display scaling for high-res monitors
    ctypes.windll.user32.SetProcessDPIAware()

    # Set up pygame window
    window_width, window_height = config['window_size'][0], config['window_size'][1]
    #gameboard_size = config['gameboard_size']

    # Position window (from config.py)
    #os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"
    window = pygame.display.set_mode((window_width, window_height), flags=pygame.NOFRAME)
    pygame.display.set_caption("MAISR Human Interface")

    # Create environment with human rendering
    env = MAISREnvVec(
        config=config,
        clock=clock,
        window=window,
        render_mode='human',
        num_agents=1,
        run_name='human_player',
        tag='test_suite'
    )

    # Mapping numpad keys to actions
    key_to_action = {
        pygame.K_KP8: 4,  # up
        pygame.K_KP9: 3,  # up-right
        pygame.K_KP6: 2,  # right
        pygame.K_KP3: 1,  # down-right
        pygame.K_KP2: 0,  # down
        pygame.K_KP1: 7,  # down-left
        pygame.K_KP4: 6,  # left
        pygame.K_KP7: 5  # up-left
    }

    all_observations = []
    episode_rewards = []
    all_actions = []

    for episode in range(3):
        obs, info = env.reset()
        episode_reward = 0
        episode_observations = []
        episode_actions = []

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
                        print(f"Action selected: {action}")

            if done:
                break

            # Store data
            episode_observations.append(obs.copy())
            episode_actions.append(action)

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward* (config['gamma'] ** step_count)
            done = terminated or truncated
            step_count += 1

            # Render environment
            env.render()
            pygame.display.flip()
            clock.tick(20)  # 60 FPS

        # Store episode data
        all_observations.extend(episode_observations)
        episode_rewards.append(episode_reward)
        all_actions.extend(episode_actions)

        print(f"Human episode {episode + 1}/3 completed. Reward: {episode_reward:.2f}, Steps: {step_count}")

    # Generate and save histograms
    save_histograms(all_observations, episode_rewards, all_actions, test_dir, "human")

    env.close()
    pygame.quit()
    print(f"Human playtest completed. Results saved to {test_dir}")


def test_env_random(config, test_dir=None):
    """Run env for 20 episodes, taking actions by randomly sampling from the action space"""
    print("Starting random test...")

    # Load config
    #config = load_env_config('config_files/testsuite_config.json')
    #test_dir = create_test_directory()

    # Create environment
    env = MAISREnvVec(
        config=config,
        render_mode='headless',
        tag='test_suite'
    )

    all_observations = []
    episode_rewards = []
    all_actions = []

    for episode in range(20):
        obs, info = env.reset()
        episode_reward = 0
        episode_observations = []
        episode_actions = []

        done = False
        step_count = 0
        max_steps = 1000  # Safety limit


        while not done and step_count < max_steps:
            # Random action from action space (0-7 for discrete)
            action = env.action_space.sample()
            episode_actions.append(action)

            # Store observation
            episode_observations.append(obs.copy())

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward * (config['gamma'] ** step_count)
            done = terminated or truncated
            step_count += 1

        # Store episode data
        all_observations.extend(episode_observations)
        episode_rewards.append(episode_reward)
        all_actions.extend(episode_actions)

        print(f"Random episode {episode + 1}/20 completed. Reward: {episode_reward:.2f}, Steps: {step_count}")

    # Generate and save histograms
    save_histograms(all_observations, episode_rewards, all_actions, test_dir, "random")

    env.close()
    print(f"Random test completed. Results saved to {test_dir}")


def test_env_train(config):
    """Run brief training run using Stable-baselines3"""
    print("Starting training test...")

    machine_name = 'testenv'
    project_name = 'maisr-rl-tests'

    #env_config = load_env_config('config_files/june7b_baseline.json')
    config['n_envs'] = multiprocessing.cpu_count()

    train(
        config,
        n_envs=multiprocessing.cpu_count(),
        load_path=None,
        machine_name=machine_name,
        project_name=project_name,
        save_model=False
    )

    print("Training test completed.")


if __name__ == "__main__":

    config = load_env_config('config_files/june8a.json')
    config['eval_freq'] = 4900
    config['n_eval_episodes'] = 5
    config['num_timesteps'] = 2e5

    print("Starting Environment Test Suite...")
    print("=" * 50)

    # Create shared test directory for all tests
    shared_test_dir = create_test_directory()
    print(f"All test results will be saved to: {shared_test_dir}")

    try:
        #test_env_humanplaytest(test_dir=shared_test_dir)
        #print("\n" + "=" * 50)

        test_env_heuristic(heuristic_policy, config, test_dir=shared_test_dir)
        print("\n" + "=" * 50)

        test_env_random(config, test_dir=shared_test_dir)
        print("\n" + "=" * 50)

        test_env_badheuristic(badheuristic_policy, config, test_dir=shared_test_dir)
        print("\n" + "=" * 50)

        #test_env_circles(config, test_dir=shared_test_dir)
        #print("\n" + "=" * 50)

        test_env_train(config)
        print("\n" + "=" * 50)

    except KeyboardInterrupt:
        print("\nTest suite interrupted by user.")
    except Exception as e:
        print(f"\nError during test execution: {e}")
        import traceback

        traceback.print_exc()

    print("Environment Test Suite completed!")