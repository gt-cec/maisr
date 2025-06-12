import warnings
warnings.filterwarnings("ignore", message="Your system is avx2 capable but pygame was not built with support for it")
import ctypes
import multiprocessing
import os
import pygame
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
import glob
from stable_baselines3 import PPO

from train_sb3 import train
from behavior_cloning.generate_heuristic_traj import heuristic_policy, heuristic_process_single_observation_vectorized, badheuristic_policy, badheuristic_process_single_observation_vectorized, reset_badheuristic_state
from heuristic_policies.greedy_heuristic_improved import improved_heuristic_policy, improved_heuristic_process_single_observation, reset_heuristic_state
from env_combined import MAISREnvVec
#from env_20targets import MAISREnvVec
from utility.data_logging import load_env_config
from utility.visualize_values import get_directional_potential_gains, draw_value_arrows


def combine_and_display_plots(test_dir):
    """Combine all saved plots into one image and display it"""
    print(f"\nCombining plots from {test_dir}...")

    # Find all PNG files in the test directory
    plot_files = glob.glob(os.path.join(test_dir, "*.png"))

    if not plot_files:
        print("No plots found to combine.")
        return

    # Sort files for consistent ordering
    plot_files.sort()

    # Load all images
    images = []
    for file_path in plot_files:
        try:
            img = Image.open(file_path)
            images.append(img)
            #print(f"Loaded: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    if not images:
        print("No valid images to combine.")
        return

    # Calculate grid dimensions (prefer wider layout)
    n_images = len(images)
    cols = min(3, n_images)  # Maximum 3 columns
    rows = (n_images + cols - 1) // cols  # Ceiling division

    # Get max dimensions for consistent sizing
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create combined image
    combined_width = cols * max_width
    combined_height = rows * max_height
    combined_image = Image.new('RGB', (combined_width, combined_height), 'white')

    # Paste images into grid
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * max_width
        y = row * max_height

        # Center the image in its allocated space
        x_offset = (max_width - img.width) // 2
        y_offset = (max_height - img.height) // 2

        combined_image.paste(img, (x + x_offset, y + y_offset))

    # Save combined image
    combined_path = os.path.join(test_dir, "combined_results.png")
    combined_image.save(combined_path, dpi=(300, 300))
    print(f"Combined image saved to: {combined_path}")

    # Display the image
    try:
        combined_image.show()
        print("Combined plot opened for viewing.")
    except Exception as e:
        print(f"Could not automatically open image: {e}")
        print(f"Please manually open: {combined_path}")

    return combined_path

def create_test_directory():
    """Create timestamped directory for test results"""
    timestamp = datetime.now().strftime("%m%d%H%M")
    test_dir = f"logs/env_tests/{timestamp}test"
    os.makedirs(test_dir, exist_ok=True)
    return test_dir

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
    ax.hist(reward_array, bins=30, alpha=0.7, edgecolor='black', range=(-17.0, 17.0))
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

    print(f"\nSaved histograms for {test_name} in {test_dir}")


def test_env_heuristic(heuristic, config, render=True, test_dir=None):
    """Run env for 20 episodes using the provided heuristic function"""
    print("Starting heuristic test...")

    if render:
        pygame.display.init()
        pygame.font.init()
        clock = pygame.time.Clock()
        config['obs_type'] = 'absolute'
        ctypes.windll.user32.SetProcessDPIAware()

        window_width, window_height = config['window_size'][0], config['window_size'][1]
        config['tick_rate'] = 80
        window = pygame.display.set_mode((window_width, window_height), flags=pygame.NOFRAME)
        pygame.display.set_caption("MAISR Human Interface")

        env = MAISREnvVec(
            config=config,
            clock=clock,
            window=window,
            render_mode='human',
            num_agents=1,
            tag='test_suite',
            seed=config['seed']
        )

    all_observations = []
    episode_rewards = []
    all_actions = []

    for episode in range(20):
        obs, info = env.reset()
        episode_reward = 0
        reward_history = []
        potential_gain_history = []
        episode_observations = []
        episode_actions = []

        done = False

        while not done:
            pygame.event.get()
            # Get action from heuristic
            action = heuristic(obs, None, False)
            episode_actions.append(action)

            # Store observation
            episode_observations.append(obs.copy())

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            reward_history.append(reward)
            potential_gain_history.append(info["outerstep_potential_gain"])
            episode_reward += reward
            done = terminated or truncated

        # Store episode data
        all_observations.extend(episode_observations)
        all_potential_gain = sum(potential_gain_history)
        episode_rewards.append(episode_reward)
        all_actions.extend(episode_actions)

        print(f"Heuristic episode {episode + 1}/20 completed. Reward: {episode_reward:.2f} ({round(100*(all_potential_gain * config["shaping_coeff_prox"])/abs(episode_reward),1)}% from shaping), Steps:")
        print(f'Total potential gain = {sum(potential_gain_history)}. History: ({potential_gain_history}')
        print(f'Total potential reward = {round(all_potential_gain,1)}*{config['shaping_coeff_prox']}={round(all_potential_gain * config["shaping_coeff_prox"],2)}\n')

    # Generate and save histograms
    save_histograms(all_observations, episode_rewards, all_actions, test_dir, "heuristic")
    env.close()

    avg_reward = np.mean(episode_rewards)
    print(f"Average heuristic reward: {avg_reward:.2f}")
    #assert avg_reward > 10, f"Heuristic average reward {avg_reward:.2f} must be > 10"
    print(f"Heuristic test completed. Results saved to {test_dir}")


def test_env_badheuristic(badheuristic, config, render=False,test_dir=None):
    """Run env for 20 episodes using the provided heuristic function"""
    print("\nStarting badheuristic test...")

    if render:
        pygame.display.init()
        pygame.font.init()
        clock = pygame.time.Clock()
        config['obs_type'] = 'absolute'
        ctypes.windll.user32.SetProcessDPIAware()

        window_width, window_height = config['window_size'][0], config['window_size'][1]
        config['tick_rate'] = 80
        window = pygame.display.set_mode((window_width, window_height), flags=pygame.NOFRAME)
        pygame.display.set_caption("MAISR Human Interface")

        env = MAISREnvVec(
            config=config,
            clock=clock,
            window=window,
            render_mode='human',
            num_agents=1,
            tag='test_suite',
            seed=config['seed']
        )

    else:
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
        potential_gain_history = []

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
            potential_gain_history.append(info["outerstep_potential_gain"])
            done = terminated or truncated
            step_count += 1
            if render: env.render()

        # Store episode data
        all_observations.extend(episode_observations)
        episode_rewards.append(episode_reward)
        all_actions.extend(episode_actions)
        all_potential_gain = sum(potential_gain_history)

        print(f"Badheuristic episode {episode + 1}/20 completed. Reward: {episode_reward:.2f} (Shaping: {round((all_potential_gain * config["shaping_coeff_prox"]),1)} / {episode_reward} = {round(100*(all_potential_gain * config["shaping_coeff_prox"])/abs(episode_reward),1)}% from shaping), Steps: {step_count}")

    # Generate and save histograms
    save_histograms(all_observations, episode_rewards, all_actions, test_dir, "badheuristic")
    env.close()

    avg_reward = np.mean(episode_rewards)
    print(f"Average badheuristic reward: {avg_reward:.2f}")
    #assert avg_reward < -3, f"Badheuristic average reward {avg_reward:.2f} must be < -3"
    print(f"Bad heuristic test completed. Results saved to {test_dir}")

def test_env_humanplaytest(config, test_dir=None):
    """Run env for 3 episodes allowing human to take actions using numpad keys"""
    print("\nStarting human playtest...")
    print("Controls: Numpad 8=Up, 2=Down, 4=Left, 6=Right")
    print("Diagonals: 7=Up-Left, 9=Up-Right, 1=Down-Left, 3=Down-Right")
    print("Press ESC to quit early")

    pygame.init()
    clock = pygame.time.Clock()

    config['obs_type'] = 'absolute'

    # Disable display scaling for high-res monitors
    ctypes.windll.user32.SetProcessDPIAware()

    # Set up pygame window
    window_width, window_height = config['window_size'][0], config['window_size'][1]
    config['tick_rate'] = 80
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
        tag='test_suite',
        seed=config['seed']
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
        potential_gain_history = []

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
                        #print(f"Action selected: {action}")

            if done:
                break

            # Store data
            episode_observations.append(obs.copy())
            episode_actions.append(action)

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            #print(f'Agent: {obs[0]}, {obs[1]} | Target 1: {obs[2]} | Target 2: {obs[3]}')
            episode_reward += reward# * (config['gamma'] ** step_count)
            print(info["outerstep_potential_gain"])
            potential_gain_history.append(info["outerstep_potential_gain"])
            done = terminated or truncated
            step_count += 1
            #print(f'step {step_count}')

            # Render environment
            env.render()

            ##################### TEMP TESTING ##########################################
            # Calculate and draw value arrows
            current_obs = obs  # obs from the last step
            #potentials = get_directional_potential_gains(env, current_obs)

            # Get agent position for arrow drawing
            # if env.agents and len(env.aircraft_ids) > 0:
            #     agent = env.agents[env.aircraft_ids[0]]
            #     map_half_size = config["gameboard_size"] / 2
            #     draw_value_arrows(window, env, potentials, agent.x, agent.y, map_half_size)

            # Add a legend in the corner
            #font = pygame.font.Font(None, 24)
            # legend_text = ["Value Arrows:", "Green = Positive", "Red = Negative", "Length = Magnitude"]
            # for i, text in enumerate(legend_text):
            #     color = (255, 255, 255) if i == 0 else (200, 200, 200)
            #     text_surface = font.render(text, True, color)
            #     window.blit(text_surface, (10, 10 + i * 25))
            ##################### TEMP TESTING ##########################################

            pygame.display.flip()
            clock.tick(20)  # 60 FPS

        # Store episode data
        all_observations.extend(episode_observations)
        episode_rewards.append(episode_reward)
        all_actions.extend(episode_actions)
        all_potential_gain = sum(potential_gain_history)

        print(f"Human episode {episode + 1}/3 completed. Reward: {episode_reward:.2f} (Shaping: {round((all_potential_gain * config["shaping_coeff_prox"]),1)} / {episode_reward} = {round(100*(all_potential_gain * config["shaping_coeff_prox"])/abs(episode_reward),1)}% from shaping), Steps: {step_count}")
        print(f'SHAPING {all_potential_gain}')

    # Generate and save histograms
    save_histograms(all_observations, episode_rewards, all_actions, test_dir, "human")

    env.close()
    pygame.quit()
    print(f"Human playtest completed. Results saved to {test_dir}")


def test_env_random(config, test_dir=None):
    """Run env for 20 episodes, taking actions by randomly sampling from the action space"""
    print("\nStarting random test...")

    # Create environment
    env = MAISREnvVec(
        config=config,
        render_mode='headless',
        tag='test_suite'
    )

    all_observations = []
    episode_rewards = []
    all_actions = []

    for episode in range(15):
        obs, info = env.reset()
        episode_reward = 0
        episode_observations = []
        episode_actions = []
        potential_gain_history = []

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
            potential_gain_history.append(info["outerstep_potential_gain"])
            done = terminated or truncated
            step_count += 1

        # Store episode data
        all_observations.extend(episode_observations)
        episode_rewards.append(episode_reward)
        all_actions.extend(episode_actions)
        all_potential_gain = sum(potential_gain_history)

        print(f"Random episode {episode + 1}/20 completed. Reward: {episode_reward:.2f} (Shaping: {round((all_potential_gain * config["shaping_coeff_prox"]),1)} / {episode_reward} = {round(100*(all_potential_gain * config["shaping_coeff_prox"])/abs(episode_reward),1)}% from shaping), Steps: {step_count}")

    # Generate and save histograms
    save_histograms(all_observations, episode_rewards, all_actions, test_dir, "random")

    env.close()

    avg_reward = np.mean(episode_rewards)
    print(f"Average random reward: {avg_reward:.2f}")
    assert avg_reward < -3, f"Random average reward {avg_reward:.2f} must be < -3"

    print(f"Random test completed. Results saved to {test_dir}")


def test_curriculum(config):
    """Test curriculum advancement by running episodes and manually incrementing difficulty"""
    print("Starting curriculum test...")

    # Create environments similar to train_sb3.py
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor

    # Training environment (vectorized and normalized like in train_sb3)
    env = MAISREnvVec(
        config=config,
        render_mode='headless',
        tag='curriculum_test_train_0'
    )
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env)

    # Evaluation environment (normalized but not training, like in train_sb3)
    eval_env = MAISREnvVec(
        config=config,
        render_mode='headless',
        tag='curriculum_test_eval_0'
    )
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(eval_env, norm_reward=False, training=False)
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms

    model = PPO(
        "MlpPolicy",
        env,
    )

    print("Environments created and normalized")

    # Test curriculum advancement through difficulty levels
    max_difficulty_levels = 3
    min_target_ids_to_advance = config.get('min_target_ids_to_advance', 8)

    for difficulty_level in range(max_difficulty_levels):
        print(f"\n--- Testing Difficulty Level {difficulty_level} ---")

        # Set difficulty on both environments (like EnhancedWandbCallback)
        env.env_method("set_difficulty", difficulty_level)
        try:
            eval_env.env_method("set_difficulty", difficulty_level)
        except Exception as e:
            print(f"Failed to set difficulty on eval env: {e}")

        if difficulty_level == 2:
            new_lr = model.policy.optimizer.param_groups[0]['lr'] / 3
            print(f'Model lr reduced from {model.policy.optimizer.param_groups[0]['lr']} to {new_lr}')
            model.policy.optimizer.param_groups[0]['lr'] = new_lr

        # Run one episode on training env
        obs = env.reset()
        done = False
        step_count = 0
        episode_reward = 0
        max_steps = 50

        while not done and step_count < max_steps:
            # Take random action (since we don't have a trained model)
            action = [env.action_space.sample()]
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            step_count += 1

        print(f"Training episode completed: Reward={episode_reward:.2f}, Steps={step_count}")

        # Run one episode on eval env (like EnhancedWandbCallback evaluation)
        obs = eval_env.reset()
        done = False
        step_count = 0
        episode_reward = 0

        while not done and step_count < max_steps:
            # Take random action (deterministic=True would be used with real model)
            action = [eval_env.action_space.sample()]
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward[0]
            step_count += 1

        print(f"Eval episode completed: Reward={episode_reward:.2f}, Steps={step_count}")
        if info and len(info) > 0 and "target_ids" in info[0]:
            target_ids = info[0]["target_ids"]
            episode_length = info[0]["episode"]["l"]
            print(f"Target IDs collected: {target_ids}")
            print(f"Episode length: {episode_length}")

            # # Simulate curriculum advancement logic (like EnhancedWandbCallback)
            # if target_ids >= min_target_ids_to_advance:
            #     print(f"✓ Would advance curriculum (target_ids {target_ids} >= threshold {min_target_ids_to_advance})")
            # else:
            #     print(
            #         f"✗ Would not advance curriculum (target_ids {target_ids} < threshold {min_target_ids_to_advance})")

    # Test difficulty getter
    try:
        current_difficulty = env.get_attr("difficulty")[0]
        print(f"\nFinal difficulty level: {current_difficulty}")
    except Exception as e:
        print(f"Could not get difficulty attribute: {e}")

    env.close()
    eval_env.close()
    print("Curriculum test completed.")


def test_env_train(config):
    """Run brief training run using Stable-baselines3"""
    print("Starting training test...")

    machine_name = 'testenv'
    project_name = 'maisr-rl-tests'

    config['n_envs'] = multiprocessing.cpu_count()
    config['num_timesteps'] = 3e5

    train(
        config,
        n_envs=multiprocessing.cpu_count(),
        load_path=None,
        machine_name=machine_name,
        project_name=project_name,
        save_model=False
    )

    print("Training test completed.")

def test_env_overfit(config):
    """Test if agent can overfit to 1 level"""
    print("Starting training test...")

    machine_name = 'home_overfittests'#'testenv'
    project_name = 'maisr-rl'

    config['n_envs'] = multiprocessing.cpu_count()
    #config["levels_per_lesson"] = {"0": 1, "1": 1, "2":  1}
    #config["num_timesteps"] = 8e5
    #config['lr'] = 0.001

    for levels_per_lesson in [{"0": 1, "1": 1, "2":  1}, {"0": 3, "1": 3, "2":  3}]:
        for obs_type in ['absolute']:
            config['obs_type'] = obs_type
            config["levels_per_lesson"] = levels_per_lesson
            config['n_eval_episodes'] = levels_per_lesson["0"]

            train(
                config,
                n_envs=multiprocessing.cpu_count(),
                load_path=None,
                machine_name=machine_name,
                project_name=project_name,
                save_model=False
            )

    print("Training test completed.")

def test_cnn_observations(config):
    """Test CNN pixel observation generation"""

    print("Testing CNN pixel observations...")

    config['obs_type'] = 'pixel'

    # Create environment with pixel observations
    env = MAISREnvVec(
        config=config,
        render_mode='headless',
        tag='test_0'
    )

    print(f"CNN observation space: {env.observation_space}")

    # Reset environment
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Observation range: [{obs.min()}, {obs.max()}]")

    # Take a few random actions and collect observations
    observations = []
    for i in range(10):
        action = 0#env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs.copy())
        print(f"Step {i+1}: Action={action}, Reward={reward:.3f}, Targets identified={info.get('target_ids', 0)}")

    # Plot the observations
    fig, axes = plt.subplots(1, 10, figsize=(30, 3))
    for i, obs in enumerate(observations):
        # Remove channel dimension for plotting (84, 84, 1) -> (84, 84)
        img = obs.squeeze()
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[i].set_title(f'Step {i+1}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('cnn_observations_test.png', dpi=150)
    print("Saved observations plot to 'cnn_observations_test.png'")

    # Test comparison between vector and pixel observations
    print("\nComparing vector vs pixel observations...")

    # Create vector environment for comparison
    env_vector = MAISREnvVec(
        config=config,
        render_mode='headless',
        tag='test_vector_0',
    )

    print(f"Vector observation space: {env_vector.observation_space}")

    # Reset both environments with same seed for comparison
    np.random.seed(42)
    obs_pixel, _ = env.reset()
    np.random.seed(42)
    obs_vector, _ = env_vector.reset()

    print(f"Pixel obs shape: {obs_pixel.shape}")
    print(f"Vector obs shape: {obs_vector.shape}")
    print(f"Vector obs: {obs_vector}")

    # Test that both environments produce same game state
    action = 2  # Move right

    # Step pixel environment
    obs_pixel_next, reward_pixel, term_pixel, trunc_pixel, info_pixel = env.step(action)

    # Step vector environment
    obs_vector_next, reward_vector, term_vector, trunc_vector, info_vector = env_vector.step(action)

    print(f"\nAfter action {action}:")
    print(f"Pixel env - Reward: {reward_pixel:.3f}, Targets: {info_pixel.get('target_ids', 0)}")
    print(f"Vector env - Reward: {reward_vector:.3f}, Targets: {info_vector.get('target_ids', 0)}")

    # Verify rewards are the same (game logic should be identical)
    assert abs(reward_pixel - reward_vector) < 1e-2, f"Rewards differ: {reward_pixel} vs {reward_vector}"
    print("✓ Rewards match between pixel and vector environments")

    env.close()
    env_vector.close()
    print("\n✓ CNN observation test completed successfully!")

if __name__ == "__main__":

    config = load_env_config('configs/june12a.json')
    config['eval_freq'] = 4900

    config['obs_type'] = 'absolute'
    config['n_eval_episodes'] = 5


    print("\nStarting Environment Test Suite...")
    print("=" * 50)

    # Create shared test directory for all tests
    shared_test_dir = create_test_directory()
    print(f"All test results will be saved to: {shared_test_dir}")

    try:
        #test_env_humanplaytest(config, test_dir=shared_test_dir)
        #test_curriculum(config)
        test_env_heuristic(improved_heuristic_policy, config, render=True, test_dir=shared_test_dir)
        test_env_random(config, test_dir=shared_test_dir)
        test_env_badheuristic(badheuristic_policy, config,test_dir=shared_test_dir)
        #test_cnn_observations(config)
        test_env_train(config)
        #test_env_overfit(config)
        pass

    except KeyboardInterrupt: print("\nTest suite interrupted by user.")
    except Exception as e:
        print(f"\nError during test execution: {e}")
        import traceback
        traceback.print_exc()

    print("Environment Test Suite completed!")

    # Combine and display all plots
    try: combine_and_display_plots(shared_test_dir)
    except Exception as e:
        print(f"Error combining plots: {e}")
        print(f"Individual plots are still available in: {shared_test_dir}")