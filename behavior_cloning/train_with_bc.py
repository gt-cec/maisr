import ctypes
import os
from datetime import datetime

import pygame
import torch
import numpy as np
import datasets
import glob
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import serialize
from imitation.data.huggingface_utils import TrajectoryDatasetSequence

from env_combined import MAISREnvVec
from utility.data_logging import load_env_config
from sb3_trainagent import make_env


def load_multiple_arrow_files(trajectory_dir_or_pattern):
    """Load and combine multiple arrow files"""
    if os.path.isdir(trajectory_dir_or_pattern): # If it's a directory, find all .arrow files
        arrow_files = glob.glob(os.path.join(trajectory_dir_or_pattern, "*.arrow"))
    else: # If it's a pattern, use it directly
        arrow_files = glob.glob(trajectory_dir_or_pattern)

    arrow_files.sort()  # Ensure consistent ordering
    print(f"Found {len(arrow_files)} arrow files: {[os.path.basename(f) for f in arrow_files]}")

    all_transitions = []
    for arrow_file in arrow_files:
        print('Loading trajectories from arrow file')
        dataset = datasets.load_dataset("arrow", data_files=arrow_file, split="train")
        trajectories = TrajectoryDatasetSequence(dataset)
        transitions = rollout.flatten_trajectories(trajectories)
        all_transitions.append(transitions)
        print(f"  Loaded {len(transitions)} transitions")

    # Combine all transitions
    if all_transitions:
        combined_transitions = all_transitions[0]
        for transitions in all_transitions[1:]:
            combined_transitions = combined_transitions.concatenate(transitions)
        print(f"Combined total: {len(combined_transitions)} transitions")
        return combined_transitions
    else:
        raise ValueError("No arrow files could be loaded successfully")


def train_with_bc(expert_trajectory_path, env_config, bc_config, run_name='none'):
    """
    Complete behavior cloning pipeline:
    1. Load expert trajectories from multiple .arrow files
    2. Train BC policy

    Args:
        expert_trajectory_path: Path to directory containing .arrow files or glob pattern
        env_config: Environment configuration
        bc_config: BC configuration dict with 'batch_size' and 'n_epochs'
        run_name: Name for this training run

    Returns:
        None (saves trained model to disk)
    """

    # Load expert trajectory from Arrow file
    transitions = load_multiple_arrow_files(expert_trajectory_path)
    print('Dataset loaded')

    # Create vectorized environment with RolloutInfoWrapper
    rng = np.random.default_rng(0)

    def make_env_with_wrapper(env_config, rank, seed, run_name):
        def _init():
            env = MAISREnvVec(
                env_config,
                None,
                render_mode='headless',
                tag='train',
                run_name=run_name,
            )
            env = Monitor(env)
            env = RolloutInfoWrapper(env)
            return env

        return _init

    env = DummyVecEnv([make_env_with_wrapper(env_config, 0, env_config['seed'], run_name)])

    ppo_model = PPO(
        "MlpPolicy",
        env,
        verbose=2,
        # tensorboard_log=f"logs/runs/{run.id}",
        batch_size=bc_config['batch_size'],
        n_steps=env_config['ppo_update_steps'],
        learning_rate=env_config['lr'],
        seed=env_config['seed'],
        device='cpu',
        gamma=env_config['gamma'],
        ent_coef=env_config['entropy_regularization']
    )

    # Initialize behavior cloning trainer
    bc_trainer = bc.BC(
        policy=ppo_model.policy,
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
        device=torch.device('cpu'),
        batch_size=bc_config['batch_size'],
    )

    # Train the BC policy
    print(f"Training behavior cloning policy for {bc_config['n_epochs']} epochs...")
    bc_trainer.train(n_epochs=bc_config['n_epochs'])

    trained_policy = bc_trainer.policy
    env.close()

    print("Training complete!")

    ####################################### Save the model for later use #######################################
    save_env = DummyVecEnv([make_env(env_config, i, env_config['seed'] + i, run_name=run_name) for i in range(1)])

    ppo_model = PPO(
        "MlpPolicy",
        save_env,
        verbose=2,
        learning_rate=env_config.get('lr', 3e-4),
        seed=env_config.get('seed', 42),
        device='cpu',
        gamma=env_config.get('gamma', 0.99)
    )
    ppo_model.policy = trained_policy

    # Save
    os.makedirs('./bc_models', exist_ok=True)
    trained_model_path = f"./bc_models/bc_policy_{run_name}"
    ppo_model.save(trained_model_path)
    print(f'Saved trained model to "./bc_models/bc_policy_{run_name}"')

    return trained_model_path

def evaluate_bc_policy(
        env_config,
        load_path,
        num_episodes=20,
        render=True):

    if render:
        pygame.init()
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
            run_name='run_name',
            tag='bc_eval'
        )
    else:
        eval_env = MAISREnvVec(
            env_config,
            None,
            render_mode='human' if render else 'headless',
            tag='bc_eval',
            run_name=run_name,
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
            action = trained_policy.predict(obs)[0]
            obs, reward, terminated, truncated, info = eval_env.step(action)
            if render: eval_env.render()
            episode_reward += reward

        reward_history.append(episode_reward)

    eval_env.close()
    print(f'Evaluated {num_episodes} episodes. Average reward: {np.mean(reward_history)}')

    return reward_history, np.mean(reward_history), np.std(reward_history)


if __name__ == "__main__":

    ################### Set parameters ###################
    config_name = '../config_files/june9_cloning.json'
    train_new = False
    evaluate = True
    # For eval, load trained BC model from path
    load_path = 'bc_models/bc_policy_bc_run_0609_1412.zip'

    ######################################################

    bc_config = {
        'batch_size': 256,
        'n_epochs': 10}

    env_config = load_env_config(config_name)
    env_config['curriculum_type'] = 'none'
    env_config['use_beginner_levels'] = False

    run_name = 'bc_run_' + datetime.now().strftime("%m%d_%H%M")

    results_dict = {} # Will be of the form batchsize,n_epochs = (reward_history, average_reward, std_reward)

    for batch_size in [256, 512, 1024, 2048]:
        for n_epochs in [10, 20]:
            bc_config['batch_size'] = batch_size
            bc_config['n_epochs'] = n_epochs

            trained_model_path = train_with_bc(
                                    expert_trajectory_path='./expert_trajectories/experttraj_mp_10000eps_0609_1506',
                                    env_config=env_config,
                                    bc_config=bc_config,
                                    run_name=run_name
                                )
            reward_history, average_reward, std_reward = evaluate_bc_policy(env_config, trained_model_path, render=False)
            results_dict[(batch_size, n_epochs)] = (reward_history, average_reward, std_reward)

    # Convert results_dict to JSON-serializable format
    import json
    json_results = {}
    for (batch_size, n_epochs), (reward_history, avg_reward, std_reward) in results_dict.items():
        json_results[f"batch_{batch_size}_epochs_{n_epochs}"] = {
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "average_reward": float(avg_reward),
            "std_reward": float(std_reward)
        }

    # Save to JSON file
    results_filename = f"bc_training_results_{run_name}.json"
    with open(results_filename, 'w') as f:
        json.dump(json_results, f, indent=4)

    print(f"Results saved to {results_filename}")

    print('#######################################################################')
    print('\nTraining results:')
    try:
        print(f'Config with highest average reward: {max(results_dict.keys(), key=lambda k: results_dict[k][1])}')
    except:
        print('max get failed. Printing results dict')
        print(results_dict)

    print('#######################################################################')

    # if train_new:
    #     train_with_bc(
    #         expert_trajectory_path = './expert_trajectories/expert_trajectory_200episodes_0609_1408',
    #         env_config=env_config,
    #         bc_config=bc_config,
    #         run_name=run_name
    #     )
    #
    # if evaluate:
    #     evaluate_bc_policy(env_config, load_path, render=False)