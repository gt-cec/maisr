import os
from datetime import datetime
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
    ppo_model.save(f"./bc_models/bc_policy_{run_name}")
    print(f'Saved trained model to "./bc_models/bc_policy_{run_name}"')

    return

def evaluate_bc_policy(env_config, load_path):
    from stable_baselines3.common.evaluation import evaluate_policy

    eval_env = MAISREnvVec(
        env_config,
        None,
        render_mode='headless',
        tag='eval',
        run_name=run_name,
    )
    eval_env = Monitor(eval_env)
    print('Instantiated eval_env')

    trained_policy = PPO(
        "MlpPolicy",
        eval_env,
        verbose=1,
        # tensorboard_log=f"logs/runs/{run.id}",
        batch_size=bc_config['batch_size'],
        n_steps=env_config['ppo_update_steps'],
        learning_rate=env_config['lr'],
        seed=env_config['seed'],
        device='cpu',
        gamma=env_config['gamma'],
        ent_coef=env_config['entropy_regularization']
    )
    print('instantiated policy')
    trained_policy = trained_policy.__class__.load(load_path, env=eval_env)
    print('loaded bc-trained policy')

    mean_reward, std_reward = evaluate_policy(trained_policy, eval_env, n_eval_episodes=10)
    return mean_reward, std_reward

if __name__ == "__main__":

    config_name = '../config_files/june9a.json'
    train_new = True
    evaluate = False
    load_path = None #'./bc_models/bc_policy_bc_run_0603_1202'

    bc_config = {
        'batch_size': 256,
        'n_epochs': 10}

    env_config = load_env_config(config_name)
    run_name = 'bc_run_' + datetime.now().strftime("%m%d_%H%M")

    if train_new:
        train_with_bc(
            expert_trajectory_path = './expert_trajectories/expert_trajectory_continuousactions_92obs_50000games',
            env_config=env_config,
            bc_config=bc_config,
            run_name=run_name
        )

    if evaluate:
        evaluate_bc_policy(env_config, load_path)