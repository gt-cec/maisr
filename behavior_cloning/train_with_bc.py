import os
from datetime import datetime
import torch
from imitation.data import serialize
import datasets
from imitation.data.huggingface_utils import TrajectoryDatasetSequence

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper

from env_combined import MAISREnvVec
from utility.data_logging import load_env_config
from sb3_trainagent import make_env

import numpy as np

def train_with_bc(expert_trajectory_path, env_config, bc_config, run_name='none'):
    """
    Complete behavior cloning pipeline:
    1. Load expert trajectories
    2. Train BC policy

    Args:
        expert_trajectory_path: Path to expert trajectory .arrow file
        env_config: Environment configuration
        n_episodes: Number of episodes to collect from expert (not used when loading from file)
        n_epochs: Number of training epochs for BC

    Returns:
        bc_trainer: Trained behavior cloning agent
    """

    # Load expert trajectory from Arrow file
    #transitions = serialize.load(expert_trajectory_path)
    try:
        # Method 1: Use imitation's serialize.load() on the directory
        transitions = serialize.load(expert_trajectory_path)
    except Exception as e:
        print(f"Method 1 failed: {e}")
        try:
            # Method 2: Load as HuggingFace dataset and convert
            print("Trying to load as HuggingFace dataset...")
            dataset = datasets.load_dataset("arrow", data_files=expert_trajectory_path, split="train")
            trajectories = TrajectoryDatasetSequence(dataset)
            # Convert to transitions
            transitions = rollout.flatten_trajectories(trajectories)
        except Exception as e2:
            print(f"Method 2 failed: {e2}")
            # Method 3: Load the entire dataset directory
            print("Trying to load dataset directory...")
            dataset = datasets.load_dataset(expert_trajectory_path)
            trajectories = TrajectoryDatasetSequence(dataset["train"])
            transitions = rollout.flatten_trajectories(trajectories)

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
        verbose=1,
        learning_rate=env_config.get('lr', 3e-4),
        seed=env_config.get('seed', 42),
        device='cpu',
        gamma=env_config.get('gamma', 0.99)
    )

    # Replace the PPO policy with the trained BC policy
    ppo_model.policy = trained_policy

    # Save using SB3 format
    os.makedirs('./bc_models', exist_ok=True)
    ppo_model.save(f"./bc_models/bc_policy_{run_name}")
    print(f'Saved trained model to "./bc_models/bc_policy_{run_name}"')

    return


if __name__ == "__main__":
    config_name = '../config_files/rl_default.json'

    train_new = True
    evaluate = True
    load_path = None #'./bc_models/bc_policy_bc_run_0603_1202'

    bc_config = {
        'batch_size': 64,
        'n_epochs': 1,
    }

    env_config = load_env_config(config_name)
    run_name = 'bc_run_' + datetime.now().strftime("%m%d_%H%M")

    # Run the behavior cloning pipeline
    if train_new:
        trained_policy = train_with_bc(
            expert_trajectory_path='./expert_trajectories/bc_run_0603_1052/data-00000-of-00001.arrow',
            env_config=env_config,
            bc_config=bc_config,
            run_name=run_name
        )

    # Evaluate the policy
    if evaluate:
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
            #tensorboard_log=f"logs/runs/{run.id}",
            batch_size=bc_config['batch_size'],
            n_steps=env_config['ppo_update_steps'],
            learning_rate=env_config['lr'],
            seed=env_config['seed'],
            device='cpu',
            gamma=env_config['gamma'],
            ent_coef=env_config['entropy_regularization']
        )
        print('instantiated trained_policy')
        if load_path is not None:
            trained_policy = trained_policy.__class__.load(load_path, env=eval_env)
            print('loaded trained policy')

        from stable_baselines3.common.evaluation import evaluate_policy
        mean_reward, std_reward = evaluate_policy(trained_policy, eval_env, n_eval_episodes=10)