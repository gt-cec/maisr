import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.monitor import Monitor

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env

from env_combined import MAISREnvVec
from utility.data_logging import load_env_config
from sb3_trainagent import make_env


"""
"""


def heuristic_policy(observation):
    # TODO vectorize (https://claude.ai/chat/9e0aa009-e9be-48b2-bc46-3192cd2dd1ee)
    """
    Heuristic policy that selects the closest unidentified target.

    Args:
        observation (np.ndarray): Observation vector in absolute format
            - observation[0]: agent_x (normalized 0-1)
            - observation[1]: agent_y (normalized 0-1)
            - observation[2+i*3]: target_i info_level (0=unknown, 0.5=low_qual, 1.0=identified)
            - observation[3+i*3]: target_i x position (normalized 0-1)
            - observation[4+i*3]: target_i y position (normalized 0-1)

    Returns:
        np.ndarray: Action as [x, y] coordinates in normalized range [-1, 1]
    """

    # Extract agent position
    agent_x = observation[0]
    agent_y = observation[1]

    # Calculate maximum number of targets based on observation size
    # obs_size = 2 + 3 * max_targets, so max_targets = (obs_size - 2) / 3
    max_targets = (len(observation) - 2) // 3

    closest_distance = float('inf')
    closest_target_x = agent_x  # Default to current position
    closest_target_y = agent_y

    # Iterate through all possible targets
    for i in range(max_targets):
        info_level_idx = 2 + i * 3
        target_x_idx = 3 + i * 3
        target_y_idx = 4 + i * 3

        # Check if this target slot has data (info_level > 0 or position != 0)
        info_level = observation[info_level_idx]
        target_x = observation[target_x_idx]
        target_y = observation[target_y_idx]

        # Skip if target doesn't exist (all zeros) or is already fully identified
        if (target_x == 0 and target_y == 0) or info_level >= 1.0:
            continue

        # Calculate distance to this target
        distance = np.sqrt((target_x - agent_x) ** 2 + (target_y - agent_y) ** 2)

        # Update closest target if this one is closer
        if distance < closest_distance:
            closest_distance = distance
            closest_target_x = target_x
            closest_target_y = target_y

    # Convert from normalized [0,1] coordinates to action space [-1,1]
    action_x = closest_target_x * 2 - 1  # Convert 0-1 to -1 to 1
    action_y = closest_target_y * 2 - 1  # Convert 0-1 to -1 to 1

    return np.array([action_x, action_y], dtype=np.float32)


def behavior_cloning_pipeline(expert_policy, env_config, n_episodes=50, n_epochs=10, run_name = 'none', save_expert_trajectory = True):
    """
    Complete behavior cloning pipeline:
    1. Load heuristic policy
    2. Generate expert trajectories
    3. Train BC policy

    Args:
        expert_policy: Policy function that takes in an observation and returns an action
        env_name: Gymnasium environment name
        n_episodes: Number of episodes to collect from expert
        n_epochs: Number of training epochs for BC

    Returns:
        bc_trainer: Trained behavior cloning agent
    """

    # Create vectorized environment with RolloutInfoWrapper
    rng = np.random.default_rng(0)

    # TODO might need to use imitation's makevecenv
    env = MAISREnvVec(
        env_config,
        None,
        render_mode='headless',
        tag='train',
        run_name=run_name,
    )
    env = Monitor(env)
    # TODO add RolloutInfoWrapper(env)



    # Generate expert trajectories using the heuristic policy
    print(f"Collecting {n_episodes} episodes from expert policy...")
    rollouts = rollout.rollout(
        heuristic_policy,
        env,
        rollout.make_sample_until(min_episodes=n_episodes),
        rng=rng
    )

    if save_expert_trajectory:
        from imitation.data import serialize
        serialize.save('/expert_trajectories/expert_trajectory.json', rollouts) # TODO check filetype

    transitions = rollout.flatten_trajectories(rollouts)
    print(f"Generated {len(transitions)} transitions from {len(rollouts)} episodes")


    # Initialize behavior cloning trainer
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng
    )

    # Train the BC policy
    print(f"Training behavior cloning policy for {n_epochs} epochs...")
    bc_trainer.train(n_epochs=n_epochs)

    print("Training complete!")
    trained_policy = bc_trainer.policy
    env.close()

    return trained_policy


if __name__ == "__main__":
    config_name = './config_files/rl_training_timepenalty.json'
    run_name = 'bc_test'

    env_config = load_env_config(config_name)

    # Create temporary environment to get action space
    temp_env = DummyVecEnv([make_env(env_config, i, env_config['seed'] + i, run_name=run_name) for i in range(1)])
    temp_env.close()

    # Run the behavior cloning pipeline
    trained_policy = behavior_cloning_pipeline(
        expert_policy=heuristic_policy,
        env_config=env_config,
        n_episodes=2000,
        n_epochs=10,
        run_name = run_name,
        save_expert_trajectory = True
    )


    ####################################### Save the model for later use #######################################

    env = DummyVecEnv([make_env(env_config, i, env_config['seed'] + i, run_name=run_name) for i in range(1)])

    ppo_model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=env_config.get('lr', 3e-4),
        seed=env_config.get('seed', 42),
        device='cpu',
        gamma=env_config.get('gamma', 0.99)
    )

    # Replace the PPO policy with the trained BC policy
    ppo_model.policy = trained_policy

    # Save using SB3 format
    ppo_model.save(f"./bc_models/bc_policy")
