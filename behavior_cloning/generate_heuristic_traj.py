import os
from datetime import datetime

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper

from env_combined import MAISREnvVec
from utility.data_logging import load_env_config
from sb3_trainagent import make_env

import numpy as np


def heuristic_policy(observation, state, dones):
    """
    Heuristic policy that selects the closest unidentified target (vectorized).

    Args:
        observation: Can be either:
            - Single observation (np.ndarray) for non-vectorized env
            - Batch of observations (np.ndarray with shape (n_envs, obs_dim)) for vectorized env

    Returns:
        np.ndarray: Action(s) as [x, y] coordinates in normalized range [-1, 1]
                   Shape: (2,) for single env or (n_envs, 2) for vectorized env
    """

    # Handle both single and batched observations
    if len(observation.shape) == 1:
        # Single observation
        return _process_single_observation(observation)
    else:
        # Batched observations from vectorized environment
        batch_size = observation.shape[0]
        actions = np.zeros((batch_size, 2), dtype=np.float32)

        for i in range(batch_size):
            actions[i] = _process_single_observation(observation[i])

        return actions, None


def _process_single_observation(observation, obs_type='relative'):
    """
    Process a single observation and return a single action.

    Args:
        observation: Single observation array
        obs_type: 'absolute' or 'relative' - should match your environment config
    """
    # Extract agent position (always absolute in [0,1] range)
    agent_x = observation[0]
    agent_y = observation[1]

    # Calculate maximum number of targets based on observation size
    max_targets = (len(observation) - 2) // 3

    if max_targets == 0:
        # No targets, stay at current position
        return np.array([0.0, 0.0], dtype=np.float32)

    # Extract target data using array slicing and reshaping
    target_data = observation[2:2 + max_targets * 3].reshape(max_targets, 3)

    # Split into separate arrays for each attribute
    info_levels = target_data[:, 0]  # Shape: (max_targets,)
    target_x_raw = target_data[:, 1]  # Shape: (max_targets,)
    target_y_raw = target_data[:, 2]  # Shape: (max_targets,)

    # Based on your environment code, both absolute and relative modes
    # store target positions in absolute [0,1] coordinates
    target_x = target_x_raw
    target_y = target_y_raw

    # Create masks for valid and unidentified targets
    # Check if target exists (non-zero coordinates OR has info level > 0)
    exists_mask = (target_x > 0) | (target_y > 0) | (info_levels > 0)
    unidentified_mask = info_levels < 1.0  # Not fully identified
    valid_mask = exists_mask & unidentified_mask  # Both conditions

    #print(f"Agent position: ({agent_x:.3f}, {agent_y:.3f})")
    #print(f"Valid targets found: {np.sum(valid_mask)}")

    if np.sum(valid_mask) > 0:
        valid_indices = np.where(valid_mask)[0]
        #print(f"Valid target positions: {[(target_x[i], target_y[i]) for i in valid_indices[:3]]}")  # Show first 3

    # If no valid targets, stay at current position (no movement)
    if not np.any(valid_mask):
        #print("No valid targets found, staying in place")
        return np.array([0.0, 0.0], dtype=np.float32)

    # Calculate distances to valid targets only
    valid_target_x = target_x[valid_mask]
    valid_target_y = target_y[valid_mask]

    distances = np.sqrt((valid_target_x - agent_x) ** 2 + (valid_target_y - agent_y) ** 2)

    # Find index of closest valid target (in the valid targets array)
    closest_valid_idx = np.argmin(distances)

    # Get coordinates of closest target
    closest_target_x = valid_target_x[closest_valid_idx]
    closest_target_y = valid_target_y[closest_valid_idx]

    closest_distance = distances[closest_valid_idx]
    #print(f"Closest target at ({closest_target_x:.3f}, {closest_target_y:.3f}), distance: {closest_distance:.3f}")

    if obs_type == 'absolute':
        # For absolute mode: convert target position [0,1] to action [-1,1]
        action_x = closest_target_x * 2 - 1
        action_y = closest_target_y * 2 - 1

    elif obs_type == 'relative':
        # For relative mode: calculate movement direction
        # Based on your environment's process_action method for relative mode
        dx = closest_target_x - agent_x
        dy = closest_target_y - agent_y

        # Normalize to action space [-1,1]
        # The environment uses max_move_distance = gameboard_size * 0.3
        # Since coordinates are normalized [0,1], max movement is 0.3
        max_movement = 0.3

        # Scale movement to fit in [-1,1] action space
        action_x = np.clip(dx / max_movement, -1, 1)
        action_y = np.clip(dy / max_movement, -1, 1)

    else:
        raise ValueError(f"Unknown obs_type: {obs_type}")

    action = np.array([action_x, action_y], dtype=np.float32)
    #print(f"Taking action: [{action_x:.3f}, {action_y:.3f}]")

    return action

def generate_heuristic_trajectories(expert_policy, env_config, n_episodes=50, run_name = 'none', save_expert_trajectory = True):
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

    # Generate expert trajectories using the heuristic policy
    print(f"Collecting {n_episodes} episodes from expert policy...")
    rollouts = rollout.rollout(
        expert_policy,
        env,
        rollout.make_sample_until(min_episodes=n_episodes),
        rng=rng
    )

    if save_expert_trajectory:
        from imitation.data import serialize
        os.makedirs('./expert_trajectories', exist_ok=True)
        serialize.save(f'./expert_trajectories/{run_name}', rollouts)

    transitions = rollout.flatten_trajectories(rollouts)
    print(f"Generated {len(transitions)} transitions from {len(rollouts)} episodes")

    return


if __name__ == "__main__":

    # Set parameters
    config_name = '../config_files/rl_default.json'
    n_episodes = 50000


    run_name = 'expert_trajectory'+str(n_episodes)+datetime.now().strftime("%m%d_%H%M")
    env_config = load_env_config(config_name)

    # Create temporary environment to get action space
    temp_env = DummyVecEnv([make_env(env_config, i, env_config['seed'] + i, run_name=run_name) for i in range(1)])
    temp_env.close()

    # Run the behavior cloning pipeline
    trained_policy = generate_heuristic_trajectories(
        expert_policy=heuristic_policy,
        env_config=env_config,
        n_episodes=n_episodes,
        run_name = run_name,
        save_expert_trajectory = True
    )
