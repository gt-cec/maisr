import os
from datetime import datetime
import math

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
        return _process_single_observation_vectorized(observation)
    else:
        # Batched observations from vectorized environment
        batch_size = observation.shape[0]
        actions = np.zeros((batch_size, 2), dtype=np.float32)

        for i in range(batch_size):
            actions[i] = _process_single_observation_vectorized(observation[i])

        return actions, None


def _process_single_observation_vectorized(observation, obs_type='relative'):
    """
    Vectorized version: Process a single observation and return a single action.

    Args:
        observation: Single observation array
        obs_type: 'absolute' or 'relative' - should match your environment config
    """

    # Convert to numpy array for vectorized operations
    obs = np.array(observation)

    # Direction mapping - convert to numpy arrays
    directions = np.array([
        [0, 1],  # up
        [1, 1],  # up-right
        [1, 0],  # right
        [1, -1],  # down-right
        [0, -1],  # down
        [-1, -1],  # down-left
        [-1, 0],  # left
        [-1, 1]  # up-left
    ], dtype=float)

    # Extract agent position
    agent_pos = obs[:2]  # [agent_x, agent_y]

    # Extract target information vectorized
    max_targets = 5
    target_data = obs[2:2 + max_targets * 3].reshape(max_targets, 3)

    # Check which targets are valid (non-zero position)
    valid_mask = (target_data[:, 1] != 0) | (target_data[:, 2] != 0)

    if not np.any(valid_mask):
        # No targets at all, just move up
        return 0

    # Filter to valid targets only
    valid_targets = target_data[valid_mask]

    # Calculate distances vectorized
    target_positions = valid_targets[:, 1:3]  # x, y coordinates
    distances = np.linalg.norm(target_positions - agent_pos, axis=1)

    # Filter to unidentified targets (info_level < 1.0)
    unidentified_mask = valid_targets[:, 0] < 1.0

    if np.any(unidentified_mask):
        # Find closest unidentified target
        unidentified_distances = distances[unidentified_mask]
        closest_idx = np.argmin(unidentified_distances)
        # Map back to original valid targets index
        unidentified_indices = np.where(unidentified_mask)[0]
        target_idx = unidentified_indices[closest_idx]
    else:
        # No unidentified targets, move toward closest identified target
        target_idx = np.argmin(distances)

    # Get target position
    target_pos = target_positions[target_idx]

    # Calculate direction vector
    direction_to_target = target_pos - agent_pos

    # Handle case where we're already at target
    if np.linalg.norm(direction_to_target) < 1e-6:
        return 0

    # Normalize direction vectors
    direction_norms = np.linalg.norm(directions, axis=1)
    normalized_directions = directions / direction_norms[:, np.newaxis]

    # Calculate dot products with all directions at once
    dot_products = np.dot(normalized_directions, direction_to_target)

    # Find best action
    best_action = np.argmax(dot_products)

    return int(best_action)

# def _process_single_observation(observation, obs_type='relative'):
#     """
#     Process a single observation and return a single action.
#
#     Args:
#         observation: Single observation array
#         obs_type: 'absolute' or 'relative' - should match your environment config
#     """
#
#     # Direction mapping from the environment
#     direction_map = {
#         0: (0, 1),  # up
#         1: (1, 1),  # up-right
#         2: (1, 0),  # right
#         3: (1, -1),  # down-right
#         4: (0, -1),  # down
#         5: (-1, -1),  # down-left
#         6: (-1, 0),  # left
#         7: (-1, 1)  # up-left
#     }
#
#     # Extract agent position
#     agent_x = observation[0]
#     agent_y = observation[1]
#
#     # Extract target information
#     max_targets = 5  # From the environment code
#     targets = []
#
#     for i in range(max_targets):
#         info_idx = 2 + i * 3
#         x_idx = 3 + i * 3
#         y_idx = 4 + i * 3
#
#         # Check if this target slot has data (non-zero position indicates a real target)
#         if info_idx < len(observation) and (observation[x_idx] != 0 or observation[y_idx] != 0):
#             info_level = observation[info_idx]
#             target_x = observation[x_idx]
#             target_y = observation[y_idx]
#
#             targets.append({
#                 'info_level': info_level,
#                 'x': target_x,
#                 'y': target_y,
#                 'distance': math.sqrt((target_x - agent_x) ** 2 + (target_y - agent_y) ** 2)
#             })
#
#     # Filter to only unidentified targets (info_level < 1.0)
#     unidentified_targets = [t for t in targets if t['info_level'] < 1.0]
#
#     # If no unidentified targets, move toward the closest identified target or stay put
#     if not unidentified_targets:
#         if targets:
#             # Move toward closest identified target
#             closest_target = min(targets, key=lambda t: t['distance'])
#         else:
#             # No targets at all, just move up
#             return 0
#     else:
#         # Find the closest unidentified target
#         closest_target = min(unidentified_targets, key=lambda t: t['distance'])
#
#     # Calculate direction vector from agent to closest target
#     target_x = closest_target['x']
#     target_y = closest_target['y']
#
#     dx = target_x - agent_x
#     dy = target_y - agent_y
#
#     # Handle the case where we're already at the target
#     if abs(dx) < 1e-6 and abs(dy) < 1e-6:
#         return 0  # Default to moving up
#
#     # Find the best direction by calculating dot product with each direction vector
#     best_action = 0
#     best_dot_product = -float('inf')
#
#     for action, (dir_x, dir_y) in direction_map.items():
#         # Normalize the direction vector
#         dir_length = math.sqrt(dir_x * dir_x + dir_y * dir_y)
#         norm_dir_x = dir_x / dir_length
#         norm_dir_y = dir_y / dir_length
#
#         # Calculate dot product with target direction
#         dot_product = dx * norm_dir_x + dy * norm_dir_y
#
#         if dot_product > best_dot_product:
#             best_dot_product = dot_product
#             best_action = action
#
#     #print(best_action)
#     return best_action

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
    config_name = '../config_files/bc_config.json'
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
