import os
from datetime import datetime
import math

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper

from env_20targets import MAISREnvVec
from policies.greedy_heuristic_improved import improved_heuristic_policy
from utility.data_logging import load_env_config
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial

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
        return heuristic_process_single_observation_vectorized(observation)
    else:
        # Batched observations from vectorized environment
        batch_size = observation.shape[0]
        actions = np.zeros((batch_size,), dtype=np.float32)

        for i in range(batch_size):
            actions[i] = np.int64(heuristic_process_single_observation_vectorized(observation[i]))

        return actions, None


def heuristic_process_single_observation_vectorized(observation, obs_type='relative'):
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

    #print(f'Chose action {best_action} {type(best_action)}')
    return int(best_action)


import numpy as np

# Global state to track current waypoint and target
_current_waypoint = None
_current_target_pos = None
_waypoint_threshold = 15.0  # Distance threshold to consider waypoint "reached"


def badheuristic_policy(observation, state, dones):
    """
    Heuristic policy that selects the FARTHEST unidentified target (vectorized).
    Maintains current waypoint until reached.

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
        return badheuristic_process_single_observation_vectorized(observation)
    else:
        # Batched observations from vectorized environment
        batch_size = observation.shape[0]
        actions = np.zeros((batch_size,), dtype=np.float32)

        for i in range(batch_size):
            actions[i] = np.int64(badheuristic_process_single_observation_vectorized(observation[i]))

        return actions, None

def badheuristic_process_single_observation_vectorized(observation, obs_type='relative'):
    """
    Vectorized version: Process a single observation and return a single action.
    Maintains current waypoint until the agent reaches it.

    Args:
        observation: Single observation array
        obs_type: 'absolute' or 'relative' - should match your environment config
    """
    global _current_waypoint, _current_target_pos, _waypoint_threshold

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

    # Check if we have a current waypoint and if we've reached it
    waypoint_reached = False
    if _current_waypoint is not None and _current_target_pos is not None:
        # Calculate distance to current waypoint target
        distance_to_current_target = np.linalg.norm(_current_target_pos - agent_pos)
        if distance_to_current_target <= _waypoint_threshold:
            waypoint_reached = True
            #print(f"Waypoint reached! Distance to target: {distance_to_current_target:.2f}")

    # Only select a new target if we don't have a current waypoint or we've reached it
    if _current_waypoint is None or waypoint_reached:
        # Extract target information vectorized
        max_targets = 5
        target_data = obs[2:2 + max_targets * 3].reshape(max_targets, 3)

        # Check which targets are valid (non-zero position)
        valid_mask = (target_data[:, 1] != 0) | (target_data[:, 2] != 0)

        if not np.any(valid_mask):
            # No targets at all, just move up
            _current_waypoint = None
            _current_target_pos = None
            return 0

        # Filter to valid targets only
        valid_targets = target_data[valid_mask]

        # Calculate distances vectorized
        target_positions = valid_targets[:, 1:3]  # x, y coordinates
        distances = np.linalg.norm(target_positions - agent_pos, axis=1)

        # Filter to unidentified targets (info_level < 1.0)
        unidentified_mask = valid_targets[:, 0] < 1.0

        if np.any(unidentified_mask):
            # Find farthest unidentified target
            unidentified_distances = distances[unidentified_mask]
            farthest_idx = np.argmax(unidentified_distances)
            # Map back to original valid targets index
            unidentified_indices = np.where(unidentified_mask)[0]
            target_idx = unidentified_indices[farthest_idx]
        else:
            # No unidentified targets, move toward closest identified target
            target_idx = np.argmin(distances)

        # Update current target position
        _current_target_pos = target_positions[target_idx].copy()

        #print(f"New target selected: pos={_current_target_pos}, distance={distances[target_idx]:.2f}")

    # Use current target position for direction calculation
    target_pos = _current_target_pos

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

    # Store the current waypoint for reference (this would be the actual waypoint sent to the agent)
    # Note: This is just for tracking - the actual waypoint calculation happens in the environment
    _current_waypoint = best_action

    return int(best_action)

def reset_badheuristic_state():
    """
    Call this function to reset the persistent state of the badheuristic policy.
    Should be called at the beginning of each episode.
    """
    global _current_waypoint, _current_target_pos
    _current_waypoint = None
    _current_target_pos = None


def generate_heuristic_trajectories_worker(args):
    """
    Worker function for multiprocessing trajectory generation.

    Args:
        args: Tuple containing (expert_policy, env_config, n_episodes, run_name, process_id)

    Returns:
        Tuple of (rollouts, process_id)
    """
    expert_policy, env_config, n_episodes, run_name, process_id = args

    # Reset global state for badheuristic if using it
    if expert_policy.__name__ == 'badheuristic_policy':
        reset_badheuristic_state()

    # Create unique run name for this process
    process_run_name = f"{run_name}_proc{process_id}"

    rng = np.random.default_rng(process_id)  # Different seed per process

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

    env = DummyVecEnv([make_env_with_wrapper(env_config, 0, env_config['seed'] + process_id, process_run_name)])

    print(f"Process {process_id}: Collecting {n_episodes} episodes...")
    rollouts = rollout.rollout(
        expert_policy,
        env,
        rollout.make_sample_until(min_episodes=n_episodes),
        rng=rng
    )

    env.close()
    print(f"Process {process_id}: Completed {len(rollouts)} episodes")

    return rollouts, process_id


def generate_heuristic_trajectories_mp(expert_policy, env_config, n_episodes=50, run_name='none',
                                       save_expert_trajectory=True, n_processes=None):
    """
    Multiprocessing version of trajectory generation.

    Args:
        expert_policy: Policy function that takes in an observation and returns an action
        env_config: Environment configuration
        n_episodes: Total number of episodes to collect
        run_name: Name for the run
        save_expert_trajectory: Whether to save trajectories
        n_processes: Number of processes (defaults to CPU count)

    Returns:
        None (saves trajectories to file)
    """
    if n_processes is None:
        n_processes = mp.cpu_count()

    print(f"Using {n_processes} processes to collect {n_episodes} total episodes")

    # Split episodes across processes
    episodes_per_process = n_episodes // n_processes
    extra_episodes = n_episodes % n_processes

    # Create arguments for each process
    process_args = []
    for i in range(n_processes):
        # Give extra episodes to first few processes
        proc_episodes = episodes_per_process + (1 if i < extra_episodes else 0)
        process_args.append((expert_policy, env_config, proc_episodes, run_name, i))

    # Run processes
    with Pool(n_processes) as pool:
        results = pool.map(generate_heuristic_trajectories_worker, process_args)

    # Combine results
    all_rollouts = []
    total_transitions = 0

    for rollouts, process_id in results:
        all_rollouts.extend(rollouts)
        transitions = rollout.flatten_trajectories(rollouts)
        total_transitions += len(transitions)
        print(f"Process {process_id}: Generated {len(transitions)} transitions from {len(rollouts)} episodes")

    print(f"Total: Generated {total_transitions} transitions from {len(all_rollouts)} episodes")

    if save_expert_trajectory:
        from imitation.data import serialize
        os.makedirs('./expert_trajectories', exist_ok=True)
        serialize.save(f'./expert_trajectories/{run_name}', all_rollouts)
        print(f"Saved trajectories to ./expert_trajectories/{run_name}")

    return all_rollouts


if __name__ == "__main__":
    # Set parameters
    config_name = '../configs/bigmap.json'
    n_episodes = 10000
    n_processes = 6  # Adjust based on your system

    run_name = f'experttraj_mp_1000map_30tgts_{str(n_episodes)}eps_{datetime.now().strftime("%m%d_%H%M")}'
    env_config = load_env_config(config_name)

    # Run the multiprocessing behavior cloning pipeline
    all_rollouts = generate_heuristic_trajectories_mp(
        expert_policy=improved_heuristic_policy,
        env_config=env_config,
        n_episodes=n_episodes,
        run_name=run_name,
        save_expert_trajectory=True,
        n_processes=n_processes
    )