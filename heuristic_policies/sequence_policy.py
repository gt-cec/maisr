# Global state to maintain sequence progress and target persistence
import ctypes

import numpy as np
import pygame

from env_combined import MAISREnvVec
from utility.data_logging import load_env_config

_current_sequence = None
_sequence_index = 0
_current_target_pos = None
_last_action = None
_action_repeat_count = 0
_max_repeat_count = 3  # Minimum steps to take in same direction
_target_reached_threshold = 0.025  # Distance threshold to consider target "reached"


def sequence_policy(observation, target_sequence, state=None, dones=None):
    """
    Sequence-based heuristic policy that visits targets in a predetermined order.
    
    Args:
        observation: Environment observation
        target_sequence: List of target indices in the order to visit them (e.g., [1, 0, 4, 3, 2])
        state: Optional state parameter (for compatibility)
        dones: Optional dones parameter (for compatibility)
    
    Returns:
        action or (actions, None) depending on observation format
    """
    if len(observation.shape) == 1:
        return sequence_process_single_observation(observation, target_sequence)
    else:
        batch_size = observation.shape[0]
        actions = np.zeros((batch_size,), dtype=np.float32)
        for i in range(batch_size):
            actions[i] = np.int64(sequence_process_single_observation(observation[i], target_sequence))
        return actions, None


def sequence_process_single_observation(observation, target_sequence):
    """
    Process observation following the target sequence with persistence.
    
    Args:
        observation: Single environment observation
        target_sequence: List of target indices in visiting order
    
    Returns:
        int: Action (0-7 for 8 directional movement)
    """
    global _current_sequence, _sequence_index, _current_target_pos, _last_action, _action_repeat_count

    obs = np.array(observation)
    
    # Direction mapping (same as greedy heuristic)
    directions = np.array([
        [0, 1],   # 0: up
        [1, 1],   # 1: up-right
        [1, 0],   # 2: right
        [1, -1],  # 3: down-right
        [0, -1],  # 4: down
        [-1, -1], # 5: down-left
        [-1, 0],  # 6: left
        [-1, 1]   # 7: up-left
    ], dtype=float)

    agent_pos = obs[:2]

    remaining_obs_size = len(obs) - 2
    actual_num_targets = remaining_obs_size // 3
    target_data = obs[2:2 + actual_num_targets * 3].reshape(actual_num_targets, 3)

    # Extract target information (matching the environment structure)
    #max_targets = 20  # Match environment's max_targets
    #target_data = obs[2:2 + max_targets * 3].reshape(max_targets, 3)
    
    # Find valid targets
    valid_mask = (target_data[:, 1] != 0) | (target_data[:, 2] != 0)
    
    if not np.any(valid_mask):
        reset_sequence_state()
        return 0

    # Initialize or update sequence
    if _current_sequence is None or not np.array_equal(_current_sequence, target_sequence):
        _current_sequence = np.array(target_sequence)
        _sequence_index = 0
        _current_target_pos = None
        _action_repeat_count = 0
        print(f"New sequence initialized: {target_sequence}")

    # Check if we've completed the sequence
    if _sequence_index >= len(_current_sequence):
        # Sequence complete, hold position or restart
        reset_sequence_state()
        return _last_action if _last_action is not None else 0

    # Get current target from sequence
    target_idx_in_sequence = _current_sequence[_sequence_index]
    
    # Validate target index
    valid_targets = target_data[valid_mask]
    if target_idx_in_sequence >= len(valid_targets):
        # Invalid target index, skip to next
        _sequence_index += 1
        return sequence_process_single_observation(observation, target_sequence)

    # Get target position
    target_idx_in_sequence = _current_sequence[_sequence_index]

    # Get valid target positions
    valid_targets = target_data[valid_mask]
    target_positions = valid_targets[:, 1:3]

    if target_idx_in_sequence < len(target_positions):
        target_pos = target_positions[target_idx_in_sequence]
    else:
        # Target index out of bounds, advance sequence
        _sequence_index += 1
        if _sequence_index < len(_current_sequence):
            return sequence_process_single_observation(observation, target_sequence)
        else:
            return _last_action if _last_action is not None else 0
    
    # # Check if target index is within bounds
    # if target_idx_in_sequence < len(target_positions):
    #     target_pos = target_positions[target_idx_in_sequence]
    # else:
    #     # Target not available, advance sequence
    #     _sequence_index += 1
    #     return sequence_process_single_observation(observation, target_sequence)

    # Update current target position
    _current_target_pos = target_pos.copy()

    # Check if we've reached the current target
    distance_to_target = np.linalg.norm(_current_target_pos - agent_pos)
    #print(f'Distance to target is {distance_to_target}')
    
    if distance_to_target <= _target_reached_threshold:
        # Target reached, advance to next in sequence
        _sequence_index += 1
        _action_repeat_count = 0
        #print(f"Target {target_idx_in_sequence} reached! Moving to next target. Progress: {_sequence_index}/{len(_current_sequence)}")
        
        # If sequence not complete, recursively call to get next target
        if _sequence_index < len(_current_sequence):
            return sequence_process_single_observation(observation, target_sequence)
        else:
            print("Sequence completed!")
            return _last_action if _last_action is not None else 0

    # Calculate direction to current target
    direction_to_target = _current_target_pos - agent_pos

    # Handle case where we're at exact target position (shouldn't happen due to threshold)
    if np.linalg.norm(direction_to_target) < 1e-6:
        return _last_action if _last_action is not None else 0

    # Normalize direction vectors
    direction_norms = np.linalg.norm(directions, axis=1)
    normalized_directions = directions / direction_norms[:, np.newaxis]

    # Normalize target direction
    direction_to_target_norm = direction_to_target / np.linalg.norm(direction_to_target)

    # Calculate dot products to find best direction
    dot_products = np.dot(normalized_directions, direction_to_target_norm)

    # Find best action
    best_action = np.argmax(dot_products)

    # Anti-oscillation: if we just took an action, continue for minimum steps
    if (_last_action is not None and
            _action_repeat_count < _max_repeat_count and
            _last_action != best_action):

        # Check if last action is still reasonable (dot product > 0.5)
        last_dot_product = dot_products[_last_action]
        if last_dot_product > 0.5:  # Still pointing roughly toward target
            best_action = _last_action
            _action_repeat_count += 1
        else:
            _action_repeat_count = 0  # Reset if direction is too far off
    else:
        _action_repeat_count = 0

    # Additional anti-oscillation: prevent direct opposite actions
    if (_last_action is not None and
            abs(_last_action - best_action) == 4):  # Opposite directions

        # Choose a compromise direction
        adjacent_actions = [(_last_action + 1) % 8, (_last_action - 1) % 8]
        adjacent_dots = [dot_products[a] for a in adjacent_actions]
        best_adjacent_idx = np.argmax(adjacent_dots)
        best_action = adjacent_actions[best_adjacent_idx]

    _last_action = best_action
    return int(best_action)


def reset_sequence_state():
    """Reset the global state for the sequence policy."""
    global _current_sequence, _sequence_index, _current_target_pos, _last_action, _action_repeat_count
    _current_sequence = None
    _sequence_index = 0
    _current_target_pos = None
    _last_action = None
    _action_repeat_count = 0


def get_sequence_progress():
    """
    Get current progress through the sequence.
    
    Returns:
        tuple: (current_index, total_length, current_target_pos)
    """
    global _current_sequence, _sequence_index, _current_target_pos
    
    total_length = len(_current_sequence) if _current_sequence is not None else 0
    return _sequence_index, total_length, _current_target_pos


def set_target_reached_threshold(threshold):
    """
    Set the distance threshold for considering a target "reached".
    
    Args:
        threshold (float): Distance threshold
    """
    global _target_reached_threshold
    _target_reached_threshold = threshold


# Example usage and testing functions
def generate_random_sequence(num_targets):
    """
    Generate a random sequence for visiting targets.
    
    Args:
        num_targets (int): Number of targets in the environment
        
    Returns:
        list: Random permutation of target indices
    """
    return np.random.permutation(num_targets).tolist()


def generate_nearest_neighbor_sequence(observation, start_target=0):
    """
    Generate a sequence using nearest neighbor heuristic.
    
    Args:
        observation: Environment observation
        start_target (int): Index of starting target
        
    Returns:
        list: Sequence visiting targets in nearest-neighbor order
    """
    obs = np.array(observation)
    
    # Extract target information
    remaining_obs_size = len(obs) - 2
    actual_num_targets = remaining_obs_size // 3

    #max_targets = 20
    #target_data = obs[2:2 + max_targets * 3].reshape(max_targets, 3)
    target_data = obs[2:2 + actual_num_targets * 3].reshape(actual_num_targets, 3)
    valid_mask = (target_data[:, 1] != 0) | (target_data[:, 2] != 0)
    
    if not np.any(valid_mask):
        return []
    
    valid_targets = target_data[valid_mask]
    target_positions = valid_targets[:, 1:3]
    num_valid_targets = len(target_positions)
    
    if start_target >= num_valid_targets:
        start_target = 0
    
    sequence = []
    remaining_targets = list(range(num_valid_targets))
    current_pos = target_positions[start_target]
    current_target = start_target
    
    # Add starting target
    sequence.append(current_target)
    remaining_targets.remove(current_target)
    
    # Build sequence using nearest neighbor
    while remaining_targets:
        distances = [np.linalg.norm(target_positions[t] - current_pos) for t in remaining_targets]
        nearest_idx = np.argmin(distances)
        nearest_target = remaining_targets[nearest_idx]
        
        sequence.append(nearest_target)
        current_pos = target_positions[nearest_target]
        remaining_targets.remove(nearest_target)
    
    return sequence

if __name__ == '__main__':

    config = load_env_config('../configs/sequence_june11.json')

    target_sequence = [1, 13, 8, 5, 4, 14, 19, 7, 3, 6, 2, 12, 17, 9, 15, 10, 16, 18, 11, 0]#[3,2,3,2,4]

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


    while True:
        terminated, truncated = False, False
        while not (terminated or truncated):
            pygame.event.get()
            current_obs = env.get_observation() if hasattr(env, 'get_observation') else env.observation

            sequence_action = sequence_policy(current_obs, target_sequence)

            obs, reward, terminated, truncated, info = env.step(sequence_action)
            env.render()

        env.reset()