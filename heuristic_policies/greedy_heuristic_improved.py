# Global state to maintain target persistence and prevent oscillation
import numpy as np

_current_target_id = None
_current_target_pos = None
_last_action = None
_action_repeat_count = 0
_max_repeat_count = 3  # Minimum steps to take in same direction
_target_switch_threshold = 20.0  # Distance threshold to consider switching targets


def improved_heuristic_policy(observation, state, dones):
    """
    Improved heuristic policy with target persistence and oscillation prevention.
    """
    if len(observation.shape) == 1:
        return improved_heuristic_process_single_observation(observation)
    else:
        batch_size = observation.shape[0]
        actions = np.zeros((batch_size,), dtype=np.float32)
        for i in range(batch_size):
            actions[i] = np.int64(improved_heuristic_process_single_observation(observation[i]))
        return actions, None


def improved_heuristic_process_single_observation(observation):
    """
    Process observation with target persistence and oscillation prevention.
    """
    global _current_target_id, _current_target_pos, _last_action, _action_repeat_count

    obs = np.array(observation)

    # Direction mapping
    directions = np.array([
        [0, 1],  # 0: up
        [1, 1],  # 1: up-right
        [1, 0],  # 2: right
        [1, -1],  # 3: down-right
        [0, -1],  # 4: down
        [-1, -1],  # 5: down-left
        [-1, 0],  # 6: left
        [-1, 1]  # 7: up-left
    ], dtype=float)

    agent_pos = obs[:2]

    # Extract target information
    max_targets = 30  # Match your max_targets from env
    target_data = obs[2:2 + max_targets * 3].reshape(max_targets, 3)

    # Find valid targets
    valid_mask = (target_data[:, 1] != 0) | (target_data[:, 2] != 0)

    if not np.any(valid_mask):
        reset_heuristic_state()
        return 0

    valid_targets = target_data[valid_mask]
    target_positions = valid_targets[:, 1:3]
    distances = np.linalg.norm(target_positions - agent_pos, axis=1)

    # Check if current target is still valid and unidentified
    current_target_valid = False
    if _current_target_id is not None and _current_target_pos is not None:
        # Find if current target still exists and is unidentified
        for i, target in enumerate(valid_targets):
            target_pos = target[1:3]
            if (np.linalg.norm(target_pos - _current_target_pos) < 5.0 and  # Same position
                    target[0] < 1.0):  # Still unidentified
                current_target_valid = True
                _current_target_pos = target_pos  # Update position
                break

    # Decide whether to switch targets
    should_switch_target = False

    if not current_target_valid:
        should_switch_target = True
    elif _current_target_pos is not None:
        # Check if we're very close to current target
        current_distance = np.linalg.norm(_current_target_pos - agent_pos)
        if current_distance < 35.0:  # Within identification range
            should_switch_target = True

        # Check if there's a much closer unidentified target
        unidentified_mask = valid_targets[:, 0] < 1.0
        if np.any(unidentified_mask):
            unidentified_distances = distances[unidentified_mask]
            closest_unidentified_distance = np.min(unidentified_distances)
            if closest_unidentified_distance < current_distance - _target_switch_threshold:
                should_switch_target = True

    # Select new target if needed
    if should_switch_target:
        unidentified_mask = valid_targets[:, 0] < 1.0

        if np.any(unidentified_mask):
            # Find closest unidentified target
            unidentified_distances = distances[unidentified_mask]
            closest_idx = np.argmin(unidentified_distances)
            unidentified_indices = np.where(unidentified_mask)[0]
            target_idx = unidentified_indices[closest_idx]
        else:
            # No unidentified targets, pick closest identified
            target_idx = np.argmin(distances)

        _current_target_pos = target_positions[target_idx].copy()
        _current_target_id = target_idx
        _action_repeat_count = 0  # Reset action persistence
        #print(f"Switched to new target at {_current_target_pos}, distance: {distances[target_idx]:.1f}")

    # Calculate direction to current target
    direction_to_target = _current_target_pos - agent_pos

    # Handle case where we're at target
    if np.linalg.norm(direction_to_target) < 1e-6:
        return _last_action if _last_action is not None else 0

    # Normalize direction vectors
    direction_norms = np.linalg.norm(directions, axis=1)
    normalized_directions = directions / direction_norms[:, np.newaxis]

    # Normalize target direction
    direction_to_target_norm = direction_to_target / np.linalg.norm(direction_to_target)

    # Calculate dot products
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


def reset_heuristic_state():
    """Reset the global state for the heuristic policy."""
    global _current_target_id, _current_target_pos, _last_action, _action_repeat_count
    _current_target_id = None
    _current_target_pos = None
    _last_action = None
    _action_repeat_count = 0