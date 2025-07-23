# Global state to maintain target persistence and prevent oscillation
import numpy as np
from requests.packages import target

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

def reset_heuristic_state():
    """Reset the global state for the heuristic policy."""
    global _current_target_id, _current_target_pos, _last_action, _action_repeat_count
    _current_target_id = None
    _current_target_pos = None
    _last_action = None
    _action_repeat_count = 0


def greedy_heuristic_nearest_n(observation):
    """
    Process observation with target persistence and oscillation prevention.
    Updated for nearest_n observation format.
    """
    global _current_target_id, _current_target_pos, _last_action, _action_repeat_count

    obs = np.array(observation)

    # Direction mapping
    directions = np.array([
        (0, 1),  # North (0°)
        (0.383, 0.924),  # NNE (22.5°)
        (0.707, 0.707),  # NE (45°)
        (0.924, 0.383),  # ENE (67.5°)
        (1, 0),  # East (90°)
        (0.924, -0.383),  # ESE (112.5°)
        (0.707, -0.707),  # SE (135°)
        (0.383, -0.924),  # SSE (157.5°)
        (0, -1),  # South (180°)
        (-0.383, -0.924),  # SSW (202.5°)
        (-0.707, -0.707),  # SW (225°)
        (-0.924, -0.383),  # WSW (247.5°)
        (-1, 0),  # West (270°)
        (-0.924, 0.383),  # WNW (292.5°)
        (-0.707, 0.707),  # NW (315°)
        (-0.383, 0.924),  # NNW (337.5°)
    ], dtype=float)

    # Extract nearest target vector (first two components)
    target_vector_x = obs[0]
    target_vector_y = obs[1]

    # Check if there's a valid target (non-zero vector)
    if target_vector_x == 0.0 and target_vector_y == 0.0:
        # No targets or at target location
        reset_heuristic_state()
        return 0

    # The observation already gives us the vector to the nearest target
    direction_to_target = np.array([target_vector_x, target_vector_y])

    # Normalize direction vectors
    direction_norms = np.linalg.norm(directions, axis=1)
    normalized_directions = directions / direction_norms[:, np.newaxis]

    # Normalize target direction
    target_norm = np.linalg.norm(direction_to_target)
    if target_norm > 0:
        direction_to_target_norm = direction_to_target / target_norm
    else:
        return _last_action if _last_action is not None else 0

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
    if (_last_action is not None and abs(_last_action - best_action) == 4):  # Opposite directions
        # Choose a compromise direction
        adjacent_actions = [(_last_action + 1) % 8, (_last_action - 1) % 8]
        adjacent_dots = [dot_products[a] for a in adjacent_actions]
        best_adjacent_idx = np.argmax(adjacent_dots)
        best_action = adjacent_actions[best_adjacent_idx]

    _last_action = best_action
    return np.int32(best_action)