import numpy as np


def heuristic_policy(observation):
    """
    Heuristic policy that selects the closest unidentified target (vectorized).

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
    max_targets = (len(observation) - 2) // 3

    if max_targets == 0:
        # No targets, stay at current position
        return np.array([agent_x * 2 - 1, agent_y * 2 - 1], dtype=np.float32)

    # Extract target data using array slicing and reshaping
    target_data = observation[2:2 + max_targets * 3].reshape(max_targets, 3)

    # Split into separate arrays for each attribute
    info_levels = target_data[:, 0]  # Shape: (max_targets,)
    target_x = target_data[:, 1]  # Shape: (max_targets,)
    target_y = target_data[:, 2]  # Shape: (max_targets,)

    # Create masks for valid and unidentified targets
    exists_mask = (target_x != 0) | (target_y != 0)  # Target exists
    unidentified_mask = info_levels < 1.0  # Not fully identified
    valid_mask = exists_mask & unidentified_mask  # Both conditions

    # If no valid targets, stay at current position
    if not np.any(valid_mask):
        return np.array([agent_x * 2 - 1, agent_y * 2 - 1], dtype=np.float32)

    # Calculate distances to all targets (vectorized)
    distances = np.sqrt((target_x - agent_x) ** 2 + (target_y - agent_y) ** 2)

    # Set invalid targets to infinite distance
    distances = np.where(valid_mask, distances, np.inf)

    # Find index of closest valid target
    closest_idx = np.argmin(distances)

    # Get coordinates of closest target
    closest_target_x = target_x[closest_idx]
    closest_target_y = target_y[closest_idx]

    # Convert from normalized [0,1] coordinates to action space [-1,1]
    action_x = closest_target_x * 2 - 1
    action_y = closest_target_y * 2 - 1

    return np.array([action_x, action_y], dtype=np.float32)


# Example usage and test
if __name__ == "__main__":
    # Create a sample observation for testing
    # Agent at (0.5, 0.5), with 3 targets:
    # Target 0: unidentified at (0.2, 0.3)
    # Target 1: identified at (0.8, 0.7)
    # Target 2: unidentified at (0.1, 0.9)

    sample_obs = np.array([
        0.5, 0.5,  # Agent position
        0.0, 0.2, 0.3,  # Target 0: unidentified
        0.0, 0.4, 0.5,  # Target 1: fully identified (should be ignored)
        0.0, 0.1, 0.9,  # Target 2: unidentified
        # ... remaining target slots would be zeros
        0.0, 0.0, 0.0,  # Empty target slot
        0.0, 0.0, 0.0,  # Empty target slot
    ])

    action = heuristic_policy(sample_obs)
    print(f"Agent position: ({sample_obs[0]}, {sample_obs[1]})")
    print(f"Recommended action: {action}")
    print(f"Target coordinates in [0,1]: ({(action[0] + 1) / 2}, {(action[1] + 1) / 2})")