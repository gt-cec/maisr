"""
Convert experiment JSON recordings of human-AI collaborative gameplay
into an imitation-library-compatible BC trajectory dataset.

Usage:
    python convert_to_bc_dataset.py <input_folder> [--output <output_path>]

Input:  Folder containing .json files with timestep recordings.
Output: A pickled list of imitation.data.types.Trajectory objects saved to disk.
"""

import argparse
import json
import math
import os
import pickle
from pathlib import Path
import numpy as np
from imitation.data.types import Trajectory
import random


# ── Direction mapping (16 discrete directions) ──────────────────────────
DIRECTION_MAP = {
    0: (0, 1),            # North (0°)
    1: (0.383, 0.924),    # NNE (22.5°)
    2: (0.707, 0.707),    # NE (45°)
    3: (0.924, 0.383),    # ENE (67.5°)
    4: (1, 0),            # East (90°)
    5: (0.924, -0.383),   # ESE (112.5°)
    6: (0.707, -0.707),   # SE (135°)
    7: (0.383, -0.924),   # SSE (157.5°)
    8: (0, -1),           # South (180°)
    9: (-0.383, -0.924),  # SSW (202.5°)
    10: (-0.707, -0.707), # SW (225°)
    11: (-0.924, -0.383), # WSW (247.5°)
    12: (-1, 0),          # West (270°)
    13: (-0.924, 0.383),  # WNW (292.5°)
    14: (-0.707, 0.707),  # NW (315°)
    15: (-0.383, 0.924),  # NNW (337.5°)
}

# Pre-compute reference angles for fast lookup
# atan2(y, x) gives the angle from the positive x-axis, counter-clockwise
_REF_ANGLES = {}
for idx, (dx, dy) in DIRECTION_MAP.items():
    _REF_ANGLES[idx] = math.atan2(dy, dx)


def direction_to_action(dx: float, dy: float) -> int:
    """
    Convert a (dx, dy) direction vector into the closest discrete action index
    from DIRECTION_MAP.

    Uses the angle between the direction vector and each reference direction,
    picking the one with the smallest angular difference.
    """
    angle = math.atan2(dy, dx)  # range [-pi, pi]

    best_idx = 0
    best_diff = float("inf")
    for idx, ref_angle in _REF_ANGLES.items():
        # Shortest angular distance (handles wraparound)
        diff = abs(math.atan2(math.sin(angle - ref_angle), math.cos(angle - ref_angle)))
        if diff < best_diff:
            best_diff = diff
            best_idx = idx
    return best_idx


def reconstruct_19dim_observation(
    human_pos: tuple[float, float],
    agent_pos: tuple[float, float],
    target_positions: list,
    target_info_levels: list,
    threat_positions: list,
    agent_action: int,
    num_observed_targets: int = 5,
    num_observed_threats: int = 2
) -> np.ndarray:
    """
    Reconstruct a 19-dimensional observation matching the RL agent format.

    Observation structure (19 dimensions):
    - [0-9]:   5 nearest unknown targets (2 coords each) - raw distance vectors
    - [10-13]: 2 nearest threats (2 coords each) - raw distance vectors
    - [14-15]: Teammate position relative to human - raw distance vector
    - [16-17]: Teammate heading (inferred from action) - unit vector
    - [18]:    Teammate priority flag (1.0 if flying toward threat, 0.0 otherwise)

    Args:
        human_pos: (x, y) position of human agent
        agent_pos: (x, y) position of AI teammate
        target_positions: List of [x, y] target coordinates
        target_info_levels: List of info levels for each target (< 1.0 means unknown)
        threat_positions: List of [x, y] threat coordinates
        agent_action: Integer action (0-15) representing teammate's direction
        num_observed_targets: Number of nearest unknown targets to include
        num_observed_threats: Number of nearest threats to include

    Returns:
        19-dimensional observation array as np.float32
    """
    observation = np.zeros(19, dtype=np.float32)

    human_pos_arr = np.array(human_pos, dtype=np.float32)
    agent_pos_arr = np.array(agent_pos, dtype=np.float32)

    # ─── A. Find 5 Nearest Unknown Targets (indices 0-9) ───
    target_positions_arr = np.array(target_positions, dtype=np.float32)
    target_info_levels_arr = np.array(target_info_levels, dtype=np.float32)

    unknown_mask = target_info_levels_arr < 1.0

    if np.any(unknown_mask):
        unknown_positions = target_positions_arr[unknown_mask]
        # Calculate distances from human position
        distances = np.sqrt(np.sum((unknown_positions - human_pos_arr) ** 2, axis=1))

        # Get indices of N nearest targets (or all if fewer than N)
        num_targets_to_use = min(num_observed_targets, len(distances))
        nearest_indices = np.argsort(distances)[:num_targets_to_use]

        # Fill observation with RAW distance vectors (not unit vectors)
        for i in range(num_targets_to_use):
            target_idx = nearest_indices[i]
            target_pos = unknown_positions[target_idx]
            vector_to_target = target_pos - human_pos_arr

            observation[i * 2] = vector_to_target[0]
            observation[i * 2 + 1] = vector_to_target[1]

    # ─── B. Get 2 Nearest Threats (indices 10-13) ───
    threat_positions_arr = np.array(threat_positions, dtype=np.float32)
    threat_distances = np.sqrt(np.sum((threat_positions_arr - human_pos_arr) ** 2, axis=1))

    num_threats_to_use = min(num_observed_threats, len(threat_distances))
    nearest_threat_indices = np.argsort(threat_distances)[:num_threats_to_use]

    start_idx = 2 * num_observed_targets
    for j in range(num_threats_to_use):
        threat_idx = nearest_threat_indices[j]
        threat_pos = threat_positions_arr[threat_idx]
        vector_to_threat = threat_pos - human_pos_arr

        observation[start_idx + j * 2] = vector_to_threat[0]
        observation[start_idx + j * 2 + 1] = vector_to_threat[1]

    # ─── C. Calculate Teammate Position (indices 14-15) ───
    teammate_idx = 2 * (num_observed_targets + num_observed_threats)
    teammate_relative_pos = agent_pos_arr - human_pos_arr
    observation[teammate_idx] = teammate_relative_pos[0]
    observation[teammate_idx + 1] = teammate_relative_pos[1]

    # ─── D. Calculate Teammate Heading (indices 16-17) ───
    # Infer heading from agent_action using DIRECTION_MAP
    if agent_action in DIRECTION_MAP:
        heading = DIRECTION_MAP[agent_action]
        observation[teammate_idx + 2] = heading[0]
        observation[teammate_idx + 3] = heading[1]
    else:
        # Invalid action - no heading
        observation[teammate_idx + 2] = 0.0
        observation[teammate_idx + 3] = 0.0

    # ─── E. Calculate Teammate Priority Flag (index 18) ───
    heading_unit = np.array([observation[teammate_idx + 2], observation[teammate_idx + 3]], dtype=np.float32)
    heading_magnitude = np.linalg.norm(heading_unit)

    if heading_magnitude > 0:
        # Heading is already a unit vector from DIRECTION_MAP
        # Combine threats and targets with labels
        entities = [(pos, "threat") for pos in threat_positions_arr] + \
                   [(pos, "target") for pos in target_positions_arr]

        closest_entity_type = None
        closest_forward_dist = float("inf")
        beam_half_width = 25.0  # 50-pixel wide beam

        for pos, etype in entities:
            vec_to_entity = pos - agent_pos_arr
            forward_dist = np.dot(vec_to_entity, heading_unit)  # projection along heading

            if forward_dist <= 0:
                continue  # Only consider entities in front

            # Perpendicular distance to heading line
            perp_dist = np.linalg.norm(vec_to_entity - forward_dist * heading_unit)
            if perp_dist <= beam_half_width:
                if forward_dist < closest_forward_dist:
                    closest_forward_dist = forward_dist
                    closest_entity_type = etype

        # 1.0 if teammate is flying toward a threat, else 0.0
        if closest_entity_type == "threat":
            observation[-1] = 1.0
        else:
            observation[-1] = 0.0
    else:
        # No valid heading - priority is 0
        observation[-1] = 0.0

    return observation


def load_json_file(path: str) -> list[dict]:
    """Load a single JSON recording and return its timestep list."""
    with open(path, "r") as f:
        data = json.load(f)
    return data["timesteps"]


def process_episode(timesteps: list[dict]) -> Trajectory | None:
    """
    Convert a list of timesteps from one JSON file into an imitation Trajectory.

    Skips timesteps where human_custom_waypoint is null (no human input).
    A Trajectory requires:
        obs:   np.ndarray of shape (T+1, obs_dim)  — observations including terminal
        acts:  np.ndarray of shape (T,)             — actions (one fewer than obs)
        infos: list[dict] of length T               — optional per-step metadata
        terminal: bool                              — whether the episode truly ended
    """
    obs_list = []
    act_list = []
    infos = []

    for ts in timesteps:
        waypoint = ts["human_custom_waypoint"]
        if waypoint is None:
            # No human waypoint set yet — skip this timestep
            continue

        pos = ts["human_position"]
        dx = waypoint[0] - pos[0]
        dy = waypoint[1] - pos[1]

        # If waypoint == position (arrived), skip — direction is undefined
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue

        action = direction_to_action(dx, dy)

        # Reconstruct 19-dimensional observation from raw game state
        observation = reconstruct_19dim_observation(
            human_pos=tuple(ts["human_position"]),
            agent_pos=tuple(ts["agent_position"]),
            target_positions=ts["target_positions"],
            target_info_levels=ts["target_info_levels"],
            threat_positions=ts["threat_positions"],
            agent_action=ts["agent_action"][0] if isinstance(ts["agent_action"], list) else ts["agent_action"],
            num_observed_targets=5,
            num_observed_threats=2
        )

        obs_list.append(observation)
        act_list.append(action)
        infos.append({"timestep": ts["timestep"]})

    if len(act_list) == 0:
        return None

    # Trajectory needs T+1 observations for T actions.
    # We use the last observation repeated as the terminal observation.
    # Alternatively, if the next timestep after the last action has an
    # observation, we could use that — but to keep it simple and robust
    # we duplicate the final obs.
    terminal_obs = obs_list[-1].copy()
    obs_array = np.array(obs_list + [terminal_obs], dtype=np.float32)  # (T+1, obs_dim)
    acts_array = np.array(act_list, dtype=np.int64)                    # (T,)

    # Determine if the episode actually terminated
    last_ts = timesteps[-1]
    terminal = bool(last_ts.get("terminated", False))

    return Trajectory(
        obs=obs_array,
        acts=acts_array,
        infos=np.array(infos),  # imitation expects array of dicts
        terminal=terminal,
    )




def build_dataset(input_folder: str, train_fraction: float = 0.8, seed: int = 42) -> list[Trajectory]:
    """
    Scan nested participant folders for .json recording files and convert each into a Trajectory.

    Performs a subject-level split:
      - Randomly selects ~80% of subjects for training
      - Remaining subjects are held out for evaluation
      - ONLY training subjects are included in the returned dataset

    Expected structure:
      input_folder/
        subject_<ID>/
          timestep_data/
            *.json
    """
    root = Path(input_folder)

    # ── Discover subjects ────────────────────────────────────────────────
    subject_dirs = sorted([p for p in root.glob("subject_*") if p.is_dir()])

    if not subject_dirs:
        raise FileNotFoundError(f"No subject_* directories found under {root}")

    subject_ids = [p.name for p in subject_dirs]

    # ── Train / eval split at SUBJECT level ──────────────────────────────
    rng = random.Random(seed)
    rng.shuffle(subject_ids)

    num_subjects = len(subject_ids)
    num_train = int(round(train_fraction * num_subjects))

    train_subjects = set(subject_ids[:num_train])
    eval_subjects = set(subject_ids[num_train:])

    print(f"\nSubject split:")
    print(f"  Total subjects: {num_subjects}")
    print(f"  Training subjects ({len(train_subjects)}): {sorted(train_subjects)}")
    print(f"  Eval subjects ({len(eval_subjects)}): {sorted(eval_subjects)}")

    # ── Collect JSON files ONLY from training subjects ───────────────────
    json_files = []
    for subj in train_subjects:
        subj_path = root / subj / "timestep_data"
        if subj_path.exists():
            json_files.extend(sorted(subj_path.glob("*.json")))

    if not json_files:
        raise FileNotFoundError(
            "No .json files found for training subjects "
            f"(expected subject_*/timestep_data/*.json)"
        )

    # ── Build trajectories ───────────────────────────────────────────────
    trajectories: list[Trajectory] = []

    for jf in json_files:
        rel = jf.relative_to(root)
        print(f"Processing {rel} ...")

        timesteps = load_json_file(jf)
        traj = process_episode(timesteps)

        if traj is not None:
            trajectories.append(traj)
            print(f"  → {len(traj.acts)} state-action pairs, terminal={traj.terminal}")
        else:
            print(f"  → Skipped (no valid state-action pairs)")

    print(f"\nFinal TRAINING dataset:")
    print(f"  Trajectories: {len(trajectories)}")
    total_pairs = sum(len(t.acts) for t in trajectories)
    print(f"  State-action pairs: {total_pairs}")

    return trajectories




def main():
    parser = argparse.ArgumentParser(
        description="Convert gameplay JSON recordings to an imitation BC trajectory dataset."
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to folder containing .json recording files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training/bc/expert_trajectories_80pct.pkl",
        help="Output path for the pickled trajectory dataset (default: expert_trajectories.pkl).",
    )
    args = parser.parse_args()

    trajectories = build_dataset(args.input_folder)

    # Save as pickle (standard for imitation library workflows)
    with open(args.output, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"\nSaved {len(trajectories)} trajectories to {args.output}")


if __name__ == "__main__":
    main()
