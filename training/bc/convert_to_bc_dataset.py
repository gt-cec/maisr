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
        observation = np.array(ts["human_observation"], dtype=np.float32)

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


def build_dataset(input_folder: str) -> list[Trajectory]:
    """
    Scan a folder for .json recording files and convert each into a Trajectory.
    Returns a list of Trajectory objects (one per file/episode).
    """
    folder = Path(input_folder)
    json_files = sorted(folder.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No .json files found in {input_folder}")

    trajectories: list[Trajectory] = []

    for jf in json_files:
        print(f"Processing {jf.name} ...")
        timesteps = load_json_file(jf)
        traj = process_episode(timesteps)
        if traj is not None:
            trajectories.append(traj)
            print(f"  → {len(traj.acts)} state-action pairs, terminal={traj.terminal}")
        else:
            print(f"  → Skipped (no valid state-action pairs)")

    print(f"\nTotal trajectories: {len(trajectories)}")
    total_pairs = sum(len(t.acts) for t in trajectories)
    print(f"Total state-action pairs: {total_pairs}")

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
        default="expert_trajectories.pkl",
        help="Output path for the pickled trajectory dataset (default: expert_trajectories.pkl).",
    )
    args = parser.parse_args()

    trajectories = build_dataset(args.input_folder)

    # Save as pickle (standard for imitation library workflows)
    with open(args.output, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"\nSaved {len(trajectories)} trajectories to {args.output}")

    # Also print a quick sanity check of how to load & use with BC
    print(
        "\n── Quick usage with imitation BC ──\n"
        "  import pickle\n"
        "  from imitation.data.types import Trajectory\n"
        "  from imitation.algorithms.bc import BC\n"
        "  from imitation.data import rollout\n"
        "\n"
        f'  with open("{args.output}", "rb") as f:\n'
        "      trajectories = pickle.load(f)\n"
        "\n"
        "  transitions = rollout.flatten_trajectories(trajectories)\n"
        "  # transitions is a Transitions object usable directly with BC\n"
    )


if __name__ == "__main__":
    main()
