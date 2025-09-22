import ctypes
from dataclasses import dataclass
from typing import List, Tuple, Dict
import os, json, math
import glob

import pygame
from matplotlib.ticker import FuncFormatter
from scipy.optimize import linear_sum_assignment
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from dataclasses import dataclass, field

import re
import itertools
import pickle
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

# For target-threat cluster analysis
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# For compare_progress_rate
from scipy.spatial.distance import cdist
#from fastdtw import fastdtw
from scipy.stats import mannwhitneyu, pearsonr, kruskal

# For action distribution comparison
from scipy.stats import chisquare

from base_env import MAISREnvVec
from utility.config_management import load_env_config
from utility.league_management import LocalSearch, GoToNearestThreat, ChangeRegions, GenericTeammatePolicy, TargetSearchLocalTSP, HeuristicAgent
from utility.localsearch_training_wrapper import MaisrLocalSearchWrapper




def perform_kruskal_wallis_test(metric_name, human_vals, rl_vals, heuristic_vals):
    """
    Perform Kruskal-Wallis H test to determine if three groups are significantly different.
    """
    # Kruskal-Wallis test
    h_stat, p_value = kruskal(human_vals, rl_vals, heuristic_vals)

    print(f"\n--- {metric_name} - Kruskal-Wallis H Test ---")
    print(f"H-statistic: {h_stat:.4f}")
    print(f"p-value: {p_value}")
    print(f"Result: {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} group differences (α = 0.05)")

    # If significant, perform post-hoc pairwise comparisons
    if p_value < 0.05:
        print(f"\nPost-hoc pairwise comparisons (Dunn's test):")

        # Combine data for post-hoc test
        all_data = human_vals + rl_vals + heuristic_vals
        groups = (['Human'] * len(human_vals) +
                  ['RL'] * len(rl_vals) +
                  ['Heuristic'] * len(heuristic_vals))

        # Create DataFrame for scikit-posthocs
        import pandas as pd
        df = pd.DataFrame({'values': all_data, 'groups': groups})

        # Dunn's test with Bonferroni correction
        try:
            import scikit_posthocs as sp
            dunn_results = sp.posthoc_dunn(df, val_col='values', group_col='groups', p_adjust='bonferroni')
            print(dunn_results)
        except ImportError:
            print("scikit-posthocs not available. Install with: pip install scikit-posthocs")

    return h_stat, p_value


def load_vecnormalize_wrapper(vecnorm_path, env):
    """Load saved VecNormalize wrapper with stats from training and apply it to the new environment."""
    print(f"Loading VecNormalize stats from: {vecnorm_path}")

    vec_normalize = VecNormalize.load(vecnorm_path, venv=env)
    vec_normalize.training = False  # Disable further normalization updates
    vec_normalize.norm_reward = False
    return vec_normalize


def save_trajectories_to_json(trajectories, output_file, full_trajectories=False):
    """Save trajectories to JSON with proper numpy array handling"""
    serializable_data = []

    if full_trajectories:
        for traj in trajectories:
            traj_dict = {
                'category': traj.category,
                'level': traj.level,
                'name': traj.name,
                'positions': [list(pos) if hasattr(pos, '__iter__') else pos for pos in (traj.positions or [])],
                'actions': [int(action) if hasattr(action, 'item') else action for action in (traj.actions or [])],
                'target_ids': [int(tid) if hasattr(tid, 'item') else tid for tid in (traj.target_ids or [])],
                'threat_ids': [int(tid) if hasattr(tid, 'item') else tid for tid in (traj.threat_ids or [])]
            }

            # Add target positions and statuses (targets 0-14)
            for target_idx in range(15):
                target_pos_attr = f'target{target_idx}_pos'
                target_status_attr = f'target{target_idx}_status'

                if hasattr(traj, target_pos_attr):
                    target_pos = getattr(traj, target_pos_attr)
                    if target_pos is not None:
                        if isinstance(target_pos, list) and len(target_pos) > 0:
                            # If it's a list of positions, convert each position to list
                            if hasattr(target_pos[0], '__iter__'):
                                traj_dict[target_pos_attr] = [list(pos) for pos in target_pos]
                            else:
                                traj_dict[target_pos_attr] = target_pos
                        else:
                            traj_dict[target_pos_attr] = target_pos
                    else:
                        traj_dict[target_pos_attr] = []
                else:
                    traj_dict[target_pos_attr] = []

                if hasattr(traj, target_status_attr):
                    target_status = getattr(traj, target_status_attr)
                    if hasattr(target_status, 'item'):
                        traj_dict[target_status_attr] = int(target_status.item())
                    elif isinstance(target_status, (list, tuple)):
                        traj_dict[target_status_attr] = [int(status) if hasattr(status, 'item') else status for status
                                                         in target_status]
                    else:
                        traj_dict[target_status_attr] = int(target_status) if target_status is not None else 0
                else:
                    traj_dict[target_status_attr] = 0

            # Add threat positions and statuses (threats 0-3)
            for threat_idx in range(4):
                threat_pos_attr = f'threat{threat_idx}_pos'
                threat_status_attr = f'threat{threat_idx}_status'

                if hasattr(traj, threat_pos_attr):
                    threat_pos = getattr(traj, threat_pos_attr)
                    if threat_pos is not None:
                        if isinstance(threat_pos, list) and len(threat_pos) > 0:
                            # If it's a list of positions, convert each position to list
                            if hasattr(threat_pos[0], '__iter__'):
                                traj_dict[threat_pos_attr] = [list(pos) for pos in threat_pos]
                            else:
                                traj_dict[threat_pos_attr] = threat_pos
                        else:
                            traj_dict[threat_pos_attr] = threat_pos
                    else:
                        traj_dict[threat_pos_attr] = []
                else:
                    traj_dict[threat_pos_attr] = []

                if hasattr(traj, threat_status_attr):
                    threat_status = getattr(traj, threat_status_attr)
                    if hasattr(threat_status, 'item'):
                        traj_dict[threat_status_attr] = int(threat_status.item())
                    elif isinstance(threat_status, (list, tuple)):
                        traj_dict[threat_status_attr] = [int(status) if hasattr(status, 'item') else status for status
                                                         in threat_status]
                    else:
                        traj_dict[threat_status_attr] = int(threat_status) if threat_status is not None else 0
                else:
                    traj_dict[threat_status_attr] = 0

            serializable_data.append(traj_dict)
    else:
        # Original behavior for basic trajectories
        for traj in trajectories:
            traj_dict = {
                'category': traj.category,
                'level': traj.level,
                'name': traj.name,
                'positions': [list(pos) if hasattr(pos, '__iter__') else pos for pos in (traj.positions or [])],
                'actions': [int(action) if hasattr(action, 'item') else action for action in (traj.actions or [])],
                'target_ids': [int(tid) if hasattr(tid, 'item') else tid for tid in (traj.target_ids or [])],
                'threat_ids': [int(tid) if hasattr(tid, 'item') else tid for tid in (traj.threat_ids or [])]
            }
            serializable_data.append(traj_dict)

    with open(output_file, "w") as f:
        json.dump(serializable_data, f, indent=2)


def load_saved_trajectories(trajectory_type: str, subsample_episodes: int = 1):
    """
    Load saved trajectories from JSON files.

    Args:
        trajectory_type: str - One of 'human', 'rl', or 'heuristic'/'strategy'

    Returns:
        List[Trajectory] - List of loaded trajectory objects
    """
    # Map trajectory types to file names and class attributes
    trajectory_mapping = {
        'human': {
            'file': 'human_trajectories_for_training.json',
            'attr': 'human_trajectories_for_training'
        },
        'rl': {
            'file': 'rl_trajectories_nondeterministic.json',
            'attr': 'rl_trajectories'
        },
        'heuristic': {
            'file': 'strategy_trajectories.json',
            'attr': 'strategy_trajectories'
        },
        'strategy': {  # Allow both 'heuristic' and 'strategy' as aliases
            'file': 'strategy_trajectories.json',
            'attr': 'strategy_trajectories'
        }
    }

    if trajectory_type not in trajectory_mapping:
        raise ValueError(
            f"Invalid trajectory_type: {trajectory_type}. Must be one of: {list(trajectory_mapping.keys())}")

    file_info = trajectory_mapping[trajectory_type]
    file_path = os.path.join('', file_info['file'])

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Trajectory file not found: {file_path}")

    with open(file_path, 'r') as f:
        trajectory_data = json.load(f)

    # For heuristic/strategy trajectories, apply subsampling by agent name
    if trajectory_type in ['heuristic', 'strategy'] and subsample_episodes > 1:
        # Group trajectories by agent name
        trajectories_by_name = {}
        for traj_dict in trajectory_data:
            agent_name = traj_dict['name']
            if agent_name not in trajectories_by_name:
                trajectories_by_name[agent_name] = []
            trajectories_by_name[agent_name].append(traj_dict)

        # Subsample trajectories for each agent name
        subsampled_trajectory_data = []
        for agent_name, agent_trajectories in trajectories_by_name.items():
            # Take every Nth trajectory for this agent
            subsampled_trajectories = agent_trajectories[::subsample_episodes]
            subsampled_trajectory_data.extend(subsampled_trajectories)

        trajectory_data = subsampled_trajectory_data
        print(f"Subsampled heuristic trajectories: taking every {subsample_episodes} episodes per agent")

    # Convert JSON data back to Trajectory objects
    trajectories = []
    for traj_dict in trajectory_data:
        # Subsample every 10th timestep for human trajectories (human gameplay used 10x refresh rate to improve user experience)
        if trajectory_type == 'human':
            positions = traj_dict.get('positions', [])
            actions = traj_dict.get('actions', [])
            target_ids = traj_dict.get('target_ids', [])
            threat_ids = traj_dict.get('threat_ids', [])

            # Apply subsampling (every 10th element, starting from 0)
            positions = positions[::10] if positions else []
            actions = actions[::10] if actions else []
            target_ids = target_ids[::10] if target_ids else []
            threat_ids = threat_ids[::10] if threat_ids else []
        else:
            # Keep all timesteps for RL and heuristic trajectories
            positions = traj_dict.get('positions', [])
            actions = traj_dict.get('actions', [])
            target_ids = traj_dict.get('target_ids', [])
            threat_ids = traj_dict.get('threat_ids', [])

        trajectory = Trajectory(
            category=traj_dict['category'],
            level=traj_dict['level'],
            name=traj_dict['name'],
            positions=[tuple(pos) if isinstance(pos, list) else pos for pos in positions],
            actions=actions,
            target_ids=target_ids,
            threat_ids=threat_ids
        )
        trajectories.append(trajectory)
        #print(f"Successfully processed {traj_dict['category']} trajectory, {len(positions)} timesteps")

    print(f"Successfully loaded {len(trajectories)} {trajectory_type} trajectories from {file_path}")
    if trajectory_type == 'human':
        print(f"  Note: Human trajectories subsampled to every 10th timestep")

    return trajectories



def waypoint_to_direction_index(current_pos, target_waypoint):
    """
    Convert a waypoint to a direction index (0-15) based on the agent's current position.

    Args:
        current_pos: tuple (x, y) of current agent position
        target_waypoint: tuple (x, y) of target waypoint

    Returns:
        int: Direction index (0-15) where:
        0: North, 1: NNE, 2: NE, 3: ENE, 4: East, etc.
    """
    import math

    # Direction vectors for 16 discrete directions
    direction_vectors = [
        (0, 1), (0.383, 0.924), (0.707, 0.707), (0.924, 0.383),
        (1, 0), (0.924, -0.383), (0.707, -0.707), (0.383, -0.924),
        (0, -1), (-0.383, -0.924), (-0.707, -0.707), (-0.924, -0.383),
        (-1, 0), (-0.924, 0.383), (-0.707, 0.707), (-0.383, 0.924)
    ]

    def normalize(vx, vy):
        """Normalize a vector to unit length"""
        mag = math.sqrt(vx ** 2 + vy ** 2)
        return (vx / mag, vy / mag) if mag > 1e-8 else (0.0, 0.0)

    # Calculate movement vector from current position to waypoint
    dx = target_waypoint[0] - current_pos[0]
    dy = target_waypoint[1] - current_pos[1]

    # Normalize the movement vector
    ndx, ndy = normalize(dx, dy)

    # Find the direction index with highest cosine similarity
    best_idx = 0
    best_dot = -float("inf")
    for i, (vx, vy) in enumerate(direction_vectors):
        dot = ndx * vx + ndy * vy  # cosine similarity
        if dot > best_dot:
            best_dot = dot
            best_idx = i

    return best_idx


def make_wrapped_env(env_config, clock = None, window = None, run_name='no_name', teammate=None):
    def _init():
        base_env = MAISREnvVec( # Create base environment
            config=env_config,
            clock=clock,
            window=window,
            render_mode='headless' if clock is None else 'human',
            run_name=run_name,
            tag=f'trajectorygen0',
        )

        base_env.teammate_active = False

        local_search_policy = LocalSearch()
        go_to_highvalue_policy = GoToNearestThreat(model_path=None)
        change_region_subpolicy = ChangeRegions(model_path=None)
        evade_policy = None

        wrapped_env = MaisrLocalSearchWrapper(
            base_env,
            env_config['obs_noise_std_localsearch'],
            local_search_policy,
            go_to_highvalue_policy,
            change_region_subpolicy,
            evade_policy,
            teammate_manager=None,
            teammate_policy=teammate
        )

        wrapped_env = Monitor(wrapped_env)
        wrapped_env.reset()
        return wrapped_env

    return _init


@dataclass
class Trajectory:
    category: str  # 'human', 'rl', or 'heuristic'
    level: int     # Integer level index
    name: str        # Subject ID or agent strategy identifier

    positions: List[Tuple[float, float]] = None  # List of (x, y) tuples per timestep
    actions: List[int] = None                    # List of actions (0–15) per timestep
    target_ids: List[int] = None                 # History of targets identified per timestep.
    threat_ids: List[int] = None                 # History of threats identified per timestep


@dataclass
class FullTrajectory:
    category: str  # 'human', 'rl', or 'heuristic'
    level: int
    name: str

    positions: List[Tuple[float, float]] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)  # List of actions (0–15) per timestep
    target_ids: List[int] = field(default_factory=list)  # History of targets identified per timestep.
    threat_ids: List[int] =field(default_factory=list)  # History of threats identified per timestep

    target0_pos: List[Tuple[float, float]] = field(default_factory=list)
    target0_status:  List[int] = field(default_factory=list)
    target1_pos: List[Tuple[float, float]] = field(default_factory=list)
    target1_status:   List[int] = field(default_factory=list)
    target2_pos: List[Tuple[float, float]]= field(default_factory=list)
    target2_status:   List[int] = field(default_factory=list)
    target3_pos: List[Tuple[float, float]]= field(default_factory=list)
    target3_status:   List[int] = field(default_factory=list)
    target4_pos: List[Tuple[float, float]]= field(default_factory=list)
    target4_status:   List[int] = field(default_factory=list)
    target5_pos: List[Tuple[float, float]]= field(default_factory=list)
    target5_status:   List[int] = field(default_factory=list)
    target6_pos: List[Tuple[float, float]]= field(default_factory=list)
    target6_status:   List[int] = field(default_factory=list)
    target7_pos: List[Tuple[float, float]]= field(default_factory=list)
    target7_status:   List[int] = field(default_factory=list)
    target8_pos: List[Tuple[float, float]]= field(default_factory=list)
    target8_status:   List[int] = field(default_factory=list)
    target9_pos: List[Tuple[float, float]]= field(default_factory=list)
    target9_status:   List[int] = field(default_factory=list)
    target10_pos: List[Tuple[float, float]]= field(default_factory=list)
    target10_status:   List[int] = field(default_factory=list)
    target11_pos: List[Tuple[float, float]]= field(default_factory=list)
    target11_status:   List[int] = field(default_factory=list)
    target12_pos: List[Tuple[float, float]]= field(default_factory=list)
    target12_status:   List[int] = field(default_factory=list)
    target13_pos: List[Tuple[float, float]]= field(default_factory=list)
    target13_status:   List[int] = field(default_factory=list)
    target14_pos: List[Tuple[float, float]]= field(default_factory=list)
    target14_status:   List[int] = field(default_factory=list)
    threat0_pos: List[Tuple[float, float]]= field(default_factory=list)
    threat0_status:   List[int] = field(default_factory=list)
    threat1_pos: List[Tuple[float, float]]= field(default_factory=list)
    threat1_status:   List[int] = field(default_factory=list)
    threat2_pos: List[Tuple[float, float]]= field(default_factory=list)
    threat2_status:   List[int] = field(default_factory=list)
    threat3_pos: List[Tuple[float, float]]= field(default_factory=list)
    threat3_status:   List[int] = field(default_factory=list)


class SimilarityAnalysis:
    def __init__(self):
        self.human_trajectories_path = '../userstudy_logs/'  # Where the human trajectory json files are stored

        #self.rl_agents_path = './trained_models/pretrained_teammates/' # Where the RL agent .zip and .pkl files are stored
         
        #self.num_rl_agents = 32
        self.level_list = [1, 3, 5, 6, 7]
	
        self.human_trajectories = []
        self.heuristic_trajectories = []
        self.rl_trajectories = []

    def load_level_layouts(self):
        """Load level layouts from JSON file for easy access to level configurations."""
        try:
            with open('../../utility/level_layouts.json', 'r') as f:
                self.level_layouts = json.load(f)
            print(f"Loaded level layouts for {len(self.level_layouts['levels'])} levels")
        except FileNotFoundError:
            print("Warning: level_layouts.json not found")
            raise ValueError
            self.level_layouts = None
        except json.JSONDecodeError as e:
            print(f"Error parsing level_layouts.json: {e}")
            raise ValueError
            self.level_layouts = None


    def process_human_trajectories(self):
        """
        Process human trajectory data from JSON files.
        The JSON structure contains a list of timesteps, each with human_position, human_action, etc.
        """
        direction_vectors = [
            (0, 1), (0.383, 0.924), (0.707, 0.707), (0.924, 0.383),
            (1, 0), (0.924, -0.383), (0.707, -0.707), (0.383, -0.924),
            (0, -1), (-0.383, -0.924), (-0.707, -0.707), (-0.924, -0.383),
            (-1, 0), (-0.924, 0.383), (-0.707, 0.707), (-0.383, 0.924)
        ]

        def normalize(vx, vy):
            mag = math.sqrt(vx ** 2 + vy ** 2)
            return (vx / mag, vy / mag) if mag > 1e-8 else (0.0, 0.0)

        def vector_to_action(dx, dy):
            # Normalize the movement vector
            ndx, ndy = normalize(dx, dy)

            # Compute cosine similarity with each direction vector
            best_idx = 0
            best_dot = -float("inf")
            for i, (vx, vy) in enumerate(direction_vectors):
                dot = ndx * vx + ndy * vy  # cosine similarity since all are normalized
                if dot > best_dot:
                    best_dot = dot
                    best_idx = i
            return best_idx

        self.human_trajectories = []

        for subject_dir in sorted(os.listdir(self.human_trajectories_path)):
            subject_path = os.path.join(self.human_trajectories_path, subject_dir, 'timestep_data')
            if not os.path.isdir(subject_path):
                continue

            for json_file in sorted(glob.glob(os.path.join(subject_path, "*.json"))):
                #print(f"Processing: {json_file}")

                # Extract level from filename (assuming format like "level_A1_data.json")
                filename = os.path.basename(json_file)
                #level_match = re.search(r'[ABCPS](\d+)', filename)
                level_match = re.search(r'[PS](\d+)', filename)
                if not level_match:
                    print(f"Could not extract level from filename: {filename}")
                    continue
                level = int(level_match.group(1))

                try:
                    with open(json_file, "r") as f:
                        data = json.load(f)

                    # The JSON structure is {"timesteps": [list of timestep objects]}
                    if "timesteps" not in data:
                        print(f"No 'timesteps' key found in {json_file}")
                        continue

                    timesteps = data["timesteps"]
                    if not timesteps:
                        print(f"Empty timesteps in {json_file}")
                        continue

                    positions, actions = [], []
                    target_counts, threat_counts = [], []

                    prev_pos = None
                    for i, step in enumerate(timesteps):
                        # Extract human position
                        if "human_position" not in step:
                            print(f"Missing human_position in timestep {i} of {json_file}")
                            continue

                        pos = tuple(step["human_position"])
                        positions.append(pos)

                        # Compute action based on movement
                        if prev_pos is not None:
                            dx = pos[0] - prev_pos[0]
                            dy = pos[1] - prev_pos[1]

                            # If there's no movement, use the stored human_action if available
                            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                                if "human_action" in step and step["human_action"] is not None:
                                    action_idx = step["human_action"]
                                else:
                                    action_idx = 0  # Default to "no action" or "stay"
                            else:
                                action_idx = vector_to_action(dx, dy)
                        else:
                            # First timestep - use stored action or default
                            if "human_action" in step and step["human_action"] is not None:
                                action_idx = step["human_action"]
                            else:
                                action_idx = 0

                        actions.append(action_idx)

                        # Extract target and threat identification counts
                        if "targets_identified_total" in step:
                            target_count = step["targets_identified_total"]
                        else:
                            # Fallback: count True values in target_identified array
                            target_identified = step.get("target_identified", [])
                            target_count = sum(1 for val in target_identified if val)

                        if "threats_identified_total" in step:
                            threat_count = step["threats_identified_total"]
                        else:
                            # Fallback: count True values in threat_identified array
                            threat_identified = step.get("threat_identified", [])
                            threat_count = sum(1 for val in threat_identified if val)

                        target_counts.append(target_count)
                        threat_counts.append(threat_count)

                        prev_pos = pos

                    # Create trajectory object
                    traj = Trajectory(
                        category="human",
                        level=level,
                        name=subject_dir,
                        positions=positions,
                        actions=actions,
                        target_ids=target_counts,
                        threat_ids=threat_counts,
                    )
                    self.human_trajectories.append(traj)
                    print(
                        f"Successfully processed trajectory: {subject_dir}, level {level}, {len(positions)} timesteps")

                except Exception as e:
                    print(f"Error processing {json_file}: {str(e)}")
                    continue

        print(f"Total human trajectories processed: {len(self.human_trajectories)}")

        out_file = os.path.join('', "human_trajectories_for_training.json")
        save_trajectories_to_json(self.human_trajectories, out_file)
        return self.human_trajectories


    def generate_rl_trajectories(self, agents_path, out_name, full_trajectories=False, deterministic=True):

        level_string_dict = {1: 'level_1a', 2: 'level_1b', 3: 'level_2a', 4: 'level_2b', 5: 'level_3a', 6: 'level_3b', 7: 'level_4'}

        render = False

        rl_trajectories = []
        agent_pairs = []

        # Step 1: Find all RL agent .zip and .pkl pairs
        model_patterns = [
            os.path.join(agents_path, "*_model.zip"),
            os.path.join(agents_path, "**/*_model.zip")
        ]

        normstats_patterns = [
            os.path.join(agents_path, "*_vecnormalize.pkl"),
            os.path.join(agents_path, "**/*_vecnormalize.pkl")
        ]

        all_checkpoints = []
        for pattern in model_patterns:
            all_checkpoints.extend(glob.glob(pattern, recursive=True))

        all_normstats = []
        for pattern in normstats_patterns:
            all_normstats.extend(glob.glob(pattern, recursive=True))

        all_checkpoints = list(set(all_checkpoints))  # Remove duplicates and sort by modification time (newest first)
        all_checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        all_normstats.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        print(all_checkpoints)

        if render:
            if hasattr(ctypes, 'windll') and hasattr(ctypes.windll, 'user32'): ctypes.windll.user32.SetProcessDPIAware()
            pygame.display.init()
            pygame.font.init()
            clock = pygame.time.Clock()

            window_width, window_height = 1000, 1100
            window = pygame.display.set_mode((window_width, window_height))
        else:
            pygame.font.init()
            window = None
            clock = None

        agent_list = []
        for agent_model_filename in all_checkpoints:
            model = PPO.load(agent_model_filename)

            # Extract the prefix by removing '_model.zip' suffix
            prefix = agent_model_filename.replace('_model.zip', '')
            expected_vecnorm_filename = f"{prefix}_vecnormalize.pkl"

            norm_stats_path = None
            if os.path.exists(expected_vecnorm_filename):
                norm_stats_path = expected_vecnorm_filename

            agent_pairs.append((model, norm_stats_path, agent_model_filename))
            
        # Step 2: Run each agent in each level and save the trajectory
        for agent_model, pkl_path, agent_model_filename in agent_pairs:
            print(f'\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print(f'%%%%%%% Generating trajectory for RL agent {agent_model_filename[28:]}, {pkl_path} %%%%%%%')
            print(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
            
            # Extract seed (number after "seed")
            seed_match = re.search(r"seed(\d+)", pkl_path)
            seed = seed_match.group(1) if seed_match else "000"
            
            for level in self.level_list:
                for run in range(8): #
                    config = self.config.copy()
                    config['force_specific_level'] = level

                    if render:
                        env = DummyVecEnv([make_wrapped_env(config, clock=clock, window=window) for _ in range(1)])
                    else:
                        env = DummyVecEnv([make_wrapped_env(config) for _ in range(1)])

                    env = load_vecnormalize_wrapper(pkl_path, env)

                    if full_trajectories:
                        trajectory = FullTrajectory(name=f'seed{seed}', level=level, category='rl', actions=[], positions=[], target_ids=[], threat_ids=[])  # instantiate the trajectory

                        level_string = level_string_dict[level]

                        # Fully populate target and threat positions at each step (they don't change)
                        for target in range(15):
                            target_pos = self.level_layouts["levels"][level_string]["targets"][target]
                            setattr(trajectory, f'target{target}_pos', target_pos)

                        for threat in range(4):
                            threat_pos = self.level_layouts["levels"][level_string]["threats"][threat]
                            setattr(trajectory, f'threat{threat}_pos', threat_pos)

                    else:
                        trajectory = Trajectory(name = f'seed{seed}', level = level, category = 'rl', actions = [], positions = [], target_ids = [], threat_ids = []) # instantiate the trajectory

                    step_count = 0
                    obs = env.reset()
                    done = False
                    base_env = env.envs[0].env
                    base_env.env.agents[1].appearance = 'invisible'  # Forces a hold

                    while not done:
                        agent_action, _ = agent_model.predict(obs, deterministic=deterministic)

                        obses, rewards, dones, infos = env.step([agent_action])
                        obs = obses[0]
                        reward = rewards[0]
                        info = infos[0]
                        done = dones[0]

                        if render: base_env.env.render()

                        trajectory.actions.append(agent_action)
                        trajectory.positions.append((base_env.env.agents[0].x, base_env.env.agents[0].y))
                        trajectory.target_ids.append(base_env.env.targets_identified)
                        trajectory.threat_ids.append(base_env.env.num_threats_identified)

                        if full_trajectories:
                            for target in range(15):
                                getattr(trajectory, f'target{target}_status').append(base_env.env.targets[target, 2])
                            for threat in range(4):
                                getattr(trajectory, f'threat{threat}_status').append(base_env.env.threat_identified[threat])
                        step_count += 1

                    rl_trajectories.append(trajectory)
            
        # 3. Save the list of trajectories to a file
        out_file = os.path.join('', out_name)
        save_trajectories_to_json(rl_trajectories, out_file, full_trajectories=full_trajectories)
        
        return rl_trajectories
        

    def generate_strategy_trajectories(self, out_name, full_trajectories = False):

        # Step 1: Create list of heuristic agent parameter combinations. Each element in the list is itself a list of three strings (risk_tolerance, action_noise, spatial_coordination)
        risk_tolerance = ["low", "medium", "high", "max_greedy"]
        action_noise = ["stable", "noisy", "very_noisy"]
        planning_horizon = ['greedy', 'clusters']
        spatial_coordination = [False, True]
        decision_speed = ['fast', 'slow']
        combinations = [list(p) for p in itertools.product(risk_tolerance, action_noise, spatial_coordination, planning_horizon, decision_speed)]
        print(f'Generated {len(combinations)} strategy combinations')

        render = False
        level_string_dict = {1: 'level_1a', 2: 'level_1b', 3: 'level_2a', 4: 'level_2b', 5: 'level_3a', 6: 'level_3b', 7: 'level_4'}

        if render:
            if hasattr(ctypes, 'windll') and hasattr(ctypes.windll, 'user32'): ctypes.windll.user32.SetProcessDPIAware()
            pygame.display.init()
            pygame.font.init()
            clock = pygame.time.Clock()

            window_width, window_height = 1000, 1100
            window = pygame.display.set_mode((window_width, window_height))
        else:
            pygame.font.init()
            window = None
            clock = None


        # Step 2: Generate game trajectories for each heuristic combination for each level
        for combination in combinations:
            print(f'Generating trajectories for heuristic {combination}')
            
            # Instantiate agent with <combination> strategy settings		
            risk_tolerance, action_noise, spatial_coordination, planning_horizon, decision_speed = combination
            
            teammate = GenericTeammatePolicy(env=None,
                local_search_policy=TargetSearchLocalTSP(search_radius=1000, spatial_coord=spatial_coordination, model_path=None, norm_stats_filepath=None, search_method=planning_horizon),
                go_to_highvalue_policy=GoToNearestThreat(model_path=None),
                change_region_subpolicy=ChangeRegions(model_path=None),
                mode_selector_agent=HeuristicAgent(mode_selector='heuristic', risk_tolerance=risk_tolerance, spatial_coord=spatial_coordination),
                use_collision_avoidance=False,
                action_stability=action_noise,
                decision_speed=decision_speed)

            # Run the agent in all 7 levels
            for level in self.level_list:
                config = self.config.copy()
                config['force_specific_level'] = level
                
                #env = DummyVecEnv([make_wrapped_env(config, teammate=teammate) for _ in range(1)])
                if render:
                    env = DummyVecEnv([make_wrapped_env(config, clock=clock, window=window) for _ in range(1)])
                else:
                    env = DummyVecEnv([make_wrapped_env(config) for _ in range(1)])

                base_env = env.envs[0].env
                base_env.current_teammate = teammate
                base_env.teammate_policy = teammate

                teammate.env = base_env.env

                if full_trajectories:
                    trajectory = FullTrajectory(name = f'{risk_tolerance}-{action_noise}_{spatial_coordination}_{planning_horizon}_{decision_speed}', level = level, category = 'heuristic', actions=[], positions=[], target_ids=[], threat_ids=[])  # instantiate the trajectory

                    level_string = level_string_dict[level]

                    # Fully populate target and threat positions at each step (they don't change)
                    for target in range(15):
                        target_pos = self.level_layouts["levels"][level_string]["targets"][target]
                        setattr(trajectory, f'target{target}_pos', target_pos)

                    for threat in range(4):
                        threat_pos = self.level_layouts["levels"][level_string]["threats"][threat]
                        setattr(trajectory, f'threat{threat}_pos', threat_pos)

                else:
                    trajectory = Trajectory(name = f'{risk_tolerance}-{action_noise}_{spatial_coordination}_{planning_horizon}_{decision_speed}', level = level, category = 'heuristic', actions = [], positions = [], target_ids = [], threat_ids = []) # instantiate the trajectory

                step_count = 0
                obs = env.reset()
                done = False

                #print(f'base_env is {base_env} (should be MaisrEnvVec, NOT LocalSearchWrapper\n\n%%%')
                
                while not done:
                    agent0_action = 0
                    base_env.env.agents[0].appearance = 'invisible' # Forces a hold

                    agent1_waypoint = base_env.get_teammate_action()#teammate_action # This is a waypoint
                    #print(f'Agent 1 waypoint is {agent1_waypoint}')

                    current_pos = (base_env.env.agents[1].x, base_env.env.agents[1].y)
                    agent1_action = waypoint_to_direction_index(current_pos, agent1_waypoint)
                    
                    obses, rewards, dones, infos = env.step([agent0_action])
                    obs = obses[0]
                    reward = rewards[0]
                    info = infos[0]
                    done = dones[0]

                    if render:
                        base_env.env.render()
                        
                    trajectory.actions.append(agent1_action)
                    trajectory.positions.append((base_env.env.agents[1].x, base_env.env.agents[1].y))
                    trajectory.target_ids.append(base_env.env.targets_identified)
                    trajectory.threat_ids.append(base_env.env.num_threats_identified)

                    if full_trajectories:
                        for target in range(15):
                            getattr(trajectory, f'target{target}_status').append(base_env.env.targets[target, 2])
                        for threat in range(4):
                            getattr(trajectory, f'threat{threat}_status').append(base_env.env.threat_identified[threat])

                    step_count += 1
                        
                self.heuristic_trajectories.append(trajectory)

        # 3. Save the list of trajectories to a file type of your choice so we don't have regenerate it if we need to re-run
        out_file = os.path.join('', out_name)
        save_trajectories_to_json(self.heuristic_trajectories, out_file, full_trajectories=full_trajectories)

        return self.heuristic_trajectories
        
    
    ################################################ Helper functions ################################################
    
    # Ready to test
    def compute_2d_emd(self, heatmap1: np.ndarray, heatmap2: np.ndarray) -> float:
        """
        Compute 2D Earth Mover's Distance (EMD) between two 2D histograms using
        Euclidean distance and the Hungarian algorithm.
        """

        # Normalize to probability distributions
        h1 = heatmap1 / np.sum(heatmap1)
        h2 = heatmap2 / np.sum(heatmap2)

        # Get coordinates of all non-zero bins
        coords = np.indices(h1.shape).reshape(2, -1).T  # (x_idx, y_idx)
        weights1 = h1.flatten()
        weights2 = h2.flatten()

        # Convert to point clouds by repeating coordinates proportional to weight
        # For performance, use integer scaling
        scale = 1000  # Higher -> better approximation
        pts1 = np.repeat(coords, (weights1 * scale).astype(int), axis=0)
        pts2 = np.repeat(coords, (weights2 * scale).astype(int), axis=0)

        n = min(len(pts1), len(pts2))
        if n == 0: n = 1
        #print(f'n = {n}')
        pts1 = pts1[:n]
        pts2 = pts2[:n]

        # Compute cost matrix (Euclidean distances)
        d = cdist(pts1, pts2)

        # Solve the linear sum assignment problem
        row_ind, col_ind = linear_sum_assignment(d)

        # Compute mean cost = EMD
        emd = d[row_ind, col_ind].sum() / n
        return emd
    
    ################################################ Analysis functions ################################################

    def compare_position_heatmaps_2d(self, human_trajectories: List[Trajectory], rl_trajectories: List[Trajectory],
                                     heuristic_trajectories: List[Trajectory], bins: int = 100,
                                     output_csv: str = "heatmap_emd_results_2d.csv"):
        """ References/justification for using 2D EMD for this analysis:
            https://stats.stackexchange.com/questions/404775/calculate-earth-movers-distance-for-two-grayscale-images
            https://stats.stackexchange.com/questions/659384/compute-p-value-of-earth-movers-distance-score-comparing-two-heatmaps-in-r
        """

        # --- 1. Aggregate positions ---
        def extract_positions(trajs: List[Trajectory], subsample_every_n: int = None) -> np.ndarray:
            all_positions = []
            for t in trajs:
                if subsample_every_n is not None:
                    # Subsample positions by taking every nth position
                    subsampled_positions = t.positions[::subsample_every_n]
                    all_positions.extend(subsampled_positions)
                else:
                    # Use all positions
                    all_positions.extend(t.positions)
            return np.array(all_positions)

        import random

        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        human_sample_fraction = 0.3

        # Sample human trajectories
        if human_sample_fraction < 1.0:
            sample_size = int(len(human_trajectories) * human_sample_fraction)
            sampled_human_trajectories = random.sample(human_trajectories, sample_size)
            print(
                f"\nUsing {sample_size} out of {len(human_trajectories)} human trajectories ({human_sample_fraction:.1%})")
        else:
            sampled_human_trajectories = human_trajectories
            print(f"\nUsing all {len(human_trajectories)} human trajectories")

        # Extract human positions (subsampled)
        human_positions = extract_positions(sampled_human_trajectories, subsample_every_n=1)

        # Group heuristic trajectories by agent type (name)
        heuristic_agents = {}
        for traj in heuristic_trajectories:
            agent_name = traj.name
            if agent_name not in heuristic_agents:
                heuristic_agents[agent_name] = []
            heuristic_agents[agent_name].append(traj)

        print(f"\nFound {len(heuristic_agents)} unique heuristic agent types:")
        for agent_name, trajs in heuristic_agents.items():
            print(f"  - {agent_name}: {len(trajs)} trajectories")

        # Extract RL positions (for overall comparison)
        rl_positions = extract_positions(rl_trajectories)

        # --- 2. Define common grid for all heatmaps ---
        x_min, y_min = -500, -500
        x_max, y_max = 500, 500

        def compute_heatmap(positions, normalize_by_density=True):
            if len(positions) == 0:
                return np.zeros((bins, bins))

            heatmap, _, _ = np.histogram2d(
                positions[:, 0], positions[:, 1],
                bins=bins,
                range=[[x_min, x_max], [y_min, y_max]])
            #heatmap = np.log(heatmap + 1)

            if normalize_by_density:
                total_points = len(positions)
                if total_points > 0:
                    heatmap = heatmap * 1000 / total_points

                # total_mass = np.sum(heatmap)
                # #print(f'Total mass is: {total_mass}\n')
                # if total_mass > 0:
                #     heatmap = heatmap / total_mass

            return np.log(heatmap + 1)

        # Compute human and RL heatmaps
        print(f'Human heatmap:')
        human_heatmap = compute_heatmap(human_positions, normalize_by_density=True)
        print(f'RL heatmap:')
        rl_heatmap = compute_heatmap(rl_positions, normalize_by_density=True)

        # Compute heatmaps for each heuristic agent type
        heuristic_heatmaps = {}
        for agent_name, trajs in heuristic_agents.items():
            agent_positions = extract_positions(trajs)
            #print(f'Heuristic agent {agent_name}:')
            heuristic_heatmaps[agent_name] = compute_heatmap(agent_positions, normalize_by_density=True)

        # --- 3. Compute pairwise 2D EMD ---
        emd_results = {}

        # Standard comparisons
        emd_results['human_vs_rl'] = self.compute_2d_emd(human_heatmap, rl_heatmap)

        # Compute all heuristic positions for overall comparison
        all_heuristic_positions = extract_positions(heuristic_trajectories)
        print(f'All heuristic heatmap:')
        all_heuristic_heatmap = compute_heatmap(all_heuristic_positions, normalize_by_density=True)
        emd_results['human_vs_all_heuristic'] = self.compute_2d_emd(human_heatmap, all_heuristic_heatmap)
        emd_results['rl_vs_all_heuristic'] = self.compute_2d_emd(rl_heatmap, all_heuristic_heatmap)

        # Compute EMD between human and each individual heuristic agent type
        for agent_name, agent_heatmap in heuristic_heatmaps.items():
            emd_key = f'human_vs_heuristic_{agent_name}'
            emd_results[emd_key] = self.compute_2d_emd(human_heatmap, agent_heatmap)

        # --- 4. Output results ---
        print("\nPairwise 2D Heatmap EMD Results:")
        print("=" * 50)

        # Print standard comparisons first
        standard_keys = ['human_vs_rl', 'human_vs_all_heuristic', 'rl_vs_all_heuristic']
        for key in standard_keys:
            if key in emd_results:
                print(f"{key}: {emd_results[key]:.6f}")

        print("\nHuman vs Individual Heuristic Agents:")
        print("-" * 40)
        # Sort heuristic agent results by EMD value
        heuristic_results = {k: v for k, v in emd_results.items() if k.startswith('human_vs_heuristic_')}
        sorted_heuristic = sorted(heuristic_results.items(), key=lambda x: x[1])

        for key, value in sorted_heuristic:
            agent_name = key.replace('human_vs_heuristic_', '')
            print(f"{agent_name}: {value:.6f}")

        # Save results to CSV
        pd.DataFrame([emd_results]).to_csv(output_csv, index=False)

        # --- 5. Plot the heatmaps ---
        # Plot overview: Human, RL, All Heuristic
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        heatmaps = [human_heatmap, rl_heatmap, all_heuristic_heatmap]
        titles = ['Human', 'RL Agent', 'All Heuristic']

        for ax, hm, title in zip(axes, heatmaps, titles):
            im = ax.imshow(hm, origin='lower', aspect='auto',extent=[x_min, x_max, y_min, y_max], cmap='hot')
            ax.set_title(f"{title} Heatmap", fontsize=20)
            #ax.set_xlabel("X Position",fontsize=14)
            #ax.set_ylabel("Y Position",fontsize=14)
            #fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig('./exp3_similarity_analysis/heatmaps_overview.png', dpi=300)
        plt.close(fig)

        # Plot individual heuristic agent heatmaps (top 6 most similar to humans)
        top_6_agents = sorted_heuristic[:6]
        if len(top_6_agents) > 0:
            n_cols = 3
            n_rows = 2
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12))
            axes = axes.flatten()

            for i, (key, emd_value) in enumerate(top_6_agents):
                if i >= len(axes):
                    break

                agent_name = key.replace('human_vs_heuristic_', '')
                heatmap = heuristic_heatmaps[agent_name]

                im = axes[i].imshow(heatmap, origin='lower', aspect='auto',
                                    extent=[x_min, x_max, y_min, y_max], cmap='hot')
                axes[i].set_title(f"{agent_name}\n(EMD = {emd_value:.3f})")
                #axes[i].set_xlabel("X Position")
                #axes[i].set_ylabel("Y Position")
                #fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

            # Hide unused subplots
            for i in range(len(top_6_agents), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.savefig('./exp3_similarity_analysis/heatmaps_top_heuristic_agents.png', dpi=300)
            plt.close(fig)

            # Plot individual heuristic agent heatmaps (top 6 LEAST similar to humans)
            worst_6_agents = sorted_heuristic[-6:]
            if len(top_6_agents) > 0:
                n_cols = 3
                n_rows = 2
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12))
                axes = axes.flatten()

                for i, (key, emd_value) in enumerate(worst_6_agents):
                    if i >= len(axes):
                        break

                    agent_name = key.replace('human_vs_heuristic_', '')
                    heatmap = heuristic_heatmaps[agent_name]

                    im = axes[i].imshow(heatmap, origin='lower', aspect='auto',
                                        extent=[x_min, x_max, y_min, y_max], cmap='hot')
                    axes[i].set_title(f"{agent_name}\n(EMD = {emd_value:.3f})")
                    #axes[i].set_xlabel("X Position")
                    #axes[i].set_ylabel("Y Position")
                    #fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

                # Hide unused subplots
                for i in range(len(worst_6_agents), len(axes)):
                    axes[i].set_visible(False)

                plt.tight_layout()
                plt.savefig('./exp3_similarity_analysis/heatmaps_worst_heuristic_agents.png', dpi=300)
                plt.close(fig)

        # Create a summary plot showing EMD values
        plt.figure(figsize=(12, 8))
        agent_names = [key.replace('human_vs_heuristic_', '') for key, _ in sorted_heuristic]
        emd_values = [value for _, value in sorted_heuristic]

        bars = plt.bar(range(len(agent_names)), emd_values, alpha=0.7)
        plt.xlabel('Heuristic Agent Type')
        plt.ylabel('EMD Distance from Human Trajectories')
        plt.title('Earth Mover\'s Distance: Human vs Individual Heuristic Agents', fontsize=20)
        plt.xticks(range(len(agent_names)), agent_names, rotation=90, ha='right')

        # Add horizontal lines for reference comparisons
        if 'human_vs_rl' in emd_results:
            plt.axhline(y=emd_results['human_vs_rl'], color='green', linestyle='-',
                        label=f"Human vs RL ({emd_results['human_vs_rl']:.2f})")
        # if 'human_vs_all_heuristic' in emd_results:
        #     plt.axhline(y=emd_results['human_vs_all_heuristic'], color='blue', linestyle='--',
        #                 label=f"Human vs All Heuristic ({emd_results['human_vs_all_heuristic']:.3f})")

        plt.legend()
        plt.grid(True, axis='y', alpha=0.6)
        plt.tight_layout()
        plt.savefig('./exp3_similarity_analysis/emd_comparison_bar_chart.png', dpi=300)
        plt.close()

        return emd_results
    
    # Ready to test
    # def compare_position_heatmaps_2d(self, human_trajectories_for_training: List[Trajectory], rl_trajectories: List[Trajectory],
    #                                  heuristic_trajectories: List[Trajectory], bins: int = 50,
    #                                  output_csv: str = "heatmap_emd_results_2d.csv"):
    #     """ References/justification for using 2D EMD for this analysis:
    #         https://stats.stackexchange.com/questions/404775/calculate-earth-movers-distance-for-two-grayscale-images
    #         https://stats.stackexchange.com/questions/659384/compute-p-value-of-earth-movers-distance-score-comparing-two-heatmaps-in-r
    #     """
    #
    #     # --- 1. Aggregate positions ---
    #     #def extract_positions(trajs: List[Trajectory]) -> np.ndarray:
    #         #return np.array([pos for t in trajs for pos in t.positions])
    #
    #     def extract_positions(trajs: List[Trajectory], subsample_every_n: int = None) -> np.ndarray:
    #         all_positions = []
    #         for t in trajs:
    #             if subsample_every_n is not None:
    #                 # Subsample positions by taking every nth position
    #                 subsampled_positions = t.positions[::subsample_every_n]
    #                 all_positions.extend(subsampled_positions)
    #             else:
    #                 # Use all positions
    #                 all_positions.extend(t.positions)
    #         return np.array(all_positions)
    #
    #     import random
    #
    #     # Set random seed for reproducibility
    #     random.seed(42)
    #     np.random.seed(42)
    #     human_sample_fraction = 0.3
    #
    #     # Sample human trajectories
    #     if human_sample_fraction < 1.0:
    #         sample_size = int(len(human_trajectories_for_training) * human_sample_fraction)
    #         sampled_human_trajectories = random.sample(human_trajectories_for_training, sample_size)
    #         print(f"Using {sample_size} out of {len(human_trajectories_for_training)} human trajectories ({human_sample_fraction:.1%})")
    #     else:
    #         sampled_human_trajectories = human_trajectories_for_training
    #         print(f"Using all {len(human_trajectories_for_training)} human trajectories")
    #
    #
    #     #human_positions = extract_positions(sampled_human_trajectories)
    #     human_positions = extract_positions(sampled_human_trajectories, subsample_every_n=10)
    #     rl_positions = extract_positions(rl_trajectories)
    #     heuristic_positions = extract_positions(heuristic_trajectories)
    #
    #     # --- 2. Define common grid for all heatmaps ---
    #     #all_positions = np.vstack([human_positions, rl_positions, heuristic_positions])
    #     x_min, y_min = -500, -500 #np.min(all_positions, axis=0)
    #     x_max, y_max = 500, 500 #np.max(all_positions, axis=0)
    #
    #     def compute_heatmap(positions):
    #         print(positions)
    #         heatmap, _, _ = np.histogram2d(
    #             positions[:,0], positions[:,1],
    #             bins=bins,
    #             range=[[x_min, x_max], [y_min, y_max]]
    #         )
    #         return heatmap
    #
    #     human_heatmap = compute_heatmap(human_positions)
    #     rl_heatmap = compute_heatmap(rl_positions)
    #     heuristic_heatmap = compute_heatmap(heuristic_positions)
    #
    #     # --- 3. Compute pairwise 2D EMD ---
    #     emd_results = {
    #         'human_vs_rl': self.compute_2d_emd(human_heatmap, rl_heatmap),
    #         'human_vs_heuristic': self.compute_2d_emd(human_heatmap, heuristic_heatmap),
    #         'rl_vs_heuristic': self.compute_2d_emd(rl_heatmap, heuristic_heatmap),
    #     }
    #
    #     # --- 4. Output results ---
    #     print("Pairwise 2D Heatmap EMD Results:")
    #     for k, v in emd_results.items():
    #         print(f"{k}: {v:.6f}")
    #
    #     pd.DataFrame([emd_results]).to_csv(output_csv, index=False)
    #
    #     # --- 5. Plot the three heatmaps ---
    #     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    #     heatmaps = [human_heatmap, rl_heatmap, heuristic_heatmap]
    #     titles = ['Human', 'RL Agent', 'Heuristic']
    #
    #     for ax, hm, title in zip(axes, heatmaps, titles):
    #         im = ax.imshow(hm, origin='lower', aspect='auto',
    #                        extent=[x_min, x_max, y_min, y_max], cmap='hot')
    #         ax.set_title(f"{title} Heatmap")
    #         ax.set_xlabel("X Position")
    #         ax.set_ylabel("Y Position")
    #         fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    #
    #     plt.tight_layout()
    #     plt.savefig('./exp3_similarity_analysis/position_heatmaps.png', dpi=300)
    #     plt.close(fig)
    #
    #     return emd_results
    #
    
    # # Ready to test
    # def compute_silhouette_scores(self,
    #     human_trajectories_for_training: List[Trajectory],
    #     rl_trajectories: List[Trajectory],
    #     heuristic_trajectories: List[Trajectory],
    #     save_dir: str = "exp3_similarity_analysis"
    # ):
    #     """
    #     Compute and visualize silhouette scores for human, RL, and heuristic gameplay trajectories.
    #     Clusters are based on the final number of targets and threats identified per trajectory.
    #     """
    #     os.makedirs(save_dir, exist_ok=True)
    #
    #     # 1. Build the feature matrix: (targets_final, threats_final)
    #     def extract_features(trajs):
    #         return np.array([
    #             [t.target_ids[-1], t.threat_ids[-1]]
    #             for t in trajs
    #             if t.target_ids and t.threat_ids
    #         ])
    #
    #     human_features = extract_features(human_trajectories_for_training)
    #     rl_features = extract_features(rl_trajectories)
    #     heuristic_features = extract_features(heuristic_trajectories)
    #
    #     # Combine for plotting
    #     all_features = np.vstack([human_features, rl_features, heuristic_features])
    #     all_labels = (
    #         ["human"] * len(human_features)
    #         + ["rl"] * len(rl_features)
    #         + ["heuristic"] * len(heuristic_features)
    #     )
    #
    #     # 2. Create scatter plot
    #     colors = {"human": "blue", "rl": "green", "heuristic": "red"}
    #     plt.figure(figsize=(8, 6))
    #     for label, features in zip(
    #         ["human", "rl", "heuristic"],
    #         [human_features, rl_features, heuristic_features]
    #     ):
    #         if len(features) > 0:
    #             plt.scatter(
    #                 features[:, 0], features[:, 1],
    #                 c=colors[label], label=label, alpha=0.7, edgecolors='k'
    #             )
    #     plt.xlabel("Final # Targets Identified")
    #     plt.ylabel("Final # Threats Identified")
    #     plt.title("Trajectory Outcome Clusters (Targets vs Threats)")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(os.path.join(save_dir, "trajectory_clusters.png"))
    #     plt.close()
    #
    #     # 3. Compute silhouette scores
    #     # Silhouette score requires >= 2 clusters
    #     def get_silhouette_score(features, n_clusters=None):
    #         if len(features) < 2:
    #             return None  # Can't compute silhouette for <2 samples
    #         if n_clusters is None:
    #             n_clusters = min(2, len(features))  # fallback
    #         # Standardize features to improve cluster separation
    #         X = StandardScaler().fit_transform(features)
    #         # Use KMeans for clustering
    #         kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    #         labels = kmeans.fit_predict(X)
    #         if len(set(labels)) < 2:
    #             return None  # silhouette score undefined for single cluster
    #         return silhouette_score(X, labels)
    #
    #     silhouette_results = {
    #         "human_only": get_silhouette_score(human_features),
    #         "rl_only": get_silhouette_score(rl_features),
    #         "heuristic_only": get_silhouette_score(heuristic_features),
    #         "human_rl": get_silhouette_score(np.vstack([human_features, rl_features])),
    #         "human_heuristic": get_silhouette_score(np.vstack([human_features, heuristic_features])),
    #     }
    #
    #     # 4. Save results
    #     results_path = os.path.join(save_dir, "silhouette_scores.txt")
    #     with open(results_path, "w") as f:
    #         for k, v in silhouette_results.items():
    #             f.write(f"{k}: {v}\n")
    #
    #     return silhouette_results

    def compute_silhouette_scores(self,
                                  human_trajectories: List[Trajectory],
                                  rl_trajectories: List[Trajectory],
                                  heuristic_trajectories: List[Trajectory],
                                  save_dir: str = "exp3_similarity_analysis"
                                  ):
        """
        Compute and visualize silhouette scores for human, RL, and heuristic gameplay trajectories.
        Clusters are based on targets and threats identified at three temporal checkpoints:
        33%, 67%, and 100% through each trajectory.
        """
        os.makedirs(save_dir, exist_ok=True)

        # Define temporal checkpoints
        checkpoints = [0.33, 0.67, 1.0]
        checkpoint_names = ["33%", "67%", "100%"]

        def extract_features_at_checkpoints(trajs):
            """Extract features at 33%, 67%, and 100% through each trajectory"""
            features_by_checkpoint = {cp: [] for cp in checkpoints}

            for traj in trajs:
                if not traj.target_ids or not traj.threat_ids:
                    continue

                traj_length = len(traj.target_ids)
                if traj_length < 2:  # Skip very short trajectories
                    continue

                for checkpoint in checkpoints:
                    # Calculate index for this checkpoint
                    if checkpoint == 1.0:
                        idx = traj_length - 1  # Use final index for 100%
                    else:
                        idx = int(checkpoint * traj_length)
                        idx = min(idx, traj_length - 1)  # Ensure we don't exceed bounds

                    # Extract features at this checkpoint
                    targets_at_checkpoint = traj.target_ids[idx]
                    threats_at_checkpoint = traj.threat_ids[idx]
                    features_by_checkpoint[checkpoint].append([targets_at_checkpoint, threats_at_checkpoint])

            # Convert to numpy arrays
            for checkpoint in checkpoints:
                if features_by_checkpoint[checkpoint]:
                    features_by_checkpoint[checkpoint] = np.array(features_by_checkpoint[checkpoint])
                else:
                    features_by_checkpoint[checkpoint] = np.array([]).reshape(0, 2)

            return features_by_checkpoint

        # Extract features for each group at each checkpoint
        human_features_by_cp = extract_features_at_checkpoints(human_trajectories)
        rl_features_by_cp = extract_features_at_checkpoints(rl_trajectories)
        heuristic_features_by_cp = extract_features_at_checkpoints(heuristic_trajectories)

        # Create subplot grid for visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        colors = {"human": "blue", "rl": "green", "heuristic": "red"}

        silhouette_results = {}

        for i, (checkpoint, cp_name) in enumerate(zip(checkpoints, checkpoint_names)):
            # Get features for this checkpoint
            human_features = human_features_by_cp[checkpoint]
            rl_features = rl_features_by_cp[checkpoint]
            heuristic_features = heuristic_features_by_cp[checkpoint]

            # Skip if any group has no data
            if len(human_features) == 0 or len(rl_features) == 0 or len(heuristic_features) == 0:
                print(f"Warning: Insufficient data for checkpoint {cp_name}")
                continue

            # Combine for plotting and analysis
            all_features = np.vstack([human_features, rl_features, heuristic_features])
            all_labels = (
                    ["human"] * len(human_features)
                    + ["rl"] * len(rl_features)
                    + ["heuristic"] * len(heuristic_features)
            )

            # Plot scatter plot for this checkpoint (top row)
            ax_scatter = axes[0, i]
            for label, features in zip(
                    ["human", "rl", "heuristic"],
                    [human_features, rl_features, heuristic_features]
            ):
                if len(features) > 0:
                    ax_scatter.scatter(
                        features[:, 0], features[:, 1],
                        c=colors[label], label=label, alpha=0.7, edgecolors='k'
                    )
            ax_scatter.set_xlabel("Targets Identified")
            ax_scatter.set_ylabel("Threats Identified")
            ax_scatter.set_title(f"Trajectory Clusters at {cp_name}")
            ax_scatter.legend()
            ax_scatter.grid(True)

            # Compute silhouette scores for this checkpoint
            def get_silhouette_score(features, n_clusters=None):
                if len(features) < 2:
                    return None
                if n_clusters is None:
                    n_clusters = min(2, len(features))

                # Standardize features
                if np.std(features) == 0:  # Handle case where all values are identical
                    return None
                X = StandardScaler().fit_transform(features)

                # Use KMeans for clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
                labels = kmeans.fit_predict(X)

                if len(set(labels)) < 2:
                    return None
                return silhouette_score(X, labels)

            checkpoint_results = {
                f"human_only_{cp_name}": get_silhouette_score(human_features),
                f"rl_only_{cp_name}": get_silhouette_score(rl_features),
                f"heuristic_only_{cp_name}": get_silhouette_score(heuristic_features),
                f"human_rl_{cp_name}": get_silhouette_score(np.vstack([human_features, rl_features])),
                f"human_heuristic_{cp_name}": get_silhouette_score(np.vstack([human_features, heuristic_features])),
                f"all_groups_{cp_name}": get_silhouette_score(all_features)
            }

            silhouette_results.update(checkpoint_results)

        # Create silhouette score comparison plot (bottom row)
        # Prepare data for comparison across checkpoints
        comparison_metrics = ["human_only", "rl_only", "heuristic_only", "human_rl", "human_heuristic", "all_groups"]

        for j, metric in enumerate(comparison_metrics):
            if j >= 3:  # Only plot first 3 metrics in bottom row
                break

            ax_bar = axes[1, j]
            scores = []
            labels = []

            for i, cp_name in enumerate(checkpoint_names):
                key = f"{metric}_{cp_name}"
                if key in silhouette_results and silhouette_results[key] is not None:
                    scores.append(silhouette_results[key])
                    labels.append(cp_name)

            if scores:
                bars = ax_bar.bar(labels, scores, alpha=0.7)
                ax_bar.set_ylabel("Silhouette Score")
                ax_bar.set_title(f"{metric.replace('_', ' ').title()}")
                ax_bar.grid(True, axis='y', alpha=0.3)

                # Add value labels on bars
                for bar, score in zip(bars, scores):
                    height = bar.get_height()
                    ax_bar.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                                f'{score:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "trajectory_clusters_temporal.png"), dpi=300)
        plt.close()

        # Create a comprehensive results table
        results_df = pd.DataFrame()
        for checkpoint, cp_name in zip(checkpoints, checkpoint_names):
            checkpoint_data = {k.replace(f'_{cp_name}', ''): v for k, v in silhouette_results.items()
                               if k.endswith(f'_{cp_name}')}
            checkpoint_data['checkpoint'] = cp_name
            results_df = pd.concat([results_df, pd.DataFrame([checkpoint_data])], ignore_index=True)

        # Save detailed results
        results_path = os.path.join(save_dir, "silhouette_scores_temporal.csv")
        results_df.to_csv(results_path, index=False)

        # Print summary
        print("\nSilhouette Scores by Temporal Checkpoint:")
        print("=" * 60)
        for checkpoint, cp_name in zip(checkpoints, checkpoint_names):
            print(f"\n{cp_name} through trajectory:")
            print("-" * 30)
            for key, value in silhouette_results.items():
                if key.endswith(f'_{cp_name}') and value is not None:
                    clean_key = key.replace(f'_{cp_name}', '').replace('_', ' ')
                    print(f"  {clean_key}: {value:.4f}")

        return silhouette_results

    def analyze_temporal_silhouette_evolution(self,
                                              human_trajectories: List[Trajectory],
                                              rl_trajectories: List[Trajectory],
                                              heuristic_trajectories: List[Trajectory],
                                              save_dir: str = "exp3_similarity_analysis",
                                              n_timepoints: int = 10
                                              ):
        """
        Analyze silhouette scores at multiple evenly-spaced time points throughout trajectories
        and plot the temporal evolution of human-RL vs human-heuristic similarity.

        Args:
            human_trajectories: List of human trajectory objects
            rl_trajectories: List of RL trajectory objects
            heuristic_trajectories: List of heuristic trajectory objects
            save_dir: Directory to save outputs
            n_timepoints: Number of evenly-spaced time points to analyze (default: 10)
        """
        os.makedirs(save_dir, exist_ok=True)

        # Generate evenly-spaced time points from 10% to 100%
        time_points = np.linspace(0.1, 0.9, n_timepoints)

        def extract_features_at_timepoints(trajs, time_points):
            """Extract features at specified time points for all trajectories"""
            features_by_timepoint = {tp: [] for tp in time_points}

            for traj in trajs:
                if not traj.target_ids or not traj.threat_ids:
                    continue

                traj_length = len(traj.target_ids)
                if traj_length < 2:  # Skip very short trajectories
                    continue

                for time_point in time_points:
                    # Calculate index for this time point
                    if time_point >= 1.0:
                        idx = traj_length - 1  # Use final index for 100%
                    else:
                        idx = int(time_point * traj_length)
                        idx = min(idx, traj_length - 1)  # Ensure we don't exceed bounds

                    # Extract features at this time point
                    targets_at_timepoint = traj.target_ids[idx]
                    threats_at_timepoint = traj.threat_ids[idx]
                    features_by_timepoint[time_point].append([targets_at_timepoint, threats_at_timepoint])

            # Convert to numpy arrays
            for time_point in time_points:
                if features_by_timepoint[time_point]:
                    features_by_timepoint[time_point] = np.array(features_by_timepoint[time_point])
                else:
                    features_by_timepoint[time_point] = np.array([]).reshape(0, 2)

            return features_by_timepoint

        def get_silhouette_score(features, n_clusters=2):
            """Compute silhouette score for given features"""
            if len(features) < n_clusters:
                return None

            # Check for zero variance (all identical values)
            if np.all(features == features[0]):
                return None

            # Standardize features
            try:
                X = StandardScaler().fit_transform(features)
            except:
                return None

            # Use KMeans for clustering
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)

                if len(set(labels)) < 2:
                    return None
                return silhouette_score(X, labels)
            except:
                return None

        # Extract features for each group at each time point
        print("Extracting features at multiple time points...")
        human_features_by_tp = extract_features_at_timepoints(human_trajectories, time_points)
        rl_features_by_tp = extract_features_at_timepoints(rl_trajectories, time_points)
        heuristic_features_by_tp = extract_features_at_timepoints(heuristic_trajectories, time_points)

        # Compute silhouette scores over time
        human_rl_scores = []
        human_heuristic_scores = []
        valid_time_points = []

        print("Computing silhouette scores at each time point...")
        for i, time_point in enumerate(time_points):
            print(f"Processing time point {i + 1}/{len(time_points)} ({time_point:.1%})")

            human_features = human_features_by_tp[time_point]
            rl_features = rl_features_by_tp[time_point]
            heuristic_features = heuristic_features_by_tp[time_point]

            # Skip if any group has insufficient data
            if len(human_features) < 2 or len(rl_features) < 2 or len(heuristic_features) < 2:
                print(f"  Insufficient data at {time_point:.1%}, skipping...")
                continue

            # Compute human-RL silhouette score
            human_rl_combined = np.vstack([human_features, rl_features])
            human_rl_score = get_silhouette_score(human_rl_combined)

            # Compute human-heuristic silhouette score
            human_heuristic_combined = np.vstack([human_features, heuristic_features])
            human_heuristic_score = get_silhouette_score(human_heuristic_combined)

            # Only keep time points where both scores are valid
            if human_rl_score is not None and human_heuristic_score is not None:
                human_rl_scores.append(human_rl_score)
                human_heuristic_scores.append(human_heuristic_score)
                valid_time_points.append(time_point)
                print(f"  H-RL: {human_rl_score:.4f}, H-Heuristic: {human_heuristic_score:.4f}")
            else:
                print(f"  Invalid scores at {time_point:.1%}, skipping...")

        # Convert to numpy arrays for easier handling
        valid_time_points = np.array(valid_time_points)
        human_rl_scores = np.array(human_rl_scores)
        human_heuristic_scores = np.array(human_heuristic_scores)

        if len(valid_time_points) == 0:
            print("Warning: No valid time points found for analysis!")
            return None

        # Create the temporal evolution plot
        plt.figure(figsize=(12, 8))

        # Plot both lines
        plt.plot(valid_time_points * 100, human_rl_scores,
                 marker='o', linewidth=2.5, markersize=8,
                 label='Human vs RL', color='blue', alpha=0.8)

        plt.plot(valid_time_points * 100, human_heuristic_scores,
                 marker='s', linewidth=2.5, markersize=8,
                 label='Human vs Heuristic', color='red', alpha=0.8)

        # Customize the plot
        plt.xlabel('Progress Through Trajectory (%)', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.title('Temporal Evolution of Agent Similarity\n(Higher scores = better separation between groups)',
                  fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        # Add some styling
        plt.xlim(valid_time_points[0] * 100 - 2, valid_time_points[-1] * 100 + 2)

        # Add horizontal line at 0 for reference
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

        # Improve tick formatting
        plt.gca().tick_params(labelsize=10)

        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(save_dir, "temporal_silhouette_evolution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Save numerical results
        results_df = pd.DataFrame({
            'time_point_percent': valid_time_points * 100,
            'human_vs_rl_silhouette': human_rl_scores,
            'human_vs_heuristic_silhouette': human_heuristic_scores
        })

        results_path = os.path.join(save_dir, "temporal_silhouette_scores.csv")
        results_df.to_csv(results_path, index=False)

        # Print summary statistics
        print("\n" + "=" * 60)
        print("TEMPORAL SILHOUETTE ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Valid time points analyzed: {len(valid_time_points)}")
        print(f"Time range: {valid_time_points[0]:.1%} to {valid_time_points[-1]:.1%}")

        print(f"\nHuman vs RL Silhouette Scores:")
        print(f"  Mean: {np.mean(human_rl_scores):.4f}")
        print(f"  Std:  {np.std(human_rl_scores):.4f}")
        print(f"  Range: {np.min(human_rl_scores):.4f} to {np.max(human_rl_scores):.4f}")

        print(f"\nHuman vs Heuristic Silhouette Scores:")
        print(f"  Mean: {np.mean(human_heuristic_scores):.4f}")
        print(f"  Std:  {np.std(human_heuristic_scores):.4f}")
        print(f"  Range: {np.min(human_heuristic_scores):.4f} to {np.max(human_heuristic_scores):.4f}")

        # Determine which comparison shows better separation on average
        avg_hr = np.mean(human_rl_scores)
        avg_hh = np.mean(human_heuristic_scores)

        if avg_hr > avg_hh:
            better_sep = "Human vs RL"
            diff = avg_hr - avg_hh
        else:
            better_sep = "Human vs Heuristic"
            diff = avg_hh - avg_hr

        print(f"\nBetter average separation: {better_sep} (by {diff:.4f})")

        # Check for temporal trends
        from scipy.stats import pearsonr

        hr_corr, hr_p = pearsonr(valid_time_points, human_rl_scores)
        hh_corr, hh_p = pearsonr(valid_time_points, human_heuristic_scores)

        print(f"\nTemporal trends:")
        print(f"  Human vs RL correlation with time: r={hr_corr:.3f}, p={hr_p:.3f}")
        print(f"  Human vs Heuristic correlation with time: r={hh_corr:.3f}, p={hh_p:.3f}")

        print(f"\nResults saved to:")
        print(f"  Plot: {plot_path}")
        print(f"  Data: {results_path}")

        return {
            'time_points': valid_time_points,
            'human_rl_scores': human_rl_scores,
            'human_heuristic_scores': human_heuristic_scores,
            'summary_stats': {
                'hr_mean': np.mean(human_rl_scores),
                'hr_std': np.std(human_rl_scores),
                'hh_mean': np.mean(human_heuristic_scores),
                'hh_std': np.std(human_heuristic_scores),
                'hr_time_corr': hr_corr,
                'hh_time_corr': hh_corr
            }
        }


    # Ready to test
    def compare_progress_rates(self, human_trajectories: List[Trajectory], rl_trajectories: List[Trajectory],
                heuristic_trajectories: List[Trajectory], save_dir: str = "exp3_similarity_analysis"):
        """
        1. Compute DTW distances for intra- and inter-group pairs based on target identification histories.
        2. Compare human-vs-heuristic vs human-vs-rl with Mann-Whitney U test.
        3. Compute Pearson correlations.
        4. Generate and save boxplot of DTW distributions.
        """
        os.makedirs(save_dir, exist_ok=True)

        # Helper: extract target time series for DTW comparison
        def extract_series(traj: Trajectory):
            return np.array(traj.target_ids)

        # Helper: compute all pairwise DTW distances in a list of trajectories
        def compute_intragroup_dtw(trajs):
            dtw_distances = []
            for i in range(len(trajs)):
                for j in range(i + 1, len(trajs)):
                    series1 = extract_series(trajs[i])
                    series2 = extract_series(trajs[j])
                    dist, _ = fastdtw(series1, series2)
                    dtw_distances.append(dist)
            return dtw_distances

        # Helper: compute inter-group DTW
        def compute_intergroup_dtw(group1, group2):
            dtw_distances = []
            for t1 in group1:
                for t2 in group2:
                    series1 = extract_series(t1)
                    series2 = extract_series(t2)
                    dist, _ = fastdtw(series1, series2)
                    dtw_distances.append(dist)
            return dtw_distances

        # Compute all distributions
        dtw_human = compute_intragroup_dtw(human_trajectories)
        dtw_rl = compute_intragroup_dtw(rl_trajectories)
        dtw_heuristic = compute_intragroup_dtw(heuristic_trajectories)

        dtw_human_rl = compute_intergroup_dtw(human_trajectories, rl_trajectories)
        dtw_human_heuristic = compute_intergroup_dtw(human_trajectories, heuristic_trajectories)

        # Mann-Whitney U test: is human-heuristic < human-rl
        u_stat, p_val = mannwhitneyu(dtw_human_heuristic, dtw_human_rl, alternative='less')
        print(f"Mann-Whitney U Test: U={u_stat}, p={p_val}")

        # Pearson correlation between the two inter-group distributions
        # Pad shorter array for correlation
        min_len = min(len(dtw_human_heuristic), len(dtw_human_rl))
        pearson_corr, pearson_p = pearsonr(
            np.array(dtw_human_heuristic[:min_len]),
            np.array(dtw_human_rl[:min_len])
        )
        print(f"Pearson Correlation: r={pearson_corr:.3f}, p={pearson_p:.3f}")

        # Boxplot of all distributions
        labels = [
            "Human-Human",
            "RL-RL",
            "Heuristic-Heuristic",
            "Human-RL",
            "Human-Heuristic"
        ]
        data = [
            dtw_human,
            dtw_rl,
            dtw_heuristic,
            dtw_human_rl,
            dtw_human_heuristic
        ]

        plt.figure(figsize=(10, 6))
        plt.boxplot(data, labels=labels)
        plt.ylabel("DTW Distance")
        plt.title("Progress Rate Similarity (DTW Distances)")
        plt.xticks(rotation=20)
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "progress_rate_dtw_boxplot.png"), dpi=300)
        plt.close()

        # Return results for further analysis if desired
        return {
            "dtw_human": dtw_human,
            "dtw_rl": dtw_rl,
            "dtw_heuristic": dtw_heuristic,
            "dtw_human_rl": dtw_human_rl,
            "dtw_human_heuristic": dtw_human_heuristic,
            "mannwhitney": (u_stat, p_val),
            "pearson": (pearson_corr, pearson_p)
        }

    def analyze_action_distributions(self, human_trajectories, rl_trajectories, heuristic_trajectories,
                                     save_dir="exp3_similarity_analysis"):
        """
        Input:
            human_trajectories_for_training: list of Trajectory objects
            rl_trajectories: list of Trajectory objects
            heuristic_trajectories: list of Trajectory objects

        Process:
            1. Generate histograms of action distributions (Frequency of each discrete action 0–15) for
               human, RL, and individual heuristic agent configurations.
            2. Plot histograms in a comprehensive multi-panel figure.
            3. Perform pairwise chi-squared comparisons between all distributions.
            4. Identify which heuristic agents are most similar to humans.

        Output:
            - Prints pairwise chi-squared statistics
            - Shows and saves comprehensive histogram figures
            - Saves detailed comparison results to CSV
        """

        os.makedirs(save_dir, exist_ok=True)
        n_actions = 16  # Actions are 0–15

        # Helper function to aggregate all actions for a group
        def get_action_hist(trajectories):
            all_actions = []
            for traj in trajectories:
                all_actions.extend(traj.actions)
            if not all_actions:
                return np.zeros(n_actions)
            hist, _ = np.histogram(all_actions, bins=np.arange(n_actions + 1) - 0.5)
            return hist

        # Group heuristic trajectories by agent type (name)
        heuristic_agents = {}
        for traj in heuristic_trajectories:
            agent_name = traj.name
            if agent_name not in heuristic_agents:
                heuristic_agents[agent_name] = []
            heuristic_agents[agent_name].append(traj)

        print(f"Found {len(heuristic_agents)} unique heuristic agent types:")
        for agent_name, trajs in heuristic_agents.items():
            print(f"  - {agent_name}: {len(trajs)} trajectories")

        # Compute histograms for all groups
        human_hist = get_action_hist(human_trajectories)
        rl_hist = get_action_hist(rl_trajectories)

        # Compute histograms for each individual heuristic agent type
        heuristic_hists = {}
        for agent_name, agent_trajs in heuristic_agents.items():
            heuristic_hists[agent_name] = get_action_hist(agent_trajs)

        # Compute overall heuristic histogram for comparison
        overall_heuristic_hist = get_action_hist(heuristic_trajectories)

        # Create comprehensive visualization
        n_heuristic_agents = len(heuristic_agents)
        n_cols = 3
        n_rows = max(2, (n_heuristic_agents + 2) // n_cols + 1)  # +2 for human and RL, +1 for overall heuristic

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        plot_idx = 0

        # Plot human distribution
        axes[plot_idx].bar(range(n_actions), human_hist, color='blue', alpha=0.7, edgecolor='black')
        axes[plot_idx].set_title('Human Action Distribution')
        axes[plot_idx].set_ylabel('Frequency')
        axes[plot_idx].set_xticks(range(n_actions))
        axes[plot_idx].grid(axis='y', linestyle='--', alpha=0.7)
        plot_idx += 1

        # Plot RL distribution
        axes[plot_idx].bar(range(n_actions), rl_hist, color='green', alpha=0.7, edgecolor='black')
        axes[plot_idx].set_title('RL Agent Action Distribution')
        axes[plot_idx].set_ylabel('Frequency')
        axes[plot_idx].set_xticks(range(n_actions))
        axes[plot_idx].grid(axis='y', linestyle='--', alpha=0.7)
        plot_idx += 1

        # Plot overall heuristic distribution
        axes[plot_idx].bar(range(n_actions), overall_heuristic_hist, color='red', alpha=0.7, edgecolor='black')
        axes[plot_idx].set_title('All Heuristic Agents Combined')
        axes[plot_idx].set_ylabel('Frequency')
        axes[plot_idx].set_xticks(range(n_actions))
        axes[plot_idx].grid(axis='y', linestyle='--', alpha=0.7)
        plot_idx += 1

        # Plot individual heuristic agent distributions
        import matplotlib.cm as cm
        colors = cm.Set3(np.linspace(0, 1, len(heuristic_agents)))

        for i, (agent_name, agent_hist) in enumerate(heuristic_hists.items()):
            if plot_idx < len(axes):
                axes[plot_idx].bar(range(n_actions), agent_hist, color=colors[i], alpha=0.7, edgecolor='black')
                # Truncate long agent names for display
                display_name = agent_name[:20] + '...' if len(agent_name) > 20 else agent_name
                axes[plot_idx].set_title(f'Heuristic: {display_name}')
                axes[plot_idx].set_ylabel('Frequency')
                axes[plot_idx].set_xticks(range(n_actions))
                axes[plot_idx].grid(axis='y', linestyle='--', alpha=0.7)
                plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)

        # Add x-label to bottom row
        for i in range(max(0, len(axes) - n_cols), len(axes)):
            if axes[i].get_visible():
                axes[i].set_xlabel('Action Index')

        plt.tight_layout()
        save_path = os.path.join(save_dir, "action_distribution_histograms_individual.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Perform pairwise chi-squared comparisons
        from scipy.stats import chisquare

        comparison_results = []

        print("\n" + "=" * 80)
        print("CHI-SQUARED ACTION DISTRIBUTION COMPARISONS")
        print("=" * 80)

        # Helper function for chi-squared test with small count handling
        def safe_chisquare(observed, expected):
            # Add small constant to avoid zero expected frequencies
            eps = 1e-6
            expected_safe = expected + eps
            observed_safe = observed + eps

            # Ensure both arrays sum to the same total for fair comparison
            expected_norm = expected_safe * np.sum(observed_safe) / np.sum(expected_safe)

            try:
                stat, p = chisquare(f_obs=observed_safe, f_exp=expected_norm)
                return stat, p
            except:
                return np.nan, np.nan

        # Compare human vs RL
        chi_stat, p_val = safe_chisquare(human_hist, rl_hist)
        comparison_results.append({
            'Comparison': 'Human vs RL',
            'Group1': 'Human',
            'Group2': 'RL',
            'Chi2_Statistic': chi_stat,
            'P_Value': p_val,
            'Significant': p_val < 0.05 if not np.isnan(p_val) else False
        })
        print(f"Human vs RL: χ² = {chi_stat:.3f}, p = {p_val:.3e}")

        # Compare human vs overall heuristic
        chi_stat, p_val = safe_chisquare(human_hist, overall_heuristic_hist)
        comparison_results.append({
            'Comparison': 'Human vs All Heuristic',
            'Group1': 'Human',
            'Group2': 'All_Heuristic',
            'Chi2_Statistic': chi_stat,
            'P_Value': p_val,
            'Significant': p_val < 0.05 if not np.isnan(p_val) else False
        })
        print(f"Human vs All Heuristic: χ² = {chi_stat:.3f}, p = {p_val:.3e}")

        # Compare human vs each individual heuristic agent
        human_heuristic_comparisons = []

        print(f"\nHuman vs Individual Heuristic Agents:")
        print("-" * 50)

        for agent_name, agent_hist in heuristic_hists.items():
            chi_stat, p_val = safe_chisquare(human_hist, agent_hist)

            comparison_results.append({
                'Comparison': f'Human vs {agent_name}',
                'Group1': 'Human',
                'Group2': agent_name,
                'Chi2_Statistic': chi_stat,
                'P_Value': p_val,
                'Significant': p_val < 0.05 if not np.isnan(p_val) else False
            })

            human_heuristic_comparisons.append((agent_name, chi_stat, p_val))
            print(f"Human vs {agent_name}: χ² = {chi_stat:.3f}, p = {p_val:.3e}")

        # Compare RL vs each individual heuristic agent
        print(f"\nRL vs Individual Heuristic Agents:")
        print("-" * 50)

        for agent_name, agent_hist in heuristic_hists.items():
            chi_stat, p_val = safe_chisquare(rl_hist, agent_hist)

            comparison_results.append({
                'Comparison': f'RL vs {agent_name}',
                'Group1': 'RL',
                'Group2': agent_name,
                'Chi2_Statistic': chi_stat,
                'P_Value': p_val,
                'Significant': p_val < 0.05 if not np.isnan(p_val) else False
            })

            print(f"RL vs {agent_name}: χ² = {chi_stat:.3f}, p = {p_val:.3e}")

        # Identify most similar heuristic agents to humans
        # Sort by chi-squared statistic (lower = more similar)
        human_heuristic_comparisons.sort(key=lambda x: x[1] if not np.isnan(x[1]) else float('inf'))

        print(f"\n" + "=" * 60)
        print("TOP 5 MOST SIMILAR HEURISTIC AGENTS TO HUMANS")
        print("=" * 60)
        print("(Ranked by Chi-squared statistic - lower is more similar)")

        for i, (agent_name, chi_stat, p_val) in enumerate(human_heuristic_comparisons[:5]):
            print(f"{i + 1}. {agent_name}")
            print(f"   χ² = {chi_stat:.3f}, p = {p_val:.3e}")
            if not np.isnan(p_val):
                significance = "significant" if p_val < 0.05 else "not significant"
                print(f"   Difference is {significance} (α = 0.05)")
            print()

        # Create a focused comparison plot for top similar agents
        top_5_agents = [comp[0] for comp in human_heuristic_comparisons[:5]]

        if len(top_5_agents) > 0:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()

            # Plot human reference
            axes[0].bar(range(n_actions), human_hist, color='blue', alpha=0.7, edgecolor='black')
            axes[0].set_title('Human (Reference)', fontweight='bold')
            axes[0].set_ylabel('Frequency')
            axes[0].set_xticks(range(n_actions))
            axes[0].grid(axis='y', linestyle='--', alpha=0.7)

            # Plot top 5 most similar heuristic agents
            for i, agent_name in enumerate(top_5_agents):
                if i + 1 < len(axes):
                    agent_hist = heuristic_hists[agent_name]
                    chi_stat = human_heuristic_comparisons[i][1]

                    axes[i + 1].bar(range(n_actions), agent_hist, color=colors[i % len(colors)],
                                    alpha=0.7, edgecolor='black')

                    display_name = agent_name[:15] + '...' if len(agent_name) > 15 else agent_name
                    axes[i + 1].set_title(f'{display_name}\n(χ² = {chi_stat:.2f})')
                    axes[i + 1].set_ylabel('Frequency')
                    axes[i + 1].set_xticks(range(n_actions))
                    axes[i + 1].grid(axis='y', linestyle='--', alpha=0.7)

            # Hide unused subplot
            if len(top_5_agents) < 5:
                axes[5].set_visible(False)

            # Add x-labels to bottom row
            for i in range(3, 6):
                if i < len(axes) and axes[i].get_visible():
                    axes[i].set_xlabel('Action Index')

            plt.suptitle('Top 5 Most Similar Heuristic Agents to Human Action Distributions',
                         fontsize=16, fontweight='bold')
            plt.tight_layout()

            top5_save_path = os.path.join(save_dir, "action_distribution_top5_similar.png")
            plt.savefig(top5_save_path, dpi=300, bbox_inches='tight')
            plt.show()

        # Save detailed results to CSV
        results_df = pd.DataFrame(comparison_results)
        results_df = results_df.sort_values('Chi2_Statistic', ascending=True)

        csv_save_path = os.path.join(save_dir, "action_distribution_comparisons.csv")
        results_df.to_csv(csv_save_path, index=False)

        # Create summary statistics
        summary_stats = {
            'total_heuristic_agents': len(heuristic_agents),
            'human_total_actions': int(np.sum(human_hist)),
            'rl_total_actions': int(np.sum(rl_hist)),
            'heuristic_total_actions': {name: int(np.sum(hist)) for name, hist in heuristic_hists.items()},
            'most_similar_to_human': top_5_agents[0] if top_5_agents else None,
            'most_similar_chi2': float(human_heuristic_comparisons[0][1]) if human_heuristic_comparisons else None
        }

        summary_path = os.path.join(save_dir, "action_distribution_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)

        print(f"\nResults saved:")
        print(f"  Main plot: {save_path}")
        if len(top_5_agents) > 0:
            print(f"  Top 5 plot: {top5_save_path}")
        print(f"  Detailed comparisons: {csv_save_path}")
        print(f"  Summary statistics: {summary_path}")

        return {
            'comparison_results': comparison_results,
            'most_similar_agents': human_heuristic_comparisons[:5],
            'summary_statistics': summary_stats
        }
        

    def analyze_metric_similarity(self):
        """
        Analyze trajectories from three groups: human, RL, and heuristic.

        Computes metrics:
            - Path smoothness (mean angular change between actions)
            - Action switch rate (action changes / steps)
            - Flying toward nearest rate (placeholder)
            - Average target distance (placeholder)
        
        Performs pairwise statistical comparisons and generates boxplots.
        """

        # 16 discrete action direction vectors
        direction_vectors = np.array([
            (0, 1), (0.383, 0.924), (0.707, 0.707), (0.924, 0.383),
            (1, 0), (0.924, -0.383), (0.707, -0.707), (0.383, -0.924),
            (0, -1), (-0.383, -0.924), (-0.707, -0.707), (-0.924, -0.383),
            (-1, 0), (-0.924, 0.383), (-0.707, 0.707), (-0.383, 0.924)
        ])

        # --- Step 1: Compute metrics per trajectory ---
        metrics: Dict[str, Dict[str, List[float]]] = {
            "human": {
                "path_smoothness": [], "action_switches_per_step": [],
                #"flying_toward_nearest_rate": [], "average_target_distance": []
            },
            "rl": {
                "path_smoothness": [], "action_switches_per_step": [],
                #"flying_toward_nearest_rate": [], "average_target_distance": []
            },
            "heuristic": {
                "path_smoothness": [], "action_switches_per_step": [],
                #"flying_toward_nearest_rate": [], "average_target_distance": []
            },
        }

        all_trajectories = self.human_trajectories + self.rl_trajectories + self.heuristic_trajectories
        for traj in all_trajectories:
            actions = np.array(traj.actions)
            if len(actions) < 2:
                continue

            # Path smoothness: Mean angular change between successive actions
            angles = []
            for i in range(1, len(actions)):
                v1 = direction_vectors[actions[i-1]]
                v2 = direction_vectors[actions[i]]
                dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle = np.arccos(dot)
                angles.append(angle)
            smoothness = np.mean(angles) if angles else 0.0

            # Action switch rate
            switches = np.sum(actions[1:] != actions[:-1])
            switch_rate = switches / (len(actions) - 1)

            # --- Placeholders for future metrics ---
            flying_toward_nearest_rate = None
            average_target_distance = None

            # Save metrics
            metrics[traj.category]["path_smoothness"].append(smoothness)
            metrics[traj.category]["action_switches_per_step"].append(switch_rate)
            #metrics[traj.category]["flying_toward_nearest_rate"].append(flying_toward_nearest_rate)
            #metrics[traj.category]["average_target_distance"].append(average_target_distance)

        # --- Step 2: Perform overall group comparison test ---
        print("\n--- (Analyze metric similarity) Overall Group Comparison (Kruskal-Wallis H Test) ---")
        for metric_name in metrics["human"].keys():
            human_vals = [v for v in metrics["human"][metric_name] if v is not None]
            rl_vals = [v for v in metrics["rl"][metric_name] if v is not None]
            heuristic_vals = [v for v in metrics["heuristic"][metric_name] if v is not None]

            if len(human_vals) > 0 and len(rl_vals) > 0 and len(heuristic_vals) > 0:
                h_stat, p_val = perform_kruskal_wallis_test(
                    metric_name, human_vals, rl_vals, heuristic_vals
                )
            print(f'Kruskal results: h = {h_stat}, p = {p_val}')

        # --- Step 3: Perform pairwise statistical tests ---
        def pairwise_tests(metric_name):
            human_vals = [v for v in metrics["human"][metric_name] if v is not None]
            rl_vals = [v for v in metrics["rl"][metric_name] if v is not None]
            heuristic_vals = [v for v in metrics["heuristic"][metric_name] if v is not None]

            #print(human_vals)
            #print(rl_vals)
            #print(heuristic_vals)

            pairs = [
                ("human", "rl", human_vals, rl_vals),
                ("human", "heuristic", human_vals, heuristic_vals),
                ("rl", "heuristic", rl_vals, heuristic_vals),
            ]
            results = []
            for name1, name2, data1, data2 in pairs:
                if len(data1) > 0 and len(data2) > 0:
                    stat, p = mannwhitneyu(data1, data2, alternative="two-sided")
                    #print(f'For {metric_name} {name1} vs {name2}: U = {stat}, p = {p}')
                    results.append(f"{metric_name} {name1} vs {name2}: U={stat:.2f}, p={p}")
            return results

        print("\n--- Metric similarity - Statistical Tests ---")
        for metric_name in metrics["human"].keys():
            for line in pairwise_tests(metric_name):
                print(line)

        # --- Step 3: Generate boxplots ---
        for metric_name in metrics["human"].keys():
            human_vals = [v for v in metrics["human"][metric_name] if v is not None]
            rl_vals = [v for v in metrics["rl"][metric_name] if v is not None]
            heuristic_vals = [v for v in metrics["heuristic"][metric_name] if v is not None]

            # Skip plotting if metric is only placeholders
            if len(human_vals) == 0 and len(rl_vals) == 0 and len(heuristic_vals) == 0:
                continue

            plt.figure(figsize=(8, 6))
            plt.grid(True, axis='y', alpha=0.6)

            #plt.boxplot([human_vals, rl_vals, heuristic_vals], labels=["Human", "RL", "Heuristic"])

            #bp = plt.violinplot([human_vals, rl_vals, heuristic_vals], tick_labels=["Human", "RL", "Heuristic"],patch_artist=True)
            parts = plt.violinplot([human_vals, rl_vals, heuristic_vals], positions=[1, 2, 3], showmeans=False,showmedians=True)

            # Color the boxes
            colors = ['lightblue', 'lightgreen', 'lightcoral']

            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(1.0)  # Increased alpha to 0.9
                pc.set_edgecolor('black')  # Added black outline around violins
                pc.set_linewidth(1)  # Set outline width
                pc.set_zorder(3)  # Set higher zorder to appear above gridlines

            # Make all box and whisker elements black instead of blue
            parts['cmedians'].set_color('black')
            parts['cmedians'].set_linewidth(2)
            parts['cmins'].set_color('black')
            parts['cmaxes'].set_color('black')
            parts['cmedians'].set_zorder(4)  # Keep medians above violins
            parts['cbars'].set_color('black')

            parts['cbars'].set_visible(False)
            parts['cmins'].set_visible(False)
            parts['cmaxes'].set_visible(False)

            plt.title(f"{metric_name.replace('_', ' ').capitalize()}", fontsize=25)
            plt.ylabel(metric_name.replace('_', ' ').capitalize(), fontsize=22)
            plt.xlabel("Agent Type", fontsize=22)
            plt.tick_params(axis='y', which='major', labelsize=20)

            plt.show()


    def analyze_progress_rate_correlations(self,
                                           human_trajectories: List[Trajectory],
                                           rl_trajectories: List[Trajectory],
                                           heuristic_trajectories: List[Trajectory],
                                           save_dir: str = "exp3_similarity_analysis",
                                           n_timepoints: int = 10
                                           ):
        """
        Analyze correlations between target and threat identification progress rates
        across human, RL, and individual heuristic agent groups.

        Args:
            human_trajectories: List of human trajectory objects
            rl_trajectories: List of RL trajectory objects
            heuristic_trajectories: List of heuristic trajectory objects
            save_dir: Directory to save outputs
            n_timepoints: Number of evenly-spaced time points to analyze (default: 10)

        Returns:
            dict: Correlation results and time series data
        """
        os.makedirs(save_dir, exist_ok=True)

        # Generate evenly-spaced time points from 10% to 100%
        time_points = np.linspace(0.1, 0.9, n_timepoints)

        def extract_identification_series(trajs, time_points, metric_type='targets'):
            """
            Extract target or threat identification counts at specified time points

            Args:
                trajs: List of trajectory objects
                time_points: Array of time points (0.1 to 1.0)
                metric_type: 'targets' or 'threats'

            Returns:
                dict: {time_point: [values_across_trajectories]}
            """
            series_by_timepoint = {tp: [] for tp in time_points}

            for traj in trajs:
                if metric_type == 'targets':
                    identification_data = traj.target_ids
                elif metric_type == 'threats':
                    identification_data = traj.threat_ids
                else:
                    raise ValueError("metric_type must be 'targets' or 'threats'")

                if not identification_data:
                    continue

                traj_length = len(identification_data)
                if traj_length < 2:  # Skip very short trajectories
                    continue

                for time_point in time_points:
                    # Calculate index for this time point
                    if time_point >= 1.0:
                        idx = traj_length - 1  # Use final index for 100%
                    else:
                        idx = int(time_point * traj_length)
                        idx = min(idx, traj_length - 1)  # Ensure we don't exceed bounds

                    # Extract identification count at this time point
                    identification_count = identification_data[idx]
                    series_by_timepoint[time_point].append(identification_count)

            return series_by_timepoint

        def compute_average_series(series_by_timepoint, time_points):
            """Compute average identification counts at each time point"""
            avg_series = []
            valid_timepoints = []

            for tp in time_points:
                values = series_by_timepoint[tp]
                if len(values) > 0:
                    avg_series.append(np.mean(values))
                    valid_timepoints.append(tp)
                else:
                    print(f"Warning: No data at time point {tp:.1%}")

            return np.array(avg_series), np.array(valid_timepoints)

        # Group heuristic trajectories by agent type (name)
        heuristic_agents = {}
        for traj in heuristic_trajectories:
            agent_name = traj.name
            if agent_name not in heuristic_agents:
                heuristic_agents[agent_name] = []
            heuristic_agents[agent_name].append(traj)

        print(f"Found {len(heuristic_agents)} unique heuristic agent types:")
        for agent_name, trajs in heuristic_agents.items():
            print(f"  - {agent_name}: {len(trajs)} trajectories")

        print("Extracting target identification time series...")
        # Extract target identification series for each group
        human_targets_series = extract_identification_series(human_trajectories, time_points, 'targets')
        rl_targets_series = extract_identification_series(rl_trajectories, time_points, 'targets')

        # Extract series for each individual heuristic agent type
        heuristic_targets_series = {}
        for agent_name, trajs in heuristic_agents.items():
            heuristic_targets_series[agent_name] = extract_identification_series(trajs, time_points, 'targets')

        print("Extracting threat identification time series...")
        # Extract threat identification series for each group
        human_threats_series = extract_identification_series(human_trajectories, time_points, 'threats')
        rl_threats_series = extract_identification_series(rl_trajectories, time_points, 'threats')

        # Extract series for each individual heuristic agent type
        heuristic_threats_series = {}
        for agent_name, trajs in heuristic_agents.items():
            heuristic_threats_series[agent_name] = extract_identification_series(trajs, time_points, 'threats')

        # Compute average series for each group
        human_targets_avg, valid_tp_targets = compute_average_series(human_targets_series, time_points)
        rl_targets_avg, _ = compute_average_series(rl_targets_series, time_points)

        # Compute average series for each heuristic agent type
        heuristic_targets_avg = {}
        for agent_name, series in heuristic_targets_series.items():
            heuristic_targets_avg[agent_name], _ = compute_average_series(series, time_points)

        human_threats_avg, valid_tp_threats = compute_average_series(human_threats_series, time_points)
        rl_threats_avg, _ = compute_average_series(rl_threats_series, time_points)

        # Compute average series for each heuristic agent type
        heuristic_threats_avg = {}
        for agent_name, series in heuristic_threats_series.items():
            heuristic_threats_avg[agent_name], _ = compute_average_series(series, time_points)

        # Ensure all series have the same length for correlation calculation
        min_len_targets = min(len(human_targets_avg), len(rl_targets_avg))
        min_len_threats = min(len(human_threats_avg), len(rl_threats_avg))

        if min_len_targets == 0 or min_len_threats == 0:
            print("Error: Insufficient data for correlation analysis")
            return None

        # Trim series to same length
        human_targets_avg = human_targets_avg[:min_len_targets]
        rl_targets_avg = rl_targets_avg[:min_len_targets]
        valid_tp_targets = valid_tp_targets[:min_len_targets]

        human_threats_avg = human_threats_avg[:min_len_threats]
        rl_threats_avg = rl_threats_avg[:min_len_threats]
        valid_tp_threats = valid_tp_threats[:min_len_threats]

        # Calculate correlations
        from scipy.stats import pearsonr

        print("Computing correlations...")
        correlation_results = {}

        # Target identification correlations
        if len(human_targets_avg) > 1:
            corr_human_rl_targets, p_human_rl_targets = pearsonr(human_targets_avg, rl_targets_avg)

            correlation_results['human_vs_rl_targets'] = {
                'correlation': corr_human_rl_targets,
                'p_value': p_human_rl_targets
            }

            # Correlations with individual heuristic agents
            for agent_name, agent_targets_avg in heuristic_targets_avg.items():
                if len(agent_targets_avg) >= min_len_targets:
                    agent_targets_trimmed = agent_targets_avg[:min_len_targets]
                    corr_human_heuristic, p_human_heuristic = pearsonr(human_targets_avg, agent_targets_trimmed)
                    correlation_results[f'human_vs_{agent_name}_targets'] = {
                        'correlation': corr_human_heuristic,
                        'p_value': p_human_heuristic
                    }

        # Threat identification correlations
        if len(human_threats_avg) > 1:
            corr_human_rl_threats, p_human_rl_threats = pearsonr(human_threats_avg, rl_threats_avg)

            correlation_results['human_vs_rl_threats'] = {
                'correlation': corr_human_rl_threats,
                'p_value': p_human_rl_threats
            }

            # Correlations with individual heuristic agents
            for agent_name, agent_threats_avg in heuristic_threats_avg.items():
                if len(agent_threats_avg) >= min_len_threats:
                    agent_threats_trimmed = agent_threats_avg[:min_len_threats]
                    corr_human_heuristic, p_human_heuristic = pearsonr(human_threats_avg, agent_threats_trimmed)
                    correlation_results[f'human_vs_{agent_name}_threats'] = {
                        'correlation': corr_human_heuristic,
                        'p_value': p_human_heuristic
                    }

        # Create plots with individual heuristic agent lines
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

        # Define colors for heuristic agents
        import matplotlib.cm as cm
        colors = cm.Set3(np.linspace(0, 1, len(heuristic_agents)))
        heuristic_colors = dict(zip(heuristic_agents.keys(), colors))

        # Plot 1: Target identification progress
        ax1.plot(valid_tp_targets * 100, human_targets_avg,
                 marker='o', linewidth=3, markersize=10,
                 label='Human', color='blue', alpha=0.9, zorder=10)
        ax1.plot(valid_tp_targets * 100, rl_targets_avg,
                 marker='s', linewidth=3, markersize=10,
                 label='RL Agent', color='green', alpha=0.9, zorder=10)

        # Plot individual heuristic agents
        for agent_name, agent_targets_avg in heuristic_targets_avg.items():
            if len(agent_targets_avg) >= min_len_targets:
                agent_targets_trimmed = agent_targets_avg[:min_len_targets]
                ax1.plot(valid_tp_targets * 100, agent_targets_trimmed,
                         marker='^', linewidth=2, markersize=6,
                         label=f'Heuristic: {agent_name}',
                         color=heuristic_colors[agent_name], alpha=0.7)

        ax1.set_xlabel('Progress Through Trajectory (%)', fontsize=12)
        ax1.set_ylabel('Average Targets Identified', fontsize=12)
        ax1.set_title('Target Identification Progress Over Time', fontsize=14)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Threat identification progress
        ax2.plot(valid_tp_threats * 100, human_threats_avg,
                 marker='o', linewidth=3, markersize=10,
                 label='Human', color='blue', alpha=0.9, zorder=10)
        ax2.plot(valid_tp_threats * 100, rl_threats_avg,
                 marker='s', linewidth=3, markersize=10,
                 label='RL Agent', color='green', alpha=0.9, zorder=10)

        # Plot individual heuristic agents
        for agent_name, agent_threats_avg in heuristic_threats_avg.items():
            if len(agent_threats_avg) >= min_len_threats:
                agent_threats_trimmed = agent_threats_avg[:min_len_threats]
                ax2.plot(valid_tp_threats * 100, agent_threats_trimmed,
                         marker='^', linewidth=2, markersize=6,
                         label=f'Heuristic: {agent_name}',
                         color=heuristic_colors[agent_name], alpha=0.7)

        ax2.set_xlabel('Progress Through Trajectory (%)', fontsize=12)
        ax2.set_ylabel('Average Threats Identified', fontsize=12)
        ax2.set_title('Threat Identification Progress Over Time', fontsize=14)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(save_dir, "progress_rate_correlations_individual.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Create a separate plot showing only the best correlating heuristic agents (top 6)
        # Find top 6 heuristic agents with highest correlation to humans for targets
        target_correlations_heuristic = {}
        for key, value in correlation_results.items():
            if key.startswith('human_vs_') and key.endswith('_targets') and 'rl' not in key:
                agent_name = key.replace('human_vs_', '').replace('_targets', '')
                target_correlations_heuristic[agent_name] = abs(value['correlation'])

        top_6_agents = sorted(target_correlations_heuristic.items(), key=lambda x: x[1], reverse=True)[:6]

        if len(top_6_agents) > 0:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # Plot top 6 for targets
            ax1.plot(valid_tp_targets * 100, human_targets_avg,
                     marker='o', linewidth=3, markersize=10,
                     label='Human', color='blue', alpha=0.9, zorder=10)
            ax1.plot(valid_tp_targets * 100, rl_targets_avg,
                     marker='s', linewidth=3, markersize=10,
                     label='RL Agent', color='green', alpha=0.9, zorder=10)

            for i, (agent_name, corr_value) in enumerate(top_6_agents):
                if agent_name in heuristic_targets_avg:
                    agent_targets_avg = heuristic_targets_avg[agent_name]
                    if len(agent_targets_avg) >= min_len_targets:
                        agent_targets_trimmed = agent_targets_avg[:min_len_targets]
                        ax1.plot(valid_tp_targets * 100, agent_targets_trimmed,
                                 marker='^', linewidth=2, markersize=6,
                                 label=f'{agent_name} (r={corr_value:.3f})',
                                 color=heuristic_colors[agent_name], alpha=0.8)

            ax1.set_xlabel('Progress Through Trajectory (%)', fontsize=12)
            ax1.set_ylabel('Average Targets Identified', fontsize=12)
            ax1.set_title('Target Identification: Top 6 Most Similar Heuristic Agents', fontsize=14)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)

            # Plot top 6 for threats (using same agents as targets for consistency)
            ax2.plot(valid_tp_threats * 100, human_threats_avg,
                     marker='o', linewidth=3, markersize=10,
                     label='Human', color='blue', alpha=0.9, zorder=10)
            ax2.plot(valid_tp_threats * 100, rl_threats_avg,
                     marker='s', linewidth=3, markersize=10,
                     label='RL Agent', color='green', alpha=0.9, zorder=10)

            for i, (agent_name, _) in enumerate(top_6_agents):
                if agent_name in heuristic_threats_avg:
                    agent_threats_avg = heuristic_threats_avg[agent_name]
                    if len(agent_threats_avg) >= min_len_threats:
                        agent_threats_trimmed = agent_threats_avg[:min_len_threats]
                        # Get threat correlation for this agent
                        threat_corr_key = f'human_vs_{agent_name}_threats'
                        threat_corr = correlation_results.get(threat_corr_key, {}).get('correlation', 0)
                        ax2.plot(valid_tp_threats * 100, agent_threats_trimmed,
                                 marker='^', linewidth=2, markersize=6,
                                 label=f'{agent_name} (r={threat_corr:.3f})',
                                 color=heuristic_colors[agent_name], alpha=0.8)

            ax2.set_xlabel('Progress Through Trajectory (%)', fontsize=12)
            ax2.set_ylabel('Average Threats Identified', fontsize=12)
            ax2.set_title('Threat Identification: Top 6 Most Similar Heuristic Agents', fontsize=14)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            top_6_plot_path = os.path.join(save_dir, "progress_rate_correlations_top6.png")
            plt.savefig(top_6_plot_path, dpi=300, bbox_inches='tight')
            plt.show()

        # Save numerical results
        results_data = {
            'time_points_percent': (valid_tp_targets * 100).tolist(),
            'human_targets_avg': human_targets_avg.tolist(),
            'rl_targets_avg': rl_targets_avg.tolist(),
            'human_threats_avg': human_threats_avg.tolist(),
            'rl_threats_avg': rl_threats_avg.tolist(),
            'correlations': correlation_results
        }

        # Add individual heuristic agent data
        for agent_name, agent_targets_avg in heuristic_targets_avg.items():
            if len(agent_targets_avg) >= min_len_targets:
                results_data[f'{agent_name}_targets_avg'] = agent_targets_avg[:min_len_targets].tolist()

        for agent_name, agent_threats_avg in heuristic_threats_avg.items():
            if len(agent_threats_avg) >= min_len_threats:
                results_data[f'{agent_name}_threats_avg'] = agent_threats_avg[:min_len_threats].tolist()

        results_path = os.path.join(save_dir, "progress_rate_correlation_results_individual.json")
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        # Create comprehensive summary DataFrame
        summary_data = []

        # Add Human vs RL comparisons
        if 'human_vs_rl_targets' in correlation_results:
            summary_data.append({
                'Comparison': 'Human vs RL (Targets)',
                'Agent_Type': 'RL',
                'Correlation': correlation_results['human_vs_rl_targets']['correlation'],
                'P-value': correlation_results['human_vs_rl_targets']['p_value']
            })

        if 'human_vs_rl_threats' in correlation_results:
            summary_data.append({
                'Comparison': 'Human vs RL (Threats)',
                'Agent_Type': 'RL',
                'Correlation': correlation_results['human_vs_rl_threats']['correlation'],
                'P-value': correlation_results['human_vs_rl_threats']['p_value']
            })

        # Add individual heuristic agent comparisons
        for key, value in correlation_results.items():
            if key.startswith('human_vs_') and ('_targets' in key or '_threats' in key) and 'rl' not in key:
                if '_targets' in key:
                    agent_name = key.replace('human_vs_', '').replace('_targets', '')
                    metric = 'Targets'
                else:
                    agent_name = key.replace('human_vs_', '').replace('_threats', '')
                    metric = 'Threats'

                summary_data.append({
                    'Comparison': f'Human vs {agent_name} ({metric})',
                    'Agent_Type': 'Heuristic',
                    'Agent_Name': agent_name,
                    'Correlation': value['correlation'],
                    'P-value': value['p_value']
                })

        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(save_dir, "progress_rate_correlation_summary_individual.csv")
        summary_df.to_csv(summary_path, index=False)

        # Print detailed results summary
        print("\n" + "=" * 80)
        print("INDIVIDUAL HEURISTIC AGENT CORRELATION ANALYSIS SUMMARY")
        print("=" * 80)

        print(
            f"\nAnalyzed {len(valid_tp_targets)} time points from {valid_tp_targets[0]:.1%} to {valid_tp_targets[-1]:.1%}")
        print(f"Analyzed {len(heuristic_agents)} individual heuristic agent types")

        print("\nTarget Identification Correlations:")
        print("-" * 50)
        if 'human_vs_rl_targets' in correlation_results:
            print(f"Human vs RL:       r = {correlation_results['human_vs_rl_targets']['correlation']:.4f}, "
                  f"p = {correlation_results['human_vs_rl_targets']['p_value']:.4f}")

        # Print top 5 heuristic agents for targets
        print("\nTop 5 Most Similar Heuristic Agents (Targets):")
        for i, (agent_name, corr_value) in enumerate(top_6_agents[:5]):
            key = f'human_vs_{agent_name}_targets'
            if key in correlation_results:
                p_val = correlation_results[key]['p_value']
                print(f"  {i + 1}. {agent_name}: r = {correlation_results[key]['correlation']:.4f}, p = {p_val:.4f}")

        print("\nThreat Identification Correlations:")
        print("-" * 50)
        if 'human_vs_rl_threats' in correlation_results:
            print(f"Human vs RL:       r = {correlation_results['human_vs_rl_threats']['correlation']:.4f}, "
                  f"p = {correlation_results['human_vs_rl_threats']['p_value']:.4f}")

        # Find and print top 5 heuristic agents for threats
        threat_correlations_heuristic = {}
        for key, value in correlation_results.items():
            if key.startswith('human_vs_') and key.endswith('_threats') and 'rl' not in key:
                agent_name = key.replace('human_vs_', '').replace('_threats', '')
                threat_correlations_heuristic[agent_name] = abs(value['correlation'])

        top_5_threats = sorted(threat_correlations_heuristic.items(), key=lambda x: x[1], reverse=True)[:5]

        print("\nTop 5 Most Similar Heuristic Agents (Threats):")
        for i, (agent_name, corr_value) in enumerate(top_5_threats):
            key = f'human_vs_{agent_name}_threats'
            if key in correlation_results:
                p_val = correlation_results[key]['p_value']
                print(f"  {i + 1}. {agent_name}: r = {correlation_results[key]['correlation']:.4f}, p = {p_val:.4f}")

        print(f"\nFiles saved:")
        print(f"  Individual agents plot: {plot_path}")
        if len(top_6_agents) > 0:
            print(f"  Top 6 agents plot: {top_6_plot_path}")
        print(f"  Detailed results: {results_path}")
        print(f"  Summary table: {summary_path}")

        return {
            'correlations': correlation_results,
            'time_series': {
                'time_points': valid_tp_targets,
                'human_targets': human_targets_avg,
                'rl_targets': rl_targets_avg,
                'heuristic_targets': heuristic_targets_avg,
                'human_threats': human_threats_avg,
                'rl_threats': rl_threats_avg,
                'heuristic_threats': heuristic_threats_avg
            },
            'summary_stats': {
                'top_target_agents': top_6_agents,
                'top_threat_agents': top_5_threats
            }
        }

    def analyze_progress_rate_mse(self,
                                  human_trajectories: List[Trajectory],
                                  rl_trajectories: List[Trajectory],
                                  heuristic_trajectories: List[Trajectory],
                                  save_dir: str = "exp3_similarity_analysis",
                                  n_timepoints: int = 10
                                  ):
        """
        Analyze Mean Squared Error (MSE) between human progress rates and both
        RL and individual heuristic agent progress rates for target and threat identification.

        Args:
            human_trajectories: List of human trajectory objects
            rl_trajectories: List of RL trajectory objects
            heuristic_trajectories: List of heuristic trajectory objects
            save_dir: Directory to save outputs
            n_timepoints: Number of evenly-spaced time points to analyze (default: 10)

        Returns:
            dict: MSE results and time series data
        """
        os.makedirs(save_dir, exist_ok=True)

        # Generate evenly-spaced time points from 10% to 90%
        time_points = np.linspace(0.1, 0.9, n_timepoints)

        def extract_identification_series(trajs, time_points, metric_type='targets'):
            """
            Extract target or threat identification counts at specified time points

            Args:
                trajs: List of trajectory objects
                time_points: Array of time points (0.1 to 0.9)
                metric_type: 'targets' or 'threats'

            Returns:
                dict: {time_point: [values_across_trajectories]}
            """
            series_by_timepoint = {tp: [] for tp in time_points}

            for traj in trajs:
                if metric_type == 'targets':
                    identification_data = traj.target_ids
                elif metric_type == 'threats':
                    identification_data = traj.threat_ids
                else:
                    raise ValueError("metric_type must be 'targets' or 'threats'")

                if not identification_data:
                    continue

                traj_length = len(identification_data)
                if traj_length < 2:  # Skip very short trajectories
                    continue

                for time_point in time_points:
                    # Calculate index for this time point
                    if time_point >= 1.0:
                        idx = traj_length - 1  # Use final index for 100%
                    else:
                        idx = int(time_point * traj_length)
                        idx = min(idx, traj_length - 1)  # Ensure we don't exceed bounds

                    # Extract identification count at this time point
                    identification_count = identification_data[idx]
                    series_by_timepoint[time_point].append(identification_count)

            return series_by_timepoint

        def compute_average_series(series_by_timepoint, time_points):
            """Compute average identification counts at each time point"""
            avg_series = []
            valid_timepoints = []

            for tp in time_points:
                values = series_by_timepoint[tp]
                if len(values) > 0:
                    avg_series.append(np.mean(values))
                    valid_timepoints.append(tp)
                else:
                    print(f"Warning: No data at time point {tp:.1%}")

            return np.array(avg_series), np.array(valid_timepoints)

        def compute_mse(series1, series2):
            """Compute Mean Squared Error between two time series"""
            if len(series1) != len(series2):
                min_len = min(len(series1), len(series2))
                series1 = series1[:min_len]
                series2 = series2[:min_len]

            if len(series1) == 0:
                return None

            return np.mean((series1 - series2) ** 2)

        # Group heuristic trajectories by agent type (name)
        heuristic_agents = {}
        for traj in heuristic_trajectories:
            agent_name = traj.name
            if agent_name not in heuristic_agents:
                heuristic_agents[agent_name] = []
            heuristic_agents[agent_name].append(traj)

        print(f"Found {len(heuristic_agents)} unique heuristic agent types:")
        for agent_name, trajs in heuristic_agents.items():
            print(f"  - {agent_name}: {len(trajs)} trajectories")

        print("Extracting target identification time series...")
        # Extract target identification series for each group
        human_targets_series = extract_identification_series(human_trajectories, time_points, 'targets')
        rl_targets_series = extract_identification_series(rl_trajectories, time_points, 'targets')

        # Extract series for each individual heuristic agent type
        heuristic_targets_series = {}
        for agent_name, trajs in heuristic_agents.items():
            heuristic_targets_series[agent_name] = extract_identification_series(trajs, time_points, 'targets')

        print("Extracting threat identification time series...")
        # Extract threat identification series for each group
        human_threats_series = extract_identification_series(human_trajectories, time_points, 'threats')
        rl_threats_series = extract_identification_series(rl_trajectories, time_points, 'threats')

        # Extract series for each individual heuristic agent type
        heuristic_threats_series = {}
        for agent_name, trajs in heuristic_agents.items():
            heuristic_threats_series[agent_name] = extract_identification_series(trajs, time_points, 'threats')

        # Compute average series for each group
        human_targets_avg, valid_tp_targets = compute_average_series(human_targets_series, time_points)
        rl_targets_avg, _ = compute_average_series(rl_targets_series, time_points)

        # Compute average series for each heuristic agent type
        heuristic_targets_avg = {}
        for agent_name, series in heuristic_targets_series.items():
            heuristic_targets_avg[agent_name], _ = compute_average_series(series, time_points)

        human_threats_avg, valid_tp_threats = compute_average_series(human_threats_series, time_points)
        rl_threats_avg, _ = compute_average_series(rl_threats_series, time_points)

        # Compute average series for each heuristic agent type
        heuristic_threats_avg = {}
        for agent_name, series in heuristic_threats_series.items():
            heuristic_threats_avg[agent_name], _ = compute_average_series(series, time_points)

        # Ensure all series have the same length for MSE calculation
        min_len_targets = min(len(human_targets_avg), len(rl_targets_avg))
        min_len_threats = min(len(human_threats_avg), len(rl_threats_avg))

        if min_len_targets == 0 or min_len_threats == 0:
            print("Error: Insufficient data for MSE analysis")
            return None

        # Trim series to same length
        human_targets_avg = human_targets_avg[:min_len_targets]
        rl_targets_avg = rl_targets_avg[:min_len_targets]
        valid_tp_targets = valid_tp_targets[:min_len_targets]

        human_threats_avg = human_threats_avg[:min_len_threats]
        rl_threats_avg = rl_threats_avg[:min_len_threats]
        valid_tp_threats = valid_tp_threats[:min_len_threats]

        # Calculate MSE values
        print("Computing MSE values...")
        mse_results = {}

        # Target identification MSE
        if len(human_targets_avg) > 0:
            mse_human_rl_targets = compute_mse(human_targets_avg, rl_targets_avg)
            mse_results['human_vs_rl_targets'] = mse_human_rl_targets

            # MSE with individual heuristic agents
            for agent_name, agent_targets_avg in heuristic_targets_avg.items():
                if len(agent_targets_avg) >= min_len_targets:
                    agent_targets_trimmed = agent_targets_avg[:min_len_targets]
                    mse_human_heuristic = compute_mse(human_targets_avg, agent_targets_trimmed)
                    mse_results[f'human_vs_{agent_name}_targets'] = mse_human_heuristic

        # Threat identification MSE
        if len(human_threats_avg) > 0:
            mse_human_rl_threats = compute_mse(human_threats_avg, rl_threats_avg)
            mse_results['human_vs_rl_threats'] = mse_human_rl_threats

            # MSE with individual heuristic agents
            for agent_name, agent_threats_avg in heuristic_threats_avg.items():
                if len(agent_threats_avg) >= min_len_threats:
                    agent_threats_trimmed = agent_threats_avg[:min_len_threats]
                    mse_human_heuristic = compute_mse(human_threats_avg, agent_threats_trimmed)
                    mse_results[f'human_vs_{agent_name}_threats'] = mse_human_heuristic

        # Create plots showing MSE comparisons
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Define colors for heuristic agents
        import matplotlib.cm as cm
        colors = cm.Set3(np.linspace(0, 1, len(heuristic_agents)))
        heuristic_colors = dict(zip(heuristic_agents.keys(), colors))

        # Plot 1: Target identification progress lines
        ax1.plot(valid_tp_targets * 100, human_targets_avg,
                 marker='o', linewidth=3, markersize=8,
                 label='Human', color='blue', alpha=0.9, zorder=10)
        ax1.plot(valid_tp_targets * 100, rl_targets_avg,
                 marker='s', linewidth=3, markersize=8,
                 label=f'RL (MSE: {mse_results.get("human_vs_rl_targets", 0):.3f})',
                 color='green', alpha=0.9, zorder=10)

        # Show only top 6 most similar heuristic agents for clarity
        target_mse_heuristic = {}
        for key, value in mse_results.items():
            if key.startswith('human_vs_') and key.endswith('_targets') and 'rl' not in key and value is not None:
                agent_name = key.replace('human_vs_', '').replace('_targets', '')
                target_mse_heuristic[agent_name] = value

        top_6_agents_targets = sorted(target_mse_heuristic.items(), key=lambda x: x[1])[:6]

        for agent_name, mse_value in top_6_agents_targets:
            if agent_name in heuristic_targets_avg:
                agent_targets_avg = heuristic_targets_avg[agent_name]
                if len(agent_targets_avg) >= min_len_targets:
                    agent_targets_trimmed = agent_targets_avg[:min_len_targets]
                    ax1.plot(valid_tp_targets * 100, agent_targets_trimmed,
                             marker='^', linewidth=2, markersize=6,
                             label=f'{agent_name[:15]}... (MSE: {mse_value:.3f})',
                             color=heuristic_colors[agent_name], alpha=0.7)

        ax1.set_xlabel('Progress Through Trajectory (%)', fontsize=12)
        ax1.set_ylabel('Average Targets Identified', fontsize=12)
        ax1.set_title('Target Identification: Top 6 Most Similar Agents (Lowest MSE)', fontsize=14)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Target MSE bar chart
        target_mse_values = []
        target_labels = []

        # Add RL MSE
        if 'human_vs_rl_targets' in mse_results and mse_results['human_vs_rl_targets'] is not None:
            target_mse_values.append(mse_results['human_vs_rl_targets'])
            target_labels.append('RL (all)')

        # Add top 10 heuristic agent MSEs
        top_10_targets = sorted(target_mse_heuristic.items(), key=lambda x: x[1])[:96]
        for agent_name, mse_value in top_10_targets:
            target_mse_values.append(mse_value)
            target_labels.append(agent_name[:20] + ('...' if len(agent_name) > 15 else ''))

        bars = ax2.bar(range(len(target_mse_values)), target_mse_values, alpha=0.7)
        bars[0].set_color('green') if len(bars) > 0 else None  # Color RL bar differently

        ax2.set_xlabel('Agent Type', fontsize=18)
        ax2.set_ylabel('MSE vs Human Targets', fontsize=18)
        ax2.set_title('Average MSE with all human trajectories – \nRegular Target IDs over time', fontsize=20)
        ax2.set_xticks(range(len(target_labels)))
        ax2.set_xticklabels(target_labels, rotation=45, ha='right')
        ax2.grid(True, axis='y', alpha=0.3)

        # Add value labels on bars
        # for bar, value in zip(bars, target_mse_values):
        #     height = bar.get_height()
        #     ax2.text(bar.get_x() + bar.get_width() / 2., height + max(target_mse_values) * 0.01,
        #              f'{value:.3f}', ha='center', va='bottom', fontsize=8)

        # Plot 3: Threat identification progress lines
        ax3.plot(valid_tp_threats * 100, human_threats_avg,
                 marker='o', linewidth=3, markersize=8,
                 label='Human', color='blue', alpha=0.9, zorder=10)
        ax3.plot(valid_tp_threats * 100, rl_threats_avg,
                 marker='s', linewidth=3, markersize=8,
                 label=f'RL (MSE: {mse_results.get("human_vs_rl_threats", 0):.3f})',
                 color='green', alpha=0.9, zorder=10)

        # Show only top 6 most similar heuristic agents for threats
        threat_mse_heuristic = {}
        for key, value in mse_results.items():
            if key.startswith('human_vs_') and key.endswith('_threats') and 'rl' not in key and value is not None:
                agent_name = key.replace('human_vs_', '').replace('_threats', '')
                threat_mse_heuristic[agent_name] = value

        top_6_agents_threats = sorted(threat_mse_heuristic.items(), key=lambda x: x[1])[:6]

        for agent_name, mse_value in top_6_agents_threats:
            if agent_name in heuristic_threats_avg:
                agent_threats_avg = heuristic_threats_avg[agent_name]
                if len(agent_threats_avg) >= min_len_threats:
                    agent_threats_trimmed = agent_threats_avg[:min_len_threats]
                    ax3.plot(valid_tp_threats * 100, agent_threats_trimmed,
                             marker='^', linewidth=2, markersize=6,
                             label=f'{agent_name[:15]}... (MSE: {mse_value:.3f})',
                             color=heuristic_colors[agent_name], alpha=0.7)

        ax3.set_xlabel('Progress Through Trajectory (%)', fontsize=12)
        ax3.set_ylabel('Average Threats Identified', fontsize=12)
        ax3.set_title('Threat Identification: Top 6 Most Similar Agents (Lowest MSE)', fontsize=14)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Threat MSE bar chart
        threat_mse_values = []
        threat_labels = []

        # Add RL MSE
        if 'human_vs_rl_threats' in mse_results and mse_results['human_vs_rl_threats'] is not None:
            threat_mse_values.append(mse_results['human_vs_rl_threats'])
            threat_labels.append('RL (all))')

        # Add top 10 heuristic agent MSEs
        top_10_threats = sorted(threat_mse_heuristic.items(), key=lambda x: x[1])[:96]
        for agent_name, mse_value in top_10_threats:
            threat_mse_values.append(mse_value)
            threat_labels.append(agent_name[:20] + ('...' if len(agent_name) > 15 else ''))

        bars = ax4.bar(range(len(threat_mse_values)), threat_mse_values, alpha=0.7)
        bars[0].set_color('green') if len(bars) > 0 else None  # Color RL bar differently

        ax4.set_xlabel('Agent Type', fontsize=18)
        ax4.set_ylabel('MSE vs Human Threats', fontsize=18)
        ax4.set_title('Average MSE with all human trajectories – \nHigh-Value Target IDs over time', fontsize=20)
        ax4.set_xticks(range(len(threat_labels)))
        ax4.set_xticklabels(threat_labels, rotation=45, ha='right')
        ax4.grid(True, axis='y', alpha=0.3)

        # Add value labels on bars
        # for bar, value in zip(bars, threat_mse_values):
        #     height = bar.get_height()
        #     ax4.text(bar.get_x() + bar.get_width() / 2., height + max(threat_mse_values) * 0.01,
        #              f'{value:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(save_dir, "progress_rate_mse_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Save numerical results
        results_data = {
            'time_points_percent': (valid_tp_targets * 100).tolist(),
            'human_targets_avg': human_targets_avg.tolist(),
            'rl_targets_avg': rl_targets_avg.tolist(),
            'human_threats_avg': human_threats_avg.tolist(),
            'rl_threats_avg': rl_threats_avg.tolist(),
            'mse_results': mse_results
        }

        # Add individual heuristic agent data
        for agent_name, agent_targets_avg in heuristic_targets_avg.items():
            if len(agent_targets_avg) >= min_len_targets:
                results_data[f'{agent_name}_targets_avg'] = agent_targets_avg[:min_len_targets].tolist()

        for agent_name, agent_threats_avg in heuristic_threats_avg.items():
            if len(agent_threats_avg) >= min_len_threats:
                results_data[f'{agent_name}_threats_avg'] = agent_threats_avg[:min_len_threats].tolist()

        results_path = os.path.join(save_dir, "progress_rate_mse_results.json")
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        # Create comprehensive summary DataFrame
        summary_data = []

        # Add Human vs RL comparisons
        if 'human_vs_rl_targets' in mse_results and mse_results['human_vs_rl_targets'] is not None:
            summary_data.append({
                'Comparison': 'Human vs RL (Targets)',
                'Agent_Type': 'RL',
                'MSE': mse_results['human_vs_rl_targets']
            })

        if 'human_vs_rl_threats' in mse_results and mse_results['human_vs_rl_threats'] is not None:
            summary_data.append({
                'Comparison': 'Human vs RL (Threats)',
                'Agent_Type': 'RL',
                'MSE': mse_results['human_vs_rl_threats']
            })

        # Add individual heuristic agent comparisons
        for key, value in mse_results.items():
            if key.startswith('human_vs_') and (
                    '_targets' in key or '_threats' in key) and 'rl' not in key and value is not None:
                if '_targets' in key:
                    agent_name = key.replace('human_vs_', '').replace('_targets', '')
                    metric = 'Targets'
                else:
                    agent_name = key.replace('human_vs_', '').replace('_threats', '')
                    metric = 'Threats'

                summary_data.append({
                    'Comparison': f'Human vs {agent_name} ({metric})',
                    'Agent_Type': 'Heuristic',
                    'Agent_Name': agent_name,
                    'MSE': value
                })

        summary_df = pd.DataFrame(summary_data)

        # Sort by MSE for easier interpretation
        summary_df = summary_df.sort_values('MSE', ascending=True)

        summary_path = os.path.join(save_dir, "progress_rate_mse_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        # Print detailed results summary
        print("\n" + "=" * 80)
        print("MEAN SQUARED ERROR (MSE) PROGRESS ANALYSIS SUMMARY")
        print("=" * 80)

        print(
            f"\nAnalyzed {len(valid_tp_targets)} time points from {valid_tp_targets[0]:.1%} to {valid_tp_targets[-1]:.1%}")
        print(f"Analyzed {len(heuristic_agents)} individual heuristic agent types")
        print("\nLower MSE values indicate better similarity to human progress patterns.")

        print("\nTarget Identification MSE Results:")
        print("-" * 50)
        if 'human_vs_rl_targets' in mse_results and mse_results['human_vs_rl_targets'] is not None:
            print(f"Human vs RL:       MSE = {mse_results['human_vs_rl_targets']:.6f}")

        # Print top 5 most similar heuristic agents for targets
        print("\nTop 5 Most Similar Heuristic Agents (Targets - Lowest MSE):")
        for i, (agent_name, mse_value) in enumerate(top_6_agents_targets[:5]):
            print(f"  {i + 1}. {agent_name}: MSE = {mse_value:.6f}")

        print("\nThreat Identification MSE Results:")
        print("-" * 50)
        if 'human_vs_rl_threats' in mse_results and mse_results['human_vs_rl_threats'] is not None:
            print(f"Human vs RL:       MSE = {mse_results['human_vs_rl_threats']:.6f}")

        # Print top 5 most similar heuristic agents for threats
        print("\nTop 5 Most Similar Heuristic Agents (Threats - Lowest MSE):")
        for i, (agent_name, mse_value) in enumerate(top_6_agents_threats[:5]):
            print(f"  {i + 1}. {agent_name}: MSE = {mse_value:.6f}")

        # Determine which agent type is more similar to humans overall
        print(f"\nOverall Similarity to Human Progress Patterns:")
        print("-" * 50)

        # Compare average ranks
        rl_target_rank = None
        rl_threat_rank = None

        if 'human_vs_rl_targets' in mse_results and mse_results['human_vs_rl_targets'] is not None:
            target_mse_all = [(name, mse) for name, mse in target_mse_heuristic.items()]
            target_mse_all.append(('RL', mse_results['human_vs_rl_targets']))
            target_mse_all.sort(key=lambda x: x[1])
            rl_target_rank = next(i for i, (name, _) in enumerate(target_mse_all) if name == 'RL') + 1
            print(f"RL agent ranks #{rl_target_rank} out of {len(target_mse_all)} for target identification similarity")

        if 'human_vs_rl_threats' in mse_results and mse_results['human_vs_rl_threats'] is not None:
            threat_mse_all = [(name, mse) for name, mse in threat_mse_heuristic.items()]
            threat_mse_all.append(('RL', mse_results['human_vs_rl_threats']))
            threat_mse_all.sort(key=lambda x: x[1])
            rl_threat_rank = next(i for i, (name, _) in enumerate(threat_mse_all) if name == 'RL') + 1
            print(f"RL agent ranks #{rl_threat_rank} out of {len(threat_mse_all)} for threat identification similarity")

        print(f"\nFiles saved:")
        print(f"  Plot: {plot_path}")
        print(f"  Detailed results: {results_path}")
        print(f"  Summary table: {summary_path}")

        return {
            'mse_results': mse_results,
            'time_series': {
                'time_points': valid_tp_targets,
                'human_targets': human_targets_avg,
                'rl_targets': rl_targets_avg,
                'heuristic_targets': heuristic_targets_avg,
                'human_threats': human_threats_avg,
                'rl_threats': rl_threats_avg,
                'heuristic_threats': heuristic_threats_avg
            },
            'summary_stats': {
                'top_target_agents': top_6_agents_targets,
                'top_threat_agents': top_6_agents_threats,
                'rl_target_rank': rl_target_rank,
                'rl_threat_rank': rl_threat_rank
            }
        }

    def analyze_cross_trajectory_position_mse(self,
                                              human_trajectories: List[Trajectory],
                                              rl_trajectories: List[Trajectory],
                                              heuristic_trajectories: List[Trajectory],
                                              save_dir: str = "exp3_similarity_analysis",
                                              n_timepoints: int = 20):
        """
        Calculate position vs time MSE between every possible cross-group trajectory pair.

        Tests the hypothesis that the full league of heuristic trajectories provides better
        human modeling than any individual heuristic trajectory by comparing:
        1. Average MSE between all human trajectories and each individual heuristic agent type
        2. Average MSE between all human trajectories and all heuristic trajectories combined

        Args:
            human_trajectories: List of human trajectory objects
            rl_trajectories: List of RL trajectory objects
            heuristic_trajectories: List of heuristic trajectory objects
            save_dir: Directory to save outputs
            n_timepoints: Number of evenly-spaced time points for comparison

        Returns:
            dict: MSE analysis results including statistical test outcomes
        """
        os.makedirs(save_dir, exist_ok=True)

        def extract_position_timeseries(traj, n_timepoints):
            """Extract position time series at evenly spaced time points"""
            if not traj.positions or len(traj.positions) < 2:
                return None

            positions = np.array(traj.positions)
            traj_length = len(positions)

            # Create evenly spaced indices
            indices = np.linspace(0, traj_length - 1, n_timepoints, dtype=int)
            sampled_positions = positions[indices]

            # Flatten to 1D array: [x1, y1, x2, y2, ..., xn, yn]
            return sampled_positions.flatten()

        def compute_position_mse(traj1, traj2, n_timepoints):
            """Compute MSE between two trajectory position time series"""
            series1 = extract_position_timeseries(traj1, n_timepoints)
            series2 = extract_position_timeseries(traj2, n_timepoints)

            if series1 is None or series2 is None:
                return None

            if len(series1) != len(series2):
                min_len = min(len(series1), len(series2))
                series1 = series1[:min_len]
                series2 = series2[:min_len]

            return np.mean((series1 - series2) ** 2)

        print("Computing cross-trajectory position MSE analysis...")
        print(f"Human trajectories: {len(human_trajectories)}")
        print(f"RL trajectories: {len(rl_trajectories)}")
        print(f"Heuristic trajectories: {len(heuristic_trajectories)}")

        # Group heuristic trajectories by agent type
        heuristic_agents = {}
        for traj in heuristic_trajectories:
            agent_name = traj.name
            if agent_name not in heuristic_agents:
                heuristic_agents[agent_name] = []
            heuristic_agents[agent_name].append(traj)

        print(f"Found {len(heuristic_agents)} unique heuristic agent types:")
        for agent_name, trajs in heuristic_agents.items():
            print(f"  - {agent_name}: {len(trajs)} trajectories")

        # 1. Compute MSE between all human-heuristic trajectory pairs for each agent type
        agent_mse_results = {}

        for agent_name, agent_trajs in heuristic_agents.items():
            print(f"\nComputing MSEs for agent type: {agent_name}")
            mse_values = []

            for i, human_traj in enumerate(human_trajectories):
                for j, heuristic_traj in enumerate(agent_trajs):
                    mse = compute_position_mse(human_traj, heuristic_traj, n_timepoints)
                    if mse is not None:
                        mse_values.append(mse)

            agent_mse_results[agent_name] = {
                'mse_values': mse_values,
                'mean_mse': np.mean(mse_values) if mse_values else None,
                'std_mse': np.std(mse_values) if mse_values else None,
                'n_comparisons': len(mse_values)
            }

            print(f"  {len(mse_values)} valid comparisons, mean MSE: {np.mean(mse_values):.6f}")

        # 2. Compute MSE between all human trajectories and ALL heuristic trajectories
        print(f"\nComputing MSEs between all human and all heuristic trajectories...")
        all_heuristic_mse_values = []

        for i, human_traj in enumerate(human_trajectories):
            for j, heuristic_traj in enumerate(heuristic_trajectories):
                mse = compute_position_mse(human_traj, heuristic_traj, n_timepoints)
                if mse is not None:
                    all_heuristic_mse_values.append(mse)

        overall_mean_mse = np.mean(all_heuristic_mse_values) if all_heuristic_mse_values else None

        print(f"  {len(all_heuristic_mse_values)} valid comparisons, mean MSE: {overall_mean_mse:.6f}")

        # 3. For comparison, compute human-RL MSEs
        print(f"\nComputing MSEs between human and RL trajectories...")
        human_rl_mse_values = []

        for i, human_traj in enumerate(human_trajectories):
            for j, rl_traj in enumerate(rl_trajectories):
                mse = compute_position_mse(human_traj, rl_traj, n_timepoints)
                if mse is not None:
                    human_rl_mse_values.append(mse)

        rl_mean_mse = np.mean(human_rl_mse_values) if human_rl_mse_values else None

        print(f"  {len(human_rl_mse_values)} valid comparisons, mean MSE: {rl_mean_mse:.6f}")

        # 4. Test the hypothesis: Is any individual agent worse than the overall average?
        print(f"\n" + "=" * 80)
        print("HYPOTHESIS TESTING: LEAGUE DIVERSITY BENEFIT")
        print("=" * 80)

        hypothesis_results = {}

        # Test: For each individual agent, is their average MSE > overall average MSE?
        # This would support the hypothesis that diversity helps
        print(f"\nOverall average MSE (all heuristic agents): {overall_mean_mse:.6f}")
        print(f"RL average MSE (for comparison): {rl_mean_mse:.6f}")

        worse_than_overall = []
        better_than_overall = []

        for agent_name, results in agent_mse_results.items():
            agent_mean = results['mean_mse']
            if agent_mean is not None and overall_mean_mse is not None:
                is_worse = agent_mean > overall_mean_mse
                difference = agent_mean - overall_mean_mse

                print(f"\n{agent_name}:")
                print(f"  Individual agent average MSE: {agent_mean:.6f}")
                print(f"  Difference from overall: {difference:+.6f}")
                print(f"  {'WORSE' if is_worse else 'BETTER'} than overall average")

                if is_worse:
                    worse_than_overall.append((agent_name, agent_mean, difference))
                else:
                    better_than_overall.append((agent_name, agent_mean, difference))

        # Statistical tests
        from scipy import stats

        # Perform statistical tests comparing each individual agent to overall distribution
        statistical_tests = {}

        for agent_name, results in agent_mse_results.items():
            if results['mse_values'] and len(results['mse_values']) > 1:
                # Mann-Whitney U test: individual agent vs all heuristic agents
                u_stat, p_val = stats.mannwhitneyu(
                    results['mse_values'],
                    all_heuristic_mse_values,
                    alternative='two-sided'
                )

                statistical_tests[agent_name] = {
                    'mann_whitney_u': u_stat,
                    'p_value': p_val,
                    'significant_difference': p_val < 0.05
                }

        # Summary statistics
        n_worse = len(worse_than_overall)
        n_better = len(better_than_overall)
        n_total = len(agent_mse_results)

        hypothesis_results = {
            'overall_mean_mse': overall_mean_mse,
            'rl_mean_mse': rl_mean_mse,
            'n_agents_worse_than_overall': n_worse,
            'n_agents_better_than_overall': n_better,
            'fraction_worse_than_overall': n_worse / n_total if n_total > 0 else 0,
            'worse_agents': worse_than_overall,
            'better_agents': better_than_overall,
            'statistical_tests': statistical_tests
        }

        print(f"\n" + "-" * 60)
        print("SUMMARY:")
        print(
            f"  {n_worse}/{n_total} ({100 * n_worse / n_total:.1f}%) individual agents perform WORSE than overall average")
        print(
            f"  {n_better}/{n_total} ({100 * n_better / n_total:.1f}%) individual agents perform BETTER than overall average")

        if n_worse > n_better:
            print(f"\n✓ HYPOTHESIS SUPPORTED: Majority of individual agents are worse than the diverse league")
            print(f"  This suggests that diversity in the heuristic agent league provides better human modeling")
        else:
            print(f"\n✗ HYPOTHESIS NOT SUPPORTED: More individual agents perform better than the overall average")

        # 5. Create visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: MSE distributions for each agent type
        agent_names = list(agent_mse_results.keys())
        agent_means = [agent_mse_results[name]['mean_mse'] for name in agent_names if
                       agent_mse_results[name]['mean_mse'] is not None]

        valid_agent_names = [name for name in agent_names if agent_mse_results[name]['mean_mse'] is not None]

        ax1.grid(True, axis='y', alpha=0.6)

        bars = ax1.bar(range(len(valid_agent_names)), agent_means, alpha=0.9)
        ax1.axhline(y=overall_mean_mse, color='red', linestyle='-', linewidth=2,label=f'Heuristic Average ({overall_mean_mse:.1f})')
        ax1.axhline(y=rl_mean_mse, color='green', linestyle='-', linewidth=2, label=f'RL Average ({rl_mean_mse:.1f})')

        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x / 1000)}k'))
        ax1.tick_params(labelsize=15)
        ax1.set_xlabel('Heuristic Configuration', fontsize=20)
        ax1.set_ylabel('Position MSE vs. Human', fontsize=20)
        ax1.set_title('Position vs. time MSE between heuristic\nagents and human trajectories', fontsize=25)
        ax1.set_xticks(range(len(valid_agent_names)))
        ax1.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in valid_agent_names], rotation=45, ha='right')
        ax1.set_ylim(bottom=100000)
        #ax1.legend()


        # Add value labels on bars
        # for bar, value in zip(bars, agent_means):
        #     height = bar.get_height()
        #     ax1.text(bar.get_x() + bar.get_width() / 2., height + max(agent_means) * 0.01,
        #              f'{value:.4f}', ha='center', va='bottom', fontsize=8)

        # Plot 2: Distribution comparison (box plot)
        box_data = []
        box_labels = []

        # Add individual agent distributions (top 6 most different from overall)
        sorted_agents = sorted([(name, abs(results['mean_mse'] - overall_mean_mse))
                                for name, results in agent_mse_results.items()
                                if results['mean_mse'] is not None],
                               key=lambda x: x[1], reverse=True)[:6]

        for agent_name, _ in sorted_agents:
            box_data.append(agent_mse_results[agent_name]['mse_values'])
            box_labels.append(agent_name[:15] + '...' if len(agent_name) > 10 else agent_name)

        # Add overall heuristic and RL distributions
        box_data.extend([all_heuristic_mse_values, human_rl_mse_values])
        box_labels.extend(['All Heuristic', 'RL'])

        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)

        # Color the boxes
        colors = ['lightcoral' if i < len(sorted_agents) else 'lightgreen' for i in range(len(bp['boxes']))]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax2.set_ylabel('Position MSE vs Humans')
        ax2.set_title('MSE Distribution Comparison')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, axis='y', alpha=0.3)

        # Plot 3: Hypothesis test results (p-values)
        test_agents = [name for name in statistical_tests.keys()]
        p_values = [statistical_tests[name]['p_value'] for name in test_agents]

        bars3 = ax3.bar(range(len(test_agents)), p_values, alpha=0.7)
        ax3.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='p = 0.05')

        # Color bars based on significance
        for bar, p_val in zip(bars3, p_values):
            if p_val < 0.05:
                bar.set_color('orange')  # Significant difference
            else:
                bar.set_color('lightgray')  # Not significant

        ax3.set_xlabel('Heuristic Agent Type')
        #ax3.set_ylabel('p-value (Mann-Whitney U Test)')
        ax3.set_title('Statistical Significance vs Overall Distribution')
        ax3.set_xticks(range(len(test_agents)))
        ax3.set_xticklabels([name[:15] + '...' if len(name) > 10 else name for name in test_agents],
                            rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, axis='y', alpha=0.3)
        ax3.set_yscale('log')  # Log scale for p-values

        # Plot 4: Summary pie chart
        labels = ['Worse than Overall', 'Better than Overall']
        sizes = [n_worse, n_better]
        colors = ['lightcoral', 'lightblue']

        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.set_title(f'Agent Performance Distribution\n({n_total} total agents)')

        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(save_dir, "cross_trajectory_position_mse_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()

        # 6. Save detailed results
        detailed_results = {
            'hypothesis_results': hypothesis_results,
            'agent_mse_results': {k: {**v, 'mse_values': v['mse_values'][:100]}  # Limit saved values for file size
                                  for k, v in agent_mse_results.items()},
            'overall_statistics': {
                'overall_mean_mse': float(overall_mean_mse) if overall_mean_mse is not None else None,
                'overall_std_mse': float(np.std(all_heuristic_mse_values)) if all_heuristic_mse_values else None,
                'rl_mean_mse': float(rl_mean_mse) if rl_mean_mse is not None else None,
                'rl_std_mse': float(np.std(human_rl_mse_values)) if human_rl_mse_values else None,
                'n_total_comparisons': int(len(all_heuristic_mse_values)),
                'n_rl_comparisons': int(len(human_rl_mse_values))
            },
            'statistical_tests': statistical_tests
        }
        # detailed_results = {
        #     'hypothesis_results': hypothesis_results,
        #     'agent_mse_results': {k: {**v, 'mse_values': v['mse_values'][:100]}  # Limit saved values for file size
        #                           for k, v in agent_mse_results.items()},
        #     'overall_statistics': {
        #         'overall_mean_mse': overall_mean_mse,
        #         'overall_std_mse': np.std(all_heuristic_mse_values) if all_heuristic_mse_values else None,
        #         'rl_mean_mse': rl_mean_mse,
        #         'rl_std_mse': np.std(human_rl_mse_values) if human_rl_mse_values else None,
        #         'n_total_comparisons': len(all_heuristic_mse_values),
        #         'n_rl_comparisons': len(human_rl_mse_values)
        #     },
        #     'statistical_tests': statistical_tests
        # }

        # results_path = os.path.join(save_dir, "cross_trajectory_position_mse_results.json")
        # with open(results_path, 'w') as f:
        #     json.dump(detailed_results, f, indent=2)

        # Create summary table
        summary_data = []
        for agent_name, results in agent_mse_results.items():
            if results['mean_mse'] is not None:
                summary_data.append({
                    'Agent_Name': agent_name,
                    'Mean_MSE': results['mean_mse'],
                    'Std_MSE': results['std_mse'],
                    'N_Comparisons': results['n_comparisons'],
                    'Diff_from_Overall': results['mean_mse'] - overall_mean_mse,
                    'Worse_than_Overall': results['mean_mse'] > overall_mean_mse,
                    'P_Value': statistical_tests.get(agent_name, {}).get('p_value', None),
                    'Significant_Diff': statistical_tests.get(agent_name, {}).get('significant_difference', None)
                })

        # Add overall and RL for comparison
        summary_data.extend([
            {
                'Agent_Name': 'OVERALL_HEURISTIC',
                'Mean_MSE': overall_mean_mse,
                'Std_MSE': np.std(all_heuristic_mse_values) if all_heuristic_mse_values else None,
                'N_Comparisons': len(all_heuristic_mse_values),
                'Diff_from_Overall': 0.0,
                'Worse_than_Overall': False,
                'P_Value': None,
                'Significant_Diff': None
            },
            {
                'Agent_Name': 'RL_AGENT',
                'Mean_MSE': rl_mean_mse,
                'Std_MSE': np.std(human_rl_mse_values) if human_rl_mse_values else None,
                'N_Comparisons': len(human_rl_mse_values),
                'Diff_from_Overall': rl_mean_mse - overall_mean_mse if rl_mean_mse and overall_mean_mse else None,
                'Worse_than_Overall': rl_mean_mse > overall_mean_mse if rl_mean_mse and overall_mean_mse else None,
                'P_Value': None,
                'Significant_Diff': None
            }
        ])

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Mean_MSE', ascending=True)

        summary_path = os.path.join(save_dir, "cross_trajectory_position_mse_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print(f"\nFiles saved:")
        print(f"  Plot: {plot_path}")
        #print(f"  Detailed results: {results_path}")
        print(f"  Summary table: {summary_path}")

        return detailed_results

    def analyze_human_similarity_comparison(self,
                                            human_trajectories: List[Trajectory],
                                            rl_trajectories: List[Trajectory],
                                            heuristic_trajectories: List[Trajectory],
                                            save_dir: str = "exp3_similarity_analysis",
                                            n_timepoints: int = 20):
        """
        Test the hypothesis that heuristic agents are more similar to humans than RL agents
        by comparing position trajectory MSE distributions.

        Performs statistical tests to determine if:
        1. Average MSE(human, heuristic) < Average MSE(human, RL)
        2. The difference is statistically significant

        Args:
            human_trajectories: List of human trajectory objects
            rl_trajectories: List of RL trajectory objects
            heuristic_trajectories: List of heuristic trajectory objects
            save_dir: Directory to save outputs
            n_timepoints: Number of evenly-spaced time points for comparison

        Returns:
            dict: Statistical test results and MSE comparisons
        """
        os.makedirs(save_dir, exist_ok=True)

        def extract_position_timeseries(traj, n_timepoints):
            """Extract position time series at evenly spaced time points"""
            if not traj.positions or len(traj.positions) < 2:
                return None

            positions = np.array(traj.positions)
            traj_length = len(positions)

            # Create evenly spaced indices
            indices = np.linspace(0, traj_length - 1, n_timepoints, dtype=int)
            sampled_positions = positions[indices]

            # Flatten to 1D array: [x1, y1, x2, y2, ..., xn, yn]
            return sampled_positions.flatten()

        def compute_position_mse(traj1, traj2, n_timepoints):
            """Compute MSE between two trajectory position time series"""
            series1 = extract_position_timeseries(traj1, n_timepoints)
            series2 = extract_position_timeseries(traj2, n_timepoints)

            if series1 is None or series2 is None:
                return None

            if len(series1) != len(series2):
                min_len = min(len(series1), len(series2))
                series1 = series1[:min_len]
                series2 = series2[:min_len]

            return np.mean((series1 - series2) ** 2)

        print("Computing Human-Heuristic vs Human-RL similarity comparison...")
        print(f"Human trajectories: {len(human_trajectories)}")
        print(f"RL trajectories: {len(rl_trajectories)}")
        print(f"Heuristic trajectories: {len(heuristic_trajectories)}")

        # 1. Compute all human-heuristic MSE values
        print("\nComputing Human-Heuristic MSE values...")
        human_heuristic_mse = []

        for i, human_traj in enumerate(human_trajectories):
            if i % 10 == 0:  # Progress indicator
                print(f"  Processing human trajectory {i + 1}/{len(human_trajectories)}")

            for j, heuristic_traj in enumerate(heuristic_trajectories):
                mse = compute_position_mse(human_traj, heuristic_traj, n_timepoints)
                if mse is not None:
                    human_heuristic_mse.append(mse)

        # 2. Compute all human-RL MSE values
        print("\nComputing Human-RL MSE values...")
        human_rl_mse = []

        for i, human_traj in enumerate(human_trajectories):
            if i % 10 == 0:  # Progress indicator
                print(f"  Processing human trajectory {i + 1}/{len(human_trajectories)}")

            for j, rl_traj in enumerate(rl_trajectories):
                mse = compute_position_mse(human_traj, rl_traj, n_timepoints)
                if mse is not None:
                    human_rl_mse.append(mse)

        # Convert to numpy arrays for easier manipulation
        human_heuristic_mse = np.array(human_heuristic_mse)
        human_rl_mse = np.array(human_rl_mse)

        # 3. Compute summary statistics
        hh_mean = np.mean(human_heuristic_mse)
        hh_std = np.std(human_heuristic_mse)
        hh_median = np.median(human_heuristic_mse)

        hr_mean = np.mean(human_rl_mse)
        hr_std = np.std(human_rl_mse)
        hr_median = np.median(human_rl_mse)

        print(f"\n" + "=" * 80)
        print("HUMAN SIMILARITY COMPARISON RESULTS")
        print("=" * 80)

        print(f"\nHuman-Heuristic MSE Statistics:")
        print(f"  N comparisons: {len(human_heuristic_mse):,}")
        print(f"  Mean: {hh_mean:.6f}")
        print(f"  Std:  {hh_std:.6f}")
        print(f"  Median: {hh_median:.6f}")

        print(f"\nHuman-RL MSE Statistics:")
        print(f"  N comparisons: {len(human_rl_mse):,}")
        print(f"  Mean: {hr_mean:.6f}")
        print(f"  Std:  {hr_std:.6f}")
        print(f"  Median: {hr_median:.6f}")

        # 4. Statistical tests
        from scipy import stats

        # Test 1: Mann-Whitney U test (non-parametric)
        # H0: The two distributions are the same
        # H1: Human-Heuristic MSE is significantly lower than Human-RL MSE
        w_stat, w_p_value = stats.wilcoxon(
            human_heuristic_mse,
            human_rl_mse,
            alternative='less'  # Test if heuristic MSE is less than RL MSE
        )

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(human_heuristic_mse) - 1) * hh_std ** 2 +
                              (len(human_rl_mse) - 1) * hr_std ** 2) /
                             (len(human_heuristic_mse) + len(human_rl_mse) - 2))
        cohens_d = (hh_mean - hr_mean) / pooled_std

        print(f"\n" + "-" * 60)
        print("HUMAN-HEURISTIC VS HUMAN-RL POSITION MSE STATISTICAL TEST RESULTS")
        print("-" * 60)

        print(f"\nDifference in means:")
        print(f"  Human-Heuristic mean - Human-RL mean = {hh_mean - hr_mean:.6f}")
        print(f"  Relative improvement: {((hr_mean - hh_mean) / hr_mean * 100):.2f}% lower MSE")

        print(f"\n &&&&& Wilcoxon Signed Rank Test (non-parametric):")
        print(f"  W statistic: {w_stat:,.0f}")
        print(f"  p-value: {w_p_value:.2e}")
        print(f"  Result: {'SIGNIFICANT' if w_p_value < 0.05 else 'NOT SIGNIFICANT'} (α = 0.05)")

        print(f"\nEffect Size:")
        print(f"  Cohen's d: {cohens_d:.4f}")

        if abs(cohens_d) < 0.2:
            effect_size_desc = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_size_desc = "small"
        elif abs(cohens_d) < 0.8:
            effect_size_desc = "medium"
        else:
            effect_size_desc = "large"

        print(f"  Effect size magnitude: {effect_size_desc}")

        # 5. Create comprehensive visualizations
        fig = plt.figure(figsize=(20, 16))

        # Create a 3x3 grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Plot 1: Box plot comparison (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        box_data = [human_heuristic_mse, human_rl_mse]
        box_labels = ['Human-Heuristic', 'Human-RL']

        bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')

        ax1.set_ylabel('Position MSE')
        ax1.set_title('MSE Distribution Comparison')
        ax1.grid(True, alpha=0.3)

        # Add statistical annotation
        # ax1.text(0.02, 0.98, f'p = {u_p_value:.2e}\n(Mann-Whitney U)',
        #          transform=ax1.transAxes, fontsize=10, verticalalignment='top',
        #          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Plot 2: Histogram comparison (top middle)
        ax2 = fig.add_subplot(gs[0, 1])

        # Use same bins for both histograms
        min_val = min(np.min(human_heuristic_mse), np.min(human_rl_mse))
        max_val = max(np.max(human_heuristic_mse), np.max(human_rl_mse))
        bins = np.linspace(min_val, max_val, 50)

        ax2.hist(human_heuristic_mse, bins=bins, alpha=0.7, label='Human-Heuristic',
                 color='lightblue', density=True)
        ax2.hist(human_rl_mse, bins=bins, alpha=0.7, label='Human-RL',
                 color='lightcoral', density=True)

        ax2.axvline(hh_mean, color='blue', linestyle='--', linewidth=2,
                    label=f'H-H Mean: {hh_mean:.4f}')
        ax2.axvline(hr_mean, color='red', linestyle='--', linewidth=2,
                    label=f'H-RL Mean: {hr_mean:.4f}')

        ax2.set_xlabel('Position MSE')
        ax2.set_ylabel('Density')
        ax2.set_title('MSE Distribution Histograms')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Cumulative distribution (top right)
        ax3 = fig.add_subplot(gs[0, 2])

        # Sort values for CDF
        hh_sorted = np.sort(human_heuristic_mse)
        hr_sorted = np.sort(human_rl_mse)

        # Compute cumulative probabilities
        hh_p = np.arange(1, len(hh_sorted) + 1) / len(hh_sorted)
        hr_p = np.arange(1, len(hr_sorted) + 1) / len(hr_sorted)

        ax3.plot(hh_sorted, hh_p, label='Human-Heuristic', color='blue', linewidth=2)
        ax3.plot(hr_sorted, hr_p, label='Human-RL', color='red', linewidth=2)

        ax3.set_xlabel('Position MSE')
        ax3.set_ylabel('Cumulative Probability')
        ax3.set_title('Cumulative Distribution Functions')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Add annotation for median crossover
        ax3.axhline(0.5, color='gray', linestyle=':', alpha=0.7)
        ax3.axvline(hh_median, color='blue', linestyle=':', alpha=0.7)
        ax3.axvline(hr_median, color='red', linestyle=':', alpha=0.7)

        # Plot 4: Statistical test results summary (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.axis('off')

        # Create text summary
        test_results_text = f"""
    Statistical Test Results Summary

    Mean Difference:
    • Human-Heuristic: {hh_mean:.6f}
    • Human-RL: {hr_mean:.6f}
    • Difference: {hh_mean - hr_mean:.6f}
    • Improvement: {((hr_mean - hh_mean) / hr_mean * 100):.2f}%

    Mann-Whitney U Test:
    • p-value: {w_p_value:.2e}
    • Result: {'✓ SIGNIFICANT' if w_p_value < 0.05 else '✗ NOT SIGNIFICANT'}

    Effect Size (Cohen's d):
    • Value: {cohens_d:.4f}
    • Magnitude: {effect_size_desc}

    Sample Sizes:
    • Human-Heuristic: {len(human_heuristic_mse):,}
    • Human-RL: {len(human_rl_mse):,}
        """

        ax4.text(0.05, 0.95, test_results_text, transform=ax4.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        # Plot 5: Q-Q plot (middle center)
        ax5 = fig.add_subplot(gs[1, 1])

        # Sample data for Q-Q plot if datasets are very large
        if len(human_heuristic_mse) > 10000:
            hh_sample = np.random.choice(human_heuristic_mse, 10000, replace=False)
        else:
            hh_sample = human_heuristic_mse

        if len(human_rl_mse) > 10000:
            hr_sample = np.random.choice(human_rl_mse, 10000, replace=False)
        else:
            hr_sample = human_rl_mse

        stats.probplot(hh_sample, dist="norm", plot=ax5)
        ax5.set_title('Q-Q Plot: Human-Heuristic MSE vs Normal')
        ax5.grid(True, alpha=0.3)

        # Plot 6: Scatter plot of random sample pairs (middle right)
        ax6 = fig.add_subplot(gs[1, 2])

        # Take a random sample for visualization
        n_sample = min(1000, len(human_heuristic_mse), len(human_rl_mse))
        hh_sample_idx = np.random.choice(len(human_heuristic_mse), n_sample, replace=False)
        hr_sample_idx = np.random.choice(len(human_rl_mse), n_sample, replace=False)

        ax6.scatter(human_heuristic_mse[hh_sample_idx], human_rl_mse[hr_sample_idx],
                    alpha=0.5, s=20)

        # Add diagonal line
        min_val = min(np.min(human_heuristic_mse[hh_sample_idx]),
                      np.min(human_rl_mse[hr_sample_idx]))
        max_val = max(np.max(human_heuristic_mse[hh_sample_idx]),
                      np.max(human_rl_mse[hr_sample_idx]))
        ax6.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7,
                 label='Equal MSE line')

        ax6.set_xlabel('Human-Heuristic MSE')
        ax6.set_ylabel('Human-RL MSE')
        ax6.set_title('MSE Correlation Scatter Plot\n(Random Sample)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # Plot 7: Percentile comparison (bottom left)
        ax7 = fig.add_subplot(gs[2, 0])

        percentiles = np.arange(5, 100, 5)
        hh_percentiles = np.percentile(human_heuristic_mse, percentiles)
        hr_percentiles = np.percentile(human_rl_mse, percentiles)

        ax7.plot(percentiles, hh_percentiles, 'o-', label='Human-Heuristic',
                 color='blue', linewidth=2)
        ax7.plot(percentiles, hr_percentiles, 's-', label='Human-RL',
                 color='red', linewidth=2)

        ax7.set_xlabel('Percentile')
        ax7.set_ylabel('MSE Value')
        ax7.set_title('Percentile Comparison')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # Plot 8: Effect size visualization (bottom middle)
        ax8 = fig.add_subplot(gs[2, 1])

        # Create overlapping distributions to show effect size
        x_range = np.linspace(min_val, max_val, 1000)

        # Approximate as normal distributions for visualization
        hh_pdf = stats.norm.pdf(x_range, hh_mean, hh_std)
        hr_pdf = stats.norm.pdf(x_range, hr_mean, hr_std)

        ax8.plot(x_range, hh_pdf, label='Human-Heuristic', color='blue', linewidth=2)
        ax8.plot(x_range, hr_pdf, label='Human-RL', color='red', linewidth=2)
        ax8.fill_between(x_range, hh_pdf, alpha=0.3, color='blue')
        ax8.fill_between(x_range, hr_pdf, alpha=0.3, color='red')

        ax8.axvline(hh_mean, color='blue', linestyle='--', alpha=0.7)
        ax8.axvline(hr_mean, color='red', linestyle='--', alpha=0.7)

        ax8.set_xlabel('Position MSE')
        ax8.set_ylabel('Probability Density')
        ax8.set_title(f'Effect Size Visualization\n(Cohen\'s d = {cohens_d:.3f})')
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        # Plot 9: Hypothesis test conclusion (bottom right)
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        # Determine overall conclusion
        if w_p_value < 0.05 and hh_mean < hr_mean:
            conclusion = "✓ HYPOTHESIS SUPPORTED"
            conclusion_color = 'green'
            conclusion_detail = "Heuristic agents are significantly\nmore similar to humans than RL agents"
        elif w_p_value >= 0.05:
            conclusion = "? INCONCLUSIVE"
            conclusion_color = 'orange'
            conclusion_detail = "No significant difference found\nbetween agent similarities"
        else:
            conclusion = "✗ HYPOTHESIS REJECTED"
            conclusion_color = 'red'
            conclusion_detail = "RL agents are more similar\nto humans than heuristic agents"

        conclusion_text = f"""
    FINAL CONCLUSION

    {conclusion}

    {conclusion_detail}

    Key Evidence:
    • Mean MSE difference: {hh_mean - hr_mean:.6f}
    • Statistical significance: {w_p_value:.2e}
    • Effect size: {effect_size_desc} ({cohens_d:.3f})

    Confidence Level: 95%
    (α = 0.05)
        """

        ax9.text(0.5, 0.5, conclusion_text, transform=ax9.transAxes, fontsize=12,
                 horizontalalignment='center', verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor=conclusion_color, alpha=0.2))

        plt.suptitle('Human Similarity Comparison: Heuristic vs RL Agents\nPosition Trajectory MSE Analysis',
                     fontsize=16, fontweight='bold')

        # Save the comprehensive plot
        plot_path = os.path.join(save_dir, "human_similarity_comparison_comprehensive.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()

        # 6. Save detailed results
        results = {
            'summary_statistics': {
                'human_heuristic_mse': {
                    'mean': float(hh_mean),
                    'std': float(hh_std),
                    'median': float(hh_median),
                    'n_comparisons': int(len(human_heuristic_mse)),
                    'min': float(np.min(human_heuristic_mse)),
                    'max': float(np.max(human_heuristic_mse)),
                    'q25': float(np.percentile(human_heuristic_mse, 25)),
                    'q75': float(np.percentile(human_heuristic_mse, 75))
                },
                'human_rl_mse': {
                    'mean': float(hr_mean),
                    'std': float(hr_std),
                    'median': float(hr_median),
                    'n_comparisons': int(len(human_rl_mse)),
                    'min': float(np.min(human_rl_mse)),
                    'max': float(np.max(human_rl_mse)),
                    'q25': float(np.percentile(human_rl_mse, 25)),
                    'q75': float(np.percentile(human_rl_mse, 75))
                }
            },
            'statistical_tests': {
                'wilcoxon_signed_rank': {
                    'statistic': float(w_stat),
                    'p_value': float(w_p_value),
                    'significant': bool(w_p_value < 0.05),
                    'interpretation': 'Human-Heuristic MSE is significantly lower' if w_p_value < 0.05 and hh_mean < hr_mean else 'No significant difference or opposite effect'
                },

            },
            'effect_size': {
                'cohens_d': float(cohens_d),
                'magnitude': effect_size_desc,
                'interpretation': f"{'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small' if abs(cohens_d) > 0.2 else 'Negligible'} practical difference"
            },
            'hypothesis_test': {
                'hypothesis': 'Human-Heuristic MSE < Human-RL MSE',
                'supported': bool(w_p_value < 0.05 and hh_mean < hr_mean),
                'confidence_level': 0.95,
                'conclusion': conclusion
            },
            'practical_significance': {
                'mean_difference': float(hh_mean - hr_mean),
                'percent_improvement': float((hr_mean - hh_mean) / hr_mean * 100) if hr_mean > 0 else None,
                'median_difference': float(hh_median - hr_median)
            }
        }

        # Save results to JSON
        results_path = os.path.join(save_dir, "human_similarity_comparison_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Create summary CSV
        summary_data = [{
            'Comparison': 'Human-Heuristic vs Human-RL',
            'H_Heuristic_Mean': hh_mean,
            'H_RL_Mean': hr_mean,
            'Mean_Difference': hh_mean - hr_mean,
            'Percent_Improvement': (hr_mean - hh_mean) / hr_mean * 100 if hr_mean > 0 else None,
            'Mann_Whitney_p': w_p_value,
            'Significant': w_p_value < 0.05,
            'Cohens_d': cohens_d,
            'Effect_Size': effect_size_desc,
            'N_HH_Comparisons': len(human_heuristic_mse),
            'N_HR_Comparisons': len(human_rl_mse),
            'Hypothesis_Supported': w_p_value < 0.05 and hh_mean < hr_mean
        }]

        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(save_dir, "human_similarity_comparison_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print(f"\n" + "=" * 80)
        print("FINAL CONCLUSION")
        print("=" * 80)
        print(f"{conclusion}")
        print(f"\nKey findings:")
        print(f"• Average Human-Heuristic MSE: {hh_mean:.6f}")
        print(f"• Average Human-RL MSE: {hr_mean:.6f}")
        print(f"• Difference: {hh_mean - hr_mean:.6f} ({((hr_mean - hh_mean) / hr_mean * 100):.2f}% improvement)")
        print(f"• Statistical significance: p = {w_p_value:.2e}")
        print(f"• Effect size: {effect_size_desc} (Cohen's d = {cohens_d:.3f})")

        print(f"\nFiles saved:")
        print(f"  Comprehensive plot: {plot_path}")
        print(f"  Detailed results: {results_path}")
        print(f"  Summary table: {summary_path}")

        return results










    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################

    def run_analysis(self):
        load_saved = True

        self.config = load_env_config('../../configs/Monolith_index_August.json')
        self.config['use_stuck_detection'] = False
        self.config['prob_detect'] = 0  # 0.0003
        self.config['action_type'] = 'Discrete16'
        self.config['league_type'] = 'strategy_diverse'

        # Load trajectories
        if load_saved:
            self.human_trajectories = load_saved_trajectories('human')
            self.heuristic_trajectories = load_saved_trajectories('strategy', subsample_episodes=1)
            self.rl_trajectories = load_saved_trajectories('rl')

        else:
            #self.human_trajectories_for_training = self.process_human_trajectories()
            #self.heuristic_trajectories = self.generate_strategy_trajectories("heuristic_trajectories_fulltrajectories.json", full_trajectories=True)
            self.rl_agents_path = 'rl_agents'  # './exp1_offline_study/offline_study_testing_agents/'  #'./trained_models/pretrained_teammates/' # Where the RL agent .zip and .pkl files are stored
            #self.rl_trajectories = self.generate_rl_trajectories('exp3_similarity_analysis/rl_agents', "rl_trajectories.json")
            self.rl_trajectories = self.generate_rl_trajectories('rl_agents', "rl_trajectories_nondeterministic.json", deterministic=False)

            #self.rl_trajectories = self.generate_rl_trajectories('exp3_similarity_analysis/rl_agents_2', "rl_trajectories_2.json")
        
        # Analyze similarity of position trajectories
        self.compare_position_heatmaps_2d(self.human_trajectories, self.rl_trajectories, self.heuristic_trajectories) # Ready to test

        # mse_results = self.analyze_progress_rate_mse(
        #     self.human_trajectories_for_training,
        #     self.rl_trajectories,
        #     self.heuristic_trajectories
        # )
        # #
        # cross_mse_results = self.analyze_cross_trajectory_position_mse(self.human_trajectories_for_training,
        #     self.rl_trajectories,
        #     self.heuristic_trajectories)

        # similarity_results = self.analyze_human_similarity_comparison(
        #     self.human_trajectories_for_training,
        #     self.rl_trajectories,
        #     self.heuristic_trajectories
        # )
        #
        # #Analyze metric similarity
        #self.analyze_metric_similarity()


        #####################################

        # Analyze threat-target priority clusters (NOT USED)
        #self.compute_silhouette_scores(self.human_trajectories_for_training, self.rl_trajectories, self.heuristic_trajectories) # Ready to test
        
        # Analyze similarity of action distributions (NOT USED)
        #self.analyze_action_distributions(self.human_trajectories_for_training, self.heuristic_trajectories, self.rl_trajectories) # Ready to test

        # self.analyze_temporal_silhouette_evolution( # NOT USED
        #     self.human_trajectories_for_training,
        #     self.rl_trajectories,
        #     self.heuristic_trajectories
        # )




if __name__ == '__main__':
    analyzer = SimilarityAnalysis()
    analyzer.load_level_layouts()
    analyzer.run_analysis()