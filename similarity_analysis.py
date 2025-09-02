import ctypes
from dataclasses import dataclass
from typing import List, Tuple, Dict
import os, json, math
import glob

import pygame
from scipy.optimize import linear_sum_assignment
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

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
from scipy.stats import mannwhitneyu, pearsonr

# For action distribution comparison
from scipy.stats import chisquare

from env_multi_new import MAISREnvVec
from utility.data_logging import load_env_config
from utility.league_management import LocalSearch, GoToNearestThreat, ChangeRegions, GenericTeammatePolicy, TargetSearchLocalTSP, HeuristicAgent
from utility.localsearch_training_wrapper import MaisrLocalSearchWrapper


def load_vecnormalize_wrapper(vecnorm_path, env):
    """Load saved VecNormalize wrapper with stats from training and apply it to the new environment."""
    print(f"Loading VecNormalize stats from: {vecnorm_path}")

    vec_normalize = VecNormalize.load(vecnorm_path, venv=env)
    vec_normalize.training = False  # Disable further normalization updates
    vec_normalize.norm_reward = False
    return vec_normalize

def save_trajectories_to_json(trajectories, output_file):
    """Save trajectories to JSON with proper numpy array handling"""
    serializable_data = []
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


def load_saved_trajectories(trajectory_type: str):
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
            'file': 'human_trajectories.json',
            'attr': 'human_trajectories'
        },
        'rl': {
            'file': 'rl_trajectories.json',
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
    file_path = os.path.join('./similarity_analysis', file_info['file'])

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Trajectory file not found: {file_path}")

    with open(file_path, 'r') as f:
        trajectory_data = json.load(f)

    # Convert JSON data back to Trajectory objects
    trajectories = []
    for traj_dict in trajectory_data:
        trajectory = Trajectory(
            category=traj_dict['category'],
            level=traj_dict['level'],
            name=traj_dict['name'],
            positions=[tuple(pos) if isinstance(pos, list) else pos for pos in traj_dict.get('positions', [])],
            actions=traj_dict.get('actions', []),
            target_ids=traj_dict.get('target_ids', []),
            threat_ids=traj_dict.get('threat_ids', [])
        )
        trajectories.append(trajectory)

    print(f"Successfully loaded {len(trajectories)} {trajectory_type} trajectories from {file_path}")
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

        base_env.teammate_active = False # TODO make sure this is good

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


class SimilarityAnalysis:
    def __init__(self):
        self.human_trajectories_path = './userstudy_logs/' # Where the human trajectory json files are stored
        self.rl_agents_path = './similarity_analysis/rl_agents'      #'./offline_study/offline_study_testing_agents/'  #'./trained_models/pretrained_teammates/' # Where the RL agent .zip and .pkl files are stored
         
        self.num_rl_agents = 32
        self.level_list = [1, 3, 5, 6, 7]
	
        self.human_trajectories = []
        self.heuristic_trajectories = []
        self.rl_trajectories = []

    # Ready to test    
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
                level_match = re.search(r'[ABCPS](\d+)', filename)
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
                        # Use the cumulative totals if available
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

        out_file = os.path.join('./similarity_analysis', "human_trajectories.json")
        save_trajectories_to_json(self.human_trajectories, out_file)
        return self.human_trajectories


    def generate_rl_trajectories(self):
        render = False

        rl_trajectories = []
        agent_pairs = []

        # Step 1: Find all RL agent .zip and .pkl pairs
        model_patterns = [
            os.path.join(self.rl_agents_path, "*_model.zip"),
            os.path.join(self.rl_agents_path, "**/*_model.zip")
        ]

        normstats_patterns = [
            os.path.join(self.rl_agents_path, "*_vecnormalize.pkl"),
            os.path.join(self.rl_agents_path, "**/*_vecnormalize.pkl")
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

            agent_pairs.append((model, norm_stats_path))
            
        # Step 2: Run each agent in each level and save the trajectory
        for agent_model, pkl_path in agent_pairs:
            print(f'Generating trajectory for RL agent {agent_model}, {pkl_path}')
            
            # Extract seed (number after "seed")
            seed_match = re.search(r"seed(\d+)", pkl_path)
            seed = seed_match.group(1) if seed_match else "000"
            
            for level in self.level_list:
                config = self.config.copy()
                config['force_specific_level'] = level

                if render:
                    env = DummyVecEnv([make_wrapped_env(config, clock=clock, window=window) for _ in range(1)])
                else:
                    env = DummyVecEnv([make_wrapped_env(config) for _ in range(1)])

                env = load_vecnormalize_wrapper(pkl_path, env)

                trajectory = Trajectory(name = f'seed{seed}', level = level, category = 'rl', actions = [], positions = [], target_ids = [], threat_ids = []) # instantiate the trajectory
                step_count = 0
                obs = env.reset()
                done = False
                base_env = env.envs[0].env
                base_env.env.agents[1].appearance = 'invisible'  # Forces a hold # TODO see if this is consistent with the others

                while not done:
                    agent_action, _ = agent_model.predict(obs, deterministic=True)
                    
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
                    step_count += 1
                        
                rl_trajectories.append(trajectory)
            
        # 3. Save the list of trajectories to a file type of your choice so we don't have regenerate it if we need to re-run
        out_file = os.path.join('./similarity_analysis', "rl_trajectories.json")

        save_trajectories_to_json(rl_trajectories, out_file)

        #with open(out_file, "w") as f:
            #json.dump([traj.__dict__ for traj in rl_trajectories], f, indent=2)
        
        return rl_trajectories
        
        
    # TODO test
    def generate_strategy_trajectories(self):
        # Step 1: Create list of heuristic agent parameter combinations. Each element in the list is itself a list of three strings (risk_tolerance, action_noise, spatial_coordination)
        risk_tolerance = ['high'] #["low", "medium", "high", "max_greedy"]
        action_noise = ["stable", "noisy", "very_noisy"]
        planning_horizon = ['greedy', 'clusters']
        spatial_coordination = [False, True]
        decision_speed = ['fast', 'slow']
        combinations = [list(p) for p in itertools.product(risk_tolerance, action_noise, spatial_coordination)]
        print(f'Generated {len(combinations)} strategy combinations')

        render = True
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
            risk_tolerance, action_noise, spatial_coordination = combination
            
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

                teammate.env = base_env

                trajectory = Trajectory(name = f'{risk_tolerance}-{action_noise}_{spatial_coordination}', level = level, category = 'heuristic', actions = [], positions = [], target_ids = [], threat_ids = []) # instantiate the trajectory
                step_count = 0
                obs = env.reset()
                done = False

                #print(f'base_env is {base_env} (should be MaisrEnvVec, NOT LocalSearchWrapper\n\n%%%')
                
                while not done:
                    agent0_action = 0
                    base_env.env.agents[0].appearance = 'invisible' # Forces a hold

                    agent1_waypoint = base_env.get_teammate_action()#teammate_action # This is a waypoint

                    current_pos = (base_env.env.agents[1].x, base_env.env.agents[1].y)
                    agent1_action = waypoint_to_direction_index(current_pos, agent1_waypoint)
                    
                    obses, rewards, dones, infos = env.step([agent0_action])
                    obs = obses[0]
                    reward = rewards[0]
                    info = infos[0]
                    done = dones[0]

                    if render: base_env.env.render()
                        
                    trajectory.actions.append(agent1_action)
                    trajectory.positions.append((base_env.env.agents[1].x, base_env.env.agents[1].y))
                    trajectory.target_ids.append(base_env.env.targets_identified)
                    trajectory.threat_ids.append(base_env.env.num_threats_identified)
                    step_count += 1
                        
                self.heuristic_trajectories.append(trajectory)

        # 3. Save the list of trajectories to a file type of your choice so we don't have regenerate it if we need to re-run
        out_file = os.path.join('./similarity_analysis', "strategy_trajectories.json")
        save_trajectories_to_json(self.heuristic_trajectories, out_file)

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
    
    # Ready to test
    def compare_position_heatmaps_2d(self, human_trajectories: List[Trajectory], rl_trajectories: List[Trajectory], 
                                     heuristic_trajectories: List[Trajectory], bins: int = 50, 
                                     output_csv: str = "heatmap_emd_results_2d.csv"):
        """ References/justification for using 2D EMD for this analysis:
            https://stats.stackexchange.com/questions/404775/calculate-earth-movers-distance-for-two-grayscale-images
            https://stats.stackexchange.com/questions/659384/compute-p-value-of-earth-movers-distance-score-comparing-two-heatmaps-in-r
        """
        
        # --- 1. Aggregate positions ---
        #def extract_positions(trajs: List[Trajectory]) -> np.ndarray:
            #return np.array([pos for t in trajs for pos in t.positions])

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
            print(f"Using {sample_size} out of {len(human_trajectories)} human trajectories ({human_sample_fraction:.1%})")
        else:
            sampled_human_trajectories = human_trajectories
            print(f"Using all {len(human_trajectories)} human trajectories")


        #human_positions = extract_positions(sampled_human_trajectories)
        human_positions = extract_positions(sampled_human_trajectories, subsample_every_n=10)
        rl_positions = extract_positions(rl_trajectories)
        heuristic_positions = extract_positions(heuristic_trajectories)

        # --- 2. Define common grid for all heatmaps ---
        #all_positions = np.vstack([human_positions, rl_positions, heuristic_positions])
        x_min, y_min = -500, -500 #np.min(all_positions, axis=0)
        x_max, y_max = 500, 500 #np.max(all_positions, axis=0)

        def compute_heatmap(positions):
            print(positions)
            heatmap, _, _ = np.histogram2d(
                positions[:,0], positions[:,1],
                bins=bins,
                range=[[x_min, x_max], [y_min, y_max]]
            )
            return heatmap

        human_heatmap = compute_heatmap(human_positions)
        rl_heatmap = compute_heatmap(rl_positions)
        heuristic_heatmap = compute_heatmap(heuristic_positions)

        # --- 3. Compute pairwise 2D EMD ---
        emd_results = {
            'human_vs_rl': self.compute_2d_emd(human_heatmap, rl_heatmap),
            'human_vs_heuristic': self.compute_2d_emd(human_heatmap, heuristic_heatmap),
            'rl_vs_heuristic': self.compute_2d_emd(rl_heatmap, heuristic_heatmap),
        }

        # --- 4. Output results ---
        print("Pairwise 2D Heatmap EMD Results:")
        for k, v in emd_results.items():
            print(f"{k}: {v:.6f}")

        pd.DataFrame([emd_results]).to_csv(output_csv, index=False)
        
        # --- 5. Plot the three heatmaps ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        heatmaps = [human_heatmap, rl_heatmap, heuristic_heatmap]
        titles = ['Human', 'RL Agent', 'Heuristic']

        for ax, hm, title in zip(axes, heatmaps, titles):
            im = ax.imshow(hm, origin='lower', aspect='auto',
                           extent=[x_min, x_max, y_min, y_max], cmap='hot')
            ax.set_title(f"{title} Heatmap")
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig('./similarity_analysis/position_heatmaps.png', dpi=300)
        plt.close(fig)
            
        return emd_results
        
    
    # Ready to test
    def compute_silhouette_scores(self,
        human_trajectories: List[Trajectory],
        rl_trajectories: List[Trajectory],
        heuristic_trajectories: List[Trajectory],
        save_dir: str = "analysis_outputs"
    ):
        """
        Compute and visualize silhouette scores for human, RL, and heuristic gameplay trajectories.
        Clusters are based on the final number of targets and threats identified per trajectory.
        """
        os.makedirs(save_dir, exist_ok=True)

        # 1. Build the feature matrix: (targets_final, threats_final)
        def extract_features(trajs):
            return np.array([
                [t.target_ids[-1], t.threat_ids[-1]]
                for t in trajs
                if t.target_ids and t.threat_ids
            ])

        human_features = extract_features(human_trajectories)
        rl_features = extract_features(rl_trajectories)
        heuristic_features = extract_features(heuristic_trajectories)

        # Combine for plotting
        all_features = np.vstack([human_features, rl_features, heuristic_features])
        all_labels = (
            ["human"] * len(human_features)
            + ["rl"] * len(rl_features)
            + ["heuristic"] * len(heuristic_features)
        )

        # 2. Create scatter plot
        colors = {"human": "blue", "rl": "green", "heuristic": "red"}
        plt.figure(figsize=(8, 6))
        for label, features in zip(
            ["human", "rl", "heuristic"],
            [human_features, rl_features, heuristic_features]
        ):
            if len(features) > 0:
                plt.scatter(
                    features[:, 0], features[:, 1],
                    c=colors[label], label=label, alpha=0.7, edgecolors='k'
                )
        plt.xlabel("Final # Targets Identified")
        plt.ylabel("Final # Threats Identified")
        plt.title("Trajectory Outcome Clusters (Targets vs Threats)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "trajectory_clusters.png"))
        plt.close()

        # 3. Compute silhouette scores
        # Silhouette score requires >= 2 clusters
        def get_silhouette_score(features, n_clusters=None):
            if len(features) < 2:
                return None  # Can't compute silhouette for <2 samples
            if n_clusters is None:
                n_clusters = min(2, len(features))  # fallback
            # Standardize features to improve cluster separation
            X = StandardScaler().fit_transform(features)
            # Use KMeans for clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
            labels = kmeans.fit_predict(X)
            if len(set(labels)) < 2:
                return None  # silhouette score undefined for single cluster
            return silhouette_score(X, labels)

        silhouette_results = {
            "human_only": get_silhouette_score(human_features),
            "rl_only": get_silhouette_score(rl_features),
            "heuristic_only": get_silhouette_score(heuristic_features),
            "human_rl": get_silhouette_score(np.vstack([human_features, rl_features])),
            "human_heuristic": get_silhouette_score(np.vstack([human_features, heuristic_features])),
        }

        # 4. Save results
        results_path = os.path.join(save_dir, "silhouette_scores.txt")
        with open(results_path, "w") as f:
            for k, v in silhouette_results.items():
                f.write(f"{k}: {v}\n")

        return silhouette_results       

    # Ready to test
    def compare_progress_rates(self, human_trajectories: List[Trajectory], rl_trajectories: List[Trajectory],
                heuristic_trajectories: List[Trajectory], save_dir: str = "analysis_outputs"):
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

    # Ready to test
    def analyze_action_distributions(self, human_trajectories, rl_trajectories, heuristic_trajectories, save_dir="analysis_outputs"):
        """
        Input: 
            human_trajectories: list of Trajectory objects
            rl_trajectories: list of Trajectory objects
            heuristic_trajectories: list of Trajectory objects
            
        Process:
            1. Generate histograms of action distributions (Frequency of each discrete action 0–15) for 
               human, RL, and strategy agents. 
            2. Plot all three histograms on a 3x1 matplotlib plot.
            3. Perform pairwise chi-squared comparisons between all three histograms.
            
        Output:
            - Prints pairwise chi-squared statistics
            - Shows and saves the histogram figure
        """
        
        os.makedirs(save_dir, exist_ok=True)
        n_actions = 16  # Actions are 0–15

        # Helper function to aggregate all actions for a group
        def get_action_hist(trajectories):
            all_actions = []
            for traj in trajectories:
                all_actions.extend(traj.actions)
            hist, _ = np.histogram(all_actions, bins=np.arange(n_actions+1)-0.5)
            return hist

        # Compute histograms
        human_hist = get_action_hist(human_trajectories)
        rl_hist = get_action_hist(rl_trajectories)
        strategy_hist = get_action_hist(heuristic_trajectories)

        # Normalize for chi-squared to avoid zeros
        # (Add small epsilon to avoid division by zero)
        eps = 1e-6
        human_hist += eps
        rl_hist += eps
        strategy_hist += eps

        # Plot histograms
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        categories = ['Human', 'RL Agent', 'Heuristic Strategy Agent']
        hists = [human_hist, rl_hist, strategy_hist]
        
        for ax, hist, cat in zip(axes, hists, categories):
            ax.bar(range(n_actions), hist, color='skyblue', edgecolor='black')
            ax.set_title(f'{cat} Action Distribution')
            ax.set_ylabel('Frequency')
            ax.set_xticks(range(n_actions))
            ax.grid(axis='y', linestyle='--', alpha=0.7)

        axes[-1].set_xlabel('Action Index')

        plt.tight_layout()
        save_path = os.path.join(save_dir, "action_distribution_histograms.png")
        plt.savefig(save_path)
        plt.show()

        # Perform pairwise chi-squared comparisons
        # Use the first histogram as observed and second as expected for chi-squared
        chi_human_rl = chisquare(f_obs=human_hist, f_exp=rl_hist)
        chi_human_strategy = chisquare(f_obs=human_hist, f_exp=strategy_hist)
        chi_rl_strategy = chisquare(f_obs=rl_hist, f_exp=strategy_hist)

        print("Chi-squared comparisons:")
        print(f"Human vs RL: χ² = {chi_human_rl.statistic:.3f}, p = {chi_human_rl.pvalue:.3e}")
        print(f"Human vs Strategy: χ² = {chi_human_strategy.statistic:.3f}, p = {chi_human_strategy.pvalue:.3e}")
        print(f"RL vs Strategy: χ² = {chi_rl_strategy.statistic:.3f}, p = {chi_rl_strategy.pvalue:.3e}")
        print(f"Histograms saved to: {save_path}")
        
    
    # TODO: Add the target-based metrics. Then test.
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
                "smoothness": [], "switch_rate": [],
                "flying_toward_nearest_rate": [], "average_target_distance": []
            },
            "rl": {
                "smoothness": [], "switch_rate": [],
                "flying_toward_nearest_rate": [], "average_target_distance": []
            },
            "heuristic": {
                "smoothness": [], "switch_rate": [],
                "flying_toward_nearest_rate": [], "average_target_distance": []
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
            metrics[traj.category]["smoothness"].append(smoothness)
            metrics[traj.category]["switch_rate"].append(switch_rate)
            metrics[traj.category]["flying_toward_nearest_rate"].append(flying_toward_nearest_rate)
            metrics[traj.category]["average_target_distance"].append(average_target_distance)

        # --- Step 2: Perform pairwise statistical tests ---
        def pairwise_tests(metric_name):
            human_vals = [v for v in metrics["human"][metric_name] if v is not None]
            rl_vals = [v for v in metrics["rl"][metric_name] if v is not None]
            heuristic_vals = [v for v in metrics["heuristic"][metric_name] if v is not None]

            pairs = [
                ("human", "rl", human_vals, rl_vals),
                ("human", "heuristic", human_vals, heuristic_vals),
                ("rl", "heuristic", rl_vals, heuristic_vals),
            ]
            results = []
            for name1, name2, data1, data2 in pairs:
                if len(data1) > 0 and len(data2) > 0:
                    stat, p = mannwhitneyu(data1, data2, alternative="two-sided")
                    results.append(f"{metric_name} {name1} vs {name2}: U={stat:.2f}, p={p:.4f}")
            return results

        print("\n--- Pairwise Statistical Tests ---")
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
            plt.boxplot([human_vals, rl_vals, heuristic_vals], labels=["Human", "RL", "Heuristic"])
            plt.title(f"{metric_name.replace('_',' ').capitalize()} by Agent Type")
            plt.ylabel(metric_name.replace('_',' ').capitalize())
            plt.grid(True, alpha=0.3)
            plt.show()
    
    
    def run_analysis(self):
        load_saved = False

        self.config = load_env_config('configs/Monolith_index_August.json')
        self.config['use_stuck_detection'] = False
        self.config['prob_detect'] = 0  # 0.0003
        self.config['action_type'] = 'Discrete16'

        # Load trajectories
        if load_saved:
            self.human_trajectories = load_saved_trajectories('human')
            self.heuristic_trajectories = load_saved_trajectories('strategy')
            self.rl_trajectories = load_saved_trajectories('rl')

        else:
            #self.human_trajectories = self.process_human_trajectories()
            self.heuristic_trajectories = self.generate_strategy_trajectories()
            #self.rl_trajectories = self.generate_rl_trajectories()
        
        # Analyze similarity of position trajectories
        self.compare_position_heatmaps_2d(self.human_trajectories, self.heuristic_trajectories, self.rl_trajectories) # Ready to test
        
        # Analyze threat-target priority clusters
        self.compute_silhouette_scores(self.human_trajectories, self.heuristic_trajectories, self.rl_trajectories) # Ready to test
        
        # Analyze rate of identifying threats and targets throughout the episode # TODO Find DTW package
        #self.compare_progress_rates(self.human_trajectories, self.heuristic_trajectories, self.rl_trajectories) # Ready to test
        
        # Analyze metric similarity
        self.analyze_metric_similarity() # Has a TODO, then test.
        
        # Analyze similarity of action distributions
        self.analyze_action_distributions(self.human_trajectories, self.heuristic_trajectories, self.rl_trajectories) # Ready to test
        
        # Final data to return and save
        # 1. 


if __name__ == '__main__':
    analyzer = SimilarityAnalysis()
    analyzer.run_analysis()