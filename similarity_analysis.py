from dataclasses import dataclass
from typing import List, Tuple
import os, json, math
from glob import glob
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
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
from fastdtw import fastdtw
from scipy.stats import mannwhitneyu, pearsonr

# For action distribution comparison
from scipy.stats import chisquare


@dataclass
class Trajectory:
    category: str  # 'human', 'rl', or 'heuristic'
    level: int     # Integer level index
    name: str        # Subject ID or agent strategy identifier

    positions: List[Tuple[float, float]]  # List of (x, y) tuples per timestep
    actions: List[int]                    # List of actions (0–15) per timestep
    target_ids: List[int]                 # History of targets identified per timestep. 
    threat_ids: List[int]                 # History of threats identified per timestep


class SimilarityAnalysis()
	def __init__():
		self.human_trajectories_path = '/placeholderpath/' # Where the human trajectory json files are stored
		self.rl_agents_path = '/trained_models/pretrained_teammates/' # Where the RL agent .zip and .pkl files are stored
		
		self.num_rl_agents = 32
        self.level_list = [1, 3, 5, 6, 7]
		
		self.human_trajectories = None
		self.strategy_agent_trajectories = None
		self.rl_agent_trajectories = None

    # Ready to test
    # TODO  use vectors 
    def process_human_trajectories(self):
        direction_vectors = [
            (0, 1), (0.383, 0.924), (0.707, 0.707), (0.924, 0.383),
            (1, 0), (0.924, -0.383), (0.707, -0.707), (0.383, -0.924),
            (0, -1), (-0.383, -0.924), (-0.707, -0.707), (-0.924, -0.383),
            (-1, 0), (-0.924, 0.383), (-0.707, 0.707), (-0.383, 0.924)
        ]

        def vector_to_action(dx, dy):
            angle = math.atan2(dy, dx)  # radians [-pi, pi], 0 along +x
            angle_deg = (math.degrees(angle) + 360) % 360
            # map 0 deg = east (index 4), but we want 0 deg = north (index 0)
            # shift so that 0 deg = north
            angle_deg = (angle_deg - 90) % 360
            sector = int((angle_deg + 11.25) // 22.5) % 16
            return sector

        self.human_trajectories = []

        for subject_dir in sorted(os.listdir(self.human_trajectories_path)):
            subject_path = os.path.join(self.human_trajectories_path, subject_dir)
            if not os.path.isdir(subject_path):
                continue

            for json_file in sorted(glob(os.path.join(subject_path, "*.json"))):
                level_str = os.path.basename(json_file).split("_")[1]  # e.g. A1
                level = int(level_str[1:])

                with open(json_file, "r") as f:
                    data = json.load(f)

                positions, actions = [], []
                target_counts, threat_counts = [], []
                cumulative_targets = 0
                cumulative_threats = 0

                for step in data:
                    pos = tuple(step["human_position"])
                    positions.append(pos)

                    # compute action
                    wp = step.get("human_custom_waypoint", pos)
                    dx, dy = wp[0] - pos[0], wp[1] - pos[1]
                    action_idx = vector_to_action(dx, dy)
                    actions.append(action_idx)

                    # count events
                    t_events = sum(1 for val in step.get("target_identified", []) if val)
                    h_events = sum(1 for val in step.get("threat_identified", []) if val)
                    cumulative_targets += t_events
                    cumulative_threats += h_events
                    target_counts.append(cumulative_targets)
                    threat_counts.append(cumulative_threats)

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

        return self.human_trajectories

    # TODO:
    # - Decide what agent 1 is. Nonexistent?
    # - Test
    def generate_rl_trajectories(self):
        
        rl_trajectories = []

        # Step 1: Find all RL agent .zip and .pkl pairs
        rl_files = glob(os.path.join(self.rl_agents_path, "*.zip"))
        agent_pairs = []
        for zip_path in rl_files:
        # Extract prefix up to and including 'checkpoint_'
        filename = os.path.basename(zip_path)
        match = re.match(r"(.+_checkpoint_)\d+_steps\.zip$", filename)
        if not match:
            print(f"Skipping unrecognized zip name: {filename}")
            continue

        prefix = match.group(1)  # e.g., "pretrainP_0731_1530_seed42_checkpoint_"

        # Construct search pattern for pkl
        pkl_pattern = os.path.join(self.rl_agents_path, prefix + "*_vecnormalize_*.pkl")
        pkl_files = glob(pkl_pattern)

        if not pkl_files:
            raise ValueError(f"Warning: No pkl found for {zip_path}")
            continue

        # Pair the first match (or choose based on step count if multiple)
        agent_pairs.append((zip_path, pkl_files[0]))
            
        # Step 2: Run each agent in each level and save the trajectory
        for zip_path, pkl_path in agent_pairs:
            
            # Load RL model
            agent_model = PPO.load(zip_path)
            
            # Extract seed (number after "seed")
            seed_match = re.search(r"seed(\d+)", zip_path)
            seed = seed_match.group(1) if seed_match else "000"
            
            for level in range(7):
                config = self.config.copy()
                config['force_specific_level'] = level_idx
                
                env = DummyVecEnv([make_wrapped_env(config) for _ in range(1)])
                
                trajectory = Trajectory(name = f'seed{seed}', level = level, category = 'rl') # instantiate the trajectory 
                step_count = 0
                obs = env.reset()
                base_env = env.envs[0].env # TODO confirm that this will update live as the original env does. Is it a shallow copy?
                while not done:
                    agent_action = agent_model.predict(obs, deterministic=True)
                    
                    obses, rewards, dones, infos = env.step([agent_action])
                    obs = obses[0]
                    reward = rewards[0]
                    info = infos[0]
                    short_round_triggered = step_count > 5 and short_rounds
                    done = dones[0]
                        
                    trajectory.actions.append(agent_action)
                    trajectory.positions.append((env.agents[0].x, env.agents[0].y))
                    trajectory.target_ids.append(env.num_targets_identified)
                    trajectory.threat_ids.append(env.num_threat_ids)
                    step_count += 1
                        
                rl_trajectories.append(trajectory)
            
        # 3. Save the list of trajectories to a file type of your choice so we don't have regenerate it if we need to re-run
        out_file = os.path.join(self.rl_agents_path, "rl_trajectories.json")
        with open(out_file, "w") as f:
            json.dump([traj.__dict__ for traj in rl_trajectories], f, indent=2)
        
        return rl_trajectories
        
        
    # TODO:
    # - Adapt get_teammate_action for agent action selection
    # Hackiest way is to set the agent as the env teammate, set it inactive, but use get_teammate_action to get action and stpe env
    # - Test 
    def generate_strategy_trajectories(self):
        # Step 1: Create list of heuristic agent parameter combinations. Each element in the list is itself a list of three strings (risk_tolerance, action_noise, spatial_coordination)
        risk_tolerance = ["low", "medium", "high", "max_greedy"]
        action_noise = ["stable", "noisy", "very_noisy"]
        spatial_coordination = [False, True]
        combinations = [list(p) for p in itertools.product(risk_tolerance, action_noise, spatial_coordination)]

        # Step 2: Generate game trajectories for each heuristic combination for each level
        for combination in combinations
            
            # Instantiate agent with <combination> strategy settings		
            risk_tolerance, action_noise, spatial_coordination = combination
            
            teammate = GenericTeammatePolicy(env=None,
                local_search_policy=TargetSearchLocalTSP(search_radius=1000, spatial_coord=spatial_coord, model_path=None, norm_stats_filepath=None, search_method=planning_horizon),
                go_to_highvalue_policy=GoToNearestThreat(model_path=None),
                change_region_subpolicy=ChangeRegions(model_path=None),
                mode_selector_agent=HeuristicAgent(mode_selector='heuristic', risk_tolerance=risk_tolerance, spatial_coord=spatial_coord),
                use_collision_avoidance=False,
                action_stability=action_stability,
                decision_speed=decision_speed)
            
            # Run the agent in all 7 levels
            for level in self.level_list:
                config = self.config.copy()
                config['force_specific_level'] = level_idx
                
                env = DummyVecEnv([make_wrapped_env(config) for _ in range(1)])
                
                trajectory = Trajectory(name = f'{risk_tolerance}-{action_stability}_{spatial_coordination}', level = level, category = 'heuristic') # instantiate the trajectory 
                step_count = 0
                obs = env.reset()
                base_env = env.envs[0].env # TODO confirm that this will update live as the original env does. Is it a shallow copy?
                print(f'base_env is {base_env} (should be MaisrEnvVec, NOT LocalSearchWrapper\n\n%%%'}
                
                while not done:
                    agent_waypoint = env.envs[0].get_teammate_action() # TODO this is wrong. 
                    agent_action = 
                    self.env.agents[self.env.aircraft_ids[1]].waypoint_override = self.teammate_action
                    
                    obses, rewards, dones, infos = env.step([agent_action])
                    obs = obses[0]
                    reward = rewards[0]
                    info = infos[0]
                    short_round_triggered = step_count > 5 and short_rounds
                    done = dones[0]
                        
                    trajectory.actions.append(agent_action)
                    trajectory.positions.append((env.agents[0].x, env.agents[0].y))
                    trajectory.target_ids.append(env.num_targets_identified)
                    trajectory.threat_ids.append(env.num_threat_ids)
                    step_count += 1
                        
                strategy_agent_trajectories.append(trajectory)

        # 3. Save the list of trajectories to a file type of your choice so we don't have regenerate it if we need to re-run
        
        return strategy_agent_trajectories
        
    
    
    ################################################ Helper functions ################################################
    
    # Ready to test
    def compute_2d_emd(heatmap1: np.ndarray, heatmap2: np.ndarray) -> float:
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
        
        
    
    ################################################ Analysis function ################################################
    
    # Ready to test
    # 
    def compare_position_heatmaps_2d(human_trajectories: List[Trajectory],
                                 rl_trajectories: List[Trajectory],
                                 heuristic_trajectories: List[Trajectory],
                                 bins: int = 50,
                                 output_csv: str = "heatmap_emd_results_2d.csv"):
        """
        References/justification for using 2D EMD for this analysis:
            https://stats.stackexchange.com/questions/404775/calculate-earth-movers-distance-for-two-grayscale-images
            https://stats.stackexchange.com/questions/659384/compute-p-value-of-earth-movers-distance-score-comparing-two-heatmaps-in-r
            
        """
        # --- 1. Aggregate positions ---
        def extract_positions(trajs: List[Trajectory]) -> np.ndarray:
            return np.array([pos for t in trajs for pos in t.positions])

        human_positions = extract_positions(human_trajectories)
        rl_positions = extract_positions(rl_trajectories)
        heuristic_positions = extract_positions(heuristic_trajectories)

        # --- 2. Define common grid for all heatmaps ---
        all_positions = np.vstack([human_positions, rl_positions, heuristic_positions])
        x_min, y_min = np.min(all_positions, axis=0)
        x_max, y_max = np.max(all_positions, axis=0)

        def compute_heatmap(positions):
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
            'human_vs_rl': compute_2d_emd(human_heatmap, rl_heatmap),
            'human_vs_heuristic': compute_2d_emd(human_heatmap, heuristic_heatmap),
            'rl_vs_heuristic': compute_2d_emd(rl_heatmap, heuristic_heatmap),
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
        plt.savefig(output_plot, dpi=300)
        plt.close(fig)
            
        return emd_results
        
    
    # Ready to test
    def compute_silhouette_scores(
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
    def compare_progress_rates(
                human_trajectories: List[Trajectory],
                rl_trajectories: List[Trajectory],
                heuristic_trajectories: List[Trajectory],
                save_dir: str = "analysis_outputs"
            ):
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
    def analyze_action_distributions(
            human_trajectories,
            rl_agent_trajectories,
            strategy_agent_trajectories,
            save_dir="analysis_outputs"
        ):
        """
        Input: 
            human_trajectories: list of Trajectory objects
            rl_agent_trajectories: list of Trajectory objects
            strategy_agent_trajectories: list of Trajectory objects
            
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
        rl_hist = get_action_hist(rl_agent_trajectories)
        strategy_hist = get_action_hist(strategy_agent_trajectories)

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
        # Load trajectories
        self.human_trajectories = self.load_human_trajectories()
        self.strategy_agent_trajectories = self.generate_strategy_trajectories()
        self.rl_agent_trajectories = self.generate_rl_trajectories()
        
        # Analyze similarity of position trajectories
        self.compare_position_heatmaps_2d() # Ready to test
        
        # Analyze threat-target priority clusters
        self.compute_silhouette_scores() # Ready to test
        
        # Analyze rate of identifying threats and targets throughout the episode
        self.compare_progress_rates() # Ready to test
        
        # Analyze metric similarity
        self.analyze_metric_similarity() # Has a TODO, then test.
        
        # Analyze similarity of action distributions
        self.analyze_action_distributions() # Ready to test
        
        # Final data to return and save
        # 1. 


if __name__ == '__main__':
    analyzer = SimilarityAnalysis()
    analyzer.run_analysis()