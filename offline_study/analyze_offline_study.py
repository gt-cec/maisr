import json
from datetime import datetime

import pandas as pd
from typing import Dict, Tuple, Optional
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon
import seaborn as sns
import numpy as np


def inspect_data_structure(rl_df, human_df):
    """
    Inspect the data structure to understand why pairing might be failing.
    """
    print("DATA STRUCTURE INSPECTION")
    print("=" * 50)

    print("\nRL DataFrame:")
    print(f"Shape: {rl_df.shape}")
    print(f"Columns: {list(rl_df.columns)}")
    print(f"Unique agents: {rl_df['agent'].unique()}")

    if 'teammate' in rl_df.columns:
        print(f"Unique teammates: {len(rl_df['teammate'].unique())}")
        print(f"Sample teammates: {list(rl_df['teammate'].unique())[:5]}")

    print(f"Unique runs: {rl_df['run'].unique()}")

    # Check data for each agent
    for agent in rl_df['agent'].unique():
        agent_data = rl_df[rl_df['agent'] == agent]
        print(f"\nAgent '{agent}': {len(agent_data)} rows")
        print(f"  Reward range: {agent_data['reward'].min():.2f} to {agent_data['reward'].max():.2f}")
        print(f"  Mean reward: {agent_data['reward'].mean():.2f}")
        if 'teammate' in rl_df.columns:
            print(f"  Unique teammates: {len(agent_data['teammate'].unique())}")
        print(f"  Runs: {sorted(agent_data['run'].unique())}")

    print("\n" + "=" * 50)
    print("Human DataFrame:")
    print(f"Shape: {human_df.shape}")
    print(f"Columns: {list(human_df.columns)}")
    print(f"Unique agents: {human_df['agent'].unique()}")

    # Check data for each agent in human_df
    for agent in human_df['agent'].unique():
        agent_data = human_df[human_df['agent'] == agent]
        print(f"\nAgent '{agent}': {len(agent_data)} rows")
        print(f"  Reward range: {agent_data['reward'].min():.2f} to {agent_data['reward'].max():.2f}")
        print(f"  Mean reward: {agent_data['reward'].mean():.2f}")


def prepare_pairwise_data(df, agent1, agent2):
    """
    Prepare data for pairwise comparison between two agents.
    This assumes that both agents were tested on the same set of teammates/conditions.
    """
    # Get data for both agents
    agent1_data = df[df['agent'] == agent1].copy()
    agent2_data = df[df['agent'] == agent2].copy()

    # For RL results: group by teammate and run
    if 'teammate' in df.columns:
        # Create a unique identifier for each test condition
        agent1_data['condition_id'] = agent1_data['teammate'].astype(str) + '_run_' + agent1_data['run'].astype(str)
        agent2_data['condition_id'] = agent2_data['teammate'].astype(str) + '_run_' + agent2_data['run'].astype(str)
    else:
        # For human trajectory results: group by trajectory file and run
        agent1_data['condition_id'] = agent1_data['teammate'].astype(str) + '_run_' + agent1_data['run'].astype(str)
        agent2_data['condition_id'] = agent2_data['teammate'].astype(str) + '_run_' + agent2_data['run'].astype(str)

    # Merge on condition_id to get paired observations
    merged = pd.merge(agent1_data[['condition_id', 'reward']],
                      agent2_data[['condition_id', 'reward']],
                      on='condition_id',
                      suffixes=(f'_{agent1}', f'_{agent2}'))

    return merged[f'reward_{agent1}'].values, merged[f'reward_{agent2}'].values, len(merged)


def wilcoxon_pairwise_test(df, agent1, agent2, test_name=""):
    """
    Perform Wilcoxon signed-rank test between two agents.
    """
    try:
        rewards1, rewards2, n_pairs = prepare_pairwise_data(df, agent1, agent2)

        if len(rewards1) == 0 or len(rewards2) == 0:
            print(f"Warning: No paired data found for {agent1} vs {agent2}")
            return None

        # Perform Wilcoxon signed-rank test
        statistic, p_value = wilcoxon(rewards1, rewards2, alternative='two-sided')

        # Calculate effect size (r = Z / sqrt(N))
        # For Wilcoxon, we can approximate Z from the statistic
        n = len(rewards1)
        if n > 20:
            # Normal approximation for large samples
            expected = n * (n + 1) / 4
            variance = n * (n + 1) * (2 * n + 1) / 24
            z_score = (statistic - expected) / np.sqrt(variance)
            effect_size = abs(z_score) / np.sqrt(n)
        else:
            effect_size = None

        # Calculate descriptive statistics
        mean1, std1 = np.mean(rewards1), np.std(rewards1)
        mean2, std2 = np.mean(rewards2), np.std(rewards2)
        mean_diff = mean1 - mean2

        print(f"\n{test_name}")
        print(f"{'=' * 50}")
        print(f"Comparison: {agent1} vs {agent2}")
        print(f"Sample size: {n} paired observations")
        print(f"{agent1}: M = {mean1:.2f}, SD = {std1:.2f}")
        print(f"{agent2}: M = {mean2:.2f}, SD = {std2:.2f}")
        print(f"Mean difference: {mean_diff:.2f}")
        print(f"Wilcoxon statistic: {statistic}")
        print(f"P-value: {p_value}")
        if effect_size is not None:
            print(f"Effect size (r): {effect_size:.3f}")

        # Interpret results
        alpha = 0.05
        if p_value < alpha:
            direction = "higher" if mean1 > mean2 else "lower"
            print(f"Result: {agent1} performs significantly {direction} than {agent2}")
        else:
            print(f"Result: No significant difference between {agent1} and {agent2}")

        return {
            'agent1': agent1,
            'agent2': agent2,
            'n_pairs': n,
            'mean1': mean1,
            'mean2': mean2,
            'mean_diff': mean_diff,
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < alpha
        }

    except Exception as e:
        print(f"Error in statistical test for {agent1} vs {agent2}: {str(e)}")
        return None


def run_all_pairwise_tests(rl_df, human_df):
    """
    Run all the specified pairwise comparisons.
    """
    results = []

    # Define the comparisons to make
    comparisons = [
        ('mixed75', 'fcp'),
        ('strat-finetuned', 'selfplay')
    ]

    print("PAIRWISE STATISTICAL TESTS")
    print("=" * 60)

    # RL evaluations
    print("\n1. RL TEAMMATE EVALUATIONS")
    for agent1, agent2 in comparisons:
        result = wilcoxon_pairwise_test(rl_df, agent1, agent2,
                                        test_name=f"RL Test: {agent1} vs {agent2}")
        if result:
            result['test_type'] = 'RL'
            results.append(result)

    # Human trajectory evaluations
    print("\n2. HUMAN TRAJECTORY EVALUATIONS")
    for agent1, agent2 in comparisons:
        result = wilcoxon_pairwise_test(human_df, agent1, agent2,
                                        test_name=f"Human Test: {agent1} vs {agent2}")
        if result:
            result['test_type'] = 'Human'
            results.append(result)

    return results



def report_summary(results):
    """
    Print a summary of all test results.
    """
    print("\n" + "=" * 60)
    print("SUMMARY OF STATISTICAL TESTS")
    print("=" * 60)

    for result in results:
        status = "SIGNIFICANT" if result['significant'] else "NOT SIGNIFICANT"
        direction = ">" if result['mean1'] > result['mean2'] else "<"
        print(
            f"{result['test_type']}: {result['agent1']} {direction} {result['agent2']} - {status} (p={result['p_value']:.4f})")


def convert_to_numeric(df, numeric_columns):
    """
    Convert specified columns to numeric, handling potential conversion errors.
    """
    for col in numeric_columns:
        if col in df.columns:
            try:
                # First try direct conversion
                df[col] = pd.to_numeric(df[col], errors='coerce')

                # Check for any NaN values that might indicate conversion issues
                if df[col].isna().any():
                    print(f"Warning: Some values in column '{col}' could not be converted to numeric")
                    print(f"Number of NaN values: {df[col].isna().sum()}")

            except Exception as e:
                print(f"Error converting column '{col}' to numeric: {str(e)}")

    return df


def load_offline_evaluation_results(filename: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Load offline evaluation results from JSON file created by main().

    Args:
        filename (str, optional): Path to the JSON results file. If None, will find the most recent
                                 offline_evaluation_results_*.json file in the current directory.

    Returns:
        Tuple containing:
        - rl_df (pd.DataFrame): DataFrame with RL teammate evaluation results
        - human_df (pd.DataFrame): DataFrame with human trajectory evaluation results
        - metadata (dict): Metadata about the evaluation run

    Raises:
        FileNotFoundError: If no results file is found
        ValueError: If the file format is invalid
    """

    # If no filename provided, find the most recent results file
    if filename is None:
        pattern = "offline_evaluation_results_*.json"
        matching_files = glob.glob(pattern)

        if not matching_files:
            raise FileNotFoundError(f"No offline evaluation results files found matching pattern: {pattern}")

        # Sort by modification time, newest first
        matching_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        filename = matching_files[0]
        print(f"Loading most recent results file: {filename}")

    # Load the JSON data
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Results file not found: {filename}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in file {filename}: {str(e)}")

    # Validate file structure
    required_keys = ['metadata', 'raw_results', 'summary_stats']
    if not all(key in data for key in required_keys):
        raise ValueError(f"Invalid file format. Expected keys: {required_keys}, found: {list(data.keys())}")

    raw_results = data['raw_results']
    if 'rl_results' not in raw_results or 'human_results' not in raw_results:
        raise ValueError("Missing rl_results or human_results in raw_results section")

    # Convert to DataFrames
    try:
        rl_df = pd.DataFrame(raw_results['rl_results'])
        human_df = pd.DataFrame(raw_results['human_results'])

        # Convert numeric columns to proper numeric types
        numeric_columns = ['reward', 'target_ids', 'threat_ids', 'num_steps', 'run']
        rl_df = convert_to_numeric(rl_df, numeric_columns)
        human_df = convert_to_numeric(human_df, numeric_columns)

    except Exception as e:
        raise ValueError(f"Error converting results to DataFrames: {str(e)}")

    # Extract metadata
    metadata = data['metadata']

    print(f"Successfully loaded results from {filename}")
    print(f"RL results: {len(rl_df)} records")
    print(f"Human trajectory results: {len(human_df)} records")
    print(f"Evaluation timestamp: {metadata.get('timestamp', 'Unknown')}")

    return rl_df, human_df, metadata


def print_results_summary(rl_df: pd.DataFrame, human_df: pd.DataFrame, metadata: Dict):
    """
    Print a summary of the loaded evaluation results.

    Args:
        rl_df: DataFrame with RL teammate evaluation results
        human_df: DataFrame with human trajectory evaluation results
        metadata: Metadata dictionary from the results file
    """

    print("\n" + "=" * 60)
    print("OFFLINE EVALUATION RESULTS SUMMARY")
    print("=" * 60)

    # Print metadata
    print(f"\nEvaluation Metadata:")
    print(f"  Timestamp: {metadata.get('timestamp', 'Unknown')}")
    print(f"  Config file: {metadata.get('config_file', 'Unknown')}")
    print(f"  Testing agents: {metadata.get('num_testing_agents', 'Unknown')}")
    print(f"  Held-out agents: {metadata.get('num_heldout_agents', 'Unknown')}")
    print(f"  Dual trajectories: {metadata.get('num_dual_trajectories', 'Unknown')}")
    print(f"  Solo trajectories: {metadata.get('num_solo_trajectories', 'Unknown')}")

    # RL results summary
    print(f"\nRL Teammate Evaluation Results:")
    try:
        rl_stats = rl_df.groupby('agent')['reward'].agg(['mean', 'std', 'count']).reset_index()
        for _, row in rl_stats.iterrows():
            print(f"  {row['agent']}: {row['mean']:.2f} ± {row['std']:.2f} (n={row['count']})")
    except Exception as e:
        print(f"  Error calculating RL statistics: {str(e)}")
        print(f"  Reward column dtype: {rl_df['reward'].dtype}")
        print(f"  Sample reward values: {rl_df['reward'].head().tolist()}")

    # Human trajectory results summary
    print(f"\nHuman Trajectory Evaluation Results:")
    try:
        human_stats = human_df.groupby('agent')['reward'].agg(['mean', 'std', 'count']).reset_index()
        for _, row in human_stats.iterrows():
            print(f"  {row['agent']}: {row['mean']:.2f} ± {row['std']:.2f} (n={row['count']})")
    except Exception as e:
        print(f"  Error calculating Human statistics: {str(e)}")
        print(f"  Reward column dtype: {human_df['reward'].dtype}")
        print(f"  Sample reward values: {human_df['reward'].head().tolist()}")


# def plot_results(rl_df, human_df):
#     """Create individual plots for each metric and teammate type combination."""
#
#     # Define consistent colors and label mapping for agent types
#     agent_colors = {
#         'fcp': '#2E86AB',  # Blue
#         'mixed75': '#A23B72',  # Purple
#         'selfplay': '#F18F01',  # Orange
#         'strat-finetuned': '#C73E1D'  # Red
#     }
#
#     agent_labels = {
#         'fcp': 'FCP',
#         'mixed75': 'Strat-FCP',
#         'selfplay': 'SP',
#         'strat-finetuned': 'Strat-SP'
#     }
#
#     # Define y-axis limits for each metric
#     y_limits = {
#         'reward': (0, 40),
#         'target_ids': (0, 15),
#         'threat_ids': (0, 2.5)
#     }
#
#     # Calculate statistics for all metrics
#     metrics = ['reward', 'target_ids', 'threat_ids']
#
#     # Calculate stats for RL teammates
#     rl_all_stats = {}
#     for metric in metrics:
#         rl_all_stats[metric] = rl_df.groupby('agent')[metric].agg(['mean', 'std', 'count']).reset_index()
#
#     # Calculate stats for human teammates
#     human_all_stats = {}
#     for metric in metrics:
#         human_all_stats[metric] = human_df.groupby('agent')[metric].agg(['mean', 'std', 'count']).reset_index()
#
#     # Create plots for each metric and teammate type
#     for metric in metrics:
#         # Plot 1: RL teammates
#         plt.figure(figsize=(8, 8))
#         rl_data = rl_all_stats[metric]
#         colors = [agent_colors.get(agent, '#808080') for agent in rl_data['agent']]
#         bars = plt.bar(range(len(rl_data)), rl_data['mean'],
#                        yerr=rl_data['std'], capsize=5, alpha=0.9, color=colors)
#         plt.xlabel('Agent Type', fontsize=12)
#         plt.ylabel(f'Average {metric.replace("_", " ").title()}', fontsize=12)
#         plt.title(f'{metric.replace("_", " ").title()} vs Testing Agent Type for RL Teammates', fontsize=20)
#         plt.xticks(range(len(rl_data)), [agent_labels.get(agent, agent) for agent in rl_data['agent']],
#                    rotation=45, ha='right')
#         plt.ylim(y_limits[metric])  # Set y-axis limits
#         plt.grid(axis='y', alpha=0.3)
#         plt.tight_layout()
#         plt.show()
#
#         # Plot 2: Human teammates
#         plt.figure(figsize=(8, 8))
#         human_data = human_all_stats[metric]
#         colors = [agent_colors.get(agent, '#808080') for agent in human_data['agent']]
#         bars = plt.bar(range(len(human_data)), human_data['mean'],
#                        yerr=human_data['std'], capsize=5, alpha=0.9, color=colors)
#         plt.xlabel('Agent Type', fontsize=12)
#         plt.ylabel(f'Average {metric.replace("_", " ").title()}', fontsize=12)
#         plt.title(f'{metric.replace("_", " ").title()} vs Testing Agent Type for Recorded Human Teammates', fontsize=20)
#         plt.xticks(range(len(human_data)), [agent_labels.get(agent, agent) for agent in human_data['agent']],
#                    rotation=45, ha='right')
#         plt.ylim(y_limits[metric])  # Set y-axis limits
#         plt.grid(axis='y', alpha=0.3)
#         plt.tight_layout()
#         plt.show()
#
#     # Print summary statistics for all metrics
#     print(f'\n====== PERFORMANCE SUMMARY FOR ALL METRICS ======')
#
#     for metric in metrics:
#         print(f'\n--- {metric.replace("_", " ").title()} ---')
#
#         print(f'RL Teammates:')
#         for _, row in rl_all_stats[metric].iterrows():
#             agent_label = agent_labels.get(row["agent"], row["agent"])
#             print(f'  {agent_label}: {row["mean"]:.2f} ± {row["std"]:.2f} (n={row["count"]})')
#
#         print(f'Human Teammates:')
#         for _, row in human_all_stats[metric].iterrows():
#             agent_label = agent_labels.get(row["agent"], row["agent"])
#             print(f'  {agent_label}: {row["mean"]:.2f} ± {row["std"]:.2f} (n={row["count"]})')

def plot_results(rl_df, human_df):
    """Create combined plots with split bars showing both RL and human teammate results."""

    # Define consistent colors and label mapping for agent types
    agent_colors = {
        'fcp': '#2E86AB',  # Blue
        'mixed75': '#A23B72',  # Purple
        'selfplay': '#F18F01',  # Orange
        'strat-finetuned': '#C73E1D'  # Red
    }

    agent_labels = {
        'fcp': 'FCP',
        'mixed75': 'Strat-FCP',
        'selfplay': 'SP',
        'strat-finetuned': 'Strat-SP'
    }

    # Define y-axis limits for each metric
    y_limits = {
        'reward': (0, 40),
        'target_ids': (0, 15),
        'threat_ids': (0, 2.5)
    }

    # Calculate statistics for all metrics
    metrics = ['reward', 'target_ids', 'threat_ids']

    # Calculate stats for RL teammates
    rl_all_stats = {}
    for metric in metrics:
        rl_all_stats[metric] = rl_df.groupby('agent')[metric].agg(['mean', 'std', 'count']).reset_index()

    # Calculate stats for human teammates
    human_all_stats = {}
    for metric in metrics:
        human_all_stats[metric] = human_df.groupby('agent')[metric].agg(['mean', 'std', 'count']).reset_index()

    # Create combined plots for each metric
    for metric in metrics:
        plt.figure(figsize=(12, 8))

        # Get data for both RL and human teammates
        rl_data = rl_all_stats[metric]
        human_data = human_all_stats[metric]

        # Find common agents between both datasets
        common_agents = set(rl_data['agent']) & set(human_data['agent'])
        common_agents = sorted(list(common_agents))  # Sort for consistent ordering

        if not common_agents:
            print(f"Warning: No common agents found for {metric}")
            continue

        # Prepare data for plotting
        x_pos = np.arange(len(common_agents))
        bar_width = 0.35

        rl_means = []
        rl_stds = []
        human_means = []
        human_stds = []
        colors = []

        for agent in common_agents:
            # Get RL data
            rl_row = rl_data[rl_data['agent'] == agent].iloc[0]
            rl_means.append(rl_row['mean'])
            rl_stds.append(rl_row['std'])

            # Get human data
            human_row = human_data[human_data['agent'] == agent].iloc[0]
            human_means.append(human_row['mean'])
            human_stds.append(human_row['std'])

            colors.append(agent_colors.get(agent, '#808080'))


        plt.grid(axis='y', alpha=0.5)

        # Create the split bars
        bars1 = plt.bar(x_pos - bar_width / 2, rl_means, bar_width,
                        yerr=rl_stds, capsize=5, alpha=0.95,
                        color=colors, label='RL Teammates',
                        edgecolor='black', linewidth=0.5)

        bars2 = plt.bar(x_pos + bar_width / 2, human_means, bar_width,
                        yerr=human_stds, capsize=5, alpha=0.6,
                        color=colors, label='Human Teammates',
                        edgecolor='black', linewidth=0.5)

        # Customize the plot
        plt.xlabel('Agent Type', fontsize=14)
        plt.ylabel(f'Average {metric.replace("_", " ").title()}', fontsize=14)
        plt.title(f'{metric.replace("_", " ").title()} Comparison: RL vs Human Teammates', fontsize=20)
        plt.xticks(x_pos, [agent_labels.get(agent, agent) for agent in common_agents])
        plt.ylim(y_limits[metric])
        #plt.legend(fontsize=12)


        # Add value labels on bars
        # def add_value_labels(bars, values):
        #     for bar, value in zip(bars, values):
        #         height = bar.get_height()
        #         plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
        #                  f'{value:.1f}', ha='center', va='bottom', fontsize=10)
        #
        # add_value_labels(bars1, rl_means)
        # add_value_labels(bars2, human_means)

        plt.tight_layout()
        plt.show()

    # Print summary statistics for all metrics
    print(f'\n====== PERFORMANCE SUMMARY FOR ALL METRICS ======')

    for metric in metrics:
        print(f'\n--- {metric.replace("_", " ").title()} ---')

        print(f'RL Teammates:')
        for _, row in rl_all_stats[metric].iterrows():
            agent_label = agent_labels.get(row["agent"], row["agent"])
            print(f'  {agent_label}: {row["mean"]:.2f} ± {row["std"]:.2f} (n={row["count"]})')

        print(f'Human Teammates:')
        for _, row in human_all_stats[metric].iterrows():
            agent_label = agent_labels.get(row["agent"], row["agent"])
            print(f'  {agent_label}: {row["mean"]:.2f} ± {row["std"]:.2f} (n={row["count"]})')

        # Print direct comparison for common agents
        common_agents = set(rl_all_stats[metric]['agent']) & set(human_all_stats[metric]['agent'])
        if common_agents:
            print(f'\nDirect Comparison (RL vs Human):')
            for agent in sorted(common_agents):
                rl_row = rl_all_stats[metric][rl_all_stats[metric]['agent'] == agent].iloc[0]
                human_row = human_all_stats[metric][human_all_stats[metric]['agent'] == agent].iloc[0]
                agent_label = agent_labels.get(agent, agent)
                diff = rl_row['mean'] - human_row['mean']
                print(f'  {agent_label}: RL={rl_row["mean"]:.2f} vs Human={human_row["mean"]:.2f} (Δ={diff:+.2f})')

# Example usage:
if __name__ == "__main__":
    # Load the most recent results file
    rl_df, human_df, metadata = load_offline_evaluation_results()

    # Print summary
    print_results_summary(rl_df, human_df, metadata)

    # You can now use the DataFrames for further analysis
    print(f"\nRL DataFrame shape: {rl_df.shape}")
    print(f"Human DataFrame shape: {human_df.shape}")
    print(f"\nRL DataFrame columns: {list(rl_df.columns)}")
    print(f"Human DataFrame columns: {list(human_df.columns)}")

    #inspect_data_structure(rl_df, human_df)

    print(f'######################## STATISTICAL TESTS ########################')

    test_results = run_all_pairwise_tests(rl_df, human_df)

    # Create and display all plots
    plot_results(rl_df, human_df)

    # Print summary
    report_summary(test_results)

    # Save statistical results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_filename = f'statistical_tests_{timestamp}.json'

    with open(stats_filename, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)

    print(f"\nStatistical test results saved to: {stats_filename}")