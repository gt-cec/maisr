import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgba

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgba


def plot_results(input_file,
                 colors=None,
                 title_fontsize=16,
                 label_fontsize=14,
                 tick_fontsize=12,
                 figsize=(15, 10),
                 save_path=None):
    """
    Create barplots comparing different metrics across agent types.

    Parameters:
    -----------
    input_file : str
        Path to the CSV file containing user study data
    colors : list or dict, optional
        Colors for each agent type. If list, should match number of agents.
        If dict, should map agent names to colors.
    title_fontsize : int, default=16
        Font size for subplot titles
    label_fontsize : int, default=14
        Font size for axis labels
    tick_fontsize : int, default=12
        Font size for tick labels
    figsize : tuple, default=(15, 10)
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure. If None, figure is displayed but not saved.
    """

    # Load the data
    df = pd.read_csv(input_file)

    # Rename agent types
    df['agent'] = df['agent'].replace({
        'SBPD-FT': 'Strat-SP',
        'SBPD-M': 'Strat-FCP',
        'SP': 'SP'
    })

    # Define default colors if none provided
    unique_agents = df['agent'].unique()
    n_agents = len(unique_agents)

    if colors is None:
        # Use a professional color palette
        default_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8E44AD', '#27AE60']
        colors = default_colors[:n_agents]
    elif isinstance(colors, dict):
        colors = [colors.get(agent, '#2E86AB') for agent in unique_agents]

    # Create color mapping
    color_map = dict(zip(unique_agents, colors))

    # Calculate summary statistics (mean and standard error)
    metrics = ['episode_duration', 'targets_identified', 'threats_identified']
    summary_stats = df.groupby('agent')[metrics].agg(['mean', 'sem']).reset_index()

    # Flatten column names
    summary_stats.columns = ['agent'] + ['_'.join(col).strip() for col in summary_stats.columns[1:]]

    # Create subplot layout
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    # fig.suptitle('User Study Results: Performance Metrics by Agent Type', fontsize=title_fontsize + 2, fontweight='bold', y=0.95)

    # Plotting configurations
    plot_configs = [
        {
            'ax': axes[0, 0],
            'metric': 'episode_duration',
            'title': 'Episode Duration',
            'ylabel': 'Duration (seconds)',
            'mean_col': 'episode_duration_mean',
            'sem_col': 'episode_duration_sem'
        },
        {
            'ax': axes[0, 1],
            'metric': 'targets_identified',
            'title': 'Targets Identified',
            'ylabel': 'Number of Targets',
            'mean_col': 'targets_identified_mean',
            'sem_col': 'targets_identified_sem'
        },
        {
            'ax': axes[1, 0],
            'metric': 'threats_identified',
            'title': 'Threats Identified',
            'ylabel': 'Number of Threats',
            'mean_col': 'threats_identified_mean',
            'sem_col': 'threats_identified_sem'
        }
    ]

    # Create individual bar plots
    for config in plot_configs:
        ax = config['ax']
        ax.grid(True, alpha=0.5, linestyle='--', axis='y')

        # Create bars
        bars = ax.bar(summary_stats['agent'],
                      summary_stats[config['mean_col']],
                      color=[color_map[agent] for agent in summary_stats['agent']],
                      alpha=0.9,
                      edgecolor='black',
                      linewidth=1)

        # Add error bars
        ax.errorbar(summary_stats['agent'],
                    summary_stats[config['mean_col']],
                    yerr=summary_stats[config['sem_col']],
                    fmt='none',
                    color='black',
                    capsize=5,
                    capthick=1,
                    linewidth=1.5)

        # Customize the plot
        ax.set_title(config['title'], fontsize=title_fontsize, fontweight='bold', pad=20)
        ax.set_ylabel(config['ylabel'], fontsize=label_fontsize)
        ax.set_xlabel('Agent Type', fontsize=label_fontsize)

        # Format ticks
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.tick_params(axis='x', rotation=0)

        # # Add value labels on bars
        # for bar, mean_val in zip(bars, summary_stats[config['mean_col']]):
        #     height = bar.get_height()
        #     ax.text(bar.get_x() + bar.get_width() / 2., height + summary_stats[config['sem_col']].max() * 0.1,
        #             f'{mean_val:.1f}',
        #             ha='center', va='bottom', fontsize=tick_fontsize - 1, fontweight='bold')

        # Add grid for better readability

        ax.set_axisbelow(True)

    # Create a composite score plot (4th subplot)
    # Calculate a simple composite score: targets_identified - threats_identified + (1/episode_duration)*1000
    df['composite_score'] = df['targets_identified'] - df['threats_identified'] + (1000 / df['episode_duration'])
    composite_stats = df.groupby('agent')['composite_score'].agg(['mean', 'sem']).reset_index()

    ax = axes[1, 1]
    ax.grid(True, alpha=0.5, linestyle='--', axis='y')

    bars = ax.bar(composite_stats['agent'],
                  composite_stats['mean'],
                  color=[color_map[agent] for agent in composite_stats['agent']],
                  alpha=0.9,
                  edgecolor='black',
                  linewidth=1)

    ax.errorbar(composite_stats['agent'],
                composite_stats['mean'],
                yerr=composite_stats['sem'],
                fmt='none',
                color='black',
                capsize=5,
                capthick=1,
                linewidth=1.5)

    ax.set_title('Composite Performance Score', fontsize=title_fontsize, fontweight='bold', pad=20)
    ax.set_ylabel('Composite Score', fontsize=label_fontsize)
    ax.set_xlabel('Agent Type', fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)

    ax.set_axisbelow(True)

    # # Add value labels
    # for bar, mean_val in zip(bars, composite_stats['mean']):
    #     height = bar.get_height()
    #     ax.text(bar.get_x() + bar.get_width() / 2., height + composite_stats['sem'].max() * 0.1,
    #             f'{mean_val:.1f}',
    #             ha='center', va='bottom', fontsize=tick_fontsize - 1, fontweight='bold')
    #
    # # Adjust layout
    plt.tight_layout()

    # Add a legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color_map[agent], alpha=0.9, edgecolor='black') for agent
                       in unique_agents]
    fig.legend(legend_elements, unique_agents,
               loc='center', bbox_to_anchor=(0.5, 0.02),
               ncol=len(unique_agents), fontsize=label_fontsize)

    plt.subplots_adjust(bottom=0.1)

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()

    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    for metric in metrics:
        print(f"\n{metric.replace('_', ' ').title()}:")
        metric_stats = df.groupby('agent')[metric].agg(['mean', 'std', 'count'])
        print(metric_stats.round(2))

    return fig


def plot_workload(input_file,
                  colors=None,
                  title_fontsize=16,
                  label_fontsize=14,
                  tick_fontsize=12,
                  figsize=(15, 10),
                  save_path=None):
    """
    Create barplots comparing workload metrics across agent types.

    Parameters:
    -----------
    input_file : str
        Path to the CSV file containing workload data
    colors : list or dict, optional
        Colors for each agent type. If list, should match number of agents.
        If dict, should map agent names to colors.
    title_fontsize : int, default=16
        Font size for subplot titles
    label_fontsize : int, default=14
        Font size for axis labels
    tick_fontsize : int, default=12
        Font size for tick labels
    figsize : tuple, default=(15, 10)
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure. If None, figure is displayed but not saved.
    """

    # Load the data
    df = pd.read_csv(input_file)

    # Rename agent types to match the main study
    df['agent'] = df['agent'].replace({
        'SBPD-FT': 'Strat-SP',
        'SBPD-M': 'Strat-FCP',
        'SP': 'SP'
    })

    # Define default colors if none provided
    unique_agents = df['agent'].unique()
    n_agents = len(unique_agents)

    if colors is None:
        # Use a professional color palette
        default_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8E44AD', '#27AE60']
        colors = default_colors[:n_agents]
    elif isinstance(colors, dict):
        colors = [colors.get(agent, '#2E86AB') for agent in unique_agents]

    # Create color mapping
    color_map = dict(zip(unique_agents, colors))

    # Define workload metrics
    workload_metrics = ['mental_demand', 'temporal_demand', 'effort', 'performance', 'frustration',
                        'composite_workload']

    # Calculate summary statistics (mean and standard error)
    summary_stats = df.groupby('agent')[workload_metrics].agg(['mean', 'sem']).reset_index()

    # Flatten column names
    summary_stats.columns = ['agent'] + ['_'.join(col).strip() for col in summary_stats.columns[1:]]

    # Create subplot layout (3x2 grid for all workload metrics)
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('User Study Results: Workload Metrics by Agent Type', fontsize=title_fontsize + 2, fontweight='bold',
                 y=0.95)

    # Plotting configurations
    plot_configs = [
        {
            'ax': axes[0, 0],
            'metric': 'mental_demand',
            'title': 'Mental Demand',
            'ylabel': 'Rating',
            'mean_col': 'mental_demand_mean',
            'sem_col': 'mental_demand_sem'
        },
        {
            'ax': axes[0, 1],
            'metric': 'temporal_demand',
            'title': 'Temporal Demand',
            'ylabel': 'Rating',
            'mean_col': 'temporal_demand_mean',
            'sem_col': 'temporal_demand_sem'
        },
        {
            'ax': axes[0, 2],
            'metric': 'effort',
            'title': 'Effort',
            'ylabel': 'Rating',
            'mean_col': 'effort_mean',
            'sem_col': 'effort_sem'
        },
        {
            'ax': axes[1, 0],
            'metric': 'performance',
            'title': 'Performance',
            'ylabel': 'Rating',
            'mean_col': 'performance_mean',
            'sem_col': 'performance_sem'
        },
        {
            'ax': axes[1, 1],
            'metric': 'frustration',
            'title': 'Frustration',
            'ylabel': 'Rating',
            'mean_col': 'frustration_mean',
            'sem_col': 'frustration_sem'
        },
        {
            'ax': axes[1, 2],
            'metric': 'composite_workload',
            'title': 'Composite Workload',
            'ylabel': 'Overall Rating',
            'mean_col': 'composite_workload_mean',
            'sem_col': 'composite_workload_sem'
        }
    ]

    # Create individual bar plots
    for config in plot_configs:
        ax = config['ax']

        ax.grid(True, alpha=0.5, linestyle='--', axis='y')

        # Create bars
        bars = ax.bar(summary_stats['agent'],
                      summary_stats[config['mean_col']],
                      color=[color_map[agent] for agent in summary_stats['agent']],
                      alpha=0.9,
                      edgecolor='black',
                      linewidth=1)

        # Add error bars
        ax.errorbar(summary_stats['agent'],
                    summary_stats[config['mean_col']],
                    yerr=summary_stats[config['sem_col']],
                    fmt='none',
                    color='black',
                    capsize=5,
                    capthick=1,
                    linewidth=1.5)

        # Customize the plot
        ax.set_title(config['title'], fontsize=title_fontsize, fontweight='bold', pad=15)
        ax.set_ylabel(config['ylabel'], fontsize=label_fontsize)
        ax.set_xlabel('Agent Type', fontsize=label_fontsize)

        # Format ticks
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.tick_params(axis='x', rotation=0)

        # Add grid for better readability

        ax.set_axisbelow(True)

        # Set consistent y-axis limits for better comparison (except composite workload)
        if config['metric'] != 'composite_workload':
            ax.set_ylim(0, 100)  # Assuming workload ratings are 0-100 scale

    # Adjust layout
    plt.tight_layout()

    # Add a legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color_map[agent], alpha=0.9, edgecolor='black') for agent
                       in unique_agents]
    fig.legend(legend_elements, unique_agents,
               loc='center', bbox_to_anchor=(0.5, 0.02),
               ncol=len(unique_agents), fontsize=label_fontsize)

    plt.subplots_adjust(bottom=0.12, top=0.90)

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Workload figure saved to: {save_path}")

    plt.show()

    # Print summary statistics
    print("\n=== WORKLOAD SUMMARY STATISTICS ===")
    for metric in workload_metrics:
        print(f"\n{metric.replace('_', ' ').title()}:")
        metric_stats = df.groupby('agent')[metric].agg(['mean', 'std', 'count'])
        print(metric_stats.round(2))

    return fig


def plot_results(input_file,
                 colors=None,
                 title_fontsize=16,
                 label_fontsize=14,
                 tick_fontsize=12,
                 figsize=(15, 10),
                 save_path=None):
    """
    Create barplots comparing different metrics across agent types.

    Parameters:
    -----------
    input_file : str
        Path to the CSV file containing user study data
    colors : list or dict, optional
        Colors for each agent type. If list, should match number of agents.
        If dict, should map agent names to colors.
    title_fontsize : int, default=16
        Font size for subplot titles
    label_fontsize : int, default=14
        Font size for axis labels
    tick_fontsize : int, default=12
        Font size for tick labels
    figsize : tuple, default=(15, 10)
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure. If None, figure is displayed but not saved.
    """

    # Load the data
    df = pd.read_csv(input_file)

    # Rename agent types
    df['agent'] = df['agent'].replace({
        'SBPD-FT': 'Strat-SP',
        'SBPD-M': 'Strat-FCP',
        'SP': 'SP'
    })

    # Define default colors if none provided
    unique_agents = df['agent'].unique()
    n_agents = len(unique_agents)

    if colors is None:
        # Use a professional color palette
        default_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8E44AD', '#27AE60']
        colors = default_colors[:n_agents]
    elif isinstance(colors, dict):
        colors = [colors.get(agent, '#2E86AB') for agent in unique_agents]

    # Create color mapping
    color_map = dict(zip(unique_agents, colors))

    # Calculate summary statistics (mean and standard error)
    metrics = ['episode_duration', 'targets_identified', 'threats_identified']
    summary_stats = df.groupby('agent')[metrics].agg(['mean', 'sem']).reset_index()

    # Flatten column names
    summary_stats.columns = ['agent'] + ['_'.join(col).strip() for col in summary_stats.columns[1:]]

    # Create subplot layout
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    #fig.suptitle('User Study Results: Performance Metrics by Agent Type', fontsize=title_fontsize + 2, fontweight='bold', y=0.95)



    # Plotting configurations
    plot_configs = [
        {
            'ax': axes[0, 0],
            'metric': 'episode_duration',
            'title': 'Episode Duration',
            'ylabel': 'Duration (seconds)',
            'mean_col': 'episode_duration_mean',
            'sem_col': 'episode_duration_sem'
        },
        {
            'ax': axes[0, 1],
            'metric': 'targets_identified',
            'title': 'Targets Identified',
            'ylabel': 'Number of Targets',
            'mean_col': 'targets_identified_mean',
            'sem_col': 'targets_identified_sem'
        },
        {
            'ax': axes[1, 0],
            'metric': 'threats_identified',
            'title': 'Threats Identified',
            'ylabel': 'Number of Threats',
            'mean_col': 'threats_identified_mean',
            'sem_col': 'threats_identified_sem'
        }
    ]

    # Create individual bar plots
    for config in plot_configs:
        ax = config['ax']

        ax.grid(True, alpha=0.5, linestyle='--', axis='y')

        # Create bars
        bars = ax.bar(summary_stats['agent'],
                      summary_stats[config['mean_col']],
                      color=[color_map[agent] for agent in summary_stats['agent']],
                      alpha=0.9,
                      edgecolor='black',
                      linewidth=1)

        # Add error bars
        ax.errorbar(summary_stats['agent'],
                    summary_stats[config['mean_col']],
                    yerr=summary_stats[config['sem_col']],
                    fmt='none',
                    color='black',
                    capsize=5,
                    capthick=1,
                    linewidth=1.5)

        # Customize the plot
        ax.set_title(config['title'], fontsize=title_fontsize, fontweight='bold', pad=20)
        ax.set_ylabel(config['ylabel'], fontsize=label_fontsize)
        ax.set_xlabel('Agent Type', fontsize=label_fontsize)

        # Format ticks
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.tick_params(axis='x', rotation=0)

        # # Add value labels on bars
        # for bar, mean_val in zip(bars, summary_stats[config['mean_col']]):
        #     height = bar.get_height()
        #     ax.text(bar.get_x() + bar.get_width() / 2., height + summary_stats[config['sem_col']].max() * 0.1,
        #             f'{mean_val:.1f}',
        #             ha='center', va='bottom', fontsize=tick_fontsize - 1, fontweight='bold')

        # Add grid for better readability

        ax.set_axisbelow(True)

    # Create a composite score plot (4th subplot)
    # Calculate a simple composite score: targets_identified - threats_identified + (1/episode_duration)*1000
    df['composite_score'] = df['targets_identified'] - df['threats_identified'] + (1000 / df['episode_duration'])
    composite_stats = df.groupby('agent')['composite_score'].agg(['mean', 'sem']).reset_index()



    ax = axes[1, 1]

    ax.grid(True, alpha=0.5, linestyle='--', axis='y')

    bars = ax.bar(composite_stats['agent'],
                  composite_stats['mean'],
                  color=[color_map[agent] for agent in composite_stats['agent']],
                  alpha=0.9,
                  edgecolor='black',
                  linewidth=1)

    ax.errorbar(composite_stats['agent'],
                composite_stats['mean'],
                yerr=composite_stats['sem'],
                fmt='none',
                color='black',
                capsize=5,
                capthick=1,
                linewidth=1.5)

    ax.set_title('Composite Performance Score', fontsize=title_fontsize, fontweight='bold', pad=20)
    ax.set_ylabel('Composite Score', fontsize=label_fontsize)
    ax.set_xlabel('Agent Type', fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)

    ax.set_axisbelow(True)

    # Add value labels
    for bar, mean_val in zip(bars, composite_stats['mean']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + composite_stats['sem'].max() * 0.1,
                f'{mean_val:.1f}',
                ha='center', va='bottom', fontsize=tick_fontsize - 1, fontweight='bold')

    # Adjust layout
    plt.tight_layout()

    # Add a legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color_map[agent], alpha=0.9, edgecolor='black')for agent in unique_agents]
    fig.legend(legend_elements, unique_agents,
               loc='center', bbox_to_anchor=(0.5, 0.02),
               ncol=len(unique_agents), fontsize=label_fontsize)

    plt.subplots_adjust(bottom=0.1)

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()

    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    for metric in metrics:
        print(f"\n{metric.replace('_', ' ').title()}:")
        metric_stats = df.groupby('agent')[metric].agg(['mean', 'std', 'count'])
        print(metric_stats.round(2))

    return fig


if __name__ == "__main__":
    # Convert the data
    input_file = "userstudy_data_tidyverse.csv"
    workload_file = "userstudy_workload.csv"

    # Example usage with custom colors and font sizes
    custom_colors = {
        'Strat-SP': '#C73E1D',  # Blue
        'Strat-FCP': '#A23B72',  # Purple/Pink
        'SP': '#F18F01'  # Orange
    }

    # Create the plots
    fig = plot_results(
        input_file=input_file,
        colors=custom_colors,
        title_fontsize=20,
        label_fontsize=16,
        tick_fontsize=14,
        figsize=(16, 12),
        save_path="userstudy_results.png"
    )

    fig2 = plot_workload(
        input_file=workload_file,
        colors=custom_colors,
        title_fontsize=18,
        label_fontsize=14,
        tick_fontsize=12,
        figsize=(18, 12),
        save_path="userstudy_workload_results.png"
    )
