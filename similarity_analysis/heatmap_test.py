import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns


def plot_trajectory_heatmap(json_file_path):
    """
    Plot a heatmap of all x,y positions from RL trajectory data

    Args:
        json_file_path: Path to the JSON file containing trajectory data
    """

    # Load the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Extract all x,y positions from all trajectories
    all_x = []
    all_y = []

    for trajectory in data:
        positions = trajectory['positions']
        for pos in positions:
            all_x.append(pos[0])
            all_y.append(pos[1])

    # Convert to numpy arrays
    x_coords = np.array(all_x)
    y_coords = np.array(all_y)

    print(f"Total number of positions: {len(x_coords)}")
    print(f"X range: [{x_coords.min():.1f}, {x_coords.max():.1f}]")
    print(f"Y range: [{y_coords.min():.1f}, {y_coords.max():.1f}]")

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('RL Trajectory Position Analysis', fontsize=16)

    # 1. Scatter plot with alpha for density visualization
    axes[0, 0].scatter(x_coords, y_coords, alpha=0.1, s=1, c='blue')
    axes[0, 0].set_title('Scatter Plot of All Positions')
    axes[0, 0].set_xlabel('X Position')
    axes[0, 0].set_ylabel('Y Position')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 2D Histogram (heatmap)
    axes[0, 1].hist2d(x_coords, y_coords, bins=50, cmap='hot', density=True)
    axes[0, 1].set_title('2D Histogram Heatmap')
    axes[0, 1].set_xlabel('X Position')
    axes[0, 1].set_ylabel('Y Position')

    # 3. Kernel Density Estimation heatmap
    # Create a grid for KDE
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # Add some padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= 0.1 * x_range
    x_max += 0.1 * x_range
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range

    # Create grid
    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    # Calculate KDE
    values = np.vstack([x_coords, y_coords])
    kernel = gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    # Plot KDE
    im = axes[1, 0].contourf(xx, yy, f, levels=20, cmap='viridis', alpha=0.8)
    axes[1, 0].set_title('Kernel Density Estimation')
    axes[1, 0].set_xlabel('X Position')
    axes[1, 0].set_ylabel('Y Position')
    plt.colorbar(im, ax=axes[1, 0])

    # 4. Hexagonal binning
    hb = axes[1, 1].hexbin(x_coords, y_coords, gridsize=30, cmap='plasma', mincnt=1)
    axes[1, 1].set_title('Hexagonal Binning')
    axes[1, 1].set_xlabel('X Position')
    axes[1, 1].set_ylabel('Y Position')
    plt.colorbar(hb, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()

    # Additional analysis: separate by category and level
    plot_by_category_level(data)


def plot_by_category_level(data):
    """
    Create separate plots for different categories and levels
    """
    # Group data by category and level
    grouped_data = {}
    for trajectory in data:
        category = trajectory['category']
        level = trajectory['level']
        key = f"{category}_level_{level}"

        if key not in grouped_data:
            grouped_data[key] = {'x': [], 'y': []}

        for pos in trajectory['positions']:
            grouped_data[key]['x'].append(pos[0])
            grouped_data[key]['y'].append(pos[1])

    # Create subplots for each unique combination
    unique_keys = list(grouped_data.keys())
    n_plots = len(unique_keys)

    # Calculate grid size
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle('Position Heatmaps by Category and Level', fontsize=16)

    # Flatten axes array for easier indexing
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    for idx, (key, coords) in enumerate(grouped_data.items()):
        if idx < len(axes):
            x_data = np.array(coords['x'])
            y_data = np.array(coords['y'])

            # Create 2D histogram
            axes[idx].hist2d(x_data, y_data, bins=20, cmap='hot', density=True)
            axes[idx].set_title(f'{key.replace("_", " ").title()}')
            axes[idx].set_xlabel('X Position')
            axes[idx].set_ylabel('Y Position')

    # Hide empty subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()


def analyze_trajectory_stats(json_file_path):
    """
    Print statistical analysis of the trajectory data
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    print("\n" + "=" * 50)
    print("TRAJECTORY STATISTICS")
    print("=" * 50)

    # Overall statistics
    total_trajectories = len(data)
    total_positions = sum(len(traj['positions']) for traj in data)

    print(f"Total trajectories: {total_trajectories}")
    print(f"Total positions: {total_positions}")
    print(f"Average positions per trajectory: {total_positions / total_trajectories:.1f}")

    # Statistics by category and level
    stats_by_group = {}
    for traj in data:
        category = traj['category']
        level = traj['level']
        key = f"{category}_level_{level}"

        if key not in stats_by_group:
            stats_by_group[key] = []

        stats_by_group[key].append(len(traj['positions']))

    print(f"\nStatistics by category and level:")
    for key, lengths in stats_by_group.items():
        print(f"  {key}: {len(lengths)} trajectories, "
              f"avg length: {np.mean(lengths):.1f}, "
              f"range: [{min(lengths)}, {max(lengths)}]")


# Example usage
if __name__ == "__main__":
    # Replace 'rl_trajectories.json' with your actual file path
    #json_file_path = 'strategy_trajectories.json'
    json_file_path = 'human_trajectories.json'

    try:
        # Generate the heatmap plots
        plot_trajectory_heatmap(json_file_path)

        # Print statistical analysis
        analyze_trajectory_stats(json_file_path)

    except FileNotFoundError:
        print(f"File '{json_file_path}' not found. Please check the file path.")
    except json.JSONDecodeError:
        print("Error decoding JSON file. Please check the file format.")
    except Exception as e:
        print(f"An error occurred: {e}")