import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import os


def load_level_data(level_name="level_1a", level_data_path="level_layouts.json"):
    """Load level data from JSON file"""
    with open(level_data_path, 'r') as f:
        level_data = json.load(f)['levels']

    level = level_data[level_name]

    # Extract positions
    agents = np.array(level['agents'], dtype=np.float32)
    targets = np.array(level['targets'], dtype=np.float32)
    threats = np.array(level['threats'], dtype=np.float32)

    return agents, targets, threats


def calculate_nearest_vectors(x, y, targets, threats):
    """Calculate unit vectors to nearest target and threat from position (x, y)"""
    current_pos = np.array([x, y])

    # Calculate distances to all targets
    target_distances = np.sqrt(np.sum((targets - current_pos) ** 2, axis=1))
    nearest_target_idx = np.argmin(target_distances)
    nearest_target = targets[nearest_target_idx]

    # Calculate distances to all threats
    threat_distances = np.sqrt(np.sum((threats - current_pos) ** 2, axis=1))
    nearest_threat_idx = np.argmin(threat_distances)
    nearest_threat = threats[nearest_threat_idx]

    # Calculate unit vectors
    target_vector = nearest_target - current_pos
    target_distance = np.linalg.norm(target_vector)
    target_unit_vector = target_vector / target_distance if target_distance > 0 else np.array([0, 0])

    threat_vector = nearest_threat - current_pos
    threat_distance = np.linalg.norm(threat_vector)
    threat_unit_vector = threat_vector / threat_distance if threat_distance > 0 else np.array([0, 0])

    return target_unit_vector, threat_unit_vector, target_distance, threat_distance


def create_value_field_plot(level_name="level_1a", grid_resolution=100, arrow_scale=0.8):
    """Create a value field plot showing arrows to nearest targets and threats"""

    # Load level data
    agents, targets, threats = load_level_data(level_name)

    # Set up map bounds (matching the environment's coordinate system)
    map_half_size = 500  # Assuming 1000x1000 map based on coordinate ranges

    # Create grid of positions
    x_positions = np.linspace(-map_half_size, map_half_size, grid_resolution)
    y_positions = np.linspace(-map_half_size, map_half_size, grid_resolution)

    # Create the plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: Target Field
    ax1.set_xlim(-map_half_size, map_half_size)
    ax1.set_ylim(-map_half_size, map_half_size)
    ax1.set_title(f'{level_name} - Target Field\n(Arrows point to nearest target)', fontsize=14)
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Threat Field
    ax2.set_xlim(-map_half_size, map_half_size)
    ax2.set_ylim(-map_half_size, map_half_size)
    ax2.set_title(f'{level_name} - Threat Field\n(Arrows point to nearest threat)', fontsize=14)
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Combined Field
    ax3.set_xlim(-map_half_size, map_half_size)
    ax3.set_ylim(-map_half_size, map_half_size)
    ax3.set_title(f'{level_name} - Combined Field\n(Green: targets, Red: threats)', fontsize=14)
    ax3.set_xlabel('X Position')
    ax3.set_ylabel('Y Position')
    ax3.grid(True, alpha=0.3)

    # Calculate step size for arrows
    step_x = (2 * map_half_size) / (grid_resolution - 1)
    step_y = (2 * map_half_size) / (grid_resolution - 1)
    arrow_length = min(step_x, step_y) * arrow_scale

    # Generate vector field
    for i, x in enumerate(x_positions):
        for j, y in enumerate(y_positions):
            target_vector, threat_vector, target_dist, threat_dist = calculate_nearest_vectors(x, y, targets, threats)

            # Plot 1: Target arrows (green)
            if np.linalg.norm(target_vector) > 0:
                ax1.arrow(x, y,
                          target_vector[0] * arrow_length,
                          target_vector[1] * arrow_length,
                          head_width=arrow_length * 0.3,
                          head_length=arrow_length * 0.3,
                          fc='green', ec='green', alpha=0.7, linewidth=0.5)

            # Plot 2: Threat arrows (red)
            if np.linalg.norm(threat_vector) > 0:
                ax2.arrow(x, y,
                          threat_vector[0] * arrow_length,
                          threat_vector[1] * arrow_length,
                          head_width=arrow_length * 0.3,
                          head_length=arrow_length * 0.3,
                          fc='red', ec='red', alpha=0.7, linewidth=0.5)

            # Plot 3: Combined arrows (scaled by distance)
            if np.linalg.norm(target_vector) > 0:
                # Make target arrows more prominent when targets are closer
                alpha_target = min(1.0, 200.0 / max(target_dist, 50))
                ax3.arrow(x, y,
                          target_vector[0] * arrow_length,
                          target_vector[1] * arrow_length,
                          head_width=arrow_length * 0.25,
                          head_length=arrow_length * 0.25,
                          fc='green', ec='green', alpha=alpha_target, linewidth=0.5)

            if np.linalg.norm(threat_vector) > 0:
                # Make threat arrows more prominent when threats are closer
                alpha_threat = min(1.0, 200.0 / max(threat_dist, 50))
                ax3.arrow(x, y,
                          threat_vector[0] * arrow_length * 0.7,
                          threat_vector[1] * arrow_length * 0.7,
                          head_width=arrow_length * 0.2,
                          head_length=arrow_length * 0.2,
                          fc='red', ec='red', alpha=alpha_threat, linewidth=0.5)

    # Add game elements to all plots
    for ax in [ax1, ax2, ax3]:
        # Plot targets
        for i, target in enumerate(targets):
            ax.scatter(target[0], target[1], s=100, color='orange',
                       marker='o', edgecolors='black', linewidth=2,
                       label='Targets' if i == 0 else "", zorder=5)

        # Plot threats
        for i, threat in enumerate(threats):
            # Draw threat with circle (representing threat radius)
            threat_radius = 80  # Approximate threat radius from the code
            circle = plt.Circle((threat[0], threat[1]), threat_radius,
                                fill=False, color='red', linewidth=2, alpha=0.7)
            ax.add_patch(circle)

            # Draw threat marker
            ax.scatter(threat[0], threat[1], s=200, color='red',
                       marker='v', edgecolors='black', linewidth=2,
                       label='Threats' if i == 0 else "", zorder=5)

        # Plot agent starting positions
        for i, agent in enumerate(agents):
            ax.scatter(agent[0], agent[1], s=150, color='blue',
                       marker='*', edgecolors='black', linewidth=2,
                       label='Agent Start' if i == 0 else "", zorder=5)

        # Add center lines
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize='small')

    # Adjust layout and save
    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)

    # Save the plot
    filename = f'plots/value_field_{level_name}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Value field plot saved to {filename}")

    plt.show()


def create_potential_field_plot(level_name="level_1a", grid_resolution=30):
    """Create a potential field plot showing the 'potential' values as a heatmap"""

    # Load level data
    agents, targets, threats = load_level_data(level_name)

    # Set up map bounds
    map_half_size = 500

    # Create grid of positions
    x_positions = np.linspace(-map_half_size, map_half_size, grid_resolution)
    y_positions = np.linspace(-map_half_size, map_half_size, grid_resolution)

    # Initialize potential grids
    target_potential = np.zeros((grid_resolution, grid_resolution))
    threat_potential = np.zeros((grid_resolution, grid_resolution))

    # Calculate potential fields
    for i, x in enumerate(x_positions):
        for j, y in enumerate(y_positions):
            current_pos = np.array([x, y])

            # Target potential (negative distance to nearest target)
            target_distances = np.sqrt(np.sum((targets - current_pos) ** 2, axis=1))
            target_potential[j, i] = -np.min(target_distances)  # Note: j,i for correct orientation

            # Threat potential (negative distance to nearest threat)
            threat_distances = np.sqrt(np.sum((threats - current_pos) ** 2, axis=1))
            threat_potential[j, i] = -np.min(threat_distances)

    # Create the plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: Target Potential
    im1 = ax1.imshow(target_potential, extent=[-map_half_size, map_half_size, -map_half_size, map_half_size],
                     origin='lower', cmap='RdYlBu_r', alpha=0.8)
    ax1.set_title(f'{level_name} - Target Potential\n(Higher values = closer to targets)', fontsize=14)
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    plt.colorbar(im1, ax=ax1, label='Potential Value')

    # Plot 2: Threat Potential
    im2 = ax2.imshow(threat_potential, extent=[-map_half_size, map_half_size, -map_half_size, map_half_size],
                     origin='lower', cmap='RdYlBu_r', alpha=0.8)
    ax2.set_title(f'{level_name} - Threat Potential\n(Higher values = closer to threats)', fontsize=14)
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    plt.colorbar(im2, ax=ax2, label='Potential Value')

    # Plot 3: Combined Potential (target potential - threat potential)
    combined_potential = target_potential - 0.5 * threat_potential  # Weight threats less
    im3 = ax3.imshow(combined_potential, extent=[-map_half_size, map_half_size, -map_half_size, map_half_size],
                     origin='lower', cmap='RdYlBu_r', alpha=0.8)
    ax3.set_title(f'{level_name} - Combined Potential\n(Target attraction - Threat repulsion)', fontsize=14)
    ax3.set_xlabel('X Position')
    ax3.set_ylabel('Y Position')
    plt.colorbar(im3, ax=ax3, label='Potential Value')

    # Add game elements to all plots
    for ax in [ax1, ax2, ax3]:
        # Plot targets
        for target in targets:
            ax.scatter(target[0], target[1], s=100, color='orange',
                       marker='o', edgecolors='black', linewidth=2, zorder=5)

        # Plot threats
        for threat in threats:
            threat_radius = 80
            circle = plt.Circle((threat[0], threat[1]), threat_radius,
                                fill=False, color='red', linewidth=2, alpha=0.9)
            ax.add_patch(circle)
            ax.scatter(threat[0], threat[1], s=200, color='red',
                       marker='v', edgecolors='black', linewidth=2, zorder=5)

        # Plot agent starting positions
        for agent in agents:
            ax.scatter(agent[0], agent[1], s=150, color='blue',
                       marker='*', edgecolors='black', linewidth=2, zorder=5)

        # Add center lines
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    filename = f'plots/potential_field_{level_name}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Potential field plot saved to {filename}")

    plt.show()


if __name__ == "__main__":
    # Create both types of plots
    print("Creating value field plot...")
    create_value_field_plot(level_name="level_1a", grid_resolution=100)

    print("Creating potential field plot...")
    create_potential_field_plot(level_name="level_1a", grid_resolution=100)

    print("Plots complete!")