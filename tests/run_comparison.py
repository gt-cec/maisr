import os
import re
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def combine_eval_figures(folder_paths, N, specific_episodes=None):
    # Extract and sort eval files for each folder
    folder_data = {}

    for folder_path in folder_paths:
        folder_name = os.path.basename(folder_path.rstrip('/'))
        eval_files = []

        # Find all eval_ep files
        for file in os.listdir(folder_path):
            match = re.match(r'eval_ep(\d{4})\.png', file)
            if match:
                episode_num = int(match.group(1))
                eval_files.append((episode_num, file))

        # Sort by episode number and take the N highest
        eval_files.sort(key=lambda x: x[0])
        selected_files = eval_files[-N:] if len(eval_files) >= N else eval_files


        folder_data[folder_name] = {
            'folder_path': folder_path,
            'files': selected_files
        }

    # Create the combined figure
    n_folders = len(folder_data)
    fig, axes = plt.subplots(n_folders, N, figsize=(N * 4, n_folders * 3))

    # Handle single row case
    if n_folders == 1:
        axes = axes.reshape(1, -1)

    for row_idx, (folder_name, data) in enumerate(folder_data.items()):
        for col_idx, (episode_num, filename) in enumerate(data['files']):
            img_path = os.path.join(data['folder_path'], filename)
            img = Image.open(img_path)

            axes[row_idx, col_idx].imshow(img)
            axes[row_idx, col_idx].set_title(f'eval_ep{episode_num:04d}')
            axes[row_idx, col_idx].axis('off')

        # # Add folder name as row label
        # axes[row_idx, 0].set_ylabel(folder_name, rotation=90,
        #                             labelpad=50, fontsize=12, fontweight='bold')

    for row_idx, folder_name in enumerate(folder_data.keys()):
        # Calculate the vertical center position for this row
        y_pos = 1 - (row_idx + 0.5) / n_folders

        # Add text label on the left side of the figure
        fig.text(0.02, y_pos, folder_name[24:-8],
                 rotation=90, ha='center', va='center',
                 fontsize=5, fontweight='bold')

    # Adjust layout to make room for row labels
    plt.subplots_adjust(left=0.1)

    plt.tight_layout()
    plt.savefig('combined_eval_figures.png', dpi=500, bbox_inches='tight')
    plt.show()


# Usage
if __name__ == "__main__":
    import sys
    root = './logs/action_histories/'
    folder_paths = [
        root + 'monolith_Monolithoverfit-greedy_planning_entreg-0.07_threat_potential_coeff0.15_0702_2243_',
        root + 'monolith_Monolithoverfit-cluster_planning_entreg-0.07_threat_potential_coeff0.15_0703_0100_',
        root + 'monolith_Monolithoverfit-high_risk_entreg-0.07_threat_potential_coeff0.15_0703_0221_',
        root + 'monolith_Monolithoverfit-low_risk_entreg-0.07_threat_potential_coeff0.15_0703_0336_',
        root + 'monolith_Monolithoverfit-greedy_planning_entreg-0.07_threat_potential_coeff0.2_0703_0451_',
        root + 'monolith_Monolithoverfit-cluster_planning_entreg-0.07_threat_potential_coeff0.2_0703_0712_'
    ]

    N = 8

    combine_eval_figures(folder_paths, N)