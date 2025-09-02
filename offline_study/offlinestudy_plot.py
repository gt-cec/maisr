import matplotlib.pyplot as plt
import numpy as np

# Data from the performance summary
# This comes from n=160 study run
rl_data = {
    'FCP': {'mean': 25.03, 'std': 0.00, 'n': 160},
    'Strat-FCP': {'mean': 30.63, 'std': 2.41, 'n': 160},
    'SP': {'mean': 24.91, 'std': 0.00, 'n': 160},
    'Strat-SP': {'mean': 26.03, 'std': 0.00, 'n': 160}
}

human_data = {
    'FCP': {'mean': 24.71, 'std': 3.30, 'n': 160},
    'Strat-FCP': {'mean': 30.76, 'std': 2.92, 'n': 160},
    'SP': {'mean': 23.92, 'std': 2.08, 'n': 160},
    'Strat-SP': {'mean': 28.45, 'std': 2.43, 'n': 160}
}

# Extract agent names and values
agents = list(rl_data.keys())
rl_means = [rl_data[agent]['mean'] for agent in agents]
rl_stds = [rl_data[agent]['std'] for agent in agents]
human_means = [human_data[agent]['mean'] for agent in agents]
human_stds = [human_data[agent]['std'] for agent in agents]

# Create 2-subplot figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Left subplot: RL results
# Add grid first (behind bars)
ax1.grid(axis='y', alpha=0.3, zorder=0)
bars1 = ax1.bar(range(len(agents)), rl_means,
                yerr=rl_stds, capsize=5, alpha=1, color='skyblue', zorder=2)
#ax1.set_xlabel('Testing Agents', fontsize=20)
ax1.set_ylabel('Average Reward', fontsize=20)
ax1.set_title('With Held-Out RL Teammates', fontsize=20)
ax1.set_xticks(range(len(agents)))
ax1.set_xticklabels(agents, rotation=0, fontsize=20)
ax1.tick_params(axis='y', labelsize=18)

# Add value labels on RL bars
# for i, (bar, mean_val, std_val) in enumerate(zip(bars1, rl_means, rl_stds)):
#     y_pos = bar.get_height() + std_val + 0.2
#     ax1.text(bar.get_x() + bar.get_width() / 2, y_pos,
#              f'{mean_val:.1f}', ha='center', va='bottom', fontsize=15)

# Right subplot: Human results
# Add grid first (behind bars)
ax2.grid(axis='y', alpha=0.3, zorder=0)
bars2 = ax2.bar(range(len(agents)), human_means,
                yerr=human_stds, capsize=5, alpha=1, color='lightcoral', zorder=2)
#ax2.set_xlabel('Testing Agents', fontsize=20)
#ax2.set_ylabel('Average Reward', fontsize=20)
ax2.set_title('With Recorded Human Teammates', fontsize=20)
ax2.set_xticks(range(len(agents)))
ax2.set_xticklabels(agents, rotation=0, fontsize=20)
ax2.tick_params(axis='y', labelsize=0)

# Add value labels on Human bars
# for i, (bar, mean_val, std_val) in enumerate(zip(bars2, human_means, human_stds)):
#     y_pos = bar.get_height() + std_val + 0.2
#     ax2.text(bar.get_x() + bar.get_width() / 2, y_pos,
#              f'{mean_val:.1f}', ha='center', va='bottom', fontsize=15)

# Set consistent y-axis limits starting from zero
y_max = max(max(np.array(rl_means) + np.array(rl_stds)),
            max(np.array(human_means) + np.array(human_stds))) + 2
ax1.set_ylim(0, y_max)
ax2.set_ylim(0, y_max)

plt.tight_layout()
plt.show()