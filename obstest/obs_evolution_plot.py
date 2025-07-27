import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Load JSON data from relative path
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, "../outputs/logs/norm_stats_history.json")
#file_path = os.path.join(script_dir, "725norm_stats_history.json")

with open(os.path.normpath(file_path), "r") as f:
    data = json.load(f)

# Extract number of observation elements
n_obs_elements = len(data[0]["training_env"]["obs_mean"]) - 8
ep_counts = [entry["training_env"]["ep_count"] for entry in data]

# Prepare storage for means and variances
training_means = [[] for _ in range(n_obs_elements)]
training_vars = [[] for _ in range(n_obs_elements)]
eval_means = [[] for _ in range(n_obs_elements)]
eval_vars = [[] for _ in range(n_obs_elements)]

# Populate the lists
for entry in data:
    for i in range(n_obs_elements):
        training_means[i].append(entry["training_env"]["obs_mean"][i])
        training_vars[i].append(entry["training_env"]["obs_var"][i])
        eval_means[i].append(entry["eval_env"]["obs_mean"][i])
        eval_vars[i].append(entry["eval_env"]["obs_var"][i])

# Create 4-row, ceil(n_obs_elements/2)-column subplot grid
n_cols = (n_obs_elements + 1) // 2
fig, axes = plt.subplots(nrows=4, ncols=n_cols, figsize=(4 * n_cols, 12), sharex=True)

for i in range(n_obs_elements):
    col = i // 2
    row_train = 0 if i % 2 == 0 else 1
    row_eval = 2 if i % 2 == 0 else 3

    axes[row_train, col].bar(ep_counts, training_means[i], yerr=np.sqrt(training_vars[i]), capsize=3,error_kw=dict(linewidth=0.6, alpha=0.2))
    axes[row_train, col].set_title(f"Obs {i} (Train)")

    axes[row_eval, col].bar(ep_counts, eval_means[i], yerr=np.sqrt(eval_vars[i]), capsize=3,error_kw=dict(linewidth=0.6, alpha=0.2))
    axes[row_eval, col].set_title(f"Obs {i} (Eval)")

# Label axes
# for ax in axes[3]:
#     ax.set_xlabel("Episode Count")

# for row in [0, 1]:
#     for ax in axes[row]:
#         ax.set_ylabel("Mean ± StdDev")

fig.tight_layout()
plt.show()
