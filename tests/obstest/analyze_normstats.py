import json
import matplotlib.pyplot as plt
import pandas as pd

# Load the JSON data
file_path = "./outputs/logs/normstat_diffs.json"
with open(file_path, "r") as f:
    data = json.load(f)

# Convert JSON into a DataFrame
df = pd.DataFrame(data)

# Get the number of observation elements
num_obs = len(df["mean_diffs"][0])

# Create individual time series for each mean_diff and var_diff
mean_diff_series = {i: [] for i in range(num_obs)}
var_diff_series = {i: [] for i in range(num_obs)}
steps = df["step"]

for row in df.itertuples():
    for i in range(num_obs):
        mean_diff_series[i].append(row.mean_diffs[i])
        var_diff_series[i].append(row.var_diffs[i])

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# Mean diffs
for i in range(num_obs):
    axs[0].plot(steps, mean_diff_series[i], label=f'mean_diff[{i}]', alpha=0.5)
axs[0].plot(steps, df["avg_mean_diff"], label='avg_mean_diff', color='black', linewidth=2)
axs[0].set_ylabel("Mean Differences")
axs[0].set_title("Mean Differences Over Time")
axs[0].legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))

# Var diffs
for i in range(num_obs):
    axs[1].plot(steps, var_diff_series[i], label=f'var_diff[{i}]', alpha=0.5)
axs[1].plot(steps, df["avg_var_diff"], label='avg_var_diff', color='black', linewidth=2)
axs[1].set_xlabel("Timesteps")
axs[1].set_ylabel("Variance Differences")
axs[1].set_title("Variance Differences Over Time")
axs[1].legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))

plt.tight_layout()
plt.show()
