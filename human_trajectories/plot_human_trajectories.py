import json
import matplotlib.pyplot as plt

# Load the JSON file
file_path = 'timesteps_A2_20250723_153039.json'
with open(file_path, 'r') as f:
    data = json.load(f)

# Extract human positions
human_positions = [timestep['human_position'] for timestep in data['timesteps']]
human_x, human_y = zip(*human_positions)

agent_positions = [timestep['agent_position'] for timestep in data['timesteps']]
agent_x, agent_y = zip(*agent_positions)

# Extract target and threat positions for plotting
initial_targets = data['timesteps'][0]['target_positions']
initial_threats = data['timesteps'][0]['threat_positions']
target_x, target_y = zip(*initial_targets)
threat_x, threat_y = zip(*initial_threats)

# Plotting
plt.figure(figsize=(10, 10))
plt.plot(human_x, human_y, marker='o', linestyle='-', color='blue', label='Human Trajectory')
#plt.plot(agent_x, agent_y, marker='o', linestyle='-', color='green', label='Agent Trajectory')

# Plot targets and threats
plt.scatter(target_x, target_y, color='green', marker='*', s=100, label='Targets')
plt.scatter(threat_x, threat_y, color='red', marker='X', s=100, label='Threats')

# Mark initial and final positions
#plt.scatter(x[0], y[0], color='cyan', edgecolor='black', s=200, label='Start')
#plt.scatter(x[-1], y[-1], color='orange', edgecolor='black', s=200, label='End')

plt.title('Human Position Trajectory')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.grid(True)
plt.axis('equal')
plt.legend()

# Display the plot
plt.show()
