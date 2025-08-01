import json
import matplotlib.pyplot as plt

# Load data
with open("./userstudy_logs/subject_99/timestep_data/timesteps_A3_20250717_172853.json") as f:
    data = json.load(f)["timesteps"]

# Extract fields
timesteps = [d["timestep"] for d in data]
agent_x = [d["agent_position"][0] for d in data]
agent_y = [d["agent_position"][1] for d in data]
agent_action = [d["agent_action"] for d in data]

# Plot agent position
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(timesteps, agent_x, label="x")
plt.plot(timesteps, agent_y, label="y")
plt.title("Agent Position vs Timestep")
plt.xlabel("Timestep")
plt.ylabel("Position")
plt.legend()

# Plot agent action
plt.subplot(1, 2, 2)
plt.plot(timesteps, agent_action)
plt.title("Agent Action vs Timestep")
plt.xlabel("Timestep")
plt.ylabel("Action")

plt.tight_layout()
plt.show()
