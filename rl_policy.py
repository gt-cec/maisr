import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

"""Draft agent RL policy. Not yet tested."""

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Use tanh to bound actions between -1 and 1


class RLPolicy:
    def __init__(self, env, aircraft_id, training=False):
        self.env = env
        self.aircraft_id = aircraft_id
        self.training = training

        # Initialize network
        self.state_dim = self._get_state_dim()
        self.action_dim = 2  # (x, y) continuous actions
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim)

        # For compatibility with game UI
        self.target_point = (0, 0)
        self.waypoint_override = False
        self.hold_commanded = False

        # For agent status display
        self.low_level_rationale = 'RL Agent Searching'
        self.high_level_rationale = 'Using learned policy'
        self.quadrant_rationale = ''
        self.nearby_threats = []
        self.three_upcoming_targets = []
        self.current_target_distance = 0
        self.risk_level = 'MEDIUM'
        self.search_quadrant = 'full'
        self.search_type = 'target'

    def _get_state_dim(self):
        """Calculate state dimension based on observations"""
        # State includes:
        # - Aircraft position (2)
        # - Aircraft damage (1)
        # - For each ship:
        #   - Relative position (2)
        #   - Observed status (1)
        #   - Observed threat level (1)
        #   - Distance (1)
        #   - Angle (1)
        ship_features = 6
        num_ships = len([a for a in self.env.agents if a.agent_class == "ship"])
        return 3 + (ship_features * num_ships)

    def get_state_representation(self):
        """Convert game state to network input"""
        aircraft = self.env.agents[self.aircraft_id]

        state = []
        # Aircraft features
        state.extend([aircraft.x / self.env.config["gameboard size"],
                      aircraft.y / self.env.config["gameboard size"],
                      aircraft.damage / 100.0])

        # Ship features
        for agent in self.env.agents:
            if agent.agent_class == "ship":
                # Calculate relative position
                rel_x = (agent.x - aircraft.x) / self.env.config["gameboard size"]
                rel_y = (agent.y - aircraft.y) / self.env.config["gameboard size"]

                # Calculate distance and angle
                dist = aircraft.distance(agent) / self.env.config["gameboard size"]
                angle = math.atan2(rel_y, rel_x) / math.pi

                state.extend([
                    rel_x,
                    rel_y,
                    float(agent.observed),
                    float(agent.observed_threat),
                    dist,
                    angle
                ])

        return torch.FloatTensor(state).unsqueeze(0)

    def act(self):
        """Generate action using policy network"""
        if self.hold_commanded:
            self.target_point = self.hold_policy()
            return

        if self.waypoint_override:
            self.target_point = self.waypoint_override
            return

        # Get state representation
        state = self.get_state_representation()

        # Get action from policy network
        with torch.no_grad():
            action = self.policy_net(state).squeeze().numpy()

        # Convert normalized actions (-1 to 1) to game coordinates
        game_size = self.env.config["gameboard size"]
        margin = self.env.config["gameboard border margin"]
        target_x = float(((action[0] + 1) / 2) * (game_size - 2 * margin) + margin)
        target_y = (float((action[1] + 1) / 2) * (game_size - 2 * margin) + margin)

        self.target_point = (float(target_x), float(target_y))

        # Update status display info
        self.update_agent_info()


    def hold_policy(self):
        """Stay in current position"""
        aircraft = self.env.agents[self.aircraft_id]
        return aircraft.x, aircraft.y


    def update_agent_info(self):
        """Update information shown in agent status window"""
        # Calculate risk level based on nearby threats
        aircraft = self.env.agents[self.aircraft_id]
        hostile_targets_nearby = sum(1 for agent in self.env.agents
                                     if agent.agent_class == "ship"
                                     and agent.threat > 0
                                     and agent.observed_threat
                                     and math.hypot(agent.x - aircraft.x, agent.y - aircraft.y) <= 30)

        risk_level_value = 10 * hostile_targets_nearby + aircraft.damage
        self.risk_level = ('LOW' if risk_level_value <= 30 else
                           'MEDIUM' if risk_level_value <= 60 else
                           'HIGH' if risk_level_value <= 80 else
                           'EXTREME')

        # Update nearby threats list
        self.nearby_threats = []
        threats = [(i, aircraft.distance(agent))
                   for i, agent in enumerate(self.env.agents)
                   if agent.agent_class == "ship" and agent.threat > 0]
        threats.sort(key=lambda x: x[1])
        self.nearby_threats = threats[:2]

        # Could also update other status info based on policy network's attention
        # or action distributions