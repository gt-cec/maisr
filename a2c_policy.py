import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MAISRActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim_continuous, act_dim_discrete, gameboard_size):
        super(MAISRActorCritic, self).__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # Actor network - waypoint head
        self.waypoint_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim_continuous),
            nn.Tanh()  # Output in range [-1, 1]
        )

        # Actor network - ID method head
        # self.id_method_head = nn.Sequential(
        #     nn.Linear(128, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, act_dim_discrete)
        # )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.gameboard_size = gameboard_size

    def forward(self, obs):
        features = self.shared(obs)

        # Actor outputs
        waypoint_raw = self.waypoint_head(features)
        #id_logits = self.id_method_head(features)

        # Scale waypoint from [-1, 1] to [0, gameboard_size]
        waypoint = (waypoint_raw + 1) * (self.gameboard_size / 2)

        # Value estimate
        value = self.critic(features)

        return waypoint, value

    def act(self, obs):
        with torch.no_grad():
            waypoint, value, = self.forward(obs)
            #id_probs = F.softmax(id_logits, dim=-1)
            #id_method = torch.multinomial(id_probs, 1).item()

        return waypoint.cpu().numpy()
        #return {'waypoint': waypoint.cpu().numpy(), 'id_method': id_method}

    def get_value(self, obs):
        with torch.no_grad():
            _, value = self.forward(obs)
        return value.item()