import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import math
import time
import json
import pygame
from datetime import datetime

from env import MAISREnv
from reinforcement_learning.rl_policy import PolicyNetwork

"""Draft training framework implementation. Not tested. """

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class PPOTrainer:
    def __init__(self, env_config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        # Initialize pygame and font system without display
        pygame.init()
        pygame.font.init()

        # Create a dummy surface since we're not rendering
        pygame.display.set_mode((1, 1), pygame.NOFRAME)

        # Initialize environment
        self.env = MAISREnv(env_config, render=False)

        # Rest of the initialization code remains the same...
        state_dim = self._get_state_dim()
        action_dim = 2  # (x, y) coordinates

        self.policy = PolicyNetwork(state_dim, action_dim * 2).to(device)  # *2 for mean and std
        self.value = ValueNetwork(state_dim).to(device)

        # Training hyperparameters
        self.lr = 3e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.c1 = 1.0  # Value loss coefficient
        self.c2 = 0.01  # Entropy coefficient
        self.batch_size = 64
        self.num_epochs = 10
        self.rollout_length = 2048

        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.lr)

        # Logging
        self.rewards_history = []
        self.value_losses = []
        self.policy_losses = []
        self.entropy_losses = []

    def _get_state_dim(self):
        """Calculate state dimension"""
        ship_features = 6  # Same as in RLPolicy
        num_ships = len([a for a in self.env.agents if a.agent_class == "ship"])
        return 3 + (ship_features * num_ships)

    def process_state(self, state_dict):
        """Convert state dictionary to flat array for RL agent"""
        aircraft = self.env.agents[self.env.aircraft_ids[0]]  # Get our aircraft
        game_size = self.env.config["gameboard size"]

        state_array = []

        # Aircraft features
        state_array.extend([
            aircraft.x / game_size,  # Normalized x position
            aircraft.y / game_size,  # Normalized y position
            aircraft.damage / 100.0  # Normalized damage
        ])

        # Process each ship
        for agent in self.env.agents:
            if agent.agent_class == "ship":
                # Calculate relative position
                rel_x = (agent.x - aircraft.x) / game_size
                rel_y = (agent.y - aircraft.y) / game_size

                # Calculate distance and angle
                dist = math.hypot(rel_x, rel_y)
                angle = math.atan2(rel_y, rel_x) / math.pi

                state_array.extend([
                    rel_x,
                    rel_y,
                    float(agent.observed),
                    float(agent.observed_threat),
                    dist,
                    angle
                ])

        return np.array(state_array, dtype=np.float32)

    def get_action(self, state_dict):
        """Sample action from current policy"""
        state = self.process_state(state_dict)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_params = self.policy(state)

            # Split into mean and log_std
            mean, log_std = torch.chunk(action_params, 2, dim=-1)
            std = log_std.exp()

            # Sample action
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        # Ensure we get scalar values for the action
        action = action.squeeze()

        # Convert normalized actions (-1 to 1) to game coordinates
        game_size = self.env.config["gameboard size"]
        margin = self.env.config["gameboard border margin"]

        # Convert to scalar values
        action_x = action[0].item()
        action_y = action[1].item()

        # Convert to game coordinates
        target_x = ((action_x + 1) / 2) * (game_size - 2 * margin) + margin
        target_y = ((action_y + 1) / 2) * (game_size - 2 * margin) + margin

        return (target_x, target_y), log_prob.cpu().detach().numpy()

    def compute_gae(self, rewards, values, next_value, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]

            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def update_policy(self, states, actions, old_log_probs, advantages, returns):
        """Update policy and value networks"""
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.num_epochs):
            # Get current policy distribution
            action_params = self.policy(states)
            mean, log_std = torch.chunk(action_params, 2, dim=-1)
            std = log_std.exp()

            # Create distribution
            dist = Normal(mean, std)

            # Calculate new log probs and entropy
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().mean()

            # Calculate policy ratio and clipped objective
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Calculate value loss
            value_pred = self.value(states).squeeze()
            value_loss = nn.MSELoss()(value_pred, returns)

            # Combined loss
            loss = policy_loss + self.c1 * value_loss - self.c2 * entropy

            # Update networks
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()

            # Optional: Clip gradients
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=0.5)

            self.policy_optimizer.step()
            self.value_optimizer.step()

            # Log losses
            self.policy_losses.append(policy_loss.item())
            self.value_losses.append(value_loss.item())
            self.entropy_losses.append(entropy.item())

    def collect_rollout(self):
        """Collect experience using current policy"""
        states, actions, rewards, dones = [], [], [], []
        log_probs, values = [], []

        state = self.env.reset()
        done = False
        episode_reward = 0

        while len(states) < self.rollout_length:
            # Get action from policy
            action, log_prob = self.get_action(state)
            processed_state = self.process_state(state)

            # Get value estimate with proper detachment
            with torch.no_grad():
                value = self.value(torch.FloatTensor(processed_state).unsqueeze(0).to(self.device))

            # Take step in environment
            next_state, reward, done, _ = self.env.step([(self.env.aircraft_ids[0], action)])

            # Convert action back to normalized form for storage
            game_size = self.env.config["gameboard size"]
            margin = self.env.config["gameboard border margin"]

            action_normalized = [
                (action[0] - margin) / (game_size - 2 * margin) * 2 - 1,
                (action[1] - margin) / (game_size - 2 * margin) * 2 - 1
            ]

            # Store experience
            states.append(processed_state)
            actions.append(action_normalized)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value.cpu().detach().numpy())

            episode_reward += reward

            if done:
                state = self.env.reset()
                self.rewards_history.append(episode_reward)
                episode_reward = 0
            else:
                state = next_state

        # Get final value for GAE calculation
        final_state = self.process_state(state)
        with torch.no_grad():
            final_value = self.value(torch.FloatTensor(final_state).unsqueeze(0).to(self.device))
        advantages, returns = self.compute_gae(rewards, values, final_value.cpu().detach().numpy(), dones)

        return np.array(states), np.array(actions), np.array(log_probs), advantages, returns


    def train(self, num_iterations=1000):
        """Main training loop"""
        start_time = time.time()

        for iteration in range(num_iterations):

            #pygame.init()  # init pygame
            #clock = pygame.time.Clock()

            # Collect experience
            states, actions, log_probs, advantages, returns = self.collect_rollout()

            # Update policy and value networks
            self.update_policy(states, actions, log_probs, advantages, returns)

            # Log progress
            if (iteration + 1) % 10 == 0:
                mean_reward = np.mean(self.rewards_history[-100:])
                mean_policy_loss = np.mean(self.policy_losses[-100:])
                mean_value_loss = np.mean(self.value_losses[-100:])
                mean_entropy = np.mean(self.entropy_losses[-100:])

                print(f"Iteration {iteration + 1}")
                print(f"Average Reward: {mean_reward:.2f}")
                print(f"Policy Loss: {mean_policy_loss:.4f}")
                print(f"Value Loss: {mean_value_loss:.4f}")
                print(f"Entropy: {mean_entropy:.4f}")
                print(f"Time Elapsed: {time.time() - start_time:.2f}s\n")

                # Save training metrics
                self.save_metrics(iteration)

            self.__del__()

            # Save model periodically
            if (iteration + 1) % 100 == 0:
                self.save_model(f"model_iter_{iteration + 1}")

    def save_model(self, filename):
        """Save model weights"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict()
        }, f'models/{filename}.pt')

    def save_metrics(self, iteration):
        """Save training metrics"""
        metrics = {
            'iteration': iteration,
            'rewards': self.rewards_history,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropy_losses': self.entropy_losses
        }

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f'logs/metrics_{timestamp}.json', 'w') as f:
            json.dump(metrics, f)

    def __del__(self):
        """Cleanup pygame resources"""
        pygame.quit()


if __name__ == "__main__":
    # Load environment config
    with open('../config_files/default_config.json', 'r') as f:
        env_config = json.load(f)

    # Initialize trainer
    trainer = PPOTrainer(env_config)

    try:
        # Train model
        trainer.train(num_iterations=1000)
    finally:
        # Ensure pygame quits properly
        pygame.quit()