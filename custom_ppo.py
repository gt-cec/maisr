import torch
import numpy as np
import torch.nn.functional as F
import os
import time
from datetime import datetime

# TODO test this

class PPOTrainer:
    def __init__(self, env, model, checkpoint_dir, lr=3e-4, gamma=0.99, clip_ratio=0.2, target_kl=0.01):
        self.env = env
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl

        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Track best performance for saving best model
        self.best_reward = float('-inf')


    def save_checkpoint(self, epoch, avg_reward, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'reward': avg_reward,
        }

        # Save regular checkpoint
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.checkpoint_dir}/checkpoint_epoch_{epoch}_{timestamp}.pt"
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")

        # Save best model if this is the best performance
        if is_best:
            best_filename = f"{self.checkpoint_dir}/best_model.pt"
            torch.save(checkpoint, best_filename)
            print(f"Best model saved to {best_filename}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} does not exist")
            return False

        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with reward {checkpoint['reward']}")
        return checkpoint['epoch']


    def compute_gae(self, rewards, values, next_value, dones, gamma=0.99, lam=0.95):
        # Generalized Advantage Estimation
        advantages = np.zeros_like(rewards)
        lastgaelam = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - dones[-1]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]

            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam

        returns = advantages + values
        return returns, advantages

    def train(self, epochs=10000, steps_per_epoch=4000, batch_size=64, checkpoint_path=None, save_freq=20):

        # Resume from checkpoint if specified
        start_epoch = 0
        if checkpoint_path is not None:
            loaded_epoch = self.load_checkpoint(checkpoint_path)
            if loaded_epoch:
                start_epoch = loaded_epoch + 1

        self.env.agents[self.env.human_idx].x, self.env.agents[self.env.human_idx].y = 2000, 2000 # TODO hack

        epoch_rewards = [] # Track rewards for saving best model
        epoch_times = [] # Track epoch times

        for epoch in range(start_epoch, epochs):

            epoch_start_time = time.time()

            # Collect experience
            observations, actions, waypoints, id_methods, rewards, values, dones = [], [], [], [], [], [], []
            episode_rewards = []
            current_episode_reward = 0

            obs = self.env.reset()
            done = False

            while not done:
                action = self.model.act(torch.tensor(obs, dtype=torch.float32))
                value = self.model.get_value(torch.tensor(obs, dtype=torch.float32))

                next_obs, reward, terminated, truncated, info = self.env.step([
                    (self.env.aircraft_ids[0], action),
                    (self.env.aircraft_ids[1], {'waypoint': (2000,2000), 'id_method': 0}),
                ])

                # Store experience
                observations.append(obs)
                waypoints.append(action['waypoint'])
                id_methods.append(action['id_method'])
                rewards.append(reward)
                values.append(value)
                dones.append(done)

                current_episode_reward += reward

                done = truncated or terminated

                if done:
                    obs = self.env.reset()
                    print(f"Episode finished: Reward = {current_episode_reward}")
                    episode_rewards.append(current_episode_reward)
                    current_episode_reward = 0
                else:
                    obs = next_obs

            # Calculate average episode reward
            avg_episode_reward = np.mean(episode_rewards) if episode_rewards else 0
            epoch_rewards.append(avg_episode_reward)

            # Compute returns and advantages
            last_value = self.model.get_value(torch.tensor(obs, dtype=torch.float32))
            returns, advantages = self.compute_gae(
                rewards, values, last_value, dones, self.gamma
            )

            # Convert to tensors
            obs_tensor = torch.tensor(np.array(observations), dtype=torch.float32)
            waypoint_tensor = torch.tensor(np.array(waypoints), dtype=torch.float32)
            id_method_tensor = torch.tensor(np.array(id_methods), dtype=torch.long)
            returns_tensor = torch.tensor(returns, dtype=torch.float32)
            advantages_tensor = torch.tensor(advantages, dtype=torch.float32)

            # Normalize advantages
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

            # PPO update
            for _ in range(10):  # Multiple epochs of optimization
                # Sample mini-batches
                indices = np.random.permutation(len(observations))
                for start in range(0, len(observations), batch_size):
                    end = start + batch_size
                    idx = indices[start:end]

                    # Get mini-batch tensors
                    mb_obs = obs_tensor[idx]
                    mb_waypoints = waypoint_tensor[idx]
                    mb_id_methods = id_method_tensor[idx]
                    mb_returns = returns_tensor[idx]
                    mb_advantages = advantages_tensor[idx]

                    # Forward pass
                    new_waypoints, new_id_logits, values = self.model(mb_obs)

                    # Compute policy loss for waypoints (MSE for continuous actions)
                    waypoint_loss = F.mse_loss(new_waypoints, mb_waypoints)

                    # Compute policy loss for id_method (cross entropy for discrete actions)
                    id_loss = F.cross_entropy(new_id_logits, mb_id_methods)

                    # Compute value loss
                    value_loss = F.mse_loss(values.squeeze(), mb_returns)

                    # Combined loss
                    loss = waypoint_loss + id_loss + 0.5 * value_loss

                    # Optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            # End timing the epoch
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_duration)

            # Log metrics
            print(f"Epoch {epoch} stats:")
            print(f"  Average episode reward: {avg_episode_reward:.4f}")
            print(f"  Value loss: {value_loss.item():.4f}")
            print(f"  Waypoint loss: {waypoint_loss.item():.4f}")
            print(f"  ID method loss: {id_loss.item():.4f}")

            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            print(f"  Average epoch time: {avg_epoch_time:.2f} seconds")

            # Check if this is the best model
            # TODO need to compare this to the saved model's best too, not just best for this run
            is_best = avg_episode_reward > self.best_reward
            if is_best:
                self.best_reward = avg_episode_reward

            # Save checkpoint based on frequency or if it's the best model
            if epoch % save_freq == 0 or is_best or epoch == epochs - 1:
                self.save_checkpoint(epoch, avg_episode_reward, is_best)

    # TODO save checkpoints
    # Integrate wanbd