import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torch.distributions import Normal
import matplotlib.pyplot as plt
from collections import deque
import wandb
import os
from datetime import datetime
from env import MAISREnv


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.batch_size = batch_size

    def store(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def get_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.values),
            np.array(self.log_probs),
            np.array(self.dones),
            batches
        )


class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Output scaled between -1 and 1 for continuous actions
        )

        # Learnable standard deviation
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, state):
        mu = self.actor(state)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        return dist


class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.critic(state)


class PPOAgent:
    def __init__(
            self,
            env,
            hidden_dim=256,
            lr_actor=3e-4,
            lr_critic=1e-3,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            batch_size=64,
            n_epochs=10,
            device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.device = device

        # Initialize networks
        self.actor = ActorNetwork(env.observation_space.shape[0], env.action_space.shape[0], hidden_dim).to(device)
        self.critic = CriticNetwork(env.observation_space.shape[0], hidden_dim).to(device)

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Initialize memory
        self.memory = PPOMemory(batch_size)

        # Training metrics
        self.actor_losses = []
        self.critic_losses = []
        self.episode_rewards = []

    def choose_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        dist = self.actor(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(state)

        return action.cpu().detach().numpy(), value.cpu().detach().numpy(), log_prob.cpu().detach().numpy()

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = np.array(advantages)
        return advantages

    def update(self):
        states, actions, rewards, values, old_log_probs, dones, batches = self.memory.get_batches()
        advantages = self.compute_gae(rewards, values, dones)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        values = torch.FloatTensor(values).to(self.device)

        for _ in range(self.n_epochs):
            for batch in batches:
                dist = self.actor(states[batch])
                new_log_probs = dist.log_prob(actions[batch]).sum(dim=-1)
                entropy = dist.entropy().mean()

                new_values = self.critic(states[batch]).squeeze()

                # Compute ratio
                ratio = torch.exp(new_log_probs - old_log_probs[batch])

                # Compute surrogate losses
                surr1 = ratio * advantages[batch]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[batch]

                # Compute actor loss
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

                # Compute critic loss
                critic_loss = nn.MSELoss()(new_values, values[batch])

                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                self.actor_losses.append(actor_loss.item())
                self.critic_losses.append(critic_loss.item())

        self.memory.clear()


def train(env_config, num_episodes=1000, max_steps=1000):
    # Initialize wandb
    wandb.init(
        project="maisr-ppo",
        config={
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "env_config": env_config,
        }
    )

    # Create environment
    env = MAISREnv(config=env_config, render=False)

    # Initialize agent
    agent = PPOAgent(env)

    # Training metrics
    best_reward = float('-inf')
    reward_history = []
    episode_length_history = []

    # Create directory for checkpoints
    checkpoint_dir = f"checkpoints/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        step = 0

        while step < max_steps:
            # Choose action
            action, value, log_prob = agent.choose_action(state)

            # Take action in environment
            next_state, reward, done, _ = env.step([(env.aircraft_ids[0], action)])

            # Store transition
            agent.memory.store(state, action, reward, value, log_prob, done)

            episode_reward += reward
            state = next_state
            step += 1

            if done:
                break

        # Update agent
        agent.update()

        # Log metrics
        reward_history.append(episode_reward)
        episode_length_history.append(step)

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'reward': best_reward
            }, f"{checkpoint_dir}/best_model.pt")

        # Regular checkpoint every 100 episodes
        if episode % 100 == 0:
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'reward': episode_reward
            }, f"{checkpoint_dir}/checkpoint_{episode}.pt")

        # Log to wandb
        wandb.log({
            "episode": episode,
            "reward": episode_reward,
            "episode_length": step,
            "actor_loss": np.mean(agent.actor_losses[-100:]),
            "critic_loss": np.mean(agent.critic_losses[-100:])
        })

        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode}")
            print(f"Reward: {episode_reward:.2f}")
            print(f"Episode Length: {step}")
            print(f"Average Actor Loss: {np.mean(agent.actor_losses[-100:]):.4f}")
            print(f"Average Critic Loss: {np.mean(agent.critic_losses[-100:]):.4f}")
            print("-" * 50)

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(reward_history)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(1, 2, 2)
    plt.plot(episode_length_history)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')

    plt.tight_layout()
    plt.savefig(f"{checkpoint_dir}/training_curves.png")
    wandb.log({"training_curves": wandb.Image(plt)})

    wandb.finish()
    return agent


if __name__ == "__main__":
    # Load environment config
    env_config = {
        "gameboard size": 1000,
        "window size": (1450, 1080),
        "gameboard border margin": 50,
        "num ships": 20,
        "num aircraft": 2,
        "game speed": 1,
        "human speed": 1,
        "time limit": 240,
        "infinite health": False,
        "missiles_enabled": True,
        "surveys_enabled": False,
        "show agent waypoint": 2,
        "show_low_level_goals": True,
        "show_high_level_goals": True,
        "show_high_level_rationale": True,
        "show_tracked_factors": True,
        "agent start location": (500, 500),
        "human start location": (500, 500),
        "verbose": False
    }

    # Train agent
    agent = train(env_config)