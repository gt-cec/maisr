import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from similarity_analysis import FullTrajectory


class TrajectoryDataset(Dataset):
    def __init__(self, observations, positions, transform=None):
        self.observations = torch.FloatTensor(observations)
        self.positions = torch.FloatTensor(positions)
        self.transform = transform

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = self.observations[idx]
        pos = self.positions[idx]

        if self.transform:
            obs = self.transform(obs)

        return obs, pos


class PositionMDN(nn.Module):
    def __init__(self, obs_dim, n_components=5, hidden_dim=128):
        super().__init__()
        self.n_components = n_components
        self.obs_dim = obs_dim

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Mixture component heads
        self.pi_head = nn.Linear(hidden_dim // 2, n_components)  # mixture weights
        self.mu_head = nn.Linear(hidden_dim // 2, n_components * 2)  # means (x,y)
        self.sigma_head = nn.Linear(hidden_dim // 2, n_components * 3)  # covariance params

    def forward(self, x):
        features = self.backbone(x)

        # Mixture weights (softmax normalized)
        pi = torch.softmax(self.pi_head(features), dim=-1)

        # Means for each component
        mu = self.mu_head(features).view(-1, self.n_components, 2)

        # Covariance parameters: [sigma_x, sigma_y, rho] for each component
        sigma_params = self.sigma_head(features).view(-1, self.n_components, 3)
        sigma_x = torch.exp(sigma_params[:, :, 0]) + 1e-6  # Ensure positive
        sigma_y = torch.exp(sigma_params[:, :, 1]) + 1e-6
        rho = torch.tanh(sigma_params[:, :, 2])  # Correlation coefficient [-1, 1]

        return pi, mu, sigma_x, sigma_y, rho

    def sample(self, x, n_samples=1000):
        """Sample positions from the mixture distribution"""
        with torch.no_grad():
            pi, mu, sigma_x, sigma_y, rho = self.forward(x)
            batch_size = x.shape[0]

            # Sample mixture components
            component_samples = torch.multinomial(pi, n_samples, replacement=True)

            samples = []
            for b in range(batch_size):
                batch_samples = []
                for s in range(n_samples):
                    comp = component_samples[b, s]
                    mean = mu[b, comp]
                    sx, sy, r = sigma_x[b, comp], sigma_y[b, comp], rho[b, comp]

                    # Create covariance matrix
                    cov = torch.tensor([[sx ** 2, r * sx * sy], [r * sx * sy, sy ** 2]])

                    # Sample from 2D Gaussian
                    sample = torch.distributions.MultivariateNormal(mean, cov).sample()
                    batch_samples.append(sample)

                samples.append(torch.stack(batch_samples))

            return torch.stack(samples)


def mdn_loss(pi, mu, sigma_x, sigma_y, rho, target):
    """Compute negative log-likelihood loss for MDN with numerical stability"""
    batch_size, n_components = pi.shape
    target_expanded = target.unsqueeze(1).expand(-1, n_components, -1)

    # Compute 2D Gaussian likelihood for each component
    dx = target_expanded[:, :, 0] - mu[:, :, 0]
    dy = target_expanded[:, :, 1] - mu[:, :, 1]

    # Clamp rho to prevent numerical issues
    rho = torch.clamp(rho, -0.99, 0.99)

    # Compute determinant and ensure it's positive
    det = sigma_x * sigma_y * torch.sqrt(1 - rho ** 2)
    det = torch.clamp(det, min=1e-6)

    # Mahalanobis distance with numerical stability
    rho_sq = rho ** 2
    inv_1_minus_rho_sq = 1.0 / torch.clamp(1 - rho_sq, min=1e-6)

    z = (dx ** 2 / (sigma_x ** 2) +
         dy ** 2 / (sigma_y ** 2) -
         2 * rho * dx * dy / (sigma_x * sigma_y)) * inv_1_minus_rho_sq

    # 2D Gaussian log probability density
    log_prob = -0.5 * (z + torch.log(torch.tensor(2 * np.pi)) + torch.log(det))

    # Weighted mixture likelihood in log space for numerical stability
    log_pi = torch.log(pi + 1e-8)
    log_weighted_prob = log_pi + log_prob

    # Log-sum-exp trick for numerical stability
    max_log_prob = torch.max(log_weighted_prob, dim=1, keepdim=True)[0]
    stable_log_prob = log_weighted_prob - max_log_prob
    mixture_prob = torch.exp(max_log_prob.squeeze()) * torch.sum(torch.exp(stable_log_prob), dim=1)

    # Negative log-likelihood with clamping
    nll = -torch.log(torch.clamp(mixture_prob, min=1e-8))

    return torch.mean(nll)


def preprocess_trajectory_data(trajectories):
    """Convert trajectory data into observation-position pairs"""
    all_obs = []
    all_pos = []
    all_categories = []

    for traj in trajectories:
        positions = np.array(traj.positions)

        # Build observation vectors for each time step
        for t in range(len(positions)):
            obs_vector = []

            # Add target positions and statuses
            for target in range(15):
                pos = getattr(traj, f'target{target}_pos')
                status = getattr(traj, f'target{target}_status')
                obs_vector.extend(pos)
                obs_vector.append(status[t])

            for threat in range(4):
                pos = getattr(traj, f'target{threat}_pos')
                status = getattr(traj, f'target{threat}_status')
                obs_vector.extend(pos)
                obs_vector.append(status[t])

            # Add level as a feature
            #obs_vector.append(float(traj.level))

            # # Pad or truncate to ensure consistent dimensionality
            # while len(obs_vector) < 10:  # Assuming 10D observation space
            #     obs_vector.append(0.0)
            # obs_vector = obs_vector[:10]

            all_obs.append(obs_vector)
            all_pos.append(positions[t])
            all_categories.append(traj.category)

    return np.array(all_obs), np.array(all_pos), all_categories


def debug_data_and_model(obs_data, pos_data, model=None):
    """Debug function to check data and model behavior"""
    print("=== DATA DEBUG INFO ===")
    print(f"Observations shape: {obs_data.shape}")
    print(f"Positions shape: {pos_data.shape}")
    print(f"Obs range: [{obs_data.min():.3f}, {obs_data.max():.3f}]")
    print(f"Pos range: [{pos_data.min(axis=0)}, {pos_data.max(axis=0)}]")
    print(f"Obs std: {obs_data.std(axis=0).mean():.3f}")
    print(f"Pos std: {pos_data.std(axis=0)}")

    # Check for NaN or inf
    print(f"Obs has NaN: {np.isnan(obs_data).any()}")
    print(f"Obs has inf: {np.isinf(obs_data).any()}")
    print(f"Pos has NaN: {np.isnan(pos_data).any()}")
    print(f"Pos has inf: {np.isinf(pos_data).any()}")

    if model is not None:
        print("\n=== MODEL DEBUG INFO ===")
        # Test forward pass
        test_obs = torch.FloatTensor(obs_data[:10])
        test_pos = torch.FloatTensor(pos_data[:10])

        #with torch.no_grad():
        pi, mu, sigma_x, sigma_y, rho = model(test_obs)
        loss = mdn_loss(pi, mu, sigma_x, sigma_y, rho, test_pos)

        print(f"Pi range: [{pi.min():.6f}, {pi.max():.6f}]")
        print(f"Mu range: [{mu.min():.3f}, {mu.max():.3f}]")
        print(f"Sigma_x range: [{sigma_x.min():.6f}, {sigma_x.max():.6f}]")
        print(f"Sigma_y range: [{sigma_y.min():.6f}, {sigma_y.max():.6f}]")
        print(f"Rho range: [{rho.min():.6f}, {rho.max():.6f}]")
        print(f"Test loss: {loss.item():.6f}")

        # Check gradients
        loss.backward()
        total_grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        print(f"Total gradient norm: {total_grad_norm:.6f}")



def train_mdn(obs_data, pos_data, n_components=5, epochs=200, lr=0.001, batch_size=256):
    """Train a single MDN model with debugging"""

    # Debug data first
    #debug_data_and_model(obs_data, pos_data)

    # Create dataset and dataloader
    dataset = TrajectoryDataset(obs_data, pos_data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = PositionMDN(obs_dim=obs_data.shape[1], n_components=n_components)

    # Debug initial model state
    #debug_data_and_model(obs_data, pos_data, model)

    # Use different optimizer settings
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.7)

    train_losses = []
    val_losses = []

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 30

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0

        for obs_batch, pos_batch in train_loader:
            optimizer.zero_grad()
            pi, mu, sigma_x, sigma_y, rho = model(obs_batch)
            loss = mdn_loss(pi, mu, sigma_x, sigma_y, rho, pos_batch)

            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss at epoch {epoch}, batch {train_batches}")
                break

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for obs_batch, pos_batch in val_loader:
                pi, mu, sigma_x, sigma_y, rho = model(obs_batch)
                loss = mdn_loss(pi, mu, sigma_x, sigma_y, rho, pos_batch)
                val_loss += loss.item()
                val_batches += 1

        avg_train_loss = train_loss / train_batches if train_batches > 0 else float('inf')
        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}, LR = {current_lr:.6f}")

        # # Early stopping
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        #
        # if patience_counter >= patience:
        #     print(f"Early stopping at epoch {epoch}")
        #     break

    return model, train_losses, val_losses


def wasserstein_distance_2d(samples1, samples2):
    """Compute 2D Wasserstein distance between two sets of samples"""
    # Flatten samples for scipy's wasserstein_distance (which is 1D)
    # We'll compute distance for x and y separately and combine
    samples1 = samples1.cpu().numpy() if torch.is_tensor(samples1) else samples1
    samples2 = samples2.cpu().numpy() if torch.is_tensor(samples2) else samples2

    # Simple approximation: average of x and y Wasserstein distances
    wd_x = wasserstein_distance(samples1[:, 0], samples2[:, 0])
    wd_y = wasserstein_distance(samples1[:, 1], samples2[:, 1])

    return (wd_x + wd_y) / 2


def compare_agent_distributions(model_A, model_B, model_C, test_obs, n_samples=1000):
    """Compare distributions using Wasserstein distance"""
    model_A.eval()
    model_B.eval()
    model_C.eval()

    similarities = []

    for obs in test_obs:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

        # Sample from each model
        samples_A = model_A.sample(obs_tensor, n_samples).squeeze(0)
        samples_B = model_B.sample(obs_tensor, n_samples).squeeze(0)
        samples_C = model_C.sample(obs_tensor, n_samples).squeeze(0)

        # Compute pairwise distances
        W_AB = wasserstein_distance_2d(samples_A, samples_B)
        W_AC = wasserstein_distance_2d(samples_A, samples_C)

        similarities.append({
            'W_AB': W_AB,
            'W_AC': W_AC,
            'A_more_similar_to_B': W_AB < W_AC
        })

    return similarities


def main():
    # Load trajectory data from separate files
    print("Loading trajectory data...")
    trajectories = []

    # Load RL trajectories
    with open('rl_trajectories_fulltrajectories.json', 'r') as f:
        rl_data = json.load(f)
    for traj_dict in rl_data:
        traj_dict['category'] = 'rl'  # Ensure category is set
        traj = FullTrajectory(**traj_dict)
        trajectories.append(traj)
    print(f"Loaded {len(rl_data)} RL trajectories")

    # Load heuristic trajectories
    with open('heuristic_trajectories_fulltrajectories.json', 'r') as f:
        heuristic_data = json.load(f)
    for traj_dict in heuristic_data:
        traj_dict['category'] = 'heuristic'  # Ensure category is set
        traj = FullTrajectory(**traj_dict)
        trajectories.append(traj)
    print(f"Loaded {len(heuristic_data)} heuristic trajectories")

    # Load human trajectories
    # with open('human_trajectories.json', 'r') as f:
    #     human_data = json.load(f)
    # for traj_dict in human_data:
    #     traj_dict['category'] = 'human'  # Ensure category is set
    #     traj = FullTrajectory(**traj_dict)
    #     trajectories.append(traj)
    # print(f"Loaded {len(human_data)} human trajectories")

    print(f"Total trajectories loaded: {len(trajectories)}")

    # Preprocess data
    print("Preprocessing data...")
    all_obs, all_pos, all_categories = preprocess_trajectory_data(trajectories)

    # Normalize observations
    scaler = StandardScaler()
    all_obs_normalized = scaler.fit_transform(all_obs)

    # Split by agent category
    #human_mask = np.array(all_categories) == 'human'
    rl_mask = np.array(all_categories) == 'rl'
    heuristic_mask = np.array(all_categories) == 'heuristic'

    #obs_A = all_obs_normalized[human_mask]  # Agent A (human)
    #pos_A = all_pos[human_mask]
    obs_B = all_obs_normalized[rl_mask]  # Agent B (RL)
    pos_B = all_pos[rl_mask]
    obs_C = all_obs_normalized[heuristic_mask]  # Agent C (heuristic)
    pos_C = all_pos[heuristic_mask]

    #print(f"Agent A (human): {len(obs_A)} samples")
    print(f"Agent B (RL): {len(obs_B)} samples")
    print(f"Agent C (heuristic): {len(obs_C)} samples")

    # Train models for each agent
    #print("Training MDN for Agent A (human)...")
    #model_A, _, _ = train_mdn(obs_A, pos_A, n_components=5)

    print("Training MDN for Agent B (RL)...")
    model_B, _, _ = train_mdn(obs_B, pos_B, n_components=5)

    print("Training MDN for Agent C (heuristic)...")
    model_C, _, _ = train_mdn(obs_C, pos_C, n_components=5)

    # Prepare test observations (sample from Agent A's observations)
    #test_indices = np.random.choice(len(obs_A), size=min(100, len(obs_A) // 10), replace=False)
    #test_obs = obs_A[test_indices]

    #print("Comparing agent distributions...")
    #similarities = compare_agent_distributions(model_A, model_B, model_C, test_obs)

    # Analyze results
    #total_comparisons = len(similarities)
    #a_more_similar_to_b = sum(1 for s in similarities if s['A_more_similar_to_B'])

    # print(f"\nResults:")
    # print(f"Total comparisons: {total_comparisons}")
    # print(f"Cases where A is more similar to B than C: {a_more_similar_to_b} ({100 * a_more_similar_to_b / total_comparisons:.1f}%)")
    # print(f"Cases where A is more similar to C than B: {total_comparisons - a_more_similar_to_b} ({100 * (total_comparisons - a_more_similar_to_b) / total_comparisons:.1f}%)")
    #
    # # Statistical significance test (binomial test)
    # from scipy.stats import binomtest
    # result = binomtest(a_more_similar_to_b, total_comparisons, p=0.5)
    # print(f"Binomial test p-value: {result.pvalue:.6f}")
    #
    # if result.pvalue < 0.05:
    #     if a_more_similar_to_b > total_comparisons / 2:
    #         print("SIGNIFICANT: Agent A's position distribution is significantly more similar to B than C")
    #     else:
    #         print("SIGNIFICANT: Agent A's position distribution is significantly more similar to C than B")
    # else:
    #     print("NOT SIGNIFICANT: No significant difference in similarity")
    #
    # # Plot some example distributions
    # plt.figure(figsize=(15, 5))
    #
    # for i, idx in enumerate(test_indices[:3]):
    #     plt.subplot(1, 3, i + 1)
    #     obs_tensor = torch.FloatTensor(obs_A[idx]).unsqueeze(0)
    #
    #     # Sample from models
    #     samples_A = model_A.sample(obs_tensor, 500).squeeze(0).cpu().numpy()
    #     samples_B = model_B.sample(obs_tensor, 500).squeeze(0).cpu().numpy()
    #     samples_C = model_C.sample(obs_tensor, 500).squeeze(0).cpu().numpy()
    #
    #     # Plot samples
    #     plt.scatter(samples_A[:, 0], samples_A[:, 1], alpha=0.3, s=2, label='Agent A (Human)', color='blue')
    #     plt.scatter(samples_B[:, 0], samples_B[:, 1], alpha=0.3, s=2, label='Agent B (RL)', color='red')
    #     plt.scatter(samples_C[:, 0], samples_C[:, 1], alpha=0.3, s=2, label='Agent C (Heuristic)', color='green')
    #
    #     # Plot actual position
    #     plt.scatter(pos_A[idx, 0], pos_A[idx, 1], s=50, color='black', marker='x', label='Actual')
    #
    #     plt.title(f'Distribution Comparison {i + 1}')
    #     plt.xlabel('X Position')
    #     plt.ylabel('Y Position')
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #
    # plt.tight_layout()
    # plt.savefig('distribution_comparison.png', dpi=300, bbox_inches='tight')
    # plt.show()

    #return model_A, model_B, model_C, similarities
    return model_B, model_C


if __name__ == "__main__":
    main()