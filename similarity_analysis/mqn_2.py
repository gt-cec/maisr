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

        # Shared backbone with batch normalization
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )

        # Mixture component heads
        self.pi_head = nn.Linear(hidden_dim // 2, n_components)  # mixture weights
        self.mu_head = nn.Linear(hidden_dim // 2, n_components * 2)  # means (x,y)
        self.sigma_head = nn.Linear(hidden_dim // 2, n_components * 3)  # covariance params

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights to prevent dead neurons"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize sigma head to produce reasonable initial variances
        nn.init.constant_(self.sigma_head.bias[:self.n_components], 0.0)  # sigma_x
        nn.init.constant_(self.sigma_head.bias[self.n_components:2 * self.n_components], 0.0)  # sigma_y
        nn.init.constant_(self.sigma_head.bias[2 * self.n_components:], 0.0)  # rho

    def forward(self, x):
        features = self.backbone(x)

        # Mixture weights (softmax normalized)
        pi = torch.softmax(self.pi_head(features), dim=-1)

        # Means for each component
        mu = self.mu_head(features).view(-1, self.n_components, 2)

        # Covariance parameters: [sigma_x, sigma_y, rho] for each component
        sigma_params = self.sigma_head(features).view(-1, self.n_components, 3)

        # More stable sigma computation with minimum variance
        sigma_x = torch.exp(sigma_params[:, :, 0]) + 1e-3  # Increased minimum
        sigma_y = torch.exp(sigma_params[:, :, 1]) + 1e-3
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

                    # Create covariance matrix with numerical stability
                    variance_x = sx ** 2
                    variance_y = sy ** 2
                    covariance = r * sx * sy

                    cov = torch.tensor([[variance_x, covariance],
                                        [covariance, variance_y]])

                    # Add small regularization to ensure positive definiteness
                    cov = cov + 1e-6 * torch.eye(2)

                    try:
                        # Sample from 2D Gaussian
                        sample = torch.distributions.MultivariateNormal(mean, cov).sample()
                        batch_samples.append(sample)
                    except:
                        # Fallback to diagonal covariance if numerical issues
                        cov_diag = torch.diag(torch.tensor([variance_x, variance_y]))
                        sample = torch.distributions.MultivariateNormal(mean, cov_diag).sample()
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
    debug_data_and_model(obs_data, pos_data)

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
    debug_data_and_model(obs_data, pos_data, model)

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

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return model, train_losses, val_losses


# Test with simpler synthetic data first
def with_synthetic_data():
    """Test the MDN with simple synthetic data to verify it works"""
    print("=== TESTING WITH SYNTHETIC DATA ===")

    # Generate simple synthetic data
    np.random.seed(42)
    n_samples = 1000
    obs_dim = 10

    # Random observations
    obs_data = np.random.randn(n_samples, obs_dim)

    # Create simple relationship: position depends on first few observations
    pos_data = np.zeros((n_samples, 2))
    pos_data[:, 0] = obs_data[:, 0] + 0.5 * obs_data[:, 1] + 0.1 * np.random.randn(n_samples)
    pos_data[:, 1] = obs_data[:, 2] - 0.3 * obs_data[:, 3] + 0.1 * np.random.randn(n_samples)

    # Normalize
    scaler = StandardScaler()
    obs_data = scaler.fit_transform(obs_data)

    print("Training on synthetic data...")
    model, train_losses, val_losses = train_mdn(obs_data, pos_data, n_components=3, epochs=100, lr=0.01, batch_size=64)

    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Synthetic Data Training')
    plt.legend()
    plt.yscale('log')

    plt.tight_layout()
    plt.show()

    return model


if __name__ == "__main__":
    # First test with synthetic data
    #test_model = with_synthetic_data()

    print("\n" + "=" * 50)
    print("If synthetic data works, proceed with your real data...")
    print("=" * 50)