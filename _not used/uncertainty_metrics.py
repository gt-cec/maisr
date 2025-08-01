import torch
import numpy as np
from stable_baselines3 import PPO


def extract_action_log_probabilities(model, observation):
    """
    Extract log probabilities for the chosen action and all actions
    """
    # Ensure observation is in correct format
    obs_tensor = torch.FloatTensor(observation).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        # Get action distribution from policy
        distribution = model.policy.get_distribution(obs_tensor)

        # Get the actual action the model would choose
        action = model.predict(observation, deterministic=False)[0]

        # Get log probability of the chosen action
        action_tensor = torch.LongTensor([action])
        chosen_action_log_prob = distribution.log_prob(action_tensor).item()

        # Get log probabilities for all possible actions
        all_actions = torch.arange(model.action_space.n)
        all_log_probs = distribution.log_prob(all_actions).cpu().numpy()

        # Convert to regular probabilities
        chosen_action_prob = np.exp(chosen_action_log_prob)
        all_probs = np.exp(all_log_probs)

    return {
        'chosen_action': action,
        'chosen_action_log_prob': chosen_action_log_prob,
        'chosen_action_prob': chosen_action_prob,
        'all_log_probs': all_log_probs,
        'all_probs': all_probs
    }

# Usage example:
# model = PPO.load('path_to_your_trained_model.zip')
# obs = env.reset()[0]  # Your observation
# uncertainty_info = extract_action_log_probabilities(model, obs)
# print(f"Action: {uncertainty_info['chosen_action']}, Log Prob: {uncertainty_info['chosen_action_log_prob']:.3f}")

def calculate_policy_entropy(model, observation):
    """
    Calculate the entropy of the action distribution (higher = more uncertain)
    """
    obs_tensor = torch.FloatTensor(observation).unsqueeze(0)

    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)

        # Calculate entropy directly from distribution
        entropy = distribution.entropy().item()

        # Alternative: calculate manually from probabilities
        all_actions = torch.arange(model.action_space.n)
        all_log_probs = distribution.log_prob(all_actions)
        all_probs = torch.exp(all_log_probs)
        manual_entropy = -(all_probs * all_log_probs).sum().item()

        # Normalize entropy (0 = certain, 1 = maximum uncertainty)
        max_entropy = np.log(model.action_space.n)
        normalized_entropy = entropy / max_entropy

    return {
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'max_entropy': max_entropy,
        'manual_entropy': manual_entropy
    }


def get_top_k_actions_uncertainty(model, observation, k=3):
    """
    Get the top-k most likely actions and their probabilities
    Useful for understanding the model's confidence distribution
    """
    obs_tensor = torch.FloatTensor(observation).unsqueeze(0)

    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)

        # Get probabilities for all actions
        all_actions = torch.arange(model.action_space.n)
        all_log_probs = distribution.log_prob(all_actions)
        all_probs = torch.exp(all_log_probs)

        # Get top-k actions
        top_k_probs, top_k_indices = torch.topk(all_probs, k)

        # Calculate probability mass of top-k actions
        top_k_mass = top_k_probs.sum().item()

        # Calculate the gap between top action and second-best
        prob_gap = (top_k_probs[0] - top_k_probs[1]).item() if k > 1 else 0.0

    return {
        'top_k_actions': top_k_indices.cpu().numpy(),
        'top_k_probs': top_k_probs.cpu().numpy(),
        'top_k_mass': top_k_mass,
        'probability_gap': prob_gap,
        'confidence_ratio': top_k_probs[0].item() / top_k_probs[1].item() if k > 1 else float('inf')
    }


def get_value_function_uncertainty(model, observation):
    """
    Extract value function estimates which can indicate uncertainty about state value
    """
    obs_tensor = torch.FloatTensor(observation).unsqueeze(0)

    with torch.no_grad():
        # Get value estimate
        value = model.policy.predict_values(obs_tensor).item()

        # If you want to get multiple samples (for stochastic policies)
        num_samples = 10
        action_samples = []
        value_samples = []

        for _ in range(num_samples):
            action, _ = model.predict(observation, deterministic=False)
            action_samples.append(action)
            # Value function doesn't change with action sampling, but included for completeness
            value_samples.append(value)

        # Calculate consistency of action sampling
        unique_actions, counts = np.unique(action_samples, return_counts=True)
        action_consistency = counts.max() / num_samples  # Fraction of times most common action was chosen

    return {
        'value_estimate': value,
        'action_samples': action_samples,
        'unique_actions': unique_actions,
        'action_counts': counts,
        'action_consistency': action_consistency
    }


def comprehensive_uncertainty_assessment(model, observation):
    """
    Combine multiple uncertainty metrics for a complete picture
    """
    # Get all uncertainty metrics
    log_prob_info = extract_action_log_probabilities(model, observation)
    entropy_info = calculate_policy_entropy(model, observation)
    top_k_info = get_top_k_actions_uncertainty(model, observation, k=3)
    value_info = get_value_function_uncertainty(model, observation)

    # Create overall uncertainty score (0 = certain, 1 = uncertain)
    uncertainty_score = (
            entropy_info['normalized_entropy'] * 0.4 +  # Policy entropy
            (1 - log_prob_info['chosen_action_prob']) * 0.3 +  # Action probability
            (1 - top_k_info['probability_gap']) * 0.2 +  # Probability gap
            (1 - value_info['action_consistency']) * 0.1  # Action consistency
    )

    return {
        'uncertainty_score': uncertainty_score,
        'log_prob_info': log_prob_info,
        'entropy_info': entropy_info,
        'top_k_info': top_k_info,
        'value_info': value_info,
        'interpretation': {
            'high_uncertainty': uncertainty_score > 0.7,
            'medium_uncertainty': 0.3 < uncertainty_score <= 0.7,
            'low_uncertainty': uncertainty_score <= 0.3
        }
    }


def deploy_with_uncertainty_logging(model_path, env, num_episodes=10):
    """
    Deploy your trained model and log uncertainty metrics
    """
    model = PPO.load(model_path)

    uncertainty_log = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_uncertainties = []

        done = False
        step = 0

        while not done:
            # Get uncertainty assessment
            uncertainty_data = comprehensive_uncertainty_assessment(model, obs)

            # Get action
            action, _ = model.predict(obs, deterministic=False)

            # Log the data
            step_data = {
                'episode': episode,
                'step': step,
                'observation': obs.copy(),
                'action': action,
                'uncertainty_score': uncertainty_data['uncertainty_score'],
                'action_log_prob': uncertainty_data['log_prob_info']['chosen_action_log_prob'],
                'entropy': uncertainty_data['entropy_info']['entropy'],
                'top_action_prob': uncertainty_data['log_prob_info']['chosen_action_prob']
            }

            episode_uncertainties.append(step_data)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

        uncertainty_log.extend(episode_uncertainties)

        # Print episode summary
        avg_uncertainty = np.mean([s['uncertainty_score'] for s in episode_uncertainties])
        print(f"Episode {episode}: Avg Uncertainty = {avg_uncertainty:.3f}")

    return uncertainty_log

# Usage:
# uncertainty_data = deploy_with_uncertainty_logging('path_to_model.zip', env)