import gymnasium as gym
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box

from env_combined import MAISREnvVec
from utility.data_logging import load_env_config


class SequenceMAISRWrapper(gym.Wrapper):
    """
    Wrapper for MAISR environment that converts it to a sequence generation task.
    
    The RL agent chooses a sequence of target indices, then the sequence policy
    executes that sequence to completion. The agent only sees the final episode outcome.
    """
    
    def __init__(self, env, sequence_policy_func, num_targets=5, max_episode_steps=None):
        """
        Initialize the sequence wrapper.
        
        Args:
            env: The base MAISR environment
            sequence_policy_func: The sequence policy function to execute sequences
            num_targets (int): Number of targets in the sequence
            max_episode_steps (int): Maximum steps for the underlying environment
        """
        super().__init__(env)
        
        self.sequence_policy_func = sequence_policy_func
        self.num_targets = num_targets
        self.max_episode_steps = max_episode_steps
        
        # Store original spaces for reference
        self.original_observation_space = env.observation_space
        self.original_action_space = env.action_space
        
        # New action space: sequence of target indices
        # Each element can be 0 to num_targets-1
        self.action_space = MultiDiscrete([num_targets] * num_targets)
        
        # Observation space remains the same (direct passthrough)
        self.observation_space = env.observation_space
        
        # Episode tracking
        self.episode_start_obs = None
        self.episode_steps = 0
        self.max_steps_reached = False
        
    def reset(self, **kwargs):
        """
        Reset the environment and return initial observation.
        
        Returns:
            observation: Initial observation from the environment
            info: Info dictionary
        """
        obs, info = self.env.reset(**kwargs)
        
        # Store the initial observation for the sequence generation agent
        self.episode_start_obs = obs.copy()
        self.episode_steps = 0
        self.max_steps_reached = False
        
        # Reset sequence policy state
        if hasattr(self.sequence_policy_func, '__globals__'):
            # Reset global state if using the sequence policy from the artifact
            reset_func = self.sequence_policy_func.__globals__.get('reset_sequence_state')
            if reset_func:
                reset_func()
        
        return obs, info
    
    def step(self, action):
        """
        Execute the sequence action by running the sequence policy to completion.
        
        Args:
            action: Array of target indices representing the sequence to follow
            
        Returns:
            observation: Final observation after sequence completion
            reward: Cumulative reward from the entire sequence execution
            terminated: Whether the episode terminated
            truncated: Whether the episode was truncated
            info: Info dictionary with execution details
        """
        # Validate action
        if not isinstance(action, (list, tuple, np.ndarray)):
            raise ValueError(f"Action must be array-like, got {type(action)}")
        
        action = np.array(action)
        if len(action) != self.num_targets:
            raise ValueError(f"Action must have length {self.num_targets}, got {len(action)}")
        
        # Convert to list for sequence policy
        target_sequence = action.tolist()
        
        # Execute the sequence using the sequence policy
        total_reward = 0.0
        step_count = 0
        terminated = False
        truncated = False
        final_obs = None
        execution_info = {
            'sequence_executed': target_sequence,
            'total_steps': 0,
            'targets_identified': 0,
            'sequence_completion': 0.0,
            'execution_details': [],
            'final_episode_info': {}
        }
        
        # Keep stepping until episode ends or max steps reached
        while not (terminated or truncated):
            # Get current observation
            current_obs = self.env.get_observation() if hasattr(self.env, 'get_observation') else self.env.observation
            
            # Get action from sequence policy
            try:
                sequence_action = self.sequence_policy_func(current_obs, target_sequence)
            except Exception as e:
                print(f"Error in sequence policy: {e}")
                sequence_action = 0  # Default action
            
            # Step the environment
            obs, reward, terminated, truncated, info = self.env.step(sequence_action)
            
            total_reward += reward
            step_count += 1
            final_obs = obs
            
            # Track execution details
            if step_count % 50 == 0:  # Log every 50 steps to avoid too much data
                execution_info['execution_details'].append({
                    'step': step_count,
                    'reward': reward,
                    'total_reward': info['episode']['r'],#total_reward,
                    'targets_identified': info.get('target_ids', 0)
                })
            
            # Safety check for maximum steps
            if self.max_episode_steps and step_count >= self.max_episode_steps:
                truncated = True
                self.max_steps_reached = True
                break
        
        # Update execution info with final results
        execution_info.update({
            'total_steps': step_count,
            'targets_identified': info.get('target_ids', 0),
            'final_episode_info': info,
            'max_steps_reached': self.max_steps_reached
        })
        
        # Calculate sequence completion rate
        if hasattr(self.sequence_policy_func, '__globals__'):
            progress_func = self.sequence_policy_func.__globals__.get('get_sequence_progress')
            if progress_func:
                current_idx, total_length, _ = progress_func()
                execution_info['sequence_completion'] = current_idx / max(total_length, 1) if total_length > 0 else 0.0
        
        self.episode_steps = step_count
        
        return final_obs, info['episode']['r'], terminated, truncated, execution_info
    
    def get_sequence_action_meanings(self):
        """
        Get human-readable meanings for sequence actions.
        
        Returns:
            list: List of strings describing what each action index means
        """
        return [f"Target {i}" for i in range(self.num_targets)]
    
    def sample_valid_sequence(self):
        """
        Sample a valid sequence action.
        
        Returns:
            np.ndarray: Valid sequence action
        """
        # Generate a random permutation of target indices
        return np.random.permutation(self.num_targets)
    
    def evaluate_sequence(self, sequence, num_episodes=10, verbose=False):
        """
        Evaluate a specific sequence over multiple episodes.
        
        Args:
            sequence: Target sequence to evaluate
            num_episodes (int): Number of episodes to run
            verbose (bool): Whether to print detailed results
            
        Returns:
            dict: Evaluation results including mean reward, success rate, etc.
        """
        rewards = []
        targets_identified = []
        sequence_completions = []
        step_counts = []
        
        for episode in range(num_episodes):
            obs, _ = self.reset()
            final_obs, reward, terminated, truncated, info = self.step(sequence)
            
            rewards.append(reward)
            targets_identified.append(info.get('targets_identified', 0))
            sequence_completions.append(info.get('sequence_completion', 0.0))
            step_counts.append(info.get('total_steps', 0))
            
            if verbose:
                print(f"Episode {episode + 1}: Reward={reward:.2f}, Targets={info.get('targets_identified', 0)}, "
                      f"Completion={info.get('sequence_completion', 0.0):.2%}, Steps={info.get('total_steps', 0)}")
        
        results = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_targets_identified': np.mean(targets_identified),
            'mean_sequence_completion': np.mean(sequence_completions),
            'mean_steps': np.mean(step_counts),
            'success_rate': np.mean([t >= self.num_targets for t in targets_identified]),
            'all_rewards': rewards,
            'all_targets': targets_identified,
            'all_completions': sequence_completions,
            'all_steps': step_counts
        }
        
        if verbose:
            print(f"\nEvaluation Results for sequence {sequence}:")
            print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"Mean Targets Identified: {results['mean_targets_identified']:.2f}")
            print(f"Mean Sequence Completion: {results['mean_sequence_completion']:.2%}")
            print(f"Success Rate: {results['success_rate']:.2%}")
            print(f"Mean Steps: {results['mean_steps']:.0f}")
        
        return results


class SequenceMAISRMultiWrapper(SequenceMAISRWrapper):
    """
    Extended wrapper that supports multiple sequence lengths and target counts.
    Useful for curriculum learning or environments with varying numbers of targets.
    """
    
    def __init__(self, env, sequence_policy_func, min_targets=3, max_targets=5, **kwargs):
        """
        Initialize wrapper with variable sequence lengths.
        
        Args:
            env: Base MAISR environment
            sequence_policy_func: Sequence policy function
            min_targets (int): Minimum number of targets in sequence
            max_targets (int): Maximum number of targets in sequence
        """
        # Initialize with max targets for action space
        super().__init__(env, sequence_policy_func, max_targets, **kwargs)
        
        self.min_targets = min_targets
        self.max_targets = max_targets
        self.current_sequence_length = max_targets
        
        # Update action space to handle variable lengths
        # Use max_targets for the action space, but pad/truncate as needed
        self.action_space = MultiDiscrete([max_targets] * max_targets)
    
    def set_sequence_length(self, length):
        """
        Set the current sequence length for episodes.
        
        Args:
            length (int): Sequence length to use
        """
        if self.min_targets <= length <= self.max_targets:
            self.current_sequence_length = length
        else:
            raise ValueError(f"Sequence length must be between {self.min_targets} and {self.max_targets}")
    
    def step(self, action):
        """
        Execute sequence with current sequence length.
        
        Args:
            action: Array of target indices (will be truncated to current_sequence_length)
        """
        # Truncate action to current sequence length
        action = np.array(action)[:self.current_sequence_length]
        
        # Temporarily update num_targets for this episode
        original_num_targets = self.num_targets
        self.num_targets = self.current_sequence_length
        
        try:
            result = super().step(action)
        finally:
            # Restore original num_targets
            self.num_targets = original_num_targets
        
        return result


# Example usage and testing functions
def create_sequence_env(base_env, sequence_policy_func, **wrapper_kwargs):
    """
    Convenience function to create a sequence environment.
    
    Args:
        base_env: Base MAISR environment
        sequence_policy_func: Sequence policy function
        **wrapper_kwargs: Additional arguments for wrapper
        
    Returns:
        SequenceMAISRWrapper: Wrapped environment
    """
    return SequenceMAISRWrapper(base_env, sequence_policy_func, **wrapper_kwargs)


# def test_sequence_wrapper(env_wrapper, num_test_episodes=5):
#     """
#     Test the sequence wrapper with random sequences.
#
#     Args:
#         env_wrapper: SequenceMAISRWrapper instance
#         num_test_episodes (int): Number of episodes to test
#     """
#     print(f"Testing Sequence Wrapper for {num_test_episodes} episodes")
#     print(f"Action space: {env_wrapper.action_space}")
#     print(f"Observation space: {env_wrapper.observation_space}")
#     print("-" * 50)
#
#     for episode in range(num_test_episodes):
#         obs, info = env_wrapper.reset()
#         print(f"\nEpisode {episode + 1}:")
#         print(f"Initial observation shape: {obs.shape}")
#
#         # Sample a random sequence
#         sequence = env_wrapper.sample_valid_sequence()
#         print(f"Testing sequence: {sequence}")
#
#         # Execute the sequence
#         final_obs, reward, terminated, truncated, exec_info = env_wrapper.step(sequence)
#
#         print(f"Final reward: {reward:.2f}")
#         print(f"Targets identified: {exec_info.get('targets_identified', 0)}")
#         print(f"Total steps: {exec_info.get('total_steps', 0)}")
#         print(f"Sequence completion: {exec_info.get('sequence_completion', 0.0):.2%}")
#         print(f"Episode ended: terminated={terminated}, truncated={truncated}")
#
#     print("\nWrapper test completed!")


# Example with different sequence generation strategies
def compare_sequence_strategies(env_wrapper, strategies, num_episodes_per_strategy=10):
    """
    Compare different sequence generation strategies.
    
    Args:
        env_wrapper: SequenceMAISRWrapper instance
        strategies: Dict of {name: sequence_generator_func}
        num_episodes_per_strategy (int): Episodes to test per strategy
        
    Returns:
        dict: Results for each strategy
    """
    results = {}
    
    for strategy_name, sequence_func in strategies.items():
        print(f"\nTesting strategy: {strategy_name}")
        strategy_rewards = []
        strategy_targets = []
        
        for episode in range(num_episodes_per_strategy):
            obs, _ = env_wrapper.reset()
            
            # Generate sequence using strategy
            if callable(sequence_func):
                sequence = sequence_func(obs) if 'obs' in sequence_func.__code__.co_varnames else sequence_func()
            else:
                sequence = sequence_func  # Fixed sequence
            
            # Ensure sequence is valid
            sequence = np.array(sequence) % env_wrapper.num_targets
            
            final_obs, reward, terminated, truncated, info = env_wrapper.step(sequence)
            
            strategy_rewards.append(reward)
            strategy_targets.append(info.get('targets_identified', 0))
        
        results[strategy_name] = {
            'mean_reward': np.mean(strategy_rewards),
            'std_reward': np.std(strategy_rewards),
            'mean_targets': np.mean(strategy_targets),
            'success_rate': np.mean([t >= env_wrapper.num_targets for t in strategy_targets])
        }
        
        print(f"  Mean reward: {results[strategy_name]['mean_reward']:.2f} ± {results[strategy_name]['std_reward']:.2f}")
        print(f"  Mean targets: {results[strategy_name]['mean_targets']:.2f}")
        print(f"  Success rate: {results[strategy_name]['success_rate']:.2%}")
    
    return results


if __name__ == '__main__':

    config = load_env_config('./configs/sequence_june11.json')

    base_env = MAISREnvVec(
        config=config,
        render_mode='headless',
        num_agents=1,
        tag='test_suite',
        #seed=config['seed']
    )

    # Create the wrapper
    from heuristic_policies.sequence_policy import sequence_policy  # Your sequence policy function

    wrapped_env = SequenceMAISRWrapper(base_env, sequence_policy, num_targets=config['num_targets'])



    # Agent chooses a sequence (this is what the RL agent learns)
    #sequence_action = [1, 0, 4, 3, 2]  # Visit targets in this order
    for trial in range(10):
        sequence_action = wrapped_env.sample_valid_sequence()

        # Execute the entire sequence
        final_obs, total_reward, done, truncated, exec_info = wrapped_env.step(sequence_action)

        print(f"Total reward: {total_reward}")
        print(f"Targets identified: {exec_info['targets_identified']}\n")
        #print(f"Steps taken: {exec_info['total_steps']}")
        obs, info = wrapped_env.reset()