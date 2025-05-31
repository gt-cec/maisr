import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

import numpy as np
import gymnasium as gym
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env

from utility.data_logging import load_env_config
from sb3_trainagent import make_env


"""
"""

class GreedyHeuristicPolicy:
    def __init__(self, )


    def act(self, observation):

        return action



def behavior_cloning_pipeline(heuristic_policy, env_config, n_episodes=50, n_epochs=10, run_name = 'none'):
    """
    Complete behavior cloning pipeline:
    1. Load heuristic policy
    2. Generate expert trajectories
    3. Train BC policy

    Args:
        heuristic_policy: Policy object with .act(observation) method
        env_name: Gymnasium environment name
        n_episodes: Number of episodes to collect from expert
        n_epochs: Number of training epochs for BC

    Returns:
        bc_trainer: Trained behavior cloning agent
    """

    # Create vectorized environment with RolloutInfoWrapper
    rng = np.random.default_rng(0)

    env = MAISREnvVec(
        env_config,
        None,
        render_mode='headless',
        tag='train',
        run_name=run_name,
    )
    env = Monitor(env)
    # TODO add RolloutInfoWrapper(env)



    # Generate expert trajectories using the heuristic policy
    print(f"Collecting {n_episodes} episodes from expert policy...")
    rollouts = rollout.rollout(
        heuristic_policy,
        env,
        rollout.make_sample_until(min_episodes=n_episodes),
        rng=rng
    )

    transitions = rollout.flatten_trajectories(rollouts)
    print(f"Generated {len(transitions)} transitions from {len(rollouts)} episodes")


    # Initialize behavior cloning trainer
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng
    )

    # TODO: Modify this to train a policy and save it
    # Train the BC policy
    print(f"Training behavior cloning policy for {n_epochs} epochs...")
    bc_trainer.train(n_epochs=n_epochs)

    print("Training complete!")
    env.close()

    return bc_trainer

if __name__ == "__main__":
    config_name = './config_files/rl_training_timepenalty.json'
    run_name = 'bc_test'

    env_config = load_env_config(config_name)

    # Create temporary environment to get action space
    temp_env = DummyVecEnv([make_env(env_config, i, env_config['seed'] + i, run_name=run_name) for i in range(1)])
    heuristic_policy = DummyHeuristicPolicy(temp_env.action_space)
    temp_env.close()

    # Run the behavior cloning pipeline
    trained_bc = behavior_cloning_pipeline(
        heuristic_policy=heuristic_policy,
        env_config=env_config,
        n_episodes=2000,  # Fewer episodes for demo
        n_epochs=10,
        run_name = run_name
    )

    # The trained policy can now be accessed via trained_bc.policy