"""
Test script to verify CNN pixel observations are working correctly.
Run this to check that pixel observations match the visual render.
"""

import numpy as np
import matplotlib.pyplot as plt
from env_combined import MAISREnvVec
from utility.data_logging import load_env_config




# def test_cnn_training_pipeline():
#     """Test that the CNN training pipeline can be imported and initialized"""
#
#     print("\nTesting CNN training pipeline...")
#
#     try:
#         # Test imports
#         from train_cnn import MAISRCNN, make_env_cnn, generate_run_name_cnn
#         from stable_baselines3 import PPO
#         from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
#         print("✓ All imports successful")
#
#         # Test CNN architecture
#         import torch
#         import gym
#
#         # Create a dummy observation space matching our CNN input
#         obs_space = gym.spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)
#         cnn = MAISRCNN(obs_space, features_dim=512)
#
#         # Test forward pass
#         dummy_input = torch.randint(0, 256, (1, 1, 84, 84), dtype=torch.float32)
#         output = cnn(dummy_input)
#         print(f"✓ CNN forward pass successful. Output shape: {output.shape}")
#
#         # Test environment creation
#         config = {
#             'obs_type': 'absolute',
#             'action_type': 'waypoint-direction',
#             'gameboard_size': 300,
#             'num_targets': 5,
#             'seed': 42,
#             'frame_skip': 4
#         }
#
#         env_fn = make_env_cnn(config, rank=0, seed=42, run_name='test')
#         env = env_fn()
#         obs, _ = env.reset()
#         print(f"✓ CNN environment creation successful. Obs shape: {obs.shape}")
#         env.close()
#
#         print("✓ CNN training pipeline test completed successfully!")
#
#     except Exception as e:
#         print(f"✗ CNN training pipeline test failed: {e}")
#         import traceback
#         traceback.print_exc()


def test_frame_skip_with_cnn():
    """Test that frame_skip works correctly with CNN observations"""

    print("\nTesting frame_skip with CNN observations...")

    config = {
        'obs_type': 'absolute',
        'action_type': 'waypoint-direction',
        'gameboard_size': 300,
        'num_targets': 5,
        'frame_skip': 4,  # Skip 4 frames
        'seed': 42,
        'time_limit': 300,
        'game_speed': 1.0,
        'highqual_regulartarget_reward': 10,
        'highqual_highvaltarget_reward': 20,
        'shaping_coeff_prox': 0.005,
        'shaping_coeff_earlyfinish': 0.02,
        'shaping_time_penalty': -0.001,
        'shaping_decay_rate': 0.99,
        'prob_detect': 0.0,
        'use_curriculum': False,
        'use_beginner_levels': False,
        'agent_start_location': 'random'
    }

    env = MAISREnvVec(
        config=config,
        render_mode='headless',
        tag='test_frameskip_0',
    )

    obs, _ = env.reset()
    print(f"Initial step count: {env.step_count_inner}")

    # Take one action with frame_skip=4
    action = env.action_space.sample()
    obs_next, reward, terminated, truncated, info = env.step(action)

    print(f"After 1 step() call:")
    print(f"  Inner step count: {env.step_count_inner} (should be 4)")
    print(f"  Outer step count: {env.step_count_outer} (should be 1)")
    print(f"  Reward: {reward:.3f}")

    # Verify frame skip worked
    assert env.step_count_inner == 4, f"Expected inner steps=4, got {env.step_count_inner}"
    assert env.step_count_outer == 1, f"Expected outer steps=1, got {env.step_count_outer}"

    print("✓ Frame skip working correctly with CNN observations")

    env.close()


if __name__ == "__main__":
    # Run all tests
    test_cnn_observations()
    #test_cnn_training_pipeline()
    test_frame_skip_with_cnn()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("You can now run CNN training with: python train_cnn.py")
    print("="*60)