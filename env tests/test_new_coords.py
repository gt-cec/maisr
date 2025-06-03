import numpy as np
from env_combined_newcoords import MAISREnvVec

# Minimal config
config = {
    'obs_type': 'absolute',
    'action_type': 'continuous-normalized',
    'verbose': 'false',
    'use_beginner_levels': False,
    'use_curriculum': True,
    'gameboard_size': 500,
    'num_targets': 10,
    'time_limit': 245,
    'frame_skip': 30,
    'game_speed': 0.07,
    'human_speed': 2.8,
    'search pattern': 'none',
    'agent_start_location': 'random',
    "window_size": [1450,1080],
    "shaping_decay_rate":1,
	"shaping_coeff_wtn": 0.02,
	"shaping_coeff_prox": 0.005,
	"shaping_coeff_earlyfinish": 2.0,
	"shaping_time_penalty": -0.0,

    'prob_detect': 0.0,
    'highqual_regulartarget_reward': 1.0,
    'highqual_highvaltarget_reward': 2.0
}

# Create environment
env = MAISREnvVec(config=config, render_mode='headless', run_name='test_run')

# Reset and run 2 steps
obs, info = env.reset()
print("Initial observation shape:", obs.shape)
print("Initial observation:", obs)

for step in range(1):
    action = np.random.uniform(-1, 1, 2)  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"\nStep {step+1}:")
    print(f"Action: {action}")
    print(f"Observation: {obs}")
    print(f'Agent location: {obs[0], obs[1]} ({obs[0]*config['gameboard_size']}, {obs[1]*config['gameboard_size']})')
    for i in [0,1,2,3,4,5,6,7,8,9]:
        print(f'Target {i} location: {obs[3+3*i], obs[4+3*i]} ({obs[3+3*i]*config['gameboard_size'], obs[4+3*i]*config['gameboard_size']})')
    print(f"Reward: {reward}")

# Save plot
env.save_action_history_plot()
print("Plot saved!")