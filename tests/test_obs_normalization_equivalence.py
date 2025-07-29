import numpy as np
from stable_baselines3 import PPO
from training_wrappers.localsearch_training_wrapper import MaisrLocalSearchWrapper
from utility.league_management import TeammateManager
from env_multi_new import MAISREnvVec
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os

from utility.data_logging import load_env_config

# === CONFIG ===
checkpoint_path = "../obstest/test_checkpoint_0_steps_model.zip"
norm_stats_path = "../obstest/test_checkpoint_vecnormalize_0_steps.pkl"

config = load_env_config('../configs/Monolith_R8H_july10.json')
config['teammate_active_at_start'] = True

# === 1. Load Environment ===
def make_env():
    #config = load_env_config('configs/Monolith_R8H_july10.json')
    env = MAISREnvVec(
        config=config,
        render_mode='headless',
        run_name='test',
        tag=f'test0',
        seed=42,
    )
    return env


base_env = make_env()

# === 2. Initialize wrapper and teammate manager ===
tm = TeammateManager(
    league_type="selfplay",
    balance_method="uniform",
    selfplay_checkpoint_dir=os.path.dirname(checkpoint_path),
    pretrained_teammate_dir=None,
)

raw_env = MaisrLocalSearchWrapper(
    base_env,
    config['obs_noise_std_localsearch'],
    teammate_manager=tm
)

env = DummyVecEnv([lambda: raw_env])
env = VecNormalize.load(norm_stats_path, venv=env)
env.training = False

# === 3. Load Main Model (Agent 0) ===
main_model = PPO.load(checkpoint_path, env=env)
tm.current_model = main_model
tm.set_normalization_stats(env.obs_rms, env.ret_rms)
teammate = tm._create_selfplay_teammate()

# === 4. Get Raw Observations ===
obs_raw_0 = raw_env.env.get_observation_nearest_n(agent_id=0)
obs_raw_1 = raw_env.env.get_observation_nearest_n(agent_id=1)

print("\n=== 🔍 RAW OBSERVATIONS ===")
print(f"Agent 0 raw obs       : {obs_raw_0[:10]}")
print(f"Agent 1 raw obs       : {obs_raw_1[:10]}")

# === 5. Get normalization stats from env ===
mean = env.obs_rms.mean
var = env.obs_rms.var
epsilon = 1e-8

print("\n=== 📊 NORMALIZATION STATS ===")
print(f"Mean (first 5): {mean[:5]}")
print(f"Var  (first 5): {var[:5]}")

# === 6. Normalize Observations Manually (Agent 0) ===
obs_norm_0 = (obs_raw_0 - mean) / np.sqrt(var + epsilon)
obs_norm_0 = np.clip(obs_norm_0, -10.0, 10.0)

# === 7. Normalize using Teammate Method (Agent 1) ===
obs_norm_1 = teammate._normalize_observation(obs_raw_1)

print("\n=== 🔄 NORMALIZED OBSERVATIONS ===")
print("Agent 0 normalized obs (VecNormalize)   : ", obs_norm_0[:5])
print("Agent 1 normalized obs (Teammate Method): ", obs_norm_1[:5])
print("↪️  Difference                          : ", (obs_norm_0 - obs_norm_1)[:5])

# === 8. Compare Actions ===
action_0, _ = main_model.predict(obs_norm_0, deterministic=True)
action_1, _ = teammate.model.predict(obs_norm_1, deterministic=True)

print("\n=== 🎯 ACTION COMPARISON ===")
print("Agent 0 action:", action_0)
print("Agent 1 action:", action_1)
print("🟰 Actions match?" if np.all(action_0 == action_1) else "❌ Actions differ!")