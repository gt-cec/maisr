import gymnasium as gym
import os
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from env_vec import MAISREnvVec
from utility.data_logging import load_env_config
from agents import *


def main(save_dir, load_dir, load_existing):
    # Instantiate the env
    config = './config_files/rl_training_config.json'
    env_config = load_env_config(config)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    run = wandb.init(
        project="maisr-training",
        config={
            "algorithm": 'PPO',
            "policy_type": "MlpPolicy",
            "total_timesteps": num_timesteps,
            "env_name": "MAISREnvVec",
            "env_config": config
        },
        sync_tensorboard=True,  # Auto-upload SB3's tensorboard metrics to wandb
    )

    vec_env = make_vec_env(MAISREnvVec, n_envs=1, env_kwargs=dict(config=env_config, render_mode='headless', reward_type='balanced-sparse', obs_type='vector', action_type='continuous'), monitor_dir=log_dir)

    eval_env = MAISREnvVec(
        env_config,
        None,
        render_mode='headless',
        reward_type='balanced-sparse',
        obs_type='vector',
        action_type='continuous',
    )
    eval_env = Monitor(eval_env)

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=save_dir,
        name_prefix=f"{algo}_maisr",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_dir}/best_model",
        log_path=log_dir,
        eval_freq=save_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=0,  # Don't save gradients to WandB
        model_save_path=f"{save_dir}/wandb/{run.id}",
        verbose=2,
    )

    # Choose algorithm
    if algo == "PPO":
        model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=log_dir)
    elif algo == "A2C":
        model = A2C("MlpPolicy", vec_env, verbose=1, tensorboard_log=log_dir)
    elif algo == "DQN":
        model = DQN("MlpPolicy", vec_env, verbose=1, tensorboard_log=log_dir)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # Check if there's a checkpoint to load
    checkpoints = sorted([f for f in os.listdir(load_dir) if f.startswith(f"{algo}_maisr_") and f.endswith(".zip")])
    if checkpoints:
        latest_checkpoint = os.path.join(load_dir, checkpoints[-1])
        print(f"Loading checkpoint: {latest_checkpoint}")
        model = model.__class__.load(latest_checkpoint, env=vec_env)
        print("Checkpoint loaded successfully!")
    #
    # if load_existing:
    #     print(f'Loading model from {load_dir}/')
    #     model = PPO.load(load_dir)
    #     model.set_env(vec_env)

    else:
        print('Training new model')
        model = PPO("MlpPolicy", vec_env, verbose=1)

    print('Beginning agent training...')
    model.learn(
        total_timesteps=num_timesteps,
        callback=[checkpoint_callback, eval_callback, wandb_callback],
        reset_num_timesteps=False,  # Set to False when resuming training
    )

    # Save the final model
    final_model_path = os.path.join(save_dir, f"{algo}_maisr_final")
    model.save(final_model_path)
    print(f"Training completed! Final model saved to {final_model_path}")

    # Run a final evaluation
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Final evaluation: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # Log final metrics to wandb
    wandb.log({"final/mean_reward": mean_reward, "final/std_reward": std_reward})

    # Close wandb run
    wandb.finish()



if __name__ == "__main__":
    save_dir = "/trained_models/"
    load_existing = False
    load_dir = "/trained_models/agent_test" # Where to load trained model from
    log_dir = "logs/" # Where to save logs
    algo = 'PPO'

    num_timesteps = 500000
    save_freq = 14400
    n_eval_episodes = 5


    main(save_dir, load_dir, load_existing)