import gymnasium as gym
import os
import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback

import multiprocessing

from env_vec_simple import MAISREnvVec
from utility.data_logging import load_env_config

class EnhancedWandbCallback(BaseCallback):
    def __init__(self, verbose=0, eval_env=None, eval_freq=14400, n_eval_episodes=5, run=None):
        super(EnhancedWandbCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.run = run

    def _on_step(self):
        # Log training metrics
        if self.locals.get("infos") and len(self.locals["infos"]) > 0:
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.run.log({
                        "train/episode_reward": info["episode"]["r"],
                        "train/episode_length": info["episode"]["l"]
                    }, step=self.num_timesteps)

        # Periodically evaluate and log evaluation metrics
        if self.eval_env is not None and self.num_timesteps % self.eval_freq == 0:
            print(f'\n#################################################\nEVALUATING (step: {self.num_timesteps})')

            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes
            )

            self.run.log({
                "eval/mean_reward": mean_reward,
                "eval/std_reward": std_reward
            }, step=self.num_timesteps)
            print(f'\nEVAL LOGGED (mean reward {mean_reward}, std {round(std_reward,2)}')
            print('#################################################\n\nReturning to training...')

        return True


def make_env(env_config, rank, seed):
    def _init():
        env = MAISREnvVec(
            config=env_config,
            render_mode='headless',
            reward_type='balanced-sparse',
            obs_type='vector',
            action_type='continuous',
        )
        env = Monitor(env)  # record stats such as returns
        env.reset(seed=seed + rank)
        return env

    return _init  # Return the function, not the environment instance


def main():
    # Instantiate the env
    config = './config_files/rl_training_config.json'
    env_config = load_env_config(config)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    run = wandb.init(
        project="maisr-rl",
        config={
            "algorithm": 'PPO',
            "policy_type": "MlpPolicy",
            "total_timesteps": num_timesteps,
            "env_name": "MAISREnvVec",
            "env_config": config
        },
        sync_tensorboard=True,
        monitor_gym=True,  # This is important for gym environment monitoring
    )

    #vec_env = make_vec_env(MAISREnvVec, n_envs=1, env_kwargs=dict(config=env_config, render_mode='headless', reward_type='balanced-sparse', obs_type='vector', action_type='continuous'), monitor_dir=log_dir)

    # Create vectorized environment for training using SubprocVecEnv
    #env_fns = [make_env(env_config, i, seed) for i in range(n_envs)]
    #vec_env = SubprocVecEnv(env_fns) # (TODO commented out to test non-vectorized env

    vec_env = MAISREnvVec(
        env_config,
        None,
        render_mode='headless',
        reward_type='balanced-sparse',
        obs_type='vector',
        action_type='continuous',
    )
    vec_env = Monitor(vec_env)

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
        name_prefix=f"simplified_checkpoint{algo}_maisr",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # eval_callback = EvalCallback( # TODO temporarily removed. May be redundant with wandb callback
    #     eval_env,
    #     best_model_save_path=f"{save_dir}/best_model",
    #     log_path=log_dir,
    #     eval_freq=eval_freq,
    #     n_eval_episodes=n_eval_episodes,
    #     deterministic=True,
    #     render=False,
    #     verbose=1,  # Set to 1 to see more output
    # )

    wandb_callback = WandbCallback( # TODO Add back if having issues logging
        gradient_save_freq=0,
        model_save_path=f"{save_dir}/wandb/{run.id}",
        verbose=2,
    )

    enhanced_wandb_callback = EnhancedWandbCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        run = run
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",  # Match tensorboard directory to wandb run.id
        batch_size=batch_size, # * n_envs,  # Scale batch size with number of environments
        n_steps= steps_per_episode, #2048 // n_envs,  # Adjust steps per environment
        learning_rate=3e-4,
        seed=seed
    )

    # Check if there's a checkpoint to load
    if load_dir:
        model = model.__class__.load(load_dir, env=vec_env)

    else:
        print('Training new model')
        #model = PPO("MlpPolicy", vec_env, verbose=1)

    print('Beginning agent training...\n#################################################')
    #run.log({"test_metric": 1.0})

    model.learn(
        total_timesteps=int(num_timesteps),
        callback=[checkpoint_callback, wandb_callback, enhanced_wandb_callback], # Removed eval_callback, wandb_callback
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
    #wandb.log({"final/mean_reward": mean_reward, "final/std_reward": std_reward})
    run.log({"final/mean_reward": mean_reward, "final/std_reward": std_reward})

    run.finish()


if __name__ == "__main__":
    global save_dir, load_dir, log_dir, algo
    global batch_size, steps_per_episode, num_timesteps
    global save_freq, eval_freq, n_eval_episodes
    global n_envs, seed

    save_dir = "./trained_models/"
    load_dir = None# "/trained_models/agent_test" # Where to load trained model from
    log_dir = "./logs/" # Where to save logs
    algo = 'PPO'

    batch_size = 128
    steps_per_episode = 14500 # Slightly higher than the max 14,400
    num_timesteps = 45e6 #500000 # Total num timesteps to train

    save_freq = 14400 # How often to save checkpoints
    eval_freq = 14400 * 3 # How often to evaluate
    n_eval_episodes = 5

    # Number of parallel environments (should not exceed number of CPU cores)
    n_envs = min(8, multiprocessing.cpu_count())  # Use at most 8 or the number of CPU cores
    print(f"Training with {n_envs} environments in parallel")

    # Set seed for reproducibility
    seed = 42

    main()