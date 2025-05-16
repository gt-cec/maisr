import gymnasium as gym
import os
import numpy as np
from sympy.physics.units import action
import multiprocessing
from typing import Dict, List, Tuple, Any

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

from utility.data_logging import load_env_config

def generate_run_name(config):
    """Generate a standardized run name from configuration."""
    # Core components that should always be included

    # name='tr9_framestack'+str(frame_skip)+'_'+str(n_envs)+'envs'+'_act' + str(action_type) + '_obs' + str(obs_type) + '_lr' + str(lr) + '_batchSize' + str(batch_size)+'_ppoupdatesteps'+str(ppo_update_steps)+('_curriculum' if use_curriculum else ''),
    # Things to add as args: n_envs, ppo_update_steps, use_curriculum

    components = [
        f"{n_envs}envs",
        f"obs-{config['obs type']}",
        f"act-{config['action type']}",
    ]

    # Add critical hyperparameters
    components.extend([
        f"lr-{config['lr']}",
        f"bs-{config['batch size']}",
        f"g-{config['gamma']}",
        f"fs-{config.get('frame skip', 1)}",
        f"ppoupdates-{config['ppo update steps']}",
        f"curriculum-{config['use curriculum']}"
        f"rew-wtn-{config['rew-shaping_coeff_wtn']}",
        f"rew-prox-{config['rew-shaping_coeff_prox']}",
    ])

    # Add a run identifier (could be auto-incremented or timestamp-based)
    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d_%H%M")

    # Combine all components
    run_name = "_".join(components) + f"_{timestamp}"

    return run_name

# New callback with difficulty
class EnhancedWandbCallback(BaseCallback):
    def __init__(self, env_config, verbose=0, eval_env=None, run=None,
                 use_curriculum=False, min_target_ids_to_advance=8, run_name = 'no_name'):
        super(EnhancedWandbCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = env_config['eval_freq']
        self.n_eval_episodes = env_config['n_eval_episodes']
        self.run = run

        self.use_curriculum = env_config['use_curriculum']
        self.min_target_ids_to_advance = env_config['min_target_ids_to_advance']

        self.current_difficulty = 0
        self.above_threshold_counter = 0  # Tracks how many evals in a row that the agent scores above the threshold to advance to next curriculum level. If >= 3, difficutly is updated. Resets if the agent scores less than 8

    def _on_step(self):

        if self.locals.get("infos") and len(self.locals["infos"]) > 0:
            # First, gather all data we want to log
            log_data = {}
            rewards = []
            lengths = []

            # Process data from all environments
            for env_idx, info in enumerate(self.locals["infos"]):
                if "episode" in info:
                    # Collect the data but don't log yet
                    rewards.append(info["episode"]["r"])
                    lengths.append(info["episode"]["l"])

                    # Add individual env data (optional)
                    log_data[f"train/env{env_idx}/episode_reward"] = info["episode"]["r"]
                    log_data[f"train/env{env_idx}/episode_length"] = info["episode"]["l"]

                    # Add target_ids and detections if available
                    if "target_ids" in info:
                        log_data[f"train/env{env_idx}/target_ids"] = info["target_ids"]
                    if "detections" in info:
                        log_data[f"train/env{env_idx}/detections"] = info["detections"]

            # Add overall average metrics
            if rewards:
                log_data["train/mean_episode_reward"] = np.mean(rewards)
                log_data["train/mean_episode_length"] = np.mean(lengths)

            # Log all the data at once - explicitly use self.num_timesteps // n_envs as step
            # to correctly align with PPO's step count
            if log_data:
                self.run.log(log_data,
                             step=self.num_timesteps // self.model.get_env().num_envs)  # TODO check the div by num_envs here

        # Periodically evaluate and log evaluation metrics
        if self.eval_env is not None and self.num_timesteps % self.eval_freq == 0:
            print(f'\n#################################################\nEVALUATING (step: {self.num_timesteps})')

            # Run evaluation
            target_ids_list = []
            mean_reward, std_reward = 0, 0

            for i in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                terminated = False
                truncated = False
                ep_reward = 0

                while not (terminated or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    ep_reward += reward

                # Collect target_ids from the info dict
                if "target_ids" in info:
                    target_ids_list.append(info["target_ids"])

                mean_reward += ep_reward / self.n_eval_episodes

            # Calculate standard deviation
            std_reward = np.std(target_ids_list) if target_ids_list else 0

            # Log evaluation results
            self.run.log({
                "eval/mean_reward": mean_reward,
                "eval/std_reward": std_reward,
                "eval/mean_target_ids": np.mean(target_ids_list) if target_ids_list else 0,
                "curriculum/difficulty_level": self.current_difficulty
            }, step=self.num_timesteps)

            print(f'\nEVAL LOGGED (mean reward {mean_reward}, std {round(std_reward, 2)}, '
                  f'mean target_ids: {np.mean(target_ids_list) if target_ids_list else 0}')

            # Check if we should increase difficulty level
            if self.use_curriculum:
                print('CURRICULUM: Checking if we should increase difficulty')
                avg_target_ids = np.mean(target_ids_list) if target_ids_list else 0
                if avg_target_ids >= self.min_target_ids_to_advance:
                    self.above_threshold_counter += 1
                else:
                    self.above_threshold_counter = 0

                if self.above_threshold_counter >= 5:
                    self.above_threshold_counter = 0
                    self.current_difficulty += 1
                    print(f'CURRICULUM: Increasing difficulty to level {self.current_difficulty}')

                    # Call the set_difficulty method on all environments
                    self.model.get_env().env_method("set_difficulty", self.current_difficulty)
                    self.eval_env.unwrapped.set_difficulty(self.current_difficulty)
                    self.run.log({"curriculum/difficulty_level": self.current_difficulty}, step=self.num_timesteps)

                else:
                    print(f'CURRICULUM: Maintaining difficulty at level {self.current_difficulty} '
                          f'(avg target_ids: {avg_target_ids} < threshold: {self.min_target_ids_to_advance})')

            print('#################################################\n\nReturning to training...')

        return True


def make_env(env_config, rank, seed, run_name='no_name'):
    def _init():
        from env_combined import MAISREnvVec  # TODO combine non-simple env into this one

        env = MAISREnvVec(
            config=env_config,
            render_mode='headless',
            run_name=run_name,
            tag=f'train_mp{rank}',
            seed=seed + rank,
        )
        env = Monitor(env)
        env.reset()
        return env

    return _init


def train(
        save_dir="./trained_models/",
        load_dir=None,
        log_dir="./logs/",
        #algo='PPO',
        #policy_type="MlpPolicy",
        #run_name='no name',
        #lr=3e-4,
        #batch_size=128,
        #steps_per_episode=14703,
        #ppo_update_steps=14703,
        #num_timesteps=30e6,
        #save_freq=14600 * 3,
        #eval_freq=14600 * 2,
        #n_eval_episodes=8,
        env_config_filename='./config_files/rl_training_default.json',
        n_envs=1,
        #seed=42,
        #frame_skip=1,
        #use_curriculum=False,
        #min_target_ids_to_advance=8,
        #max_difficulty_level=5,
        #gamma=0.998
):
    from env_combined import MAISREnvVec

    # Load env config
    env_config = load_env_config(env_config_filename)

    n_envs = min(n_envs, multiprocessing.cpu_count())
    env_config['n_envs'] = n_envs
    env_config['config filename'] = env_config_filename

    # Generate run name (To be consistent between WandB, model saving, and action history plots
    run_name = generate_run_name(env_config)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    run = wandb.init(
        project="maisr-rl-pace",
        name=f'home_{n_envs}envs'+run_name,
        config=env_config,
        sync_tensorboard=True,
        monitor_gym=True,
    )

    if n_envs > 1:
        #n_envs = min(n_envs, multiprocessing.cpu_count())  # Use at most 8 or the number of CPU cores
        print(f"Training with {n_envs} environments in parallel")

        # Create environment creation functions for each process
        env_fns = [make_env(env_config, i, env_config['seed'] + i, run_name=run_name)for i in range(n_envs)]
        env = SubprocVecEnv(env_fns)
        env = VecMonitor(env, filename=os.path.join(log_dir, 'vecmonitor'))

    else:
        env = MAISREnvVec(
            env_config,
            None,
            render_mode='headless',
            tag='train',
            run_name=run_name,
        )
        env = Monitor(env)

    eval_env = MAISREnvVec(
        env_config,
        None,
        render_mode='headless',
        tag='eval',
        run_name=run_name,
    )
    eval_env = Monitor(eval_env)

    print('Envs created')

    ################################################# Setup callbacks #################################################
    checkpoint_callback = CheckpointCallback(
        save_freq=env_config['save_freq'] // n_envs,  # Adjust for number of environments
        save_path=save_dir,
        name_prefix=f"checkpoint{env_config['algo']}_maisr",
        save_replay_buffer=True, save_vecnormalize=True,
    )

    wandb_callback = WandbCallback(gradient_save_freq=0, model_save_path=f"{save_dir}/wandb/{run.id}", verbose=2)

    enhanced_wandb_callback = EnhancedWandbCallback(env_config, eval_env=eval_env, run=run)

    print('Callbacks created')

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        batch_size=env_config['batch_size'],  # * n_envs,  # Scale batch size with number of environments TODO testing
        n_steps=env_config['ppo_update_steps'],  # // n_envs,  # Adjust steps per environment TODO testing
        learning_rate=env_config['lr'],
        seed=env_config['seed'],
        device='cpu',
        gamma=env_config['gamma']
    )

    print('Model instantiated')

    # Check if there's a checkpoint to load
    if load_dir:
        model = model.__class__.load(load_dir, env=env)
    else:
        print('Training new model')


    print('####################################### Beginning agent training... #########################################\n')

    # Log initial difficulty
    run.log({"curriculum/difficulty_level": 0}, step=0)
    print(f'Starting with difficulty level {0}')

    model.learn(
        total_timesteps=int(env_config['num_timesteps']),
        callback=[checkpoint_callback, wandb_callback, enhanced_wandb_callback],
        reset_num_timesteps=True,  # Set to False when resuming training
    )


    print('####################################### TRAINING COMPLETE #########################################\n')
    env.close()
    eval_env.close()

    # Save the final model
    final_model_path = os.path.join(save_dir, f"{env_config['algo']}_maisr_final_diff")
    model.save(final_model_path)
    print(f"Training completed! Final model saved to {final_model_path}")

    # Run a final evaluation
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=env_config['n_eval_episodes'])
    print(f"Final evaluation: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # Log final metrics to wandb
    run.log({"final/mean_reward": mean_reward, "final/std_reward": std_reward, })
    run.finish()


if __name__ == "__main__":
    # for reward_type in ['proximity and waypoint-to-nearest']:#['proximity and target', 'waypoint-to-nearest', 'proximity and waypoint-to-nearest']:
    #for obs_type in ['relative', 'absolute']:
        #for action_type in ['continuous-normalized']:  # , 'discrete-downsampled']:
            #for n_envs in [24]:
                #for lr in [5e-5]:
                    #for ppo_update_steps in [2048, 1024]:
                        #for batch_size in [128]:
                            #for gamma in [0.998]:
    print('\n################################################################################')
    print('################################################################################')
    print(f'################## STARTING TRAINING RUN ##################')
    print('################################################################################')
    print('################################################################################')

    n_envs = 24

    train(
        n_envs=n_envs,
    )