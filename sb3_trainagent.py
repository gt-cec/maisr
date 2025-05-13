import gymnasium as gym
import os
import numpy as np

from sympy.physics.units import action

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
from typing import Dict, List, Tuple, Any

from utility.data_logging import load_env_config


# class EnhancedWandbCallback(BaseCallback):
#     def __init__(self, verbose=0, eval_env=None, eval_freq=14400, n_eval_episodes=8, run=None):
#         super(EnhancedWandbCallback, self).__init__(verbose)
#         self.eval_env = eval_env
#         self.eval_freq = eval_freq
#         self.n_eval_episodes = n_eval_episodes
#         self.run = run
#
#     def _on_step(self):
#         # Log training metrics
#         if self.locals.get("infos") and len(self.locals["infos"]) > 0:
#             # For vectorized environments, infos is a list of dicts, one for each env
#             for env_idx, info in enumerate(self.locals["infos"]):
#                 if "episode" in info:
#                     # Log each environment's episode metrics
#                     self.run.log({
#                         f"train/env{env_idx}/episode_reward": info["episode"]["r"],
#                         f"train/env{env_idx}/episode_length": info["episode"]["l"],
#                     }, step=self.num_timesteps)
#
#                     # Also log target_ids and detections if available
#                     if "target_ids" in info:
#                         self.run.log({f"train/env{env_idx}/target_ids": info["target_ids"]}, step=self.num_timesteps)
#                     if "detections" in info:
#                         self.run.log({f"train/env{env_idx}/detections": info["detections"]}, step=self.num_timesteps)
#
#                     # Log overall average across environments
#                     if env_idx == 0:  # Only log this once per step
#                         rewards = [info["episode"]["r"] for info in self.locals["infos"] if "episode" in info]
#                         lengths = [info["episode"]["l"] for info in self.locals["infos"] if "episode" in info]
#                         if rewards:
#                             self.run.log({
#                                 "train/mean_episode_reward": np.mean(rewards),
#                                 "train/mean_episode_length": np.mean(lengths),
#                             }, step=self.num_timesteps)
#
#         # Periodically evaluate and log evaluation metrics
#         if self.eval_env is not None and self.num_timesteps % self.eval_freq == 0:
#             print(f'\n#################################################\nEVALUATING (step: {self.num_timesteps})')
#
#             mean_reward, std_reward = evaluate_policy(
#                 self.model,
#                 self.eval_env,
#                 n_eval_episodes=self.n_eval_episodes
#             )
#
#             self.run.log({
#                 "eval/mean_reward": mean_reward,
#                 "eval/std_reward": std_reward
#             }, step=self.num_timesteps)
#             print(f'\nEVAL LOGGED (mean reward {mean_reward}, std {round(std_reward, 2)}')
#             print('#################################################\n\nReturning to training...')
#
#         return True


# New callback with difficulty
class EnhancedWandbCallback(BaseCallback):
    def __init__(self, verbose=0, eval_env=None, eval_freq=14400, n_eval_episodes=8, run=None,
                 use_curriculum=False, min_target_ids_to_advance=8, difficulty_increase_callback=None):
        super(EnhancedWandbCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.run = run
        self.use_curriculum = use_curriculum
        self.min_target_ids_to_advance = min_target_ids_to_advance
        self.difficulty_increase_callback = difficulty_increase_callback
        self.current_difficulty = 0

    def _on_step(self):
        # Log training metrics
        if self.locals.get("infos") and len(self.locals["infos"]) > 0:
            # For vectorized environments, infos is a list of dicts, one for each env
            for env_idx, info in enumerate(self.locals["infos"]):
                if "episode" in info:
                    # Log each environment's episode metrics
                    self.run.log({
                        f"train/env{env_idx}/episode_reward": info["episode"]["r"],
                        f"train/env{env_idx}/episode_length": info["episode"]["l"],
                    }, step=self.num_timesteps)

                    # Also log target_ids and detections if available
                    if "target_ids" in info:
                        self.run.log({f"train/env{env_idx}/target_ids": info["target_ids"]}, step=self.num_timesteps)
                    if "detections" in info:
                        self.run.log({f"train/env{env_idx}/detections": info["detections"]}, step=self.num_timesteps)

                    # Log overall average across environments
                    if env_idx == 0:  # Only log this once per step
                        rewards = [info["episode"]["r"] for info in self.locals["infos"] if "episode" in info]
                        lengths = [info["episode"]["l"] for info in self.locals["infos"] if "episode" in info]
                        if rewards:
                            self.run.log({
                                "train/mean_episode_reward": np.mean(rewards),
                                "train/mean_episode_length": np.mean(lengths),
                            }, step=self.num_timesteps)

        # Periodically evaluate and log evaluation metrics
        if self.eval_env is not None and self.num_timesteps % self.eval_freq == 0:
            print(f'\n#################################################\nEVALUATING (step: {self.num_timesteps})')

            # Run evaluation
            target_ids_list = []
            mean_reward, std_reward = 0, 0

            for i in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                truncated = False
                ep_reward = 0

                while not (done or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = self.eval_env.step(action)
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
            if self.use_curriculum and self.difficulty_increase_callback is not None:
                print('CURRICULUM: Checking if we should increase difficulty')
                avg_target_ids = np.mean(target_ids_list) if target_ids_list else 0
                if avg_target_ids >= self.min_target_ids_to_advance:
                    self.current_difficulty += 1
                    print(f'CURRICULUM: Increasing difficulty to level {self.current_difficulty}')
                    self.difficulty_increase_callback(self.current_difficulty)
                    self.run.log({"curriculum/difficulty_level": self.current_difficulty}, step=self.num_timesteps)
                else:
                    print(f'CURRICULUM: Maintaining difficulty at level {self.current_difficulty} '
                          f'(avg target_ids: {avg_target_ids} < threshold: {self.min_target_ids_to_advance})')

            print('#################################################\n\nReturning to training...')

        return True


def make_env(env_config, rank, seed, obs_type, action_type, difficulty=0):
    def _init():
        from env_vec_simple import MAISREnvVec  # TODO combine non-simple env into this one

        env = MAISREnvVec(
            config=env_config,
            render_mode='headless',
            reward_type=env_config['reward type'],
            obs_type=obs_type,
            tag=f'train_mp{rank}',
            action_type=action_type,
            seed=seed + rank,
            difficulty=difficulty  # Add difficulty parameter
        )
        env = Monitor(env)
        env.reset()
        return env

    return _init


def train(
        obs_type,  # Must be "absolute" or "relative"
        action_type,  # Must be 'continuous-normalized' or 'discrete-downsampled'
        # reward_type, # Must be 'proximity', 'waypoint-to-nearest'

        save_dir="./trained_models/",
        load_dir=None,
        log_dir="./logs/",
        algo='PPO',
        policy_type="MlpPolicy",
        lr=3e-4,
        batch_size=128,
        steps_per_episode=14703,
        num_timesteps=30e6,
        save_freq=14600 * 3,
        eval_freq=14600 * 2,
        n_eval_episodes=8,
        env_config_filename='./config_files/rl_cl_phase1.json',
        n_envs=1,
        seed=42,
        use_curriculum=False,
        min_target_ids_to_advance=8,
        max_difficulty_level=5
):
    from env_vec_simple import MAISREnvVec
    # if use_simple: from env_vec_simple import MAISREnvVec
    # else: from env_vec import MAISREnvVec

    # Load env config
    env_config = load_env_config(env_config_filename)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_config = {
        "env_type": "simple_v1",  # if use_simple else "normal",
        # "reward_type": reward_type,
        "action_type": action_type,
        "obs_type": obs_type,
        "algorithm": algo,
        "policy_type": policy_type,
        "lr": lr,
        "batch_size": batch_size,
        "steps_per_episode": steps_per_episode,
        "num_timesteps": num_timesteps,
        "save_freq": save_freq,
        "checkpoint_freq": eval_freq,
        "n_eval_episodes": n_eval_episodes,
        "env_config": env_config,
        "n_envs": n_envs,
        "seed": seed,
        "env_name": "MAISREnvVec",
        "use_curriculum": use_curriculum,
        "min_target_ids_to_advance": min_target_ids_to_advance,
        "max_difficulty_level": max_difficulty_level,
        "initial_difficulty": 0
    }

    run = wandb.init(
        project="maisr-rl",
        name='tr3_subproc' + '_act' + str(action_type) + '_obs' + str(obs_type) + '_lr' + str(lr) + '_batchSize' + str(
            batch_size) + ('_curriculum' if use_curriculum else ''),
        config=train_config,
        sync_tensorboard=True,
        monitor_gym=True,
    )

    # Current difficulty level for curriculum learning
    current_difficulty = 0

    # Function to update difficulty level in environments
    def update_difficulty(new_difficulty):
        nonlocal current_difficulty
        nonlocal env
        nonlocal eval_env

        if new_difficulty > max_difficulty_level:
            print(f"Maximum difficulty level {max_difficulty_level} reached. Maintaining difficulty.")
            return

        current_difficulty = new_difficulty

        # For vectorized environments
        if isinstance(env, VecMonitor):
            # We need to close current environments and create new ones with updated difficulty
            env.close()

            # Create environment creation functions for each process with new difficulty
            env_fns = [make_env(env_config, i, seed + i, obs_type, action_type, difficulty=current_difficulty)
                       for i in range(n_envs)]

            # Create new vectorized environment
            env = SubprocVecEnv(env_fns)
            env = VecMonitor(env, filename=os.path.join(log_dir, 'vecmonitor'))

        # For single environment
        else:
            env.close()
            env = MAISREnvVec(
                env_config,
                None,
                render_mode='headless',
                obs_type=obs_type,
                action_type=action_type,
                tag='train',
                seed=seed,
                difficulty=current_difficulty
            )
            env = Monitor(env)

        # Update evaluation environment as well
        eval_env.close()
        eval_env = MAISREnvVec(
            env_config,
            None,
            render_mode='headless',
            obs_type=obs_type,
            action_type=action_type,
            tag='eval',
            seed=seed,
            difficulty=current_difficulty
        )
        eval_env = Monitor(eval_env)

        # Update model's environment
        model.set_env(env)

        print(f"Updated environments to difficulty level {current_difficulty}")

    if n_envs > 1:  # TODO implement multiprocessing
        # raise ValueError('Multiprocessing not supported yet')
        n_envs = min(n_envs, multiprocessing.cpu_count())  # Use at most 8 or the number of CPU cores
        print(f"Training with {n_envs} environments in parallel")
        # env = make_vec_env(MAISREnvVec, n_envs=1, env_kwargs=dict(config=env_config, render_mode='headless', reward_type='balanced-sparse', obs_type='vector', action_type='continuous'), monitor_dir=log_dir)

        # Create environment creation functions for each process
        env_fns = [make_env(env_config, i, seed + i, obs_type, action_type, difficulty=current_difficulty)
                   for i in range(n_envs)]

        # Create vectorized environment
        env = SubprocVecEnv(env_fns)
        env = VecMonitor(env, filename=os.path.join(log_dir, 'vecmonitor'))

    else:
        env = MAISREnvVec(
            env_config,
            None,
            render_mode='headless',
            obs_type=obs_type,
            action_type=action_type,
            # reward_type=reward_type,
            tag='train',
            seed=seed,
            difficulty=current_difficulty
        )
        env = Monitor(env)

    eval_env = MAISREnvVec(
        env_config,
        None,
        render_mode='headless',
        obs_type=obs_type,
        action_type=action_type,
        # reward_type=reward_type,
        tag='eval',
        seed=seed,
        difficulty=current_difficulty
    )
    eval_env = Monitor(eval_env)

    ################################################# Setup callbacks #################################################
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,  # Adjust for number of environments
        save_path=save_dir,
        name_prefix=f"checkpoint{algo}_maisr",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=0,
        model_save_path=f"{save_dir}/wandb/{run.id}",
        verbose=2,
    )

    enhanced_wandb_callback = EnhancedWandbCallback(
        eval_env=eval_env,
        eval_freq=eval_freq // n_envs,  # Adjust for number of environments
        n_eval_episodes=n_eval_episodes,
        run=run,
        use_curriculum=use_curriculum,
        min_target_ids_to_advance=min_target_ids_to_advance,
        difficulty_increase_callback=update_difficulty if use_curriculum else None
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        batch_size=batch_size * n_envs,  # Scale batch size with number of environments
        n_steps=steps_per_episode,# // n_envs,  # Adjust steps per environment
        learning_rate=lr,
        seed=seed,
        device='cpu'  # You can change to 'cuda' if you have a GPU
    )

    # Check if there's a checkpoint to load
    if load_dir: model = model.__class__.load(load_dir, env=env)
    else: print('Training new model')

    print('####################################### Beginning agent training... #########################################\n')
    # Log initial difficulty
    run.log({"curriculum/difficulty_level": current_difficulty}, step=0)
    print(f'Starting with difficulty level {current_difficulty}')

    model.learn(
        total_timesteps=int(num_timesteps),
        callback=[checkpoint_callback, wandb_callback, enhanced_wandb_callback],
        reset_num_timesteps=True,  # Set to False when resuming training
    )

    print('####################################### TRAINING COMPLETE #########################################\n')
    # Save the final model
    final_model_path = os.path.join(save_dir, f"{algo}_maisr_final_diff{current_difficulty}")
    model.save(final_model_path)
    print(f"Training completed! Final model saved to {final_model_path}")

    # Run a final evaluation
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes)
    print(f"Final evaluation: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # Log final metrics to wandb
    run.log({
        "final/mean_reward": mean_reward,
        "final/std_reward": std_reward,
        "final/difficulty_level": current_difficulty
    })
    run.finish()


#
# def train(
#         obs_type, # Must be "absolute" or "relative"
#         action_type, # Must be 'continuous-normalized' or 'discrete-downsampled'
#         #reward_type, # Must be 'proximity', 'waypoint-to-nearest'
#
#         save_dir = "./trained_models/",
#         load_dir= None,
#         log_dir = "./logs/",
#         algo='PPO',
#         policy_type = "MlpPolicy",
#         lr=3e-4,
#         batch_size=128,
#         steps_per_episode=14703,
#         num_timesteps=30e6,
#         save_freq=14600*3,
#         eval_freq=14600*2,
#         n_eval_episodes = 8,
#         env_config_filename = './config_files/rl_cl_phase1.json',
#         n_envs = 1,
#         seed=42,
#         use_curriculum=False,
#         min_target_ids_to_advance=8,
#         max_difficulty_level=4
#     ):
#
#     from env_vec_simple import MAISREnvVec
#     #if use_simple: from env_vec_simple import MAISREnvVec
#     #else: from env_vec import MAISREnvVec
#
#     # Load env config
#     env_config = load_env_config(env_config_filename)
#
#     os.makedirs(save_dir, exist_ok=True)
#     os.makedirs(log_dir, exist_ok=True)
#
#     train_config = {
#         "env_type": "simple_v1",# if use_simple else "normal",
#         #"reward_type": reward_type,
#         "action_type": action_type,
#         "obs_type": obs_type,
#         "algorithm": algo,
#         "policy_type": policy_type,
#         "lr": lr,
#         "batch_size": batch_size,
#         "steps_per_episode": steps_per_episode,
#         "num_timesteps": num_timesteps,
#         "save_freq": save_freq,
#         "checkpoint_freq": eval_freq,
#         "n_eval_episodes": n_eval_episodes,
#         "env_config": env_config,
#         "n_envs": n_envs,
#         "seed": seed,
#         "env_name": "MAISREnvVec"
#         }
#
#     run = wandb.init(
#         project="maisr-rl",
#         name='tr3_subproc_'+'_act'+str(action_type)+'_obs'+str(obs_type)+'_lr'+str(lr)+'_batchSize'+str(batch_size),
#         config=train_config,
#         sync_tensorboard=True,
#         monitor_gym=True,
#     )
#
#     if n_envs > 1: # TODO implement multiprocessing
#         #raise ValueError('Multiprocessing not supported yet')
#         n_envs = min(n_envs, multiprocessing.cpu_count())  # Use at most 8 or the number of CPU cores
#         print(f"Training with {n_envs} environments in parallel")
#         #env = make_vec_env(MAISREnvVec, n_envs=1, env_kwargs=dict(config=env_config, render_mode='headless', reward_type='balanced-sparse', obs_type='vector', action_type='continuous'), monitor_dir=log_dir)
#
#         # Create environment creation functions for each process
#         env_fns = [make_env(env_config, i, seed + i, obs_type, action_type) for i in range(n_envs)]
#
#         # Create vectorized environment
#         env = SubprocVecEnv(env_fns)
#         env = VecMonitor(env, filename=os.path.join(log_dir, 'vecmonitor'))
#
#     else:
#         env = MAISREnvVec(
#             env_config,
#             None,
#             render_mode='headless',
#             obs_type=obs_type,
#             action_type=action_type,
#             #reward_type=reward_type,
#             tag='train',
#             seed = 42
#         )
#         env = Monitor(env)
#
#     eval_env = MAISREnvVec(
#         env_config,
#         None,
#         render_mode='headless',
#         obs_type=obs_type,
#         action_type=action_type,
#         #reward_type=reward_type,
#         tag='eval',
#         seed = 42
#     )
#     eval_env = Monitor(eval_env)
#
#
#     ################################################# Setup callbacks #################################################
#     checkpoint_callback = CheckpointCallback(
#         save_freq=save_freq // n_envs,  # Adjust for number of environments
#         save_path=save_dir,
#         name_prefix=f"checkpoint{algo}_maisr",
#         save_replay_buffer=True,
#         save_vecnormalize=True,
#     )
#
#     wandb_callback = WandbCallback(
#         gradient_save_freq=0,
#         model_save_path=f"{save_dir}/wandb/{run.id}",
#         verbose=2,
#     )
#
#     enhanced_wandb_callback = EnhancedWandbCallback(
#         eval_env=eval_env,
#         eval_freq=eval_freq // n_envs,  # Adjust for number of environments
#         n_eval_episodes=n_eval_episodes,
#         run=run
#     )
#
#     model = PPO(
#         "MlpPolicy",
#         env,
#         verbose=1,
#         tensorboard_log=f"runs/{run.id}",
#         batch_size=batch_size * n_envs,  # Scale batch size with number of environments
#         n_steps=2048, # // n_envs,  # Adjust steps per environment
#         learning_rate=lr,
#         seed=seed,
#         device='cpu'  # You can change to 'cuda' if you have a GPU
#     )
#
#     # Check if there's a checkpoint to load
#     if load_dir: model = model.__class__.load(load_dir, env=env)
#     else: print('Training new model')
#
#     print('####################################### Beginning agent training... #########################################\n')
#     if use_curriculum:
#         raise ValueError('Curriculum learning not set up yet')
#         # phase_configs = [
#         #     './config_files/rl_phase1',
#         #     './config_files/rl_phase2',
#         #     './config_files/rl_phase3',
#         #     './config_files/rl_phase4',
#         # ]
#         # current_phase = 1
#         # while current_phase <= 4:
#         #     print(f"Training on phase {current_phase}")
#         #     model.learn(
#         #         total_timesteps=int(num_timesteps),
#         #         callback=[checkpoint_callback, wandb_callback, enhanced_wandb_callback],
#         #         reset_num_timesteps=True,  # Set to False when resuming training
#         #     )
#         #
#         #     if current_phase < 4:
#         #         current_phase += 1
#         #
#         #         env_config = load_env_config(phase_configs[current_phase])
#         #         env = MAISREnvVec(
#         #             env_config,
#         #             None,
#         #             render_mode='headless',
#         #             obs_type=obs_type,
#         #             action_type=action_type,
#         #             # reward_type=reward_type,
#         #             tag='train'
#         #         )
#         #         env = Monitor(env)
#         #
#         #         eval_env = MAISREnvVec(
#         #             env_config,
#         #             None,
#         #             render_mode='headless',
#         #             obs_type=obs_type,
#         #             action_type=action_type,
#         #             # reward_type=reward_type,
#         #             tag='eval'
#         #         )
#         #         eval_env = Monitor(eval_env)
#         #
#         #         model.set_env(create_env(phase_configs[current_phase]))
#         #         print(f"Advancing to phase {current_phase}")
#         #     else:
#         #         print("Curriculum completed!")
#         #         break
#
#     else:
#         model.learn(
#             total_timesteps=int(num_timesteps),
#             callback=[checkpoint_callback, wandb_callback, enhanced_wandb_callback], # Removed eval_callback, wandb_callback
#             reset_num_timesteps=True,  # Set to False when resuming training
#         )
#
#     print('####################################### TRAINING COMPLETE #########################################\n')
#     # Save the final model
#     final_model_path = os.path.join(save_dir, f"{algo}_maisr_final")
#     model.save(final_model_path)
#     print(f"Training completed! Final model saved to {final_model_path}")
#
#     # Run a final evaluation
#     mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes)
#     print(f"Final evaluation: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
#
#     # Log final metrics to wandb
#     run.log({"final/mean_reward": mean_reward, "final/std_reward": std_reward})
#     run.finish()


if __name__ == "__main__":


    #for reward_type in ['proximity and waypoint-to-nearest']:#['proximity and target', 'waypoint-to-nearest', 'proximity and waypoint-to-nearest']:
    for obs_type in ['relative', 'absolute']:
        for action_type in ['continuous-normalized']:#, 'discrete-downsampled']:
            for lr in [5e-5]:
                for batch_size in [128, 64, 256, 512]:

                    print('\n################################################################################')
                    print('################################################################################')
                    print(f'STARTING TRAINING RUN: obs type {obs_type}, action_type {action_type}, lr {lr}')
                    print('################################################################################')
                    print('################################################################################')

                    train(
                        obs_type,
                        action_type,
                        #reward_type,
                        num_timesteps=20e6,
                        batch_size=batch_size,
                        n_eval_episodes=8,
                        lr = lr,
                        eval_freq=14703*50,
                        use_curriculum=False,
                        seed = 42,
                        n_envs=6
                    )