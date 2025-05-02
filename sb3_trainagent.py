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

class EnhancedWandbCallback(BaseCallback):
    def __init__(self, verbose=0, eval_env=None, eval_freq=14400, n_eval_episodes=8, run=None):
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
                        "train/episode_length": info["episode"]["l"],
                        "train/target_ids":info["target_ids"],
                        "train/detections": info["detections"],
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


def make_env(use_simple,env_config,rank, seed):
    def _init():
        if use_simple:
            from env_vec_simple import MAISREnvVec
        else:
            from env_vec import MAISREnvVec

        env = MAISREnvVec(
            config=env_config,
            render_mode='headless',
            reward_type='proximity and target',
            obs_type='absolute',
            action_type='continuous-normalized',
        )
        env = Monitor(env)  # record stats such as returns
        env.reset(seed=seed + rank)
        return env

    return _init  # Return the function, not the environment instance


def train(
        use_simple,
        obs_type, # Must be "absolute" or "relative"
        action_type, # Must be 'continuous-normalized' or 'discrete-downsampled'
        #reward_type, # Must be 'proximity', 'waypoint-to-nearest'

        save_dir = "./trained_models/",
        load_dir= None,
        log_dir = "./logs/",
        algo='PPO',
        policy_type = "MlpPolicy",
        lr=3e-4,
        batch_size=128,
        steps_per_episode=14703,
        num_timesteps=20e6,
        save_freq=14600*3,
        eval_freq=14600*2,
        n_eval_episodes = 8,
        env_config_filename = './config_files/rl_training_config.json',
        n_envs = 1,
        seed=42,
        force_seed = False,
        use_curriculum=False
    ):

    if use_simple: from env_vec_simple import MAISREnvVec
    else: from env_vec import MAISREnvVec

    # Load env config
    env_config = load_env_config(env_config_filename)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_config = {
        "env_type": "simple_v1" if use_simple else "normal",
        #"reward_type": reward_type,
        "action_type": action_type,
        "obs_type": obs_type,
        "algorithm": algo,
        "policy_type": policy_type,
        "lr": lr,
        "batch_size": 128,
        "steps_per_episode": steps_per_episode,
        "num_timesteps": num_timesteps,
        "save_freq": save_freq,
        "checkpoint_freq": eval_freq,
        "n_eval_episodes": n_eval_episodes,
        "env_config": env_config,
        "n_envs": n_envs,
        "seed": seed,
        "env_name": "MAISREnvVec"
        }

    run = wandb.init(
        project="maisr-rl",
        name=str(algo)+'_'+'forceseed'+str(force_seed)+str('_simpleV1' if use_simple else 'normal')+'_act'+str(action_type)+'_obs'+str(obs_type)+'_lr'+str(lr)+'_batchSize'+str(batch_size),
        config=train_config,
        sync_tensorboard=True,
        monitor_gym=True,
    )

    if n_envs > 1: # TODO implement multiprocessing
        raise ValueError('Multiprocessing not supported yet')
        #n_envs = min(n_envs, multiprocessing.cpu_count())  # Use at most 8 or the number of CPU cores
        #print(f"Training with {n_envs} environments in parallel")
        #env = make_vec_env(MAISREnvVec, n_envs=1, env_kwargs=dict(config=env_config, render_mode='headless', reward_type='balanced-sparse', obs_type='vector', action_type='continuous'), monitor_dir=log_dir)
        #env_fns = [make_env(env_config, i, seed) for i in range(n_envs)]
        #env = SubprocVecEnv(env_fns)
        #env = VecMonitor(env)

    else:
        env = MAISREnvVec(
            env_config,
            None,
            render_mode='headless',
            obs_type=obs_type,
            action_type=action_type,
            #reward_type=reward_type,
            tag='train',
            seed = 42 if force_seed else None, # TODO currently forcing this seed
        )
        env = Monitor(env)

    eval_env = MAISREnvVec(
        env_config,
        None,
        render_mode='headless',
        obs_type=obs_type,
        action_type=action_type,
        #reward_type=reward_type,
        tag='eval',
        seed = 42 if force_seed else None, # TODO currently forcing this seed
    )
    eval_env = Monitor(eval_env)


    ################################################# Setup callbacks #################################################
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=save_dir,
        name_prefix=f"checkpoint{algo}_maisr",
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

    wandb_callback = WandbCallback(
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
        env,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",  # Match tensorboard directory to wandb run.id
        batch_size=batch_size, # * n_envs,  # Scale batch size with number of environments
        n_steps= steps_per_episode, #2048 // n_envs,  # Adjust steps per environment
        learning_rate=lr,
        seed=seed,
        device='cpu'
    )

    # Check if there's a checkpoint to load
    if load_dir: model = model.__class__.load(load_dir, env=env)
    else: print('Training new model')


    print('####################################### Beginning agent training... #########################################\n')

    if use_curriculum:
        raise ValueError('Curriculum learning not set up yet')
        # phase_configs = [
        #     './config_files/rl_phase1',
        #     './config_files/rl_phase2',
        #     './config_files/rl_phase3',
        #     './config_files/rl_phase4',
        # ]
        # current_phase = 1
        # while current_phase <= 4:
        #     print(f"Training on phase {current_phase}")
        #     model.learn(
        #         total_timesteps=int(num_timesteps),
        #         callback=[checkpoint_callback, wandb_callback, enhanced_wandb_callback],
        #         reset_num_timesteps=True,  # Set to False when resuming training
        #     )
        #
        #     if current_phase < 4:
        #         current_phase += 1
        #
        #         env_config = load_env_config(phase_configs[current_phase])
        #         env = MAISREnvVec(
        #             env_config,
        #             None,
        #             render_mode='headless',
        #             obs_type=obs_type,
        #             action_type=action_type,
        #             # reward_type=reward_type,
        #             tag='train'
        #         )
        #         env = Monitor(env)
        #
        #         eval_env = MAISREnvVec(
        #             env_config,
        #             None,
        #             render_mode='headless',
        #             obs_type=obs_type,
        #             action_type=action_type,
        #             # reward_type=reward_type,
        #             tag='eval'
        #         )
        #         eval_env = Monitor(eval_env)
        #
        #         model.set_env(create_env(phase_configs[current_phase]))
        #         print(f"Advancing to phase {current_phase}")
        #     else:
        #         print("Curriculum completed!")
        #         break

    else:
        model.learn(
            total_timesteps=int(num_timesteps),
            callback=[checkpoint_callback, wandb_callback, enhanced_wandb_callback], # Removed eval_callback, wandb_callback
            reset_num_timesteps=True,  # Set to False when resuming training
        )

    print('####################################### TRAINING COMPLETE #########################################\n')
    # Save the final model
    final_model_path = os.path.join(save_dir, f"{algo}_maisr_final")
    model.save(final_model_path)
    print(f"Training completed! Final model saved to {final_model_path}")

    # Run a final evaluation
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes)
    print(f"Final evaluation: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # Log final metrics to wandb
    run.log({"final/mean_reward": mean_reward, "final/std_reward": std_reward})
    run.finish()


if __name__ == "__main__":

    # use_simple = True
    # reward_type = 'proximity and target'
    # action_type = 'continuous'
    # obs_type = 'absolute'


    for use_simple in [True]:
        #for reward_type in ['proximity and waypoint-to-nearest']:#['proximity and target', 'waypoint-to-nearest', 'proximity and waypoint-to-nearest']:
        for obs_type in ['relative', 'absolute']:
            for action_type in ['direct-control', 'continuous-normalized', 'discrete-downsampled']:
                for lr in [1e-5, 1e-4, 1e-3]:
                    for force_seed in [True, False]:

                        print('\n################################################################################')
                        print('################################################################################')
                        print(f'STARTING TRAINING RUN: obs type {obs_type}, action_type {action_type}, lr {lr}')
                        print('################################################################################')
                        print('################################################################################')

                        train(
                            use_simple,
                            obs_type,
                            action_type,
                            #reward_type,
                            num_timesteps=30e6,
                            n_eval_episodes=8,
                            lr = lr,
                            eval_freq=14703*50,
                            use_curriculum=False,
                            force_seed=force_seed,
                        )