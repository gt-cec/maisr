"""New version of sb3_trainagent.py
Renamed to prepare for future training pipelines using RLlib etc, and also implemented cleaner config management"""

import warnings
warnings.filterwarnings("ignore", message="Your system is avx2 capable but pygame was not built with support for it")

import gymnasium as gym
import os
import numpy as np
import multiprocessing
import socket
import torch

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from env_combined import MAISREnvVec
from utility.data_logging import load_env_config
from utility.config_management import load_env_config_with_sweeps, generate_sweep_run_name


def generate_run_name(config):
    """Generate a unique, descriptive name for this training run. Will be shared across logs, WandB, and action
    history plots to make it easy to match them."""

    components = [
        f"{config['n_envs']}envs",
        #f"obs-{config['obs_type']}",
        #f"act-{config['action_type']}",
    ]

    # Add critical hyperparameters
    components.extend([
        #f"lr-{config['lr']}",
#        f"bs-{config['batch_size']}",
        #f"g-{config['gamma']}",
        # f"fs-{config.get('frame_skip', 1)}",
        # f"ppoupdates-{config['ppo_update_steps']}",
        # f"curriculum-{config['use_curriculum']}",
        # f"rew-wtn-{config['shaping_coeff_wtn']}",
        # f"rew-prox-{config['shaping_coeff_prox']}",
        # f"rew-timepenalty-{config['shaping_time_penalty']}",
        # f"rew-shapedecay-{config['shaping_decay_rate']}",
    ])

    # Add a run identifier (could be auto-incremented or timestamp-based)
    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d_%H%M")

    # Combine all components
    run_name = f"{timestamp}_" + "_".join(components)

    return run_name


class EnhancedWandbCallback(BaseCallback):
    """Custom Callback that:
    1. Logs training metrics to WandB
    2. Evaluates the agent periodically (also logged to WandB)
    3. Determines if the agent should progress to the next stage of the training curriculum
    4. Logs additional PPO training metrics
    """

    def __init__(self, env_config, verbose=0, eval_env=None, run=None,
                 use_curriculum=False, min_target_ids_to_advance=8, run_name='no_name',
                 log_freq=2):  # New parameter: log every N steps
        super(EnhancedWandbCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = env_config['eval_freq']
        self.n_eval_episodes = env_config['n_eval_episodes']
        self.run = run
        self.log_freq = log_freq  # Log every N steps instead of every step

        self.use_curriculum = env_config['curriculum_type'] != "none"
        self.min_target_ids_to_advance = env_config['min_target_ids_to_advance']
        self.max_ep_len_to_advance = 150

        self.current_difficulty = 0
        self.above_threshold_counter = 0

        # Buffer for accumulating data between log events
        self.episode_buffer = {
            'rewards': [],
            'lengths': [],
            'target_ids': [],
            'detections': []
        }

    def _on_step(self):
        # Only log on the specified frequency
        should_log_episode_data = self.num_timesteps % self.log_freq == 0

        # Always collect episode data when available (lightweight)
        if self.locals.get("infos") and len(self.locals["infos"]) > 0:
            for env_idx, info in enumerate(self.locals["infos"]):
                if "episode" in info:
                    # Always buffer the data
                    self.episode_buffer['rewards'].append(info["episode"]["r"])
                    self.episode_buffer['lengths'].append(info["episode"]["l"])

                    if "target_ids" in info:
                        self.episode_buffer['target_ids'].append(info["target_ids"])
                    if "detections" in info:
                        self.episode_buffer['detections'].append(info["detections"])

        # Only log episode data at the specified frequency
        if should_log_episode_data and any(len(v) > 0 for v in self.episode_buffer.values()):
            log_data = {}

            # Log aggregated data from buffer
            if self.episode_buffer['rewards']:
                log_data["train/mean_episode_reward"] = np.mean(self.episode_buffer['rewards'])
                log_data["train/mean_episode_length"] = np.mean(self.episode_buffer['lengths'])

            if self.episode_buffer['target_ids']:
                log_data["train/mean_target_ids"] = np.mean(self.episode_buffer['target_ids'])

            if self.episode_buffer['detections']:
                log_data["train/mean_detections"] = np.mean(self.episode_buffer['detections'])

            # Log the aggregated data
            if log_data:
                self.run.log(log_data, step=self.num_timesteps // self.model.get_env().num_envs)

            # Clear the buffer after logging
            self.episode_buffer = {
                'rewards': [],
                'lengths': [],
                'target_ids': [],
                'detections': []
            }

        # Log training metrics less frequently (e.g., every 10 steps)
        should_log_training_metrics = self.num_timesteps % (self.log_freq * 2) == 0

        if should_log_training_metrics:
            training_metrics = {}

            if hasattr(self.model, '_n_updates'):
                training_metrics["train/n_updates"] = self.model._n_updates

            if hasattr(self.logger, 'name_to_value'):
                logger_dict = self.logger.name_to_value

                # Log the core training metrics
                metrics_to_log = [
                    "train/approx_kl", "train/entropy_loss", "train/explained_variance",
                    "train/n_updates", "train/policy_gradient_loss", "train/value_loss",
                    "train/clip_fraction", "train/clip_range", "train/learning_rate"
                ]

                for metric in metrics_to_log:
                    if metric in logger_dict:
                        training_metrics[metric] = logger_dict[metric]

            # Log training metrics if any are available
            if training_metrics:
                self.run.log(training_metrics, step=self.num_timesteps // self.model.get_env().num_envs)

        # Evaluation logic remains the same (already infrequent)
        if self.eval_env is not None and self.num_timesteps % self.eval_freq == 0:
            print(f'\n#################################################\nEVALUATING (step: {self.num_timesteps})')

            if hasattr(self.model.get_env(), 'obs_rms'):
                self.eval_env.obs_rms = self.model.get_env().obs_rms
                self.eval_env.ret_rms = self.model.get_env().ret_rms

            # ... rest of evaluation code remains unchanged ...
            target_ids_list = []
            target_ids_per_step_list = []
            mean_reward, std_reward = 0, 0
            eval_lengths = []

            obs = self.eval_env.reset() # We call reset() at the beginning (vec envs reset automatically after this)
            for i in range(self.n_eval_episodes):
                #obs = self.eval_env.reset()
                # try:
                #     #episode_counter = self.eval_env.get_attr('episode_counter')[0]
                #     episode_counter = self.eval_env.get_wrapper_attr('episode_counter')
                #     print(f'Reset to eval env episode {episode_counter}')
                # except (AttributeError, IndexError):
                #     print('Reset to eval env episode (counter unavailable)')

                done = False
                ep_reward = 0

                while not done:
                    action, other = self.model.predict(obs, deterministic=True)
                    obses, rewards, dones, infos = self.eval_env.step([action])
                    obs = obses[0]
                    reward = rewards[0]
                    info = infos[0]
                    done = dones[0]
                    ep_reward += reward

                if "target_ids" in info:
                    target_ids_list.append(info["target_ids"])

                mean_reward += ep_reward / self.n_eval_episodes
                eval_lengths.append(info["episode"]["l"])
                target_ids_per_step_list.append(info["target_ids"] / info["episode"]["l"])

            std_reward = np.std(target_ids_list) if target_ids_list else 0

            # Log evaluation results
            self.run.log({
                "eval/mean_reward": mean_reward,
                "eval/std_reward": std_reward,
                "eval/mean_target_ids": np.mean(target_ids_list) if target_ids_list else 0,
                "eval/mean_episode_length": np.mean(eval_lengths) if eval_lengths else 0,
                "eval/mean_target_ids_per_step": np.mean(target_ids_per_step_list) if target_ids_per_step_list else 0,
                "curriculum/difficulty_level": self.current_difficulty
            }, step=self.num_timesteps)

            print(f'\nEVAL LOGGED (mean reward {mean_reward}, std {round(std_reward, 2)}, '
                  f'mean target_ids: {np.mean(target_ids_list) if target_ids_list else 0}')

            # Curriculum logic remains unchanged
            if self.use_curriculum:
                print('CURRICULUM: Checking if we should increase difficulty')
                avg_target_ids = np.mean(target_ids_list) if target_ids_list else 0
                avg_eval_len = np.mean(eval_lengths) if eval_lengths else 0

                if self.model.get_env().get_attr("difficulty")[0] == 0:
                    if avg_target_ids >= self.min_target_ids_to_advance:
                        self.above_threshold_counter += 1
                    else:
                        self.above_threshold_counter = 0
                else:
                    if avg_target_ids >= self.min_target_ids_to_advance and avg_eval_len <= self.max_ep_len_to_advance:
                        self.above_threshold_counter += 1
                    else:
                        self.above_threshold_counter = 0

                if self.above_threshold_counter >= 5:
                    self.above_threshold_counter = 0
                    self.current_difficulty += 1
                    print(f'CURRICULUM: Increasing difficulty to level {self.current_difficulty}')

                    self.model.get_env().env_method("set_difficulty", self.current_difficulty)
                    try:
                        self.eval_env.env_method("set_difficulty", self.current_difficulty)
                    except Exception as e:
                        print(f"Failed to set difficulty on eval env: {e}")

                    self.run.log({"curriculum/difficulty_level": self.current_difficulty}, step=self.num_timesteps)
                else:
                    print(f'CURRICULUM: Maintaining difficulty at level {self.current_difficulty} '
                          f'(avg target_ids: {avg_target_ids} < threshold: {self.min_target_ids_to_advance})')

            print('#################################################\n\nReturning to training...')

        return True


def make_env(env_config, rank, seed, run_name='no_name'):
    """
    Callable function that creates a MAISR environment. This function is passed to the vectorized environment
    instantiation in train()
    """

    def _init():
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
        env_config,
        n_envs,
        project_name,
        save_dir="./trained_models/",
        load_path=None,
        log_dir="./logs/",
        machine_name='machine',
        save_model=True
):
    """
    Main training pipeline. Does the following:
    1. Loads training and env config from env_config filename
    2. Sets up WandB for training logging
    3. Instantiates environments (multiprocessed vectorized environments for traning, and 1 env for eval)
    4. Instantiates training callbacks (WandB logging, checkpointing)
    5. Sets up Stable-Baselines3 PPO training
    6. Loads a prior checkpoint if provided
    7. Runs PPO training and saves checkpoints and the final model
    """

    # Generate run name (To be consistent between WandB, model saving, and action history plots)
    run_name = generate_run_name(env_config)

    os.makedirs(f"{save_dir}/{run_name}", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f'./logs/action_histories/{run_name}', exist_ok=True)

    run = wandb.init(
        project=project_name,
        name=f'{machine_name}_{n_envs}envs_' + run_name,
        config=env_config,
        sync_tensorboard=True,
        monitor_gym=True,
    )
    run.log_code(".")

    
    ################################################ Initialize envs ################################################
    if n_envs > 1:
        print(f"Training with {n_envs} environments in parallel")

        env_fns = [make_env(env_config, i, env_config['seed'] + i, run_name=run_name) for i in range(n_envs)]
        env = SubprocVecEnv(env_fns)
        env = VecMonitor(env, filename=os.path.join(log_dir, 'vecmonitor'))
        env = VecNormalize(env)

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
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(eval_env, norm_reward=False, training=False)
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms
    print('Envs created')

    ################################################# Setup callbacks #################################################
    checkpoint_callback = CheckpointCallback(
        save_freq=env_config['save_freq'] // n_envs,
        save_path=f"{save_dir}/{run_name}",
        name_prefix=f"maisr_checkpoint_{run_name}",
        save_replay_buffer=True, save_vecnormalize=True,
    )
    wandb_callback = WandbCallback(gradient_save_freq=50,
                                   model_save_path=f"{save_dir}/wandb/{run.id}" if save_model else None,
                                   verbose=1)
    enhanced_wandb_callback = EnhancedWandbCallback(env_config, eval_env=eval_env, run=run, log_freq=20)

    print('Callbacks created')

    ################################################# Setup model #################################################

    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(
            pi=[env_config['policy_network_size'], env_config['policy_network_size']],
            vf=[env_config['value_network_size'], env_config['value_network_size']]
        ))

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=2,
        tensorboard_log=f"logs/runs/{run.id}",
        batch_size=env_config['batch_size'],
        n_steps=env_config['ppo_update_steps'],
        learning_rate=env_config['lr'],
        seed=env_config['seed'],
        device='cpu',
        gamma=env_config['gamma'],
        ent_coef=env_config['entropy_regularization']
    )
    print('Model instantiated')
    print(model.policy)

    ################################################# Load checkpoint ##################################################
    if load_path:
        print(f'LOADING FROM {load_path}')
        model = model.__class__.load(load_path, env=env)
    else:
        print('Training new model')

    print('##################################### Beginning agent training... #######################################\n')

    # Log initial difficulty
    run.log({"curriculum/difficulty_level": 0}, step=0)
    print(f'Starting with difficulty level {0}')

    model.learn(
        total_timesteps=int(env_config['num_timesteps']),
        callback=[checkpoint_callback, wandb_callback, enhanced_wandb_callback],
        reset_num_timesteps=True if load_path else False  # TODO check this
    )

    print('########################################## TRAINING COMPLETE ############################################\n')
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

    ############## ---- SETTINGS ---- ##############
    # Specify a checkpoint to load
    load_path = None  # './trained_models/6envs_obs-relative_act-continuous-normalized_lr-5e-05_bs-128_g-0.99_fs-1_ppoupdates-2048_curriculum-Truerew-wtn-0.02_rew-prox-0.005_rew-timepenalty--0.0_0516_1425/maisr_checkpoint_6envs_obs-relative_act-continuous-normalized_lr-5e-05_bs-128_g-0.99_fs-1_ppoupdates-2048_curriculum-Truerew-wtn-0.02_rew-prox-0.005_rew-timepenalty--0.0_0516_1425_156672_steps'
    config_filename = 'config_files/june9b.json'
    ###############################################

    # Get machine name to add to run name
    #print(f'machine is {socket.gethostname()}')
    machine_name = 'home' if socket.gethostname() == 'DESKTOP-3Q1FTUP' else 'lab_pc' if socket.gethostname() == 'isye-ae-2023pc3' else 'pace'
    project_name = 'maisr-rl' if machine_name in ['home', 'lab_pc'] else 'maisr-rl-pace'
    print(f'Setting machine_name to {machine_name}. Using project {project_name}')

    print(f'\n############################ STARTING TRAINING RUN ############################')

    all_configs, param_names = load_env_config_with_sweeps(config_filename)
    #print(f"Found {len(all_configs)} configurations to run (sweeping over {param_names})")

    for i, env_config in enumerate(all_configs):
        print(f'\n--- Starting training run {i + 1}/{len(all_configs)} ---')

        env_config['n_envs'] = multiprocessing.cpu_count()
        env_config['config_filename'] = config_filename
        final_run_name = generate_run_name(env_config) + f'{"".join('_'+str(name)+'-'+str(env_config[name]) for name in param_names)}'
        print(f"Running with config: {final_run_name}")

        train(
            env_config,
            n_envs=multiprocessing.cpu_count(),
            load_path=load_path,
            machine_name=machine_name,
            project_name=project_name
        )
        print(f"✓ Completed training run {i + 1}/{len(all_configs)}")