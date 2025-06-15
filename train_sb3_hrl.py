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
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from env_multi_new import MAISREnvVec
from training_wrappers.localsearch_training_wrapper import MaisrLocalSearchWrapper
from training_wrappers.modeselector_training_wrapper import MaisrModeSelectorWrapper

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

        self.use_curriculum = env_config['use_curriculum']
        self.min_target_ids_to_advance = env_config['min_target_ids_to_advance']
        self.max_ep_len_to_advance = 130
        self.max_difficulty = env_config['max_difficulty']
        self.cl_lr_decrease = env_config['cl_lr_decrease'] # Divide learning rate by this every time we increase difficulty

        self.current_difficulty = 0
        self.above_threshold_counter = 0

        # Buffer for accumulating data between log events
        self.episode_buffer = {
            'rewards': [],
            'lengths': [],
            'target_ids': [],
            'detections': []
        }

        # Early stopping based on performance degradation
        self.best_eval_performance = -np.inf
        self.performance_crash_counter = 0
        self.performance_crash_threshold = 20  # Number of consecutive poor evals before stopping
        self.performance_crash_ratio = 0.4  # Performance must drop below 50% of best
        self.should_stop_training = False

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

                    if "target_ids" in info: self.episode_buffer['target_ids'].append(info["target_ids"])
                    if "detections" in info: self.episode_buffer['detections'].append(info["detections"])

        # Only log episode data at the specified frequency
        if should_log_episode_data and any(len(v) > 0 for v in self.episode_buffer.values()):
            log_data = {}

            # Log aggregated data from buffer
            if self.episode_buffer['rewards']:
                log_data["train/mean_episode_reward"] = np.mean(self.episode_buffer['rewards'])
                log_data["train/mean_episode_length"] = np.mean(self.episode_buffer['lengths'])

            if self.episode_buffer['target_ids']: log_data["train/mean_target_ids"] = np.mean(self.episode_buffer['target_ids'])
            if self.episode_buffer['detections']: log_data["train/mean_detections"] = np.mean(self.episode_buffer['detections'])

            if log_data: # Log the aggregated data
                self.run.log(log_data, step=self.num_timesteps // self.model.get_env().num_envs)

            # Clear the buffer after logging
            self.episode_buffer = {'rewards': [], 'lengths': [], 'target_ids': [], 'detections': []}

        # Log training metrics less frequently (e.g., every 10 steps)
        should_log_training_metrics = self.num_timesteps % (self.log_freq * 2) == 0

        if should_log_training_metrics:
            training_metrics = {}

            if hasattr(self.model, '_n_updates'): training_metrics["train/n_updates"] = self.model._n_updates

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

                done = False
                ep_reward = 0

                while not done:
                    action, other = self.model.predict(obs, deterministic=True)
                    #print(f'agent action: {action} (type: {type(action)})')
                    obses, rewards, dones, infos = self.eval_env.step([action])
                    obs = obses[0]
                    reward = rewards[0]
                    info = infos[0]
                    done = dones[0]
                    ep_reward += reward

                if "target_ids" in info: target_ids_list.append(info["target_ids"])

                mean_reward += ep_reward / self.n_eval_episodes
                eval_lengths.append(info["episode"]["l"])
                target_ids_per_step_list.append(info["target_ids"] / info["episode"]["l"])

            std_reward = np.std(target_ids_list) if target_ids_list else 0

            # Log evaluation results
            eval_metrics = {
                "eval/mean_reward": mean_reward,
                "misc/std_reward": std_reward,
                "eval/mean_target_ids": np.mean(target_ids_list) if target_ids_list else 0,
                "eval/mean_episode_length": np.mean(eval_lengths) if eval_lengths else 0,
                "eval/mean_target_ids_per_step": np.mean(target_ids_per_step_list) if target_ids_per_step_list else 0,
                "curriculum/difficulty_level": self.current_difficulty
            }

            #################################### Early stopping ####################################
            # Check for performance crash
            current_performance = np.mean(target_ids_list) if target_ids_list else 0
            if current_performance > self.best_eval_performance:
                self.best_eval_performance = current_performance
                self.performance_crash_counter = 0
                eval_metrics["early_stopping/best_performance"] = self.best_eval_performance
                print(f'NEW BEST PERFORMANCE: {self.best_eval_performance:.3f}')

            performance_threshold = self.best_eval_performance * self.performance_crash_ratio
            if current_performance < performance_threshold and self.best_eval_performance > 0:
                self.performance_crash_counter += 1
                print(f'PERFORMANCE CRASH WARNING: {current_performance:.3f} < {performance_threshold:.3f} '
                      f'({self.performance_crash_counter}/{self.performance_crash_threshold})')
            else:
                self.performance_crash_counter = 0

            eval_metrics["early_stopping/performance_crash_counter"] = self.performance_crash_counter
            eval_metrics["early_stopping/performance_threshold"] = performance_threshold

            # Check if we should stop training
            if self.performance_crash_counter >= self.performance_crash_threshold:
                self.should_stop_training = True
                print(f'\n{"=" * 80}')
                print(f'EARLY STOPPING TRIGGERED!')
                print(f'Performance has been below {self.performance_crash_ratio * 100}% of best for {self.performance_crash_counter} consecutive evaluations')
                print(f'Best performance: {self.best_eval_performance:.3f}')
                print(f'Current performance: {current_performance:.3f}')
                print(f'Threshold: {performance_threshold:.3f}')
                print(f'{"=" * 80}\n')

            self.run.log(eval_metrics, step=self.num_timesteps)

            print(f'\nEVAL LOGGED (mean reward {mean_reward}, std {round(std_reward, 2)}, 'f'mean target_ids: {np.mean(target_ids_list) if target_ids_list else 0}')


            #################################### Curriculum learning ####################################
            if self.use_curriculum:
                print('CURRICULUM: Checking if we should increase difficulty')
                avg_target_ids = np.mean(target_ids_list) if target_ids_list else 0
                avg_eval_len = np.mean(eval_lengths) if eval_lengths else 0

                # if self.model.get_env().get_attr("difficulty")[0] == 0:
                #     if avg_target_ids >= self.min_target_ids_to_advance:
                #         self.above_threshold_counter += 1
                #     else:
                #         self.above_threshold_counter = 0
                #else:
                if avg_target_ids >= self.min_target_ids_to_advance and avg_eval_len <= self.max_ep_len_to_advance:
                    self.above_threshold_counter += 1
                else:
                    self.above_threshold_counter = 0

                if self.above_threshold_counter >= 5 and self.current_difficulty < self.max_difficulty:
                    self.above_threshold_counter = 0
                    self.current_difficulty += 1
                    print(f'CURRICULUM: Increasing difficulty to level {self.current_difficulty}')

                    self.model.get_env().env_method("set_difficulty", self.current_difficulty)
                    try: self.eval_env.env_method("set_difficulty", self.current_difficulty)
                    except Exception as e: print(f"Failed to set difficulty on eval env: {e}")

                    print(f'CURRICULUM: Resetting performance tracking due to difficulty increase')
                    try: self.best_eval_performance = current_performance  # Reset best to current performance
                    except: self.best_eval_performance = -np.inf
                    self.performance_crash_counter = 0  # Reset crash counter

                    self.run.log({"curriculum/difficulty_level": self.current_difficulty}, step=self.num_timesteps)

                    # Decrease LR
                    print(f'Model lr reduced from {self.model.policy.optimizer.param_groups[0]['lr']} to {self.model.policy.optimizer.param_groups[0]['lr']/self.cl_lr_decrease}')
                    self.model.policy.optimizer.param_groups[0]['lr'] = self.model.policy.optimizer.param_groups[0]['lr']/self.cl_lr_decrease

                else:
                    print(f'CURRICULUM: Maintaining difficulty at level {self.current_difficulty} '
                          f'(avg target_ids: {avg_target_ids} < threshold: {self.min_target_ids_to_advance})')

            print('#################################################\n\nReturning to training... \n')

        if self.should_stop_training:
            return False
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


def train_hrl(
        env_config,
        n_envs,
        project_name,
        policy_to_train,
        use_normalize,
        run_name='norunname',
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

    print(f'Setting machine_name to {machine_name}. Using project {project_name}')



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
    if policy_to_train == 'local_search':
        if n_envs > 1:
            print(f"Training with {n_envs} environments in parallel")

            def make_wrapped_env(env_config, rank, seed, run_name='no_name'):
                def _init():
                    # Create base environment
                    base_env = MAISREnvVec(
                        config=env_config,
                        render_mode='headless',
                        run_name=run_name,
                        tag=f'train_mp{rank}',
                        seed=seed + rank,
                    )
                    # Wrap with local search wrapper
                    wrapped_env = MaisrLocalSearchWrapper(base_env)
                    wrapped_env = Monitor(wrapped_env)
                    wrapped_env.reset()
                    return wrapped_env

                return _init

            env_fns = [make_wrapped_env(env_config, i, env_config['seed'] + i, run_name=run_name) for i in
                       range(n_envs)]
            env = SubprocVecEnv(env_fns)
            env = VecMonitor(env, filename=os.path.join(log_dir, 'vecmonitor'))
            if use_normalize:
                env = VecNormalize(env)

        else:
            # Single environment setup
            base_env = MAISREnvVec(
                env_config,
                None,
                render_mode='headless',
                tag='train',
                run_name=run_name,
            )
            env = MaisrLocalSearchWrapper(base_env)
            env = Monitor(env)

        # Create eval environment with wrapper
        base_eval_env = MAISREnvVec(
            env_config,
            None,
            render_mode='headless',
            tag='eval',
            run_name=run_name,
        )
        eval_env = MaisrLocalSearchWrapper(base_eval_env)
        eval_env = Monitor(eval_env)
        eval_env = DummyVecEnv([lambda: eval_env])
        if use_normalize:
            eval_env = VecNormalize(eval_env, norm_reward=False, training=False)
            eval_env.obs_rms = env.obs_rms
            eval_env.ret_rms = env.ret_rms

    else:
        raise ValueError('Invalid policy to train')

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

    if env_config['algo'] == 'PPO':
        model = PPO(
            "CnnPolicy" if env_config['obs_type'] == 'pixel' else "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=2,
            tensorboard_log=f"logs/tb_runs/{run.id}",
            batch_size=env_config['batch_size'],
            n_steps=env_config['ppo_update_steps'],
            learning_rate=env_config['lr'],
            seed=env_config['seed'],
            device='cpu',
            gamma=env_config['gamma'],
            ent_coef=env_config['entropy_regularization'],
            clip_range=env_config['clip_range']
        )
    elif env_config['algo'] == 'SAC':
        model = SAC(
            "CnnPolicy" if env_config['obs_type'] == 'pixel' else "MlpPolicy",
            env,
            #policy_kwargs=policy_kwargs,
            verbose=2,
            tensorboard_log=f"logs/tb_runs/{run.id}",
            batch_size=env_config['batch_size'],
            learning_rate=env_config['lr'],
            seed=env_config['seed'],
            device='cpu',
            gamma=env_config['gamma'],
            ent_coef=env_config['entropy_regularization'],
        )
    else: raise ValueError('Unsupported algo')

    print('Model instantiated')
    print(model.policy)

    ################################################# Load checkpoint ##################################################
    if load_path:
        print(f'LOADING FROM {load_path}')
        model = model.__class__.load(load_path, env=env)
    else: print('Training new model')

    print('##################################### Beginning agent training... #######################################\n')

    # Log initial difficulty
    run.log({"curriculum/difficulty_level": 0}, step=0)
    print(f'Starting with difficulty level {0}')

    model.learn(
        total_timesteps=int(env_config['num_timesteps']),
        callback=[checkpoint_callback, wandb_callback, enhanced_wandb_callback],
        reset_num_timesteps=True if load_path else False  # TODO check this
    )

    # Save normalization stats for deployment
    stats = {
        'obs_mean': env.obs_rms.mean,
        'obs_var': env.obs_rms.var,
        'obs_count': env.obs_rms.count,
        'ret_mean': env.ret_rms.mean,
        'ret_var': env.ret_rms.var,
    }
    np.save(f"trained_models/{run_name}local_search_norm_stats.npy", stats)
    env.save(f"trained_models/{run_name}local_search_vecnormalize.pkl")
    print("Training Normalization Stats:")
    print(f"Obs mean: {env.obs_rms.mean}")
    print(f"Obs std: {np.sqrt(env.obs_rms.var + 1e-8)}")
    print(f"Obs count: {env.obs_rms.count}")

    print('########################################## TRAINING COMPLETE ############################################\n')
    env.close()
    eval_env.close()

    # Save the final model
    final_model_path = os.path.join(save_dir, f"{run_name}_maisr_trained_model")
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
    load_path = None  # './trained_models/6envs_obs-relative_act-continuous-normalized_lr-5e-05_bs-128_g-0.99_fs-1_ppoupdates-2048_curriculum-Truerew-wtn-0.02_rew-prox-0.005_rew-timepenalty--0.0_0516_1425/maisr_checkpoint_6envs_obs-relative_act-continuous-normalized_lr-5e-05_bs-128_g-0.99_fs-1_ppoupdates-2048_curriculum-Truerew-wtn-0.02_rew-prox-0.005_rew-timepenalty--0.0_0516_1425_156672_steps'
    config_filename = 'configs/june15.json'

    ################################################

    print(f'\n############################ STARTING TRAINING ############################')
    config = load_env_config(config_filename)
    config['n_envs'] = multiprocessing.cpu_count()
    config['config_filename'] = config_filename
    config['policy_to_train'] = 'local-search'

    for num_timesteps in [5e5, 2e6]:
        for inside_threat_penalty in [0.03, 0.1, 0.15, 0.25]:
            config['num_timesteps'] = num_timesteps
            config['inside_threat_penalty'] = inside_threat_penalty

            # Generate run name (To be consistent between WandB, model saving, and action history plots)
            run_name = f'local_search_{num_timesteps}timesteps_{inside_threat_penalty}threatpenalty_'+generate_run_name(config)

            print(f'\n--- Starting training run  ---')
            train_hrl(
                config,
                run_name=run_name,
                use_normalize=True,
                n_envs=multiprocessing.cpu_count(),
                policy_to_train = 'local_search',
                load_path=load_path,
                machine_name='localsearch_home'+('home' if socket.gethostname() == 'DESKTOP-3Q1FTUP' else 'lab_pc' if socket.gethostname() == 'isye-ae-2023pc3' else 'pace'),
                project_name='maisr-rl-lab', #'maisr-rl' if socket.gethostname() in ['DESKTOP-3Q1FTUP', 'isye-ae-2023pc3'] else 'maisr-rl-pace'
                save_model = False,
            )
            print(f"✓ Completed training run")