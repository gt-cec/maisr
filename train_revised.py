from stable_baselines3.common.policies import ActorCriticPolicy

import_complete = False
while not import_complete:
    try:
        import copy
        import itertools
        import ctypes
        import json
        import warnings
        import random
        import os, glob
        import pygame
        from PIL import Image
        from datetime import datetime

        from utility.localsearch_training_wrapper import MaisrLocalSearchWrapper
        import gymnasium as gym
        import numpy as np
        import multiprocessing
        import socket
        import torch
        import argparse
        import time

        import wandb
        from wandb.integration.sb3 import WandbCallback
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.callbacks import BaseCallback

        from base_env import MaisrEnv
        from utility.league_management import TeammateManager, ConfigurableHeuristicTeammate, SubPolicy, LocalSearch, ChangeRegions, GoToNearestThreat, TargetSearchLocalTSP, RecordedTrajectoryTeammate
        from utility.config_management import load_env_config
        from utility.callbacks import LeagueTypeTransitionCallback, PrintObsCallback, EnhancedWandbCallback
        import_complete = True
    except Exception as e:
        print(f'Import exception: {e}')
        import_complete = False



def get_latest_checkpoint_and_vecnorm(seed: int, note_prefix: str = "pretrain") -> tuple[str, str]:
    """
    Automatically find the latest checkpoint .zip and VecNormalize .pkl for a given seed.

    Args:
        seed (int): The training seed used in the run folder name.
        note_prefix (str): The prefix used in the run folder name, e.g., "pretrainH".

    Returns:
        (load_path, vecnorm_path): Tuple of strings with the latest checkpoint and vecnormalize file paths.
    """
    # Pattern for run folder: <note>_MMDD_HHMM_seed<seed>/checkpoints
    pattern = f"outputs/{note_prefix}_*_seed{seed}/checkpoints"
    checkpoint_dirs = glob.glob(pattern)

    if not checkpoint_dirs:
        raise FileNotFoundError(f"[AutoLoad] No checkpoint directories found for seed {seed} using pattern {pattern}")

    # Use the most recently modified directory if multiple matches
    latest_dir = max(checkpoint_dirs, key=os.path.getmtime)

    # Get all checkpoint zips and vecnorm pkls
    checkpoint_zips = glob.glob(os.path.join(latest_dir, "*_checkpoint_*_steps.zip"))
    vecnorm_pkls = glob.glob(os.path.join(latest_dir, "*_checkpoint_vecnormalize_*_steps.pkl"))

    if not checkpoint_zips or not vecnorm_pkls:
        raise FileNotFoundError(f"[AutoLoad] No valid checkpoints or vecnormalize files found in {latest_dir}")

    # Helper to extract step number from file names
    def extract_step(path: str, vecnorm: bool = False) -> int:
        base = os.path.basename(path)
        if vecnorm:
            # <run>_checkpoint_vecnormalize_<steps>_steps.pkl
            step_str = base.split("_vecnormalize_")[-1].replace("_steps.pkl", "")
        else:
            # <run>_checkpoint_<steps>_steps.zip
            step_str = base.split("_checkpoint_")[-1].replace("_steps.zip", "")
        return int(step_str)

    # Pick the files with the highest step count
    latest_zip = max(checkpoint_zips, key=lambda p: extract_step(p, vecnorm=False))
    latest_pkl = max(vecnorm_pkls, key=lambda p: extract_step(p, vecnorm=True))

    print(f"[AutoLoad] Using latest checkpoint: {latest_zip}")
    print(f"[AutoLoad] Using latest vecnorm stats: {latest_pkl}")

    return latest_zip, latest_pkl



# def make_env(env_config, rank, seed, run_name='no_name', save_episode_plots = True):
#     """
#     Callable function that creates a MAISR environment. This function is passed to the vectorized environment
#     instantiation in train()
#     """
#     def _init():
#         env = MAISREnvVec(
#             config=env_config,
#             render_mode='headless',
#             run_name=run_name,
#             tag=f'train_mp{rank}',
#             seed=seed + rank,
#             save_episode_plots = save_episode_plots
#         )
#         env = Monitor(env)
#         env.reset()
#         return env
#     return _init


def setup_teammate_pool(league_type, balance_method, selfplay_checkpoint_dir, pretrained_teammate_dir, overfit_test, fcp_ratio=1.0):
    """Setup teammate manager with specified league type"""


    teammate_manager = TeammateManager(
        league_type,
        balance_method,
        #subpolicies=subpolicies,
        selfplay_checkpoint_dir=selfplay_checkpoint_dir,
        pretrained_teammate_dir=pretrained_teammate_dir,
        overfit_test=overfit_test
    )

    #print(f"        Teammate manager setup with league_type: {league_type}")
    return teammate_manager

def train_generic(
        env_config,
        n_envs,
        project_name,
        use_normalize,
        use_teammate_manager,
        train_type, # "mode_selector" or "monolith"
        run_name='norunname',
        load_path=None,
        vecnorm_load_path=None,
        render=False,
        machine_name='machine',
        save_model=True,
        save_checkpoints = False,
        overfit_test=None,
        save_episode_plots = True

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

    if vecnorm_load_path is None and load_path is not None:
        raise ValueError('Provided model path without vecnorm stats')

    print('\n[train_generic] Initializing...')
    print(f'        Setting machine_name = {machine_name} \n        WandB project = {project_name}')

    if render:
        pygame.display.init()
        pygame.font.init()
        clock = pygame.time.Clock()
        ctypes.windll.user32.SetProcessDPIAware()
        window_width, window_height = config['window_size'][0], config['window_size'][1]
        config['tick_rate'] = 30
        window = pygame.display.set_mode((window_width, window_height), flags=pygame.NOFRAME)
        pygame.display.set_caption("MAISR Human Interface")

    print('\nCreating output folders:')
    for subfolder in ['episode_plots','trained_models', 'checkpoints','vecnorm_stats','logs']:
        folder_name = f"outputs/{run_name}/{subfolder}"
        try:
            os.makedirs(folder_name, exist_ok=True)
        except:
            print(f'failed to create folder {subfolder}, retrying...')
            os.makedirs(folder_name, exist_ok=True)
        print(f'        {folder_name}')
    print('\n')

    init_successful = False
    while not init_successful:
        try:
            run = wandb.init(
                project=project_name,
                name=run_name+f'{machine_name}_{n_envs}envs',
                config=env_config,
                sync_tensorboard=True,
                monitor_gym=True,
            )
            init_successful = True
        except:
            print('         WandB init failed, retrying')
            init_successful = False
        if init_successful:
            print(f'        WandB init successful')
            break

    run.log_code(".")

    ################################################ Initialize envs ################################################

    if env_config['num_aircraft'] > 1 and use_teammate_manager:
        teammate_manager = setup_teammate_pool(
            league_type=env_config['league_type'],
            balance_method = env_config['balance_method'],
            selfplay_checkpoint_dir=f"outputs/{run_name}/checkpoints",
            pretrained_teammate_dir=f'trained_models/pretrained_teammates',
            overfit_test=overfit_test,
        )
        print('        Instantiated teammate manager')
    else:
        teammate_manager = None
        print('        Not using a teammate manager')

    print(f"Training with {n_envs} environments in parallel\n")

    def make_wrapped_env(env_config, rank, seed, run_name='no_name', render=False, save_episode_plots=True):
        def _init():

            if rank != 0:
                import sys
                import os
                sys.stdout = open(os.devnull, 'w')

            base_env = MaisrEnv( # Create base environment
                config=env_config,
                render_mode='headless',
                run_name=run_name,
                tag=f'train_mp{rank}',
                seed=seed + rank,
                save_episode_plots = save_episode_plots
            )

            #localsearch_model = PPO.load('trained_models/local_search_2000000.0timesteps_0.1threatpenalty_0615_1541_6envs_maisr_trained_model.zip')
            local_search_policy = LocalSearch()
            go_to_highvalue_policy = GoToNearestThreat(model_path=None)
            change_region_subpolicy = ChangeRegions(model_path=None)
            evade_policy = None

            wrapped_env = MaisrLocalSearchWrapper(
                base_env,
                env_config['obs_noise_std_localsearch'],
                local_search_policy,
                go_to_highvalue_policy,
                change_region_subpolicy,
                evade_policy,
                teammate_manager=teammate_manager
            )

            wrapped_env = Monitor(wrapped_env)
            wrapped_env.reset()
            return wrapped_env

        return _init

    # Instantiate main env
    env_fns = [make_wrapped_env(env_config, i, env_config['seed'] + i, run_name=run_name, save_episode_plots=save_episode_plots) for i in range(n_envs)]
    if n_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    # SB3 wrappers for main env
    env = VecMonitor(env, filename=f'outputs/{run_name}/logs/{run_name}vecmonitor')

    if use_normalize:
        if vecnorm_load_path is not None:
            env = VecNormalize.load(vecnorm_load_path, venv=env)
            env.training = True
            env.norm_reward = True
        else:
            env = VecNormalize(env)
            env.training = True
            env.norm_reward = True


    # Create and wrap eval environment
    base_eval_env = MaisrEnv(env_config, None, render_mode='headless', tag='eval', run_name=run_name, save_episode_plots=save_episode_plots)
    eval_env = MaisrLocalSearchWrapper(
        base_eval_env,
        env_config['obs_noise_std_localsearch'],
        LocalSearch(model_path=None),
        GoToNearestThreat(model_path=None),
        ChangeRegions(model_path=None),
        None,
        teammate_manager=teammate_manager)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])


    base_human_eval_env = MaisrEnv(env_config, None, render_mode='headless', tag='human_eval0', run_name=run_name)
    human_eval_env = MaisrLocalSearchWrapper(
        base_human_eval_env,
        env_config['obs_noise_std_localsearch'],
        LocalSearch(model_path=None),
        GoToNearestThreat(model_path=None),
        ChangeRegions(model_path=None),
        None,
        teammate_manager=teammate_manager)
    human_eval_env = Monitor(human_eval_env)
    human_eval_env = DummyVecEnv([lambda: human_eval_env])


    if use_normalize:
        if vecnorm_load_path is not None:
            eval_env = VecNormalize.load(vecnorm_load_path, venv=eval_env)
            eval_env.norm_reward = False
            eval_env.training = False
        else:
            eval_env = VecNormalize(eval_env, norm_reward=False, training=False)
        eval_env.obs_rms = env.obs_rms
        eval_env.ret_rms = env.ret_rms

    print('        Envs created')

    ################################################# Setup callbacks #################################################

    wandb_callback = WandbCallback(gradient_save_freq=50, verbose=1, model_save_path = None) #f"{save_dir}/{run_name}/wandb_modelsave" if save_model else None)

    enhanced_wandb_callback = EnhancedWandbCallback(
        env_config,
        eval_env=eval_env,
        human_eval_env=human_eval_env,
        run=run,
        log_freq=75,
        teammate_manager=teammate_manager
    )

    #printcallback = PrintObsCallback(verbose=1)

    callbacks = [wandb_callback, enhanced_wandb_callback]  # printcallback

    if env_config['switch_leagues']:
        league_transition_callback = LeagueTypeTransitionCallback(
            transition_timesteps=2e6,  # Transition after this many steps
            initial_league_type='selfplay',
            target_league_type='strategy_diverse',
            eval_env=eval_env,
            run=run,
            verbose=1
        )
        callbacks.append(league_transition_callback)

    if save_checkpoints:
        checkpoint_callback = CheckpointCallback(
            save_freq=env_config['save_freq'] // n_envs,
            save_path=f"outputs/{run_name}/checkpoints",
            name_prefix=f"{run_name}_checkpoint",
            save_replay_buffer=True, save_vecnormalize=True,
        )
        callbacks.append(checkpoint_callback)

    print('        Callbacks created')

    ################################################# Setup model #################################################

    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(
            pi=[env_config['network_size']] * env_config['network_numlayers'],
            vf=[env_config['network_size']] * env_config['network_numlayers']
        ))

    algo = env_config['algo']
    if algo == 'PPO':
        model = PPO(
            "CnnPolicy" if env_config['obs_type'] == 'pixel' else "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=2,
            tensorboard_log=f"outputs/logs/tb_runs/{run.id}",
            batch_size=env_config['batch_size'],
            n_steps=env_config['ppo_update_steps'],
            learning_rate=env_config['lr'],
            seed=env_config['seed'],
            device='cpu',
            gamma=env_config['gamma'],
            ent_coef=env_config['entropy_regularization'],
            clip_range=env_config['clip_range']
        )

    elif algo == 'trajedi':

        model = ActorCriticPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda _: 0.0,  # required but unused
            net_arch=policy_kwargs['net_arch'],  # TODO need to make sure this parses the pi and vf arch correectly
            activation_fn=policy_kwargs['activation_fn']
        )


    else:
        raise ValueError('Unsupported algo')

    print('        Model instantiated\n')
    print(model.policy)

    if teammate_manager is not None:
        teammate_manager.set_current_model(model)
        if use_normalize and hasattr(env, 'obs_rms'):
            teammate_manager.set_normalization_stats(env.obs_rms, env.ret_rms)

    ################################################# Load checkpoint ##################################################
    if load_path:
        print(f'        Checkpoint: Loading from {load_path}')
        model = PPO.load(load_path, env=env) # TODO make this work even if not using PPO
    else:
        print('        Checkpoint: None provided, training new model')

    # Log initial difficulty
    run.log({"curriculum/difficulty_level": 0}, step=0)

    # === Save initial checkpoint immediately ===
    if save_checkpoints:
        initial_checkpoint_path = f"outputs/{run_name}/checkpoints/{run_name}_checkpoint_0_steps.zip"
        vecnormalize_path =  f"outputs/{run_name}/checkpoints/{run_name}_checkpoint_vecnormalize_0_steps.pkl"
        model.save(initial_checkpoint_path)
        if isinstance(env, VecNormalize):
            env.save(vecnormalize_path)
        print(f"[Startup] Initial checkpoint saved to {initial_checkpoint_path}")

    teammate_manager._create_selfplay_teammate()
    teammate_manager.current_teammate.env = env

    print('\n\n###### Running model.learn... ######\n')

    # TODO make this work even if not using PPO
    if algo == 'PPO':
        model.learn(
            total_timesteps=int(env_config['num_timesteps']),
            callback=callbacks,
            reset_num_timesteps=False if load_path else True
        )

    # Save normalization stats for deployment
    stats = {
        'obs_mean': env.obs_rms.mean,
        'obs_var': env.obs_rms.var,
        'obs_count': env.obs_rms.count,
        'ret_mean': env.ret_rms.mean,
        'ret_var': env.ret_rms.var,
    }

    print("Training Normalization Stats:")
    print(f"Obs mean: {env.obs_rms.mean}")
    print(f"Obs std: {np.sqrt(env.obs_rms.var + 1e-8)}")
    print(f"Obs count: {env.obs_rms.count}")

    print('\n#########################################################################################################')
    print('########################################## TRAINING COMPLETE ############################################\n')
    print('#########################################################################################################')
    env.close()
    eval_env.close()

    # Save the final model
    if save_model:
        try:
            np.save(f"outputs/{run_name}/trained_models/{run_name}_norm_stats.npy", stats)
            env.save(f"outputs/{run_name}/vecnorm_stats/{run_name}local_search_vecnormalize.pkl")
            final_model_path = f'outputs/{run_name}/trained_models/{run_name}_model.zip'  #os.path.join(save_dir, f"{run_name}/{run_name}_model.zip")

            model.save(final_model_path)
            print(f"Training completed!\nFinal model saved to {final_model_path}")
        except:
            print('Failed to save model and norm stats')

    # Run a final evaluation
    print('Running final eval:')
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=env_config['n_eval_episodes'])
    print(f"\nFinal evaluation: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # Log final metrics to wandb
    run.log({"final/mean_reward": mean_reward, "final/std_reward": std_reward, })
    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='MAISR RL training script')
    parser.add_argument('--version', required=False, help='Which training version to run. You can define multiple versions later in this script')
    parser.add_argument('--seed', required=False, help='Seed to run')
    parser.add_argument('--testing', action='store_true', help='Set to testing mode. Simplifies some aspects of training for faster debugging')
    parser.add_argument('--league_type', required=False, help='Override league_type from config (e.g., selfplay, strategy_diverse, fcp, mixed50, etc.)')

    args = parser.parse_args()
    version = args.version if args.version else 'main'

    print(f'\n############################ STARTING TRAINING ############################')

    ############## ---- SETTINGS ---- ##############
    config_filename = 'configs/main_config.json'
    num_envs = 2 if args.testing else multiprocessing.cpu_count() # Use all CPU cores for multiprocessing, but only use 2 if args.testing (for faster init)
    train_type = 'monolith' # What type of agent to train. "monolith" for a single policy that chooses directional or target index control. "mode_selector" for a hybrid agent that chooses subpolicies (not currently implemented)
    project_name = 'maisr-rl-mixedtraining'#'insert_wandb_project_name'
    save_episode_plots = True # If True, episode plots are saved to outputs/{run name}/episode_plots
    verbosity = {
        'league_manager': False,
        'rl_teammates': False,

    }

    machine = socket.gethostname() # If you want to label you runs based on which machine they were trained on. Can also replace with a string or '' to skip
    config = load_env_config(config_filename)

    # Add parameters to the config so they're logged
    config['seed'] = int(args.seed) if args.seed else 99
    config['n_envs'] = num_envs
    config['config_filename'] = config_filename

    # Override league_type from CLI if provided
    if args.league_type:
        config['league_type'] = args.league_type
        print(f"[CLI Override] league_type set to: {args.league_type}")

    # An example of different training versions you can set up here. Specify using the --version arg.
    if version == 'main':
        run_prefix = '' + machine[0].upper()
        project_name = 'maisr-bc' # For WandB

        # If you want to sweep over multiple hyperparameter settings, you can define them here. These will override the values in the config.json
        # Note: All dictionary keys need to be enclosed in lists, even if they are single items.
        hyperparams = {
            "network_size": [64],
            "lr": [0.001],
            'entropy_regularization': [0.07],
            "teammate_reward_scale": [0.75],
            "teammate_active_at_start": [True],
            "league_type": ['strategy_diverse'], # ["baseline", "vanilla", "strategy_diverse", "selfplay", 'fcp','mixed50','mixed25','mixed75']
            "obs_noise": [0.01],
        }

        load_path = None # You can specify a policy .zip file here if you want to continue training from a prior run
        vecnorm_load_path = None # Specify the path to the above policy's vecnormalize .pkl file here.
        config['load_path'] = load_path

    # Shorthand names for hyperparameters to reduce length of run names
    param_shorthand = {
        'entropy_regularization': 'entreg',
        'teammate_reward_scale': 'trs',
        'team_spread_bonus_coeff': 'spreadbns',
        'num_observed_targets': 'obstgts',
        'num_observed_threats': 'obstrts',
        'obs_noise': 'noise',
        'network_size': 'modelsize',
        "observe_teammate_direction": "obs-tmt-dir",
        "force_specific_level": "frclvl",
        "entropy_decay_schedule": "entdcy",
        "use_stuck_detection": "stuckdtct",
        "lr": "lr",
        "potential_ratio":"potratio",
        "max_steps": 'mxstps',
        'threat_reward_scaling': 'thrtrwdscl',
        'shaping_coeff_earlyfinish': 'erlyfnsh',
        'entropy_decay_steps': 'entdcystps',
        'seed': 'seed',
        "gamma":"gamma",
        'league_type':'lgtype',
        'quick_id_shaping_coeff':'quick_id_cf',
        'use_dynamic_potential':'dynpotential',
        "use_teammate_priority_shaping":"tmtprishaping",
        "switch_leagues":"switch_lgs"
    }

    if args.testing:
        config["eval_freq"] = 50
        config['num_eval_episodes'] = 5
        config['save_freq'] = 500
        config['num_timesteps'] = 5e5
        project_name = 'maisr-tests'

    ################################################

    param_names = list(hyperparams.keys())
    param_values = list(hyperparams.values())
    print(param_values)

    # Loop through the hyperparameters selected in the version block above
    for param_combination in itertools.product(*param_values):
        current_params = dict(zip(param_names, param_combination))
        for param_name, param_value in current_params.items():
            config[param_name] = param_value

        # CLI overrides take precedence over hyperparameters
        if args.league_type:
            config['league_type'] = args.league_type

        param_strings = []
        for param_name, param_value in current_params.items():
            try: param_key = param_shorthand[param_name]
            except: param_key = param_name
            param_strings.append(f'{param_key}-{param_value}')

        temp_identifier = '_'.join([s for s in param_strings if not s.startswith('overfittest-')])
        #
        run_name = f'{run_prefix}_' + datetime.now().strftime("%m%d_%H%M") + f'_seed{str(args.seed)}' + temp_identifier


        print(f'\n--- Starting training run with params: {current_params} ---')
        train_generic(
            config,
            run_name=run_name,
            use_normalize=True,
            use_teammate_manager=True,
            train_type = train_type,
            render=False,
            n_envs=num_envs,
            load_path=load_path,
            vecnorm_load_path=vecnorm_load_path,
            machine_name=('home' if socket.gethostname() == 'DESKTOP-3Q1FTUP' else 'lab' if socket.gethostname() == 'isye-ae-2023pc3' else 'pace'),
            project_name=project_name,
            save_model = True,
            save_checkpoints = True,
            #save_dir=f'./outputs/trained_models/',
        )
        print(f"✓ Completed training run")