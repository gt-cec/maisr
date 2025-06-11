import socket
import multiprocessing
import argparse

from train_sb3 import train, generate_run_name
from utility.config_management import load_env_config_with_sweeps
from utility.data_logging import load_env_config


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description='Run training with specified config file')
    # parser.add_argument('--config', '-c',
    #                     type=str,
    #                     default='',
    #                     help='Path to the config file (default: config_files/june11a.json)')
    #
    # args = parser.parse_args()
    # config_filename = args.config

    ############## ---- SETTINGS ---- ##############
    # Specify a checkpoint to load
    load_path = None  # './trained_models/6envs_obs-relative_act-continuous-normalized_lr-5e-05_bs-128_g-0.99_fs-1_ppoupdates-2048_curriculum-Truerew-wtn-0.02_rew-prox-0.005_rew-timepenalty--0.0_0516_1425/maisr_checkpoint_6envs_obs-relative_act-continuous-normalized_lr-5e-05_bs-128_g-0.99_fs-1_ppoupdates-2048_curriculum-Truerew-wtn-0.02_rew-prox-0.005_rew-timepenalty--0.0_0516_1425_156672_steps'
    config_filename = 'configs/june11a.json'
    ###############################################

    # Get machine name to add to run name
    machine_name = 'home' if socket.gethostname() == 'DESKTOP-3Q1FTUP' else 'lab_pc' if socket.gethostname() == 'isye-ae-2023pc3' else 'pace'
    project_name = 'maisr-rl' if machine_name in ['home', 'lab_pc'] else 'maisr-rl-pace'
    print(f'Setting machine_name to {machine_name}. Using project {project_name}')

    print(f'\n############################ STARTING TRAINING ############################')

    config = load_env_config(config_filename)
    config['n_envs'] = multiprocessing.cpu_count()
    config['config_filename'] = config_filename


    for levels_per_lesson in [{"0": 1, "1": 1, "2": 1}, {"0": 3, "1": 3, "2": 3}]:
        config["n_eval_episodes"] = levels_per_lesson["0"]

        print(f'\n--- Starting training run  ---')

        train(
            config,
            n_envs=multiprocessing.cpu_count(),
            load_path=load_path,
            machine_name=machine_name,
            project_name=project_name
        )
        print(f"✓ Completed training run")