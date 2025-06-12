import socket
import multiprocessing
import argparse

from train_sb3_Sequence import train, generate_run_name
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
    config_filename = 'configs/sequence_june11.json'

    print(f'\n############################ STARTING TRAINING ############################')
    config = load_env_config(config_filename)
    config['n_envs'] = multiprocessing.cpu_count()
    config['config_filename'] = config_filename


    for learning_rate in [0.001, 0.005, 0.0005]:
        for batch_size in [128, 256]:
            config['learning_rate'] = learning_rate
            config['batch_size'] = batch_size


            print(f'\n--- Starting training run  ---')

            train(
                config,
                n_envs=multiprocessing.cpu_count(),
                load_path=load_path,
                machine_name='home-sequence' if socket.gethostname() == 'DESKTOP-3Q1FTUP' else 'lab_pc' if socket.gethostname() == 'isye-ae-2023pc3' else 'pace',
                project_name='maisr-rl' if socket.gethostname() in ['DESKTOP-3Q1FTUP', 'isye-ae-2023pc3'] else 'maisr-rl-pace'
            )
            print(f"✓ Completed training run")