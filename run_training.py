import socket
import multiprocessing
from train_sb3 import train, generate_run_name
from utility.data_logging import load_env_config


if __name__ == "__main__":

    ############## ---- SETTINGS ---- ##############
    load_path = None  # './trained_models/6envs_obs-relative_act-continuous-normalized_lr-5e-05_bs-128_g-0.99_fs-1_ppoupdates-2048_curriculum-Truerew-wtn-0.02_rew-prox-0.005_rew-timepenalty--0.0_0516_1425/maisr_checkpoint_6envs_obs-relative_act-continuous-normalized_lr-5e-05_bs-128_g-0.99_fs-1_ppoupdates-2048_curriculum-Truerew-wtn-0.02_rew-prox-0.005_rew-timepenalty--0.0_0516_1425_156672_steps'
    config_filename = 'configs/sac_test.json'

    print(f'\n############################ STARTING TRAINING ############################')
    config = load_env_config(config_filename)
    config['n_envs'] = multiprocessing.cpu_count()
    config['config_filename'] = config_filename

    for learning_rate in [0.005]:
        for batch_size in [256]:
            for levels_per_lesson in [{"0": 3, "1": 3, "2":  3}]: # {"0": 1, "1": 1, "2":  1}
                config['learning_rate'] = learning_rate
                config['batch_size'] = batch_size
                config["levels_per_lesson"] = levels_per_lesson

                print(f'\n--- Starting training run  ---')
                train(
                    config,
                    n_envs=multiprocessing.cpu_count(),
                    load_path=load_path,
                    machine_name='home-sequence' if socket.gethostname() == 'DESKTOP-3Q1FTUP' else 'lab_pc' if socket.gethostname() == 'isye-ae-2023pc3' else 'pace',
                    project_name='maisr-rl-lab' #'maisr-rl' if socket.gethostname() in ['DESKTOP-3Q1FTUP', 'isye-ae-2023pc3'] else 'maisr-rl-pace'
                )
                print(f"✓ Completed training run")