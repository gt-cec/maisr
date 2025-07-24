import os
import shutil
import numpy as np
import glob
import argparse
import re

def collect_teammate_files(directory):
    input_dir = f'./outputs/{directory}'
    output_dir = f'./selfplayteammates/{directory}_selected'
    os.makedirs(output_dir, exist_ok=True)

    # 1. trained_models
    trained_models_dir = os.path.join(input_dir, 'trained_models')
    zip_files = glob.glob(os.path.join(trained_models_dir, '*.zip'))
    npy_files = glob.glob(os.path.join(trained_models_dir, '*.npy'))
    seed = None

    if zip_files:
        zip_file = zip_files[0]
        seed = get_seed_from_filename(zip_file)
        shutil.copy(zip_file, os.path.join(output_dir, f'teammate{seed}model.zip'))
    else:
        print("No model zip file found in trained_models.")

    if npy_files and seed:
        npy_file = npy_files[0]
        shutil.copy(npy_file, os.path.join(output_dir, f'teammate{seed}normstats.npy'))
    else:
        print("No normstats npy file found in trained_models or seed not set.")

    # 2. vecnorm_stats
    vecnorm_dir = os.path.join(input_dir, 'vecnorm_stats')
    pkl_files = glob.glob(os.path.join(vecnorm_dir, '*.pkl'))
    if pkl_files and seed:
        shutil.copy(pkl_files[0], os.path.join(output_dir, f'teammate{seed}vecnormstats.pkl'))
    else:
        print("No vecnorm .pkl file found or seed not set.")

    # 3. checkpoints
    checkpoints_dir = os.path.join(input_dir, 'checkpoints')
    checkpoint_models = sorted(glob.glob(os.path.join(checkpoints_dir, '*_checkpoint_*_steps.zip')))
    checkpoint_stats = sorted(glob.glob(os.path.join(checkpoints_dir, '*_checkpoint_vecnormalize_*_steps.pkl')))

    if len(checkpoint_models) >= 10 and seed:
        indices = np.round(np.linspace(0, len(checkpoint_models) - 1, 10)).astype(int)
        for idx in indices:
            model_path = checkpoint_models[idx]
            stepnum = extract_step_number(model_path)

            stats_path = match_vecnorm_file_by_step(checkpoint_stats, stepnum)
            if model_path and stats_path and stepnum:
                model_out = f'teammate{seed}checkpoint_{stepnum}steps.zip'
                stats_out = f'teammate{seed}checkpoint_{stepnum}vecstats.pkl'

                shutil.copy(model_path, os.path.join(output_dir, model_out))
                shutil.copy(stats_path, os.path.join(output_dir, stats_out))
    else:
        print(f"Found only {len(checkpoint_models)} checkpoint models or seed not set. Skipping checkpoint sampling.")

    print(f"✅ Files saved to {output_dir}")

def get_seed_from_filename(path):
    base = os.path.basename(path)
    parts = base.split('_')
    for part in parts:
        if 'seed' in part:
            return ''.join(filter(str.isdigit, part))
    return 'unknown'

def extract_step_number(filename):
    match = re.search(r'checkpoint_(\d+)_steps', filename)
    return match.group(1) if match else None

def match_vecnorm_file_by_step(stats_files, stepnum):
    for stats_file in stats_files:
        if f'_vecnormalize_{stepnum}_steps' in os.path.basename(stats_file):
            return stats_file
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect selected self-play teammate files.")
    parser.add_argument('directory', type=str, help='Name of the output directory (inside ./outputs/)')

    args = parser.parse_args()
    collect_teammate_files(args.directory)
