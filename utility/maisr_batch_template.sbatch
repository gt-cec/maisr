#!/bin/bash
#SBATCH --job-name=maisr-p99
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --account=gts-kf52
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=general
#SBATCH --mail-user=rbowers32@gatech.edu
#SBATCH --mail-type=END,FAIL

# >>> Conda setup <<<
source ~/.bashrc
source ~/miniconda/bin/activate
conda activate maisr-rl

# >>> Run the training script <<<
cd maisr-rl
python train_generic.py --version 'pretrained_agents' --seed 99

status=$?
if [ $status -ne 0 ]; then
    tail -n 50 logs/array_job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err | mail -s "SLURM Job ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} FAILED" rbowers32@gatech.edu
fi