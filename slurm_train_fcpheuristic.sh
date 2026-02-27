#!/bin/bash
#SBATCH --job-name=maisr_heuristic_fcp       # Job name
#SBATCH --account=gts-kf52                # Tracking account
#SBATCH --output=logs/heuristic_fcp%j.out    # Standard output log (%j = job ID)
#SBATCH --error=logs/heuristic_fcp%j.err     # Standard error log
#SBATCH --ntasks=1                     # Number of tasks (processes)
#SBATCH --cpus-per-task=16             # Number of CPU cores per task
#SBATCH --mem=32G                      # Memory per node
#SBATCH --time=32:00:00                # Time limit (48 hours)
#SBATCH --mail-type=END,FAIL           # Email notifications
#SBATCH --mail-user=rbowers32@gatech.edu # Email address (update this!)


# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment (adjust path as needed)
source ~/scratch/miniconda/bin/activate
source activate maisr-rl

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd ~/scratch/MAISR_revisions/maisr

# Run training
python train_revised.py -- league_type 'heuristic_fcp'

# Print completion time
echo "End time: $(date)"
echo "Job completed successfully!"
