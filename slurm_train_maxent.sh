#!/bin/bash
#SBATCH --job-name=maisr_maxent        # Job name
#SBATCH --output=logs/maxent_%j.out    # Standard output log (%j = job ID)
#SBATCH --error=logs/maxent_%j.err     # Standard error log
#SBATCH --ntasks=1                     # Number of tasks (processes)
#SBATCH --cpus-per-task=16             # Number of CPU cores per task
#SBATCH --mem=32G                      # Memory per node
#SBATCH --time=48:00:00                # Time limit (48 hours)
#SBATCH --partition=cpu                # Partition name (adjust for your cluster)
#SBATCH --mail-type=END,FAIL           # Email notifications
#SBATCH --mail-user=rbowers32@gatech.edu # Email address (update this!)

# Load required modules (adjust for your HPC environment)
# module load python/3.10
# module load cuda/11.8  # If GPU needed

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment (adjust path as needed)
# source /path/to/your/venv/bin/activate
# OR if using conda:
# source activate maisr

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Training parameters
TOTAL_TIMESTEPS=18e6
POPULATION_SIZE=6
NUM_CHECKPOINTS=18
SEED=42
ENT_COEF=0.01
CONFIG_PATH="configs/maxent_config.json"
PROJECT_NAME="maisr-mep-hpc"

# Run training
python train_maxent.py \
    --total_timesteps $TOTAL_TIMESTEPS \
    --population_size $POPULATION_SIZE \
    --num_checkpoints $NUM_CHECKPOINTS \
    --seed $SEED \
    --ent_coef $ENT_COEF \
    --config $CONFIG_PATH \
    --project_name $PROJECT_NAME \
    --n_envs $SLURM_CPUS_PER_TASK

# Print completion time
echo "End time: $(date)"
echo "Job completed successfully!"
