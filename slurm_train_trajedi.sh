#!/bin/bash
#SBATCH --job-name=maisr_trajedi       # Job name
#SBATCH --output=logs/trajedi_%j.out   # Standard output log (%j = job ID)
#SBATCH --error=logs/trajedi_%j.err    # Standard error log
#SBATCH --ntasks=1                     # Number of tasks (processes)
#SBATCH --cpus-per-task=24             # Number of CPU cores per task
#SBATCH --mem=48G                      # Memory per node
#SBATCH --time=72:00:00                # Time limit (72 hours)
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
SEED=42
PROJECT_NAME="maisr-trajedi-hpc"

# NOTE: Total timesteps ≈ 3e6 is configured via configs/trajedi_config.json
# The total timesteps = training_rounds × steps_per_phase × (n_populations + 1) × n_seeds
# Example: 50 rounds × 10000 steps × (3 pop + 1 BR) × 2 seeds = 4e6 timesteps
# Adjust trajedi_config.json to achieve desired timesteps:
#   - training_rounds: Number of training iterations
#   - steps_per_phase: Timesteps per training phase
#   - n_populations: Number of population agents per seed
#   - n_seeds: Number of independent seed pools

echo "Configuration:"
echo "  Target total timesteps: ~3e6"
echo "  Edit configs/trajedi_config.json to adjust:"
echo "    - training_rounds"
echo "    - steps_per_phase"
echo "    - n_populations"
echo "    - n_seeds"

# Run training
python train_trajedi_ppo.py \
    --seed $SEED \
    --project $PROJECT_NAME

# Print completion time
echo "End time: $(date)"
echo "Job completed successfully!"
