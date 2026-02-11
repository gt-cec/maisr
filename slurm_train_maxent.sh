#!/bin/bash
#SBATCH --job-name=maisr_maxent        # Job name
#SBATCH --account=ae                # Tracking account
#SBATCH --output=logs/maxent_%j.out    # Standard output log (%j = job ID)
#SBATCH --error=logs/maxent_%j.err     # Standard error log
#SBATCH --ntasks=1                     # Number of tasks (processes)
#SBATCH --cpus-per-task=16             # Number of CPU cores per task
#SBATCH --mem=32G                      # Memory per node
#SBATCH --time=18:00:00                # Time limit (48 hours)
#SBATCH --mail-type=END,FAIL           # Email notifications
#SBATCH --mail-user=rbowers32@gatech.edu # Email address (update this!)


# Print job information
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=================================================="

# Create logs directory if it doesn't exist
mkdir -p logs
echo "[OK] Logs directory ready."

# Activate virtual environment (adjust path as needed)
echo "Activating conda base environment..."
source ~/scratch/miniconda/bin/activate
echo "[OK] Conda base activated. ($(conda --version))"

echo "Activating maisr-rl conda environment..."
source activate maisr-rl
echo "[OK] Conda environment activated: $CONDA_DEFAULT_ENV"

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[OK] Thread environment variables set (OMP/MKL threads: $SLURM_CPUS_PER_TASK)."

# Training parameters
TOTAL_TIMESTEPS=1e6 #18e6
POPULATION_SIZE=2 # 6
NUM_CHECKPOINTS=9
SEED=42
ENT_COEF=0.01
CONFIG_PATH="configs/maxent_config.json"
PROJECT_NAME="maisr-mep-hpc"

echo "--------------------------------------------------"
echo "Training configuration:"
echo "  Total timesteps:  $TOTAL_TIMESTEPS"
echo "  Population size:  $POPULATION_SIZE"
echo "  Num checkpoints:  $NUM_CHECKPOINTS"
echo "  Seed:             $SEED"
echo "  Entropy coef:     $ENT_COEF"
echo "  Config path:      $CONFIG_PATH"
echo "  Project name:     $PROJECT_NAME"
echo "  Num envs:         $SLURM_CPUS_PER_TASK"
echo "--------------------------------------------------"

cd ~/scratch/MAISR_revisions/maisr
echo "[OK] Changed directory to: $(pwd)"

# Run training
echo "Starting training at $(date)..."
python train_maxent.py \
    --total_timesteps $TOTAL_TIMESTEPS \
    --population_size $POPULATION_SIZE \
    --num_checkpoints $NUM_CHECKPOINTS \
    --seed $SEED \
    --ent_coef $ENT_COEF \
    --config $CONFIG_PATH \
    --project_name $PROJECT_NAME \
    --n_envs $SLURM_CPUS_PER_TASK

# Capture exit code to confirm success/failure
TRAIN_EXIT_CODE=$?
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "[OK] Training script completed successfully."
else
    echo "[ERROR] Training script exited with code $TRAIN_EXIT_CODE."
    exit $TRAIN_EXIT_CODE
fi

# Print completion time
echo "=================================================="
echo "End time: $(date)"
echo "Job completed successfully!"
echo "=================================================="