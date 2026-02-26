#!/bin/bash
#SBATCH --job-name=ego_maxent        # Job name
#SBATCH --account=gts-kf52                # Tracking account
#SBATCH --output=logs/egomaxent_%j.out    # Standard output log (%j = job ID)
#SBATCH --error=logs/egomaxent_%j.err     # Standard error log
#SBATCH --ntasks=1                     # Number of tasks (processes)
#SBATCH --cpus-per-task=24            # Number of CPU cores per task
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
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=offline   # Prevents hang on internet-restricted HPC nodes
echo "[OK] Thread environment variables set (OMP/MKL=1, WANDB_MODE=offline)."

cd ~/scratch/MAISR_revisions/maisr
echo "[OK] Changed directory to: $(pwd)"

# Run training
echo "Starting training at $(date)..."
python train_revised.py --league_type 'maxent'

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
