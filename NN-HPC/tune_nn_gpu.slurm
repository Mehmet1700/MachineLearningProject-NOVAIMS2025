#!/bin/bash
#SBATCH --job-name=car_price_nn_tune
#SBATCH --partition=dev-a100-40
#SBATCH --account=f202500002hpcvlabistulg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
export PYTHONUNBUFFERED=1

# Point to the promoted Phase 2 grid produced by a prior warm-up run.
PHASE1_GRID="artifacts/phase1_grids/phase1_grid_20251124_112446.joblib"

if [[ ! -f "${PHASE1_GRID}" ]]; then
  echo "Missing Phase 1 grid at ${PHASE1_GRID}. Update the path before submitting." >&2
  exit 1
fi

module purge
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1

cd /projects/F202500002HPCVLABISTUL/Mehmet1700/MachineLearningProject-NOVAIMS2025/NN-HPC

mkdir -p logs artifacts

source ../.venv/bin/activate

python - << 'EOF'
import tensorflow as tf
print("Visible GPUs:", tf.config.list_physical_devices("GPU"))
EOF

echo "Running main.py (hyperparameter tuning only) ..."
echo "----------------------------------------------------"

nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv --loop=5 &
GPU_MONITOR_PID=$!
cleanup() {
  if ps -p $GPU_MONITOR_PID >/dev/null 2>&1 ; then
    kill $GPU_MONITOR_PID
  fi
}
trap cleanup EXIT

python main.py \
  --train-path ../data/train.csv \
  --test-path ../data/test.csv \
  --mapping-dir ../mapping \
  --n-splits 5 \
  --random-state 42 \
  --two-phase-search \
  --phase1-grid-path "${PHASE1_GRID}" \
  --n-jobs 8 \
  --save-cv-summary \
  --preview-feature-selection

echo "Job finished at $(date)"
