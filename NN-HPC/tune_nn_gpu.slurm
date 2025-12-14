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

# ==============================================================================
# Car Price Prediction - Hyperparameter Tuning Job
# ==============================================================================
# This script runs an extensive hyperparameter search.
# It is configured to run for a longer duration with more iterations.
#
# Usage: sbatch tune_nn_gpu.slurm
# ==============================================================================

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
export PYTHONUNBUFFERED=1

# 1. Environment Setup
# ------------------------------------------------------------------------------
cd /projects/F202500002HPCVLABISTUL/Mehmet1700/MachineLearningProject-NOVAIMS2025/NN-HPC

mkdir -p logs artifacts

source ../.venv/bin/activate

# 2. GPU Verification
# ------------------------------------------------------------------------------
python - << 'EOF'
import torch
import sys
print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA not available for PyTorch!")
EOF

echo "----------------------------------------------------"
echo "Starting Hyperparameter Tuning..."
echo "----------------------------------------------------"

# 3. GPU Monitoring
# ------------------------------------------------------------------------------
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv --loop=5 &
GPU_MONITOR_PID=$!
cleanup() {
  if ps -p $GPU_MONITOR_PID >/dev/null 2>&1 ; then
    kill $GPU_MONITOR_PID
  fi
}
trap cleanup EXIT

# 4. Execution
# ------------------------------------------------------------------------------
# We increase --n-iter to explore more hyperparameter combinations.
# We use --n-splits 3 for a robust Cross-Validation during tuning.

python main.py \
  --model-type ft_transformer \
  --train-path ../data/train.csv \
  --test-path ../data/test.csv \
  --mapping-dir ../mapping \
  --n-splits 3 \
  --random-state 42 \
  --n-iter 50 \
  --save-best-model \
  --save-cv-summary \
  --remove-outliers

echo "Job finished at $(date)"
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
