#!/bin/bash
#SBATCH --job-name=car_price_nn_gpu
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
# Car Price Prediction - Training Job
# ==============================================================================
# This script submits a job to the HPC cluster to train the car price prediction model.
# It supports various architectures defined in main.py (FT-Transformer, Keras MLP).
#
# Usage: sbatch train_nn_gpu.slurm
# ==============================================================================

echo "Job started on $(hostname) at $(date)"
export PYTHONUNBUFFERED=1

# 1. Environment Setup
# ------------------------------------------------------------------------------
# Navigate to the project directory
cd /projects/F202500002HPCVLABISTUL/Mehmet1700/MachineLearningProject-NOVAIMS2025/NN-HPC

# Create directories for logs and artifacts if they don't exist
mkdir -p logs artifacts

# Activate the virtual environment
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
echo "Starting Training Pipeline..."
echo "----------------------------------------------------"

# 3. GPU Monitoring (Background Process)
# ------------------------------------------------------------------------------
# Log GPU utilization every 5 seconds
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv --loop=5 &
GPU_MONITOR_PID=$!

# Ensure the monitor is killed when the script exits
cleanup() {
  if ps -p $GPU_MONITOR_PID >/dev/null 2>&1 ; then
    kill $GPU_MONITOR_PID
  fi
}
trap cleanup EXIT

# 4. Execution
# ------------------------------------------------------------------------------
# Available --model-type options:
#   - ft_transformer (PyTorch FT-Transformer)
#   - keras          (Keras MLP)
#
# Adjust --n-iter for hyperparameter search iterations.
# Use --make-submission to generate a submission file.

python main.py \
  --model-type ft_transformer \
  --train-path ../data/train.csv \
  --test-path ../data/test.csv \
  --mapping-dir ../mapping \
  --n-splits 5 \
  --random-state 42 \
  --n-iter 10 \
  --make-submission \
  --save-best-model \
  --save-cv-summary \
  --save-train-residuals \
  --remove-outliers

echo "Job finished at $(date)"
