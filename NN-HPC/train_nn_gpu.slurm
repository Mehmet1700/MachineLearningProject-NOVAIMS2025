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

echo "Job started on $(hostname) at $(date)"
export PYTHONUNBUFFERED=1


# Module
module purge
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1

# 1. Go to YOUR NN-Notebooks folder (correct!)
cd /projects/F202500002HPCVLABISTUL/Mehmet1700/MachineLearningProject-NOVAIMS2025/NN-HPC

mkdir -p logs artifacts

source ../.venv/bin/activate


# 4. Check GPU visibility
python - << 'EOF'
import tensorflow as tf
print("Visible GPUs:", tf.config.list_physical_devices("GPU"))
EOF

echo "Running main.py with Keras + GPU ..."
echo "----------------------------------------------------"

# Stream GPU utilization into the job logs every 5 seconds
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv --loop=5 &
GPU_MONITOR_PID=$!
cleanup() {
  if ps -p $GPU_MONITOR_PID >/dev/null 2>&1 ; then
    kill $GPU_MONITOR_PID
  fi
}
trap cleanup EXIT

# 5. IMPORTANT: data/mapping are OUTSIDE NN-Notebooks → use ../data
python main.py \
  --train-path ../data/train.csv \
  --test-path ../data/test.csv \
  --mapping-dir ../mapping \
  --n-splits 5 \
  --random-state 42 \
  --two-phase-search \
  --phase1-folds 2 \
  --phase1-iters 12 \
  --phase1-top-k 3 \
  --phase1-n-jobs 1 \
  --phase1-grid-path artifacts/phase1_grids/phase1_grid_20251124_112446.joblib \
  --phase2-epochs 120 150 \
  --make-submission \
  --save-best-model

echo "Job finished at $(date)"
