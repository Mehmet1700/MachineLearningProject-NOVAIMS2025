#!/bin/bash
#SBATCH --job-name=car_price_tree
#SBATCH --partition=dev-a100-40
#SBATCH --account=f202500002hpcvlabistulg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ==============================================================================
# Car Price Prediction - Tree Models Training Job
# ==============================================================================
# This script submits a job to the HPC cluster to train tree-based models.
# Supported models: rf (RandomForest), et (ExtraTrees), dt (DecisionTree)
#
# Usage: sbatch train_tree.slurm
# ==============================================================================

echo "Job started on $(hostname) at $(date)"
export PYTHONUNBUFFERED=1

# 1. Environment Setup
# ------------------------------------------------------------------------------
# Navigate to the project directory
cd /projects/F202500002HPCVLABISTUL/Mehmet1700/MachineLearningProject-NOVAIMS2025/Tree-HPC

# Create directories for logs and artifacts if they don't exist
mkdir -p logs artifacts

# Activate the virtual environment
# Load the same module used to create the venv
module load Python/3.11.3-GCCcore-12.3.0
# Using local .venv
source .venv/bin/activate

# 2. Run Training
# ------------------------------------------------------------------------------

# --- 1. Extra Trees (Refined Search) ---
echo "Starting Extra Trees Training (Refined)..."
python main.py \
    --model-type et \
    --n-iter 50 \
    --n-splits 5 \
    --n-jobs 32 \
    --make-submission \
    --save-cv-summary \
    --train-path ../data/train.csv \
    --test-path ../data/test.csv \
    --mapping-dir ../mapping

# --- 2. Random Forest (Optional / Baseline) ---
# echo "Starting Random Forest Training..."
# python main.py \
#     --model-type rf \
#     --n-iter 20 \
#     --n-splits 1 \
#     --n-jobs 64 \
#     --make-submission \
#     --save-cv-summary \
#     --train-path ../data/train.csv \
#     --test-path ../data/test.csv \
#     --mapping-dir ../mapping

# --- 3. Decision Tree (Optional / Baseline) ---
# echo "Starting Decision Tree Training..."
# python main.py \
#     --model-type dt \
#     --n-iter 50 \
#     --n-splits 1 \
#     --n-jobs 64 \
#     --make-submission \
#     --save-cv-summary \
#     --train-path ../data/train.csv \
#     --test-path ../data/test.csv \
#     --mapping-dir ../mapping

echo "Job finished at $(date)"
