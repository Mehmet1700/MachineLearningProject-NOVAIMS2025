# Neural Network & Machine Learning HPC Pipeline

This repository contains a professional-grade High-Performance Computing (HPC) pipeline designed to train, tune, and evaluate various regression models for car price prediction. The pipeline is optimized for execution on SLURM-managed clusters and implements a robust workflow including data ingestion, advanced preprocessing, feature engineering, hyperparameter optimization, and model persistence.

## Architecture Overview

The pipeline follows a modular design pattern to ensure reproducibility, scalability, and maintainability:

1.  **Data Ingestion (`data_loaders/`)**:
    *   Handles loading of raw training and test datasets.
    *   Performs initial rule-based cleaning and mapping normalization.
    *   Ensures data integrity by preserving unique identifiers (`carID`).

2.  **Feature Engineering (`utils/feature_engineering.py`)**:
    *   Implements domain-specific feature extraction (e.g., `efficiency_ratio`, `mileage_per_year`).
    *   Calculates statistical aggregates (e.g., median MPG per transmission) to enrich the dataset.
    *   Designed as a plug-and-play transformer for scikit-learn pipelines.

3.  **Preprocessing & Pipelines (`wrappers/baseline_nn_pipeline.py`)**:
    *   Constructs comprehensive `scikit-learn` Pipelines for different model architectures.
    *   **Imputation**: Implements a hybrid strategy using rule-based logic (Model Mode) and model-based prediction (Random Forest) for missing values.
    *   **Encoding & Scaling**: Applies One-Hot/Ordinal Encoding and Robust Scaling appropriate for each model type.
    *   **Supported Architectures**:
        *   **PyTorch FT-Transformer**: A state-of-the-art Transformer model for tabular data.
        *   **Keras MLP**: A flexible Multi-Layer Perceptron.

4.  **Hyperparameter Management (`hyperparameters.py`)**:
    *   Centralized configuration for model search spaces.
    *   Defines parameter grids for RandomizedSearchCV and GridSearchCV.
    *   Ensures clean separation between configuration and execution logic.

5.  **Orchestration (`main.py`)**:
    *   The central entry point for the training process.
    *   Parses command-line arguments to select models, tuning strategies, and output options.
    *   Implements **Log-Target Regression** (`TransformedTargetRegressor`) to handle skewed price distributions.
    *   Executes **Cross-Validation** to ensure robust performance estimation.

6.  **Artifact Management (`artifacts/`, `logs/`)**:
    *   **Models**: Persists best-performing models as timestamped `.joblib` files.
    *   **Submissions**: Generates Kaggle-ready CSV submission files.
    *   **Reports**: Saves detailed Cross-Validation summaries and residual analysis reports.

## Workflow & Usage

### 1. Environment Setup
Ensure the required modules and virtual environment are active. The provided SLURM scripts handle this automatically on the cluster.

### 2. Job Submission
To launch a training job on the HPC cluster, use the `sbatch` command.

```bash
sbatch train_nn_gpu.slurm
```


**Training (Single Run / Small Grid):**
```bash
sbatch train_nn_gpu.slurm
```
*   Runs `main.py` with selected model (default: FT-Transformer).
*   Generates a submission file.

**Hyperparameter Tuning (Extensive Search):**
```bash
sbatch tune_nn_gpu.slurm
```
*   Runs a longer `RandomizedSearchCV` session.
*   Saves the best found parameters and model.

### 3. Command Line Interface (CLI)
You can also run `main.py` directly (e.g., in an interactive session):

```bash
python main.py \
  --model-type ft_transformer \  # Options: ft_transformer, keras
  --n-splits 5 \                 # Number of CV folds
  --n-iter 20 \                  # Number of search iterations
  --make-submission \            # Generate submission.csv
  --save-best-model              # Save the trained model
```

## Project Structure

```
NN-HPC/
├── main.py                     # Main orchestration script
├── hyperparameters.py          # Hyperparameter search spaces
├── train_nn_gpu.slurm          # SLURM script for training
├── tune_nn_gpu.slurm           # SLURM script for tuning
├── wrappers/
│   ├── baseline_nn_pipeline.py # Pipeline definitions
│   ├── pytorch_wrapper.py      # PyTorch FT-Transformer wrapper
│   └── keras_nn.py             # Keras MLP wrapper
├── utils/                      # Helper modules (cleaning, engineering, etc.)
└── data_loaders/               # Data loading logic
```
*   `--n-iter`: Number of parameter settings to sample (e.g., 5 or 10).
*   `--remove-outliers`: Activates IQR-based outlier removal for improved stability.
*   `--save-best-model`: Saves the best estimator to `artifacts/models/`.
*   `--make-submission`: Generates predictions for the test set.

### 3. Monitoring
Monitor the job's progress and resource usage:

*   **Live Logs**: `tail -f logs/car_price_nn_gpu_<JOBID>.out`
*   **Queue Status**: `squeue -u $USER`

### 4. Analysis & Inference
After training completes:
*   **CV Results**: Check `artifacts/cv_reports/` for a summary of the best hyperparameters and MAE scores.
*   **Error Analysis**: Inspect `artifacts/error_analysis/` for detailed residual plots (if enabled).
*   **Inference**: To generate predictions using a saved model without retraining:
    ```bash
    python main.py --load-model-path artifacts/models/best_model_<TIMESTAMP>.joblib --submission-path my_submission.csv
    ```

## Repository Structure

*   `configs/`: Configuration files for experiments.
*   `data_loaders/`: Data reading and cleaning logic.
*   `utils/`: Helper functions for imputation, feature selection, and engineering.
*   `wrappers/`: Scikit-learn wrappers for Keras models and pipelines.
*   `main.py`: The primary entry point for training and evaluation.
*   `train_nn_gpu.slurm`: SLURM batch script for HPC execution.

````