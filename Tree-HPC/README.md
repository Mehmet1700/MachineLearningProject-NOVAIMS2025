# Tree-Based Models for Car Price Prediction (HPC Pipeline)

## 1. Project Overview
This directory contains the High-Performance Computing (HPC) implementation for the "Car Price Prediction" project. It focuses on training, tuning, and evaluating tree-based ensemble models using the Scikit-Learn framework. The pipeline is optimized for execution on the Deucalion cluster using SLURM workloads.

## 2. Methodology

### 2.1 Data Pipeline
The data processing pipeline ensures robust handling of the car dataset:
1.  **Data Loading**: Merges training and test datasets for consistent preprocessing.
2.  **Cleaning**: 
    - Filters out inconsistent records (e.g., `year < 2000`).
    - Handles missing values using statistical imputation.
3.  **Feature Engineering**:
    - **Categorical Encoding**: Uses `TargetEncoder` for high-cardinality features (Brand, Model, Transmission) to capture the relationship with price.
    - **Numerical Scaling**: Applies `QuantileTransformer` to normalize skewed distributions (e.g., mileage, tax).
4.  **Target Transformation**: The target variable (`price`) is modeled directly, but the pipeline supports Log-transformation if needed (via `TransformedTargetRegressor`).

### 2.2 Model Architectures
We explore three distinct tree-based architectures to find the optimal balance between bias and variance:
-   **Random Forest Regressor (`rf`)**: A bagging ensemble that reduces variance by averaging multiple deep decision trees trained on bootstrap samples.
-   **Extra Trees Regressor (`et`)**: "Extremely Randomized Trees" which introduce further randomness in split selection, often yielding lower variance and faster training times than Random Forest.
-   **Decision Tree Regressor (`dt`)**: A single tree baseline to assess the value added by ensemble methods.

### 2.3 Hyperparameter Optimization
-   **Search Strategy**: `RandomizedSearchCV` is used to efficiently explore the hyperparameter space.
-   **Metric**: The primary optimization metric is **Mean Absolute Error (MAE)**, aligned with the project's success criteria (< 1000 MAE).
-   **Validation**: 
    -   Uses **Hold-Out Validation (ShuffleSplit)** with a 20% test size for rapid iteration on large datasets.
    -   Supports **K-Fold Cross-Validation** for more robust final evaluation.

## 3. Repository Structure

```text
Tree-HPC/
├── main.py                 # Entry point for the training pipeline
├── pipelines.py            # Scikit-learn pipeline definitions (Preprocessing + Model)
├── hyperparameters.py      # Hyperparameter search spaces for each model
├── train_tree.slurm        # SLURM submission script for HPC execution
├── setup_env.sh            # Helper script to create the virtual environment
├── requirements.txt        # Python dependencies
├── data_loaders/           # Modules for loading and merging CSV data
├── utils/                  # Utility functions (preprocessing, logging)
└── artifacts/              # Output directory for models, logs, and submissions
```

## 4. Setup & Installation

### 4.1 Prerequisites
-   Python 3.11+
-   Access to a SLURM-based HPC cluster (optional, can run locally)

### 4.2 Environment Setup
Run the setup script to create a virtual environment and install dependencies:
```bash
bash setup_env.sh
```

## 5. Usage

### 5.1 Running on HPC (Recommended)
Submit the job to the cluster using the SLURM script. This will train all configured models sequentially.
```bash
sbatch train_tree.slurm
```
*Note: Check `train_tree.slurm` to adjust resources (CPUs, Memory) or partition.*

### 5.2 Running Locally
You can run the `main.py` script directly for debugging or local training:
```bash
source .venv/bin/activate
python main.py --model-type rf --n-iter 5 --n-splits 1
```

## 6. Results & Artifacts
After a successful run, the following artifacts are generated in the `artifacts/` directory:
-   **Models**: Saved `.joblib` files for the best estimators.
-   **Submissions**: CSV files ready for Kaggle submission.
-   **Reports**: Text files summarizing the Cross-Validation results and best hyperparameters.
