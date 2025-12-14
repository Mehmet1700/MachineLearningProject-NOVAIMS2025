"""
Main script to train Neural Network price models.

This script orchestrates the entire training pipeline:
1. Data Loading & Cleaning (Rule-based)
2. Preprocessing (Imputation, Encoding, Scaling)
3. Model Selection (Keras MLP, PyTorch FT-Transformer)
4. Hyperparameter Tuning (RandomizedSearchCV)
5. Evaluation & Submission Generation

It supports Log-Target Regression (TransformedTargetRegressor) to handle skewed price distributions.
"""

import argparse
import time
import shutil
from datetime import datetime
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd
from joblib import dump, load, parallel_backend
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, KFold, ShuffleSplit

# Internal Modules
from data_loaders.cars_data import load_full_train_and_test
from wrappers.baseline_nn_pipeline import (
    build_flexible_keras_pipeline, 
    build_flexible_ft_transformer_pipeline
)
from utils.feature_selection import candidate_selectors
from utils.preprocessing import remove_price_outliers_per_model
from hyperparameters import (
    build_keras_mlp_param_grid, 
    build_ft_transformer_param_grid, 
    build_phase_one_search_space
)

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------

def mae_func(y_true, y_pred):
    """Compute MAE for Kaggle submission metric."""
    mae = mean_absolute_error(y_true, y_pred)
    # print(f"   [DEBUG] Calculated MAE: {mae:.4f}")
    return mae

def timestamped_path(directory: str | Path, prefix: str, suffix: str) -> Path:
    """Generates a unique file path with a timestamp."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{prefix}_{ts}{suffix}"

def save_best_model(model, path: str):
    """Persist the fitted best estimator to disk via joblib."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, target_path)
    print(f"  -> Saved best model to: {target_path}")

def write_submission_file(test_ids, predictions, submission_path: str, id_column: str | None):
    """Create the Kaggle submission CSV from predictions."""
    column_name = id_column or getattr(test_ids, "name", "carID") or "carID"
    ids = pd.Series(test_ids, name=column_name).reset_index(drop=True)
    prices = pd.Series(predictions, name="price").reset_index(drop=True)

    submission = pd.DataFrame({column_name: ids, "price": prices})

    out_path = Path(submission_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(f"  -> Saved submission to: {out_path}")

def save_cv_summary_report(path: str, best_params: dict, best_mae: float, best_estimator):
    """Persist the key Cross-Validation details to a text file."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "====================================================",
        "CV RESULTS SUMMARY",
        "----------------------------------------------------",
        "Best parameters:",
        pformat(best_params, width=100),
        "",
        f"Best mean CV MAE: {best_mae:.3f}",
        "",
        "Best estimator (pipeline):",
        pformat(best_estimator, width=100),
    ]
    target_path.write_text("\n".join(lines))
    print(f"  -> Saved CV summary to: {target_path}")

def _fit_with_optional_threading(search, X, y, n_jobs: int):
    """Helper to handle joblib backend for threading vs multiprocessing."""
    backend = "threading" if n_jobs != 1 else None
    if backend:
        with parallel_backend(backend, n_jobs=n_jobs):
            search.fit(X, y)
    else:
        search.fit(X, y)

# -------------------------------------------------------------------
# Main Training Routine
# -------------------------------------------------------------------

def run_training(
    train_path: str,
    test_path: str,
    mapping_dir: str,
    n_splits: int = 5,
    random_state: int = 42,
    make_submission: bool = False,
    submission_path: str | None = None,
    id_column: str | None = None,
    model_path: str | None = None,
    n_jobs: int = 8,
    cv_summary_path: str | None = None,
    save_cv_summary: bool = False,
    debug_cleaning: bool = False,
    debug_imputation: bool = False,
    remove_outliers: bool = False,
    n_iter: int = 5,
    model_type: str = "keras",
):
    t0 = time.time()
    print("====================================================")
    print("STEP 1: Load Data & Rule-Based Cleaning")
    print("----------------------------------------------------")
    
    X_full, y_full, X_test, test_ids = load_full_train_and_test(
        train_path=train_path,
        test_path=test_path,
        mapping_dir=mapping_dir,
        return_test_ids=True,
        debug_cleaning=debug_cleaning,
    )
    
    if remove_outliers:
        print("  -> Removing price outliers per model (IQR method)...")
        X_full, y_full = remove_price_outliers_per_model(X_full, y_full)

    print(f"  -> Training Samples: {X_full.shape[0]}")
    print(f"  -> Test Samples:     {X_test.shape[0]}")
    print(f"  -> Features:         {X_full.shape[1]}")

    print("\n====================================================")
    print(f"STEP 2: Build Pipeline ({model_type})")
    print("----------------------------------------------------")
    
    # Select Pipeline Architecture
    if model_type == "ft_transformer":
        base_pipe = build_flexible_ft_transformer_pipeline(
            X_full, random_state=random_state, debug_imputation=debug_imputation
        )
        param_grid = build_ft_transformer_param_grid(random_state=random_state)
    else:
        base_pipe = build_flexible_keras_pipeline(
            X_full, random_state=random_state, debug_imputation=debug_imputation
        )
        param_grid = build_keras_mlp_param_grid(random_state=random_state)
    
    # Wrap in TransformedTargetRegressor for log-target regression
    # This ensures we train on log(price) but predict price.
    # Why? Price distributions are typically right-skewed (long tail of expensive cars).
    # Log-transforming makes the target more normal (Gaussian), which helps the NN converge faster and better.
    print("  -> Wrapping pipeline in TransformedTargetRegressor (log1p/expm1)...")
    wrapped_pipe = TransformedTargetRegressor(
        regressor=base_pipe,
        func=np.log1p,
        inverse_func=np.expm1
    )

    # Prefix param_grid keys with 'regressor__' because of TransformedTargetRegressor
    def _prefix_keys(grid, prefix="regressor__"):
        if isinstance(grid, list):
            return [{prefix + k: v for k, v in d.items()} for d in grid]
        return {prefix + k: v for k, v in grid.items()}

    param_grid = _prefix_keys(param_grid)

    print("\n====================================================")
    print(f"STEP 3: Cross-Validation Setup ({n_splits}-Fold)")
    print("----------------------------------------------------")
    
    mae_scorer = make_scorer(mae_func, greater_is_better=False)
    
    if n_splits <= 1:
        cv_strategy = ShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        print("  -> Strategy: ShuffleSplit (Hold-Out, 20% Test)")
    else:
        cv_strategy = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        print(f"  -> Strategy: KFold (k={n_splits})")

    print(f"  -> Hyperparameter Combinations: {len(param_grid)}")
    print(f"  -> Search Iterations: {n_iter}")

    search = RandomizedSearchCV(
        estimator=wrapped_pipe,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=mae_scorer,
        cv=cv_strategy,
        n_jobs=n_jobs,
        verbose=2,
        random_state=random_state,
        error_score="raise",
    )

    print("\n====================================================")
    print("STEP 4: Hyperparameter Tuning")
    print("----------------------------------------------------")
    t_fit_start = time.time()
    _fit_with_optional_threading(search, X_full, y_full, n_jobs)
    t_fit_end = time.time()

    print("\n====================================================")
    print("STEP 5: Evaluation Results")
    print("----------------------------------------------------")
    print("Best Parameters:")
    print(pformat(search.best_params_))

    best_mae_cv = -search.best_score_
    print(f"\nBest Mean CV MAE: {best_mae_cv:.3f}")

    best_model = search.best_estimator_

    if save_cv_summary:
        dest = cv_summary_path or str(timestamped_path("artifacts/cv_reports", "cv_summary", ".txt"))
        save_cv_summary_report(dest, search.best_params_, best_mae_cv, best_model)

    if save_best_model or model_path:
        dest = model_path or str(timestamped_path("artifacts/models", "best_model", ".joblib"))
        save_best_model(best_model, dest)

    # ----------------------------------------------------------------
    # Submission Generation
    # ----------------------------------------------------------------
    if make_submission:
        print("\n====================================================")
        print("STEP 6: Final Training & Submission")
        print("----------------------------------------------------")
        # Refit on full data (X_full) is done automatically by RandomizedSearchCV.best_estimator_
        # but we can do it explicitly if needed. Here we trust the search object.
        
        print("  -> Predicting on Test Set...")
        test_preds = best_model.predict(X_test)
        
        submission_dest = submission_path or timestamped_path(
            "artifacts/submissions", "submission_kaggle", ".csv"
        )
        write_submission_file(test_ids, test_preds, submission_dest, id_column)

    print("\n====================================================")
    print(f"DONE. Total Runtime: {time.time() - t0:.1f}s")
    print("====================================================")


# -------------------------------------------------------------------
# CLI Argument Parsing
# -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Neural Network Car Price Models (MLP, Transformer, Trees)."
    )
    
    # Data Paths
    parser.add_argument("--train-path", type=str, default="data/train.csv", help="Path to training data.")
    parser.add_argument("--test-path", type=str, default="data/test.csv", help="Path to test data.")
    parser.add_argument("--mapping-dir", type=str, default="mapping", help="Directory containing JSON mappings.")
    
    # Training Configuration
    parser.add_argument("--model-type", type=str, default="keras", choices=["keras", "ft_transformer"],
                        help="Model architecture to use: 'keras' (MLP), 'ft_transformer' (PyTorch).")
    parser.add_argument("--n-splits", type=int, default=1, help="Number of CV folds (1 = Hold-out).")
    parser.add_argument("--n-iter", type=int, default=5, help="Number of random search iterations.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    
    # Preprocessing Options
    parser.add_argument("--remove-outliers", action="store_true", help="Apply IQR outlier removal on training data.")
    
    # Output Options
    parser.add_argument("--make-submission", action="store_true", help="Generate submission file for Kaggle.")
    parser.add_argument("--submission-path", type=str, default=None, help="Explicit path for submission file.")
    parser.add_argument("--save-best-model", action="store_true", help="Save the best trained model.")
    parser.add_argument("--save-model-path", type=str, default=None, help="Explicit path for saving the model.")
    parser.add_argument("--save-cv-summary", action="store_true", help="Save a text report of CV results.")
    parser.add_argument("--cv-summary-path", type=str, default=None, help="Explicit path for CV report.")
    
    # Debugging
    parser.add_argument("--debug-cleaning", action="store_true", help="Verbose logging for data cleaning.")
    parser.add_argument("--debug-imputation", action="store_true", help="Verbose logging for imputation.")
    
    # Legacy/Unused (kept for compatibility if needed, or can be removed)
    parser.add_argument("--save-train-residuals", action="store_true", help="Save training residuals (deprecated).")
    parser.add_argument("--train-residuals-path", type=str, default=None, help="Path for residuals.")
    parser.add_argument("--load-model-path", type=str, default=None, help="Load existing model for inference.")
    parser.add_argument("--id-column", type=str, default="carID", help="ID column name.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    run_training(
        train_path=args.train_path,
        test_path=args.test_path,
        mapping_dir=args.mapping_dir,
        n_splits=args.n_splits,
        random_state=args.random_state,
        make_submission=args.make_submission,
        submission_path=args.submission_path,
        id_column=args.id_column,
        model_path=args.save_model_path,
        n_jobs=args.n_jobs,
        cv_summary_path=args.cv_summary_path,
        save_cv_summary=args.save_cv_summary,
        debug_cleaning=args.debug_cleaning,
        debug_imputation=args.debug_imputation,
        remove_outliers=args.remove_outliers,
        n_iter=args.n_iter,
        model_type=args.model_type,
    )
