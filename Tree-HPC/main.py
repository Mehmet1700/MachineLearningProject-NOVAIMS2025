"""
Main script to train Tree-based price models (Random Forest, Extra Trees, HistGradientBoosting).

This script orchestrates the entire training pipeline:
1. Data Loading & Cleaning
2. Preprocessing (Imputation, Encoding, Scaling)
3. Model Selection (RF, ET, HGB)
4. Hyperparameter Tuning (RandomizedSearchCV)
5. Evaluation & Submission Generation
"""

import argparse
import time
import shutil
from datetime import datetime
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import RandomizedSearchCV, KFold, ShuffleSplit

# Internal Modules
from data_loaders.cars_data import load_full_train_and_test
from utils.preprocessing import build_preprocessor, get_feature_types
from pipelines import build_rf_pipeline, build_et_pipeline, build_hgb_pipeline, build_dt_pipeline
from hyperparameters import (
    build_random_forest_param_grid,
    build_extra_trees_param_grid,
    build_hist_gradient_boosting_param_grid,
    build_decision_tree_param_grid
)

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------

def mae_func(y_true, y_pred):
    """Compute MAE for Kaggle submission metric."""
    return mean_absolute_error(y_true, y_pred)

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

def write_submission_file(test_ids, predictions, submission_path: str, id_column: str = "carID"):
    """Create the Kaggle submission CSV from predictions."""
    column_name = id_column or "carID"
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

def save_feature_importance(model, output_path: str):
    """
    Extracts and saves feature importances from the fitted pipeline.
    Handles the specific structure: Pipeline -> TransformedTargetRegressor -> Regressor
    """
    try:
        # 1. Unwrap the model
        # Pipeline steps: ['preprocessor', 'model']
        # 'model' is TransformedTargetRegressor
        ttr = model.named_steps['model']
        regressor = ttr.regressor_
        
        # Check if model has feature_importances_
        if not hasattr(regressor, "feature_importances_"):
            print("  -> Model does not support feature importance (skipping).")
            return

        importances = regressor.feature_importances_

        # 2. Get feature names from preprocessor
        preprocessor = model.named_steps['preprocessor']
        
        # For scikit-learn < 1.2, get_feature_names_out might be tricky with some transformers
        # But we are using 1.3+ in requirements.
        try:
            feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
            # Fallback if get_feature_names_out fails (e.g. custom transformers)
            print("  -> Could not extract feature names (using indices).")
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        # 3. Create DataFrame
        if len(feature_names) != len(importances):
            print(f"  -> Mismatch: {len(feature_names)} names vs {len(importances)} values.")
            return

        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        # 4. Save
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fi_df.to_csv(out_path, index=False)
        print(f"  -> Saved feature importance to: {out_path}")
        
    except Exception as e:
        print(f"  -> Error saving feature importance: {e}")

# -------------------------------------------------------------------
# Main Training Routine
# -------------------------------------------------------------------

def run_training(
    train_path: str,
    test_path: str,
    mapping_dir: str,
    n_splits: int,
    random_state: int,
    make_submission: bool,
    submission_path: str,
    id_column: str,
    model_path: str,
    n_jobs: int,
    cv_summary_path: str,
    save_cv_summary: bool,
    n_iter: int,
    model_type: str,
):
    t0 = time.time()
    print("====================================================")
    print(f"STARTING TRAINING PIPELINE (Model: {model_type})")
    print("====================================================")

    # 1. Load Data
    print(f"  -> Loading data from {train_path} and {test_path}...")
    
    X_full, y_full, X_test, test_ids = load_full_train_and_test(
        train_path=train_path,
        test_path=test_path,
        mapping_dir=mapping_dir,
        return_test_ids=True,
        debug_cleaning=False
    )
    
    print(f"  -> Training Samples: {X_full.shape[0]}")
    print(f"  -> Test Samples:     {X_test.shape[0]}")

    # 2. Build Preprocessor
    numeric_features, categorical_features = get_feature_types(X_full)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # 3. Build Pipeline & Param Grid
    if model_type == "rf":
        pipeline = build_rf_pipeline(preprocessor, random_state)
        param_grid = build_random_forest_param_grid()
    elif model_type == "et":
        pipeline = build_et_pipeline(preprocessor, random_state)
        param_grid = build_extra_trees_param_grid()
    elif model_type == "hgb":
        pipeline = build_hgb_pipeline(preprocessor, random_state)
        param_grid = build_hist_gradient_boosting_param_grid()
    elif model_type == "dt":
        pipeline = build_dt_pipeline(preprocessor, random_state)
        param_grid = build_decision_tree_param_grid()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 4. CV Setup
    # We use MAE (Mean Absolute Error) as the primary metric, consistent with the competition goal.
    mae_scorer = make_scorer(mae_func, greater_is_better=False)
    
    if n_splits <= 1:
        # Hold-Out Strategy: Faster, good for quick debugging or very large datasets.
        cv_strategy = ShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        print("  -> Strategy: ShuffleSplit (Hold-Out, 20% Test)")
    else:
        # K-Fold Cross-Validation: More robust, reduces variance in performance estimation.
        cv_strategy = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        print(f"  -> Strategy: KFold (k={n_splits})")

    print(f"  -> Search Iterations: {n_iter}")
    print(f"  -> Total Fits: {n_iter * n_splits} (n_iter={n_iter} * n_splits={n_splits})")

    # Note on Parallelization:
    # We set n_jobs here for the SearchCV. The individual models (in pipelines.py) 
    # should have n_jobs=1 to avoid oversubscription (too many threads fighting for CPU).
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=mae_scorer,
        cv=cv_strategy,
        n_jobs=n_jobs,
        verbose=10,
        random_state=random_state,
        error_score="raise",
    )

    # 5. Hyperparameter Tuning
    print("\n====================================================")
    print("STEP 4: Hyperparameter Tuning")
    print("----------------------------------------------------")
    search.fit(X_full, y_full)

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

    # Save Feature Importance
    fi_dest = str(timestamped_path("artifacts/feature_importance", f"fi_{model_type}", ".csv"))
    save_feature_importance(best_model, fi_dest)

    if model_path: # Always save if path provided, or if flag is set (handled by caller)
        dest = model_path or str(timestamped_path("artifacts/models", "best_model", ".joblib"))
        save_best_model(best_model, dest)
    else:
        # Default save
        dest = str(timestamped_path("artifacts/models", f"best_model_{model_type}", ".joblib"))
        save_best_model(best_model, dest)

    # 6. Submission
    if make_submission:
        print("\n====================================================")
        print("STEP 6: Final Training & Submission")
        print("----------------------------------------------------")
        print("  -> Predicting on Test Set...")
        test_preds = best_model.predict(X_test)
        
        submission_dest = submission_path or str(timestamped_path(
            "artifacts/submissions", f"submission_{model_type}", ".csv"
        ))
        write_submission_file(test_ids, test_preds, submission_dest, id_column)

    print("\n====================================================")
    print(f"DONE. Total Runtime: {time.time() - t0:.1f}s")
    print("====================================================")

# -------------------------------------------------------------------
# CLI Argument Parsing
# -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Tree-based Car Price Models."
    )
    
    # Data Paths (Defaults assume running from Tree-HPC and data is in ../data)
    parser.add_argument("--train-path", type=str, default="../data/train.csv", help="Path to training data.")
    parser.add_argument("--test-path", type=str, default="../data/test.csv", help="Path to test data.")
    parser.add_argument("--mapping-dir", type=str, default="../mapping", help="Directory containing JSON mappings.")
    
    # Training Configuration
    parser.add_argument("--model-type", type=str, default="rf", choices=["rf", "et", "hgb", "dt"],
                        help="Model architecture: 'rf' (RandomForest), 'et' (ExtraTrees), 'hgb' (HistGradientBoosting), 'dt' (DecisionTree).")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--n-iter", type=int, default=10, help="Number of random search iterations.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    
    # Output Options
    parser.add_argument("--make-submission", action="store_true", help="Generate submission file.")
    parser.add_argument("--submission-path", type=str, default=None, help="Explicit path for submission file.")
    parser.add_argument("--save-model-path", type=str, default=None, help="Explicit path for saving the model.")
    parser.add_argument("--save-cv-summary", action="store_true", help="Save a text report of CV results.")
    parser.add_argument("--cv-summary-path", type=str, default=None, help="Explicit path for CV report.")
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
        n_iter=args.n_iter,
        model_type=args.model_type,
    )
