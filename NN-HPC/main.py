"""
Main script to train the NN price model with:

- rule-based cleaning + mappings
- model/engine rarity feature
- preprocessing (impute + scale + one-hot)
- feature selection (filter / wrapper / embedded)
- MLPRegressor
- log(price) target via TransformedTargetRegressor
- K-Fold (K=5) cross-validation via GridSearchCV
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV, KFold

from data_loaders.cars_data import load_full_train_and_test
from utils.feature_engineering import add_model_engine_rarity_cv
from wrappers.baseline_nn_pipeline import build_flexible_nn_pipeline
from wrappers.baseline_nn_pipeline import build_flexible_keras_pipeline

from utils.feature_selection import candidate_selectors


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def rmse_func(y_true, y_pred):
    """Compute RMSE (since this sklearn version has no squared=False)."""
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


def build_param_grid(random_state: int = 42):

    selectors = candidate_selectors(random_state=random_state)

    param_grid = [
        {
            # Feature-Selection-Step IM Pipeline-Schritt "feature_sel"
            "regressor__feature_sel": list(selectors.values()),

            # Keras NN hyperparams IM Pipeline-Schritt "model"
            "regressor__model__hidden_layer_sizes": [(128, 64)],
            "regressor__model__learning_rate": [1e-3],
            "regressor__model__batch_size": [256],
            "regressor__model__epochs": [40],
        }
    ]
    return param_grid




# -------------------------------------------------------------------
# Main training routine
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
):
    t0 = time.time()
    print("====================================================")
    print("STEP 1: Load full train & test (rule-based cleaning only)")
    print("----------------------------------------------------")
    X_full, y_full, X_test = load_full_train_and_test(
        train_path=train_path,
        test_path=test_path,
        mapping_dir=mapping_dir,
    )
    print(f"  -> X_full shape: {X_full.shape}")
    print(f"  -> y_full shape: {y_full.shape}")
    print(f"  -> X_test shape: {X_test.shape}")
    print(f"  -> Columns: {list(X_full.columns)}")

    print("\n====================================================")
    print("STEP 2: Add (model, engineSize) rarity feature")
    print("----------------------------------------------------")
    X_full, X_test = add_model_engine_rarity_cv(
        X_full,
        X_test,
        model_col="model",
        engine_col="engineSize",
        new_col="model_engine_freq",
        log_scale=True,
    )
    print("  -> Added column 'model_engine_freq'.")
    print("  -> Example of new column (first 5 rows):")
    print(X_full["model_engine_freq"].head())

    print("\n====================================================")
    print("STEP 3: Build base pipeline [preprocess -> feature_sel -> MLP]")
    print("----------------------------------------------------")
    base_pipe = build_flexible_keras_pipeline(X_full, random_state=random_state)
    log_pipe = TransformedTargetRegressor(
        regressor=base_pipe,
        func=np.log1p,
        inverse_func=np.expm1,
    )
    print("  -> Pipeline structure:")
    print("     TransformedTargetRegressor(")
    print("       regressor=Pipeline(steps=[")
    print("           ('preprocess', ColumnTransformer(...)),")
    print("           ('feature_sel', <to be tuned>),")
    print("           ('model', MLPRegressor(...))")
    print("       ]),")
    print("       func=log1p, inverse_func=expm1")
    print("     )")

    print("\n====================================================")
    print(f"STEP 4: Set up {n_splits}-Fold CV + GridSearchCV with feature selection")
    print("----------------------------------------------------")
    rmse_scorer = make_scorer(rmse_func, greater_is_better=False)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    param_grid = build_param_grid(random_state=random_state)

    # Calculate approximate number of configs
    n_fs = len(param_grid[0]["regressor__feature_sel"])

    total_configs = n_fs


    print(f"  -> Feature selection candidates: {n_fs}")
    print(f"  -> Total hyperparameter configs: {total_configs}")
    print(f"  -> KFold splits:                 {n_splits}")
    print(f"  -> Total model fits (approx):    {total_configs * n_splits}")

    search = GridSearchCV(
        estimator=log_pipe,
        param_grid=param_grid,
        scoring=rmse_scorer,
        cv=kf,
        n_jobs=1,
        verbose=2,  # GridSearchCV internal progress
        error_score="raise",
    )

    print("\n====================================================")
    print("STEP 5: Run GridSearchCV (this may take a while)")
    print("----------------------------------------------------")
    t_fit_start = time.time()
    search.fit(X_full, y_full)
    t_fit_end = time.time()

    print("\n====================================================")
    print("STEP 6: Results of CV")
    print("----------------------------------------------------")
    print("Best parameters (including feature selection):")
    print(search.best_params_)

    # best_score_ is negative RMSE because greater_is_better=False
    best_rmse_cv = -search.best_score_
    print(f"\nBest mean CV RMSE: {best_rmse_cv:.3f}")

    # Extract which feature selector was chosen
    best_fs = search.best_params_.get("regressor__feature_sel", None)
    print(f"Chosen feature selection object:\n  {best_fs}")


    best_model = search.best_estimator_
    print("\nBest estimator (pipeline + log-transform):")
    print(best_model)

    # Optional: quick CV-like performance on the same data as a sanity check
    y_pred_full = best_model.predict(X_full)
    mse_full = mean_squared_error(y_full, y_pred_full)
    rmse_full = np.sqrt(mse_full)
    r2_full = r2_score(y_full, y_pred_full)
    print("\nPerformance on full training data (not a proper test, just sanity):")
    print(f"  RMSE (full train): {rmse_full:.3f}")
    print(f"  R²   (full train): {r2_full:.3f}")

    # ----------------------------------------------------------------
    # Optional: Train final model on all labeled data and predict test
    # ----------------------------------------------------------------
    if make_submission:
        print("\n====================================================")
        print("STEP 7: Fit best model on ALL training data & predict test")
        print("----------------------------------------------------")
        # It is already fitted on X_full, y_full by GridSearchCV.best_estimator_
        # but we can refit explicitly to be clear:
        best_model.fit(X_full, y_full)

        print("  -> Predicting on X_test...")
        test_preds = best_model.predict(X_test)

        if id_column is not None and id_column in X_test.columns:
            ids = X_test[id_column]
        else:
            # If the id column was dropped earlier, you might need to load it separately.
            # For now, we just use a simple range index.
            ids = np.arange(len(X_test))

        submission = pd.DataFrame({
            "id": ids,
            "price": test_preds,
        })

        out_path = submission_path or "submission_kfold_fs_nn.csv"
        submission.to_csv(out_path, index=False)
        print(f"  -> Saved submission to: {out_path}")

    t1 = time.time()
    print("\n====================================================")
    print("DONE.")
    print("----------------------------------------------------")
    print(f"Total runtime: {t1 - t0:.1f} seconds")
    print(f"  - GridSearchCV fit time: {t_fit_end - t_fit_start:.1f} seconds")
    print("====================================================")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train NN car price model with K-Fold CV and feature selection."
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default="data/train.csv",
        help="Path to Kaggle train.csv",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="data/test.csv",
        help="Path to Kaggle test.csv",
    )
    parser.add_argument(
        "--mapping-dir",
        type=str,
        default="mapping",
        help="Directory with JSON mapping files.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds for K-Fold CV.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for CV and models.",
    )
    parser.add_argument(
        "--make-submission",
        action="store_true",
        help="If set, train final model on all data and write a submission CSV.",
    )
    parser.add_argument(
        "--submission-path",
        type=str,
        default="submission_kfold_fs_nn.csv",
        help="Output path for the submission CSV.",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default=None,
        help="Name of ID column in test data (if it still exists).",
    )
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
    )
