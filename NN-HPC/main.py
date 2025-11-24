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
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load, parallel_backend
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold

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


def mae_func(y_true, y_pred):
    """Compute MAE for Kaggle submission metric."""
    return mean_absolute_error(y_true, y_pred)


def timestamped_path(directory: str | Path, prefix: str, suffix: str) -> Path:
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


def load_saved_model(path: str):
    """Load a previously persisted estimator."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found at {model_path}")
    print(f"Loading model from {model_path} ...")
    return load(model_path)


def save_phase1_grid(param_grid, path: str):
    """Persist the promoted Phase 2 grid for reuse."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    dump(param_grid, target_path)
    print(f"  -> Saved Phase 1 grid to: {target_path}")


def load_phase1_grid(path: str):
    grid_path = Path(path)
    if not grid_path.exists():
        raise FileNotFoundError(f"Phase 1 grid not found at {grid_path}")
    print(f"  -> Loading Phase 1 grid from {grid_path}")
    return load(grid_path)


def write_submission_file(test_ids, predictions, submission_path: str, id_column: str | None):
    """Create the Kaggle submission CSV from predictions."""
    column_name = id_column or getattr(test_ids, "name", "carID") or "carID"
    ids = pd.Series(test_ids, name=column_name).reset_index(drop=True)

    prices = pd.Series(predictions, name="price").reset_index(drop=True)

    submission = pd.DataFrame({
        column_name: ids,
        "price": prices,
    })

    out_path = Path(submission_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(f"  -> Saved submission to: {out_path}")


def build_param_grid(random_state: int = 42):

    selectors = candidate_selectors(random_state=random_state)

    param_grid = [
        {
            "regressor__feature_sel": [selectors["filter_freg_k100"]],
            "regressor__model__hidden_layer_sizes": [
                (256, 128, 64),
            ],
            "regressor__model__learning_rate": [1e-3, 5e-4],
            "regressor__model__dropout_rate": [0.0, 0.2],
            "regressor__model__use_batchnorm": [True],
            "regressor__model__batch_size": [512],
            "regressor__model__epochs": [120],
        },
        {
            "regressor__feature_sel": [selectors["lasso_embedded"]],
            "regressor__model__hidden_layer_sizes": [
                (256, 128, 64, 32),
            ],
            "regressor__model__learning_rate": [5e-4],
            "regressor__model__dropout_rate": [0.2],
            "regressor__model__use_batchnorm": [True],
            "regressor__model__batch_size": [512],
            "regressor__model__epochs": [150],
        },
        {
            "regressor__feature_sel": [selectors["filter_freg_k100"]],
            "regressor__model__hidden_layer_sizes": [
                (512, 256, 128, 64),
            ],
            "regressor__model__learning_rate": [7e-4, 5e-4],
            "regressor__model__dropout_rate": [0.2, 0.3],
            "regressor__model__use_batchnorm": [True],
            "regressor__model__batch_size": [512],
            "regressor__model__epochs": [150],
        },
    ]
    return param_grid


def build_phase_one_search_space(random_state: int = 42):
    """Parameter distributions for the warm-up RandomizedSearch."""
    selectors = candidate_selectors(random_state=random_state)
    hidden_layer_options = [
        (512, 256, 128),
        (384, 192, 96),
        (256, 128, 64),
        (256, 128, 64, 32),
        (384, 192, 96, 48),
    ]
    return {
        "regressor__feature_sel": [
            selectors["filter_freg_k100"],
            selectors["lasso_embedded"],
            selectors["none"],
        ],
        "regressor__model__hidden_layer_sizes": hidden_layer_options,
        "regressor__model__learning_rate": [1e-3, 7e-4, 5e-4, 3e-4, 1e-4],
        "regressor__model__dropout_rate": [0.0, 0.1, 0.2, 0.3],
        "regressor__model__use_batchnorm": [True, False],
        "regressor__model__batch_size": [512, 768],
        "regressor__model__epochs": [40, 60, 80],
    }


def _count_param_grid_combinations(param_grid):
    """Estimate number of concrete configs + feature selectors in a grid."""
    if isinstance(param_grid, dict):
        grid_list = [param_grid]
    else:
        grid_list = list(param_grid)

    total_configs = 0
    feature_selectors = set()

    for grid in grid_list:
        count = 1
        for key, values in grid.items():
            if isinstance(values, (list, tuple, np.ndarray)):
                options = list(values)
            else:
                options = [values]

            count *= max(len(options), 1)

            if key == "regressor__feature_sel":
                feature_selectors.update(options)

        total_configs += count

    return total_configs, len(feature_selectors)


def _extract_top_param_dicts(cv_results: dict, top_k: int):
    if not cv_results or top_k <= 0:
        return []
    df = pd.DataFrame(cv_results)
    if df.empty:
        return []
    df = df.sort_values("rank_test_score")
    param_cols = [c for c in df.columns if c.startswith("param_")]
    top_param_dicts = []
    for _, row in df.head(top_k).iterrows():
        config = {}
        for col in param_cols:
            param_name = col.replace("param_", "")
            value = row[col]
            if pd.isna(value):
                continue
            config[param_name] = value
        top_param_dicts.append(config)
    return top_param_dicts


def _fit_with_optional_threading(search, X, y, n_jobs: int):
    backend = "threading" if n_jobs != 1 else None
    if backend:
        with parallel_backend(backend, n_jobs=n_jobs):
            search.fit(X, y)
    else:
        search.fit(X, y)


def _wrap_param_value(value):
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return [value]
    if isinstance(value, np.ndarray):
        return list(value)
    return [value]


def _normalize_param_grid_entry(config: dict):
    normalized = {}
    for key, value in config.items():
        normalized[key] = _wrap_param_value(value)
    return normalized


def prepare_two_phase_param_grid(
    X,
    y,
    estimator,
    scoring,
    random_state,
    n_jobs,
    phase1_n_jobs,
    phase1_n_splits,
    phase1_n_iter,
    phase1_top_k,
    phase2_epoch_options,
):
    print(f"  -> Phase 1: RandomizedSearchCV warm-up (n_jobs={phase1_n_jobs})")
    warmup_cv = KFold(
        n_splits=phase1_n_splits,
        shuffle=True,
        random_state=random_state,
    )
    param_distributions = build_phase_one_search_space(random_state=random_state)
    warmup_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=phase1_n_iter,
        scoring=scoring,
        cv=warmup_cv,
        n_jobs=phase1_n_jobs,
        random_state=random_state,
        verbose=2,
        error_score="raise",
    )

    t_warm_start = time.time()
    _fit_with_optional_threading(warmup_search, X, y, phase1_n_jobs)
    t_warm_end = time.time()

    best_phase1_mae = -warmup_search.best_score_
    print(f"    Phase 1 best mean MAE: {best_phase1_mae:.3f}")
    print(f"    Phase 1 duration: {t_warm_end - t_warm_start:.1f}s")

    top_configs = _extract_top_param_dicts(
        warmup_search.cv_results_, top_k=phase1_top_k
    )
    if not top_configs:
        raise RuntimeError("Phase 1 search returned no valid configurations.")

    print("  -> Promoting top configurations to Phase 2:")
    for idx, cfg in enumerate(top_configs, start=1):
        lr = cfg.get("regressor__model__learning_rate")
        hl = cfg.get("regressor__model__hidden_layer_sizes")
        fs = type(cfg.get("regressor__feature_sel")).__name__
        print(f"     {idx}) feature_sel={fs}, hl={hl}, lr={lr}")

    epoch_options = phase2_epoch_options or [120, 150]
    phase2_grid = []
    for cfg in top_configs:
        base_cfg = _normalize_param_grid_entry(cfg)
        for epoch in epoch_options:
            refined = {k: list(v) for k, v in base_cfg.items()}
            refined["regressor__model__epochs"] = [epoch]
            phase2_grid.append(refined)

    return phase2_grid




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
    model_path: str | None = None,
    n_jobs: int = 8,
    two_phase_search: bool = False,
    phase1_n_splits: int = 2,
    phase1_n_iter: int = 12,
    phase1_top_k: int = 3,
    phase1_n_jobs: int = 1,
    phase2_epochs: list[int] | None = None,
    phase1_grid_path: str | None = None,
    phase1_grid_out: str | None = None,
    phase1_save_grid: bool = False,
):
    t0 = time.time()
    print("====================================================")
    print("STEP 1: Load full train & test (rule-based cleaning only)")
    print("----------------------------------------------------")
    X_full, y_full, X_test, test_ids = load_full_train_and_test(
        train_path=train_path,
        test_path=test_path,
        mapping_dir=mapping_dir,
        return_test_ids=True,
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
    mae_scorer = make_scorer(mae_func, greater_is_better=False)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    phase1_output_path = phase1_grid_out
    if phase1_output_path is None and phase1_save_grid:
        phase1_output_path = str(
            timestamped_path("artifacts/phase1_grids", "phase1_grid", ".joblib")
        )

    if two_phase_search:
        if phase1_grid_path:
            print("  -> Two-phase search enabled (loading saved Phase 1 grid)")
            param_grid = load_phase1_grid(phase1_grid_path)
            effective_phase2_epochs = None
        else:
            print(
                f"  -> Two-phase search enabled (warm-up folds={phase1_n_splits}, "
                f"iters={phase1_n_iter}, top_k={phase1_top_k})"
            )
            effective_phase2_epochs = phase2_epochs or [120, 150]
            param_grid = prepare_two_phase_param_grid(
                X_full,
                y_full,
                estimator=log_pipe,
                scoring=mae_scorer,
                random_state=random_state,
                n_jobs=n_jobs,
                phase1_n_jobs=phase1_n_jobs,
                phase1_n_splits=phase1_n_splits,
                phase1_n_iter=phase1_n_iter,
                phase1_top_k=phase1_top_k,
                phase2_epoch_options=effective_phase2_epochs,
            )
            print(f"  -> Phase 2 epoch candidates:     {effective_phase2_epochs}")
            if phase1_output_path:
                save_phase1_grid(param_grid, phase1_output_path)
    else:
        param_grid = build_param_grid(random_state=random_state)
        effective_phase2_epochs = None

    total_configs, n_fs = _count_param_grid_combinations(param_grid)

    print(f"  -> Feature selection candidates: {n_fs}")
    print(f"  -> Total hyperparameter configs: {total_configs}")
    print(f"  -> KFold splits:                 {n_splits}")
    print(f"  -> Total model fits (approx):    {total_configs * n_splits}")

    search = GridSearchCV(
        estimator=log_pipe,
        param_grid=param_grid,
        scoring=mae_scorer,
        cv=kf,
        n_jobs=n_jobs,
        verbose=2,  # GridSearchCV internal progress
        error_score="raise",
    )

    print("\n====================================================")
    print("STEP 5: Run GridSearchCV (this may take a while)")
    print("----------------------------------------------------")
    t_fit_start = time.time()
    _fit_with_optional_threading(search, X_full, y_full, n_jobs)
    t_fit_end = time.time()

    print("\n====================================================")
    print("STEP 6: Results of CV")
    print("----------------------------------------------------")
    print("Best parameters (including feature selection):")
    print(search.best_params_)

    # best_score_ is negative MAE because greater_is_better=False
    best_mae_cv = -search.best_score_
    print(f"\nBest mean CV MAE: {best_mae_cv:.3f}")

    # Extract which feature selector was chosen
    best_fs = search.best_params_.get("regressor__feature_sel", None)
    print(f"Chosen feature selection object:\n  {best_fs}")


    best_model = search.best_estimator_
    print("\nBest estimator (pipeline + log-transform):")
    print(best_model)

    if model_path is not None:
        print("\n====================================================")
        print("STEP 6b: Persist best estimator")
        print("----------------------------------------------------")
        save_best_model(best_model, model_path)

    # Optional: quick CV-like performance on the same data as a sanity check
    y_pred_full = best_model.predict(X_full)
    mse_full = mean_squared_error(y_full, y_pred_full)
    rmse_full = np.sqrt(mse_full)
    mae_full = mean_absolute_error(y_full, y_pred_full)
    r2_full = r2_score(y_full, y_pred_full)
    print("\nPerformance on full training data (not a proper test, just sanity):")
    print(f"  MAE (full train): {mae_full:.3f}")
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
        submission_dest = submission_path or timestamped_path(
            "artifacts/submissions", "submission_kfold_fs_nn", ".csv"
        )
        write_submission_file(test_ids, test_preds, submission_dest, id_column)

    t1 = time.time()
    print("\n====================================================")
    print("DONE.")
    print("----------------------------------------------------")
    print(f"Total runtime: {t1 - t0:.1f} seconds")
    print(f"  - GridSearchCV fit time: {t_fit_end - t_fit_start:.1f} seconds")
    print("====================================================")


def generate_submission_from_saved_model(
    model_path: str,
    train_path: str,
    test_path: str,
    mapping_dir: str,
    submission_path: str | None = None,
    id_column: str | None = None,
):
    """Load a persisted estimator and produce a Kaggle-ready submission."""
    print("====================================================")
    print("INFERENCE MODE: Load saved model and predict test set")
    print("----------------------------------------------------")

    X_full, _, X_test, test_ids = load_full_train_and_test(
        train_path=train_path,
        test_path=test_path,
        mapping_dir=mapping_dir,
        return_test_ids=True,
    )
    _, X_test = add_model_engine_rarity_cv(
        X_full,
        X_test,
        model_col="model",
        engine_col="engineSize",
        new_col="model_engine_freq",
        log_scale=True,
    )

    model = load_saved_model(model_path)

    print("Predicting on X_test with loaded model ...")
    preds = model.predict(X_test)

    submission_dest = submission_path or timestamped_path(
        "artifacts/submissions", "submission_saved_model", ".csv"
    )
    write_submission_file(test_ids, preds, submission_dest, id_column)

    print("====================================================")
    print("DONE (inference mode)")
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
        default=None,
        help="Optional explicit output path for the submission CSV. Defaults to a timestamped file in artifacts/submissions.",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="carID",
        help="Name of ID column in test data (if it still exists).",
    )
    parser.add_argument(
        "--save-model-path",
        type=str,
        default=None,
        help="If set, persist the best fitted estimator to this path (.joblib).",
    )
    parser.add_argument(
        "--save-best-model",
        action="store_true",
        help="Persist the best estimator using a timestamped path under artifacts/models (overridden by --save-model-path).",
    )
    parser.add_argument(
        "--load-model-path",
        type=str,
        default=None,
        help="If set, skip training and load an existing estimator from this path to create a submission.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="Number of parallel jobs for GridSearchCV (use 1 to disable threading backend).",
    )
    parser.add_argument(
        "--two-phase-search",
        action="store_true",
        help="Run a warm-up RandomizedSearch (few folds) before the full GridSearch to keep runtime under control.",
    )
    parser.add_argument(
        "--phase1-folds",
        type=int,
        default=2,
        help="Number of folds for the Phase 1 RandomizedSearch warm-up.",
    )
    parser.add_argument(
        "--phase1-iters",
        type=int,
        default=12,
        help="Number of sampled configurations during Phase 1 RandomizedSearchCV.",
    )
    parser.add_argument(
        "--phase1-top-k",
        type=int,
        default=3,
        help="How many top configs from Phase 1 move on to the full GridSearch.",
    )
    parser.add_argument(
        "--phase1-n-jobs",
        type=int,
        default=1,
        help="Parallel jobs to use during the warm-up RandomizedSearch (set >1 only if TensorFlow stability is confirmed).",
    )
    parser.add_argument(
        "--phase2-epochs",
        type=int,
        nargs="+",
        default=None,
        help="Epoch counts evaluated during Phase 2 for every promoted config (defaults to 120 and 150).",
    )
    parser.add_argument(
        "--phase1-grid-path",
        type=str,
        default=None,
        help="If set, skip Phase 1 and load a previously saved parameter grid for the full GridSearch phase.",
    )
    parser.add_argument(
        "--phase1-grid-out",
        type=str,
        default=None,
        help="Optional explicit path to save the promoted Phase 2 grid (joblib).",
    )
    parser.add_argument(
        "--phase1-save-grid",
        action="store_true",
        help="Save the promoted Phase 2 grid to artifacts/phase1_grids with a timestamped name.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_path = args.save_model_path
    if args.save_best_model and model_path is None:
        model_path = str(
            timestamped_path("artifacts/models", "best_model", ".joblib")
        )
    if args.load_model_path:
        generate_submission_from_saved_model(
            model_path=args.load_model_path,
            train_path=args.train_path,
            test_path=args.test_path,
            mapping_dir=args.mapping_dir,
            submission_path=args.submission_path,
            id_column=args.id_column,
        )
    else:
        run_training(
            train_path=args.train_path,
            test_path=args.test_path,
            mapping_dir=args.mapping_dir,
            n_splits=args.n_splits,
            random_state=args.random_state,
            make_submission=args.make_submission,
            submission_path=args.submission_path,
            id_column=args.id_column,
            model_path=model_path,
            n_jobs=args.n_jobs,
            two_phase_search=args.two_phase_search,
            phase1_n_splits=args.phase1_folds,
            phase1_n_iter=args.phase1_iters,
            phase1_top_k=args.phase1_top_k,
            phase1_n_jobs=args.phase1_n_jobs,
            phase2_epochs=args.phase2_epochs,
            phase1_grid_path=args.phase1_grid_path,
            phase1_grid_out=args.phase1_grid_out,
            phase1_save_grid=args.phase1_save_grid,
        )
