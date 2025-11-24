````markdown
# NN-HPC Pipeline Overview

This project trains a neural-network regressor for the Kaggle car-price challenge on the HPC cluster. Below is the high-level workflow followed by the purpose of each major folder.

## End-to-End Flow
1. **Data ingestion (`data_loaders/`)** – `load_full_train_and_test` reads raw Kaggle CSVs, applies rule-based cleaning/mappings, keeps the `carID`, and delivers `X_full`, `y_full`, `X_test`, and the IDs.
2. **Feature engineering (`utils/feature_engineering.py`)** – `add_model_engine_rarity_cv` computes the `(model, engineSize)` rarity feature on the full training data and mirrors it to the test set.
3. **Preprocessing pipeline (`wrappers/baseline_nn_pipeline.py`)** – Builds a scikit-learn `Pipeline` with numeric/categorical preprocessing, a placeholder feature-selection step, a sparse-to-dense transformer, and a SciKeras `KerasRegressor`.
4. **Hyperparameter tuning (`main.py`)** – `run_training` wraps the pipeline in `TransformedTargetRegressor`, runs `GridSearchCV` (KFold=5) or an optional two-phase search (RandomizedSearch warm-up + focused GridSearch) with MAE as the scoring metric, and logs the best configuration.
5. **Model persistence (`artifacts/models/`)** – Passing `--save-best-model` (or `--save-model-path`) stores the best estimator as a timestamped `.joblib` file for reuse.
6. **Submission generation (`artifacts/submissions/`)** – With `--make-submission` during training or `--load-model-path` later, predictions are written to timestamped CSVs that retain the original `carID` column.
7. **Reproducible runs (`configs/`, `wrappers/`, `utils/`)** – Configurable feature selectors, neural-net architectures, callbacks, and helper utilities keep experiments parameterized and scriptable.

## Logged Training Steps (`main.py`)
Each SLURM run prints the following numbered stages so you can follow progress in the `.out` log:
- **STEP 1 – Load full train & test**: invokes `load_full_train_and_test`, reports shapes, duplicate counts, and column names.
- **STEP 2 – Add rarity feature**: `add_model_engine_rarity_cv` augments both train and test with `model_engine_freq`.
- **STEP 3 – Build base pipeline**: shows the `TransformedTargetRegressor` structure wrapping preprocess → feature selector → Keras model.
- **STEP 4 – Configure CV & grid**: prints how many feature selectors, hidden-layer configs, dropout/LR choices, and total fits (configs × folds). If you pass `--n-splits 1`, the script automatically switches to a single hold-out split via `ShuffleSplit` (20% validation) so you can do quick sanity checks without full 5-fold CV.
- **STEP 5 – Run GridSearchCV**: MAE-based KFold fitting with SciKeras callbacks; verbose output lists per-config timing.
- **STEP 6 – CV results**: displays best hyperparameters, best mean CV MAE, the chosen feature selector, and a summary of the fitted estimator.
- **Save STEP 6 report**: pass `--save-cv-summary` (or `--cv-summary-path <file>`) to write the same details to `artifacts/cv_reports/` for quick sharing or comparison without scraping the `.out` logs.
- **STEP 6b – Persist model (optional)**: if `--save-best-model`/`--save-model-path` is set, writes the `.joblib` path under `artifacts/models/`.
- **STEP 7 – Final fit & submission (optional)**: refits on all data, predicts the Kaggle test set, and stores a timestamped CSV under `artifacts/submissions/` when `--make-submission` is supplied.

## Repository Layout Highlights
- `Base/` – Abstract interfaces/shared logic (for example `base_model.py`, `base_data_loader.py`).
- `configs/` – Experiment settings so training scripts stay parameter-light.
- `data_loaders/` – Input reading, cleaning, mapping, and splitting helpers.
- `models/` – (Reserved) definitions for model builders beyond the baseline NN.
- `optimizers/` – Factories for optimizers or LR schedules.
- `utils/` – General-purpose helpers (feature engineering, preprocessing, etc.).
- `wrappers/` – Adaptations for scikit-learn/SciKeras pipelines and CLI integration.
- `artifacts/` – Auto-created folders where timestamped models and submissions land.

Inspiration from: https://github.com/Python-templates/sklearn-project-template

## SLURM Workflow Checklist
1. **Submit the job** – run the `sbatch` command below. The SLURM script activates the `.venv`, sets CUDA visibility, and calls `python main.py` with the desired flags (`--save-best-model`, `--make-submission`, etc.). Note the job ID printed by `sbatch`.
2. **Follow live logs** – `tail -f logs/car_price_nn_gpu_<JOBID>.out` streams the stdout produced by `main.py` (dataset shapes, GridSearch progress, MAE results). Use the matching `.err` file if you need stack traces.
3. **Track queue status** – `squeue -u $USER` shows whether the job is pending, running, or finished. Once it disappears from the queue, stop the `tail` command.
4. **Collect artifacts** – after completion, check `artifacts/models/` for timestamped `.joblib` weights (if `--save-best-model` was enabled) and `artifacts/submissions/` for the generated Kaggle CSV. The log also prints the exact paths.
5. **(Optional) Reuse the model** – to regenerate submissions without another SLURM run, launch a lightweight job (or run locally) with `python main.py --load-model-path <joblib> --submission-path <csv>` using the saved artifacts.

## HPC Run Commands
```
sbatch   --account=f202500002hpcvlabistulg   --partition=dev-a100-40   --gpus=1   train_nn_gpu.slurm

tail -f logs/car_price_nn_gpu_686331.out

squeue -u $USER
```

## Two-Phase Search Quickstart

Use `--two-phase-search` when each batch must finish in under ~30 minutes:

1. **Phase 1 (RandomizedSearchCV)** – 2-fold CV, shorter epoch counts (40–80), and ~12 sampled configs explore a broad mix of feature selectors, hidden sizes, dropout, learning rate, batch size, and batch norm. Tune with `--phase1-folds`, `--phase1-iters`, and `--phase1-top-k`.
2. **Phase 2 (GridSearchCV)** – Promotes the top-k configs from Phase 1 and re-evaluates them with the full 5-fold CV plus longer epochs provided via `--phase2-epochs` (defaults to `120 150`).
3. **Outputs & artifacts** – Promoted configs are logged before Phase 2 starts, and standard flags (`--save-best-model`, `--make-submission`) continue to work without changes. By default `--phase1-n-jobs 1` keeps the warm-up sequential so TensorFlow builds remain stable; increase only after verifying your environment can handle threaded RandomizedSearch. Use `--phase1-save-grid` (or `--phase1-grid-out <path>`) to persist the promoted grid under `artifacts/phase1_grids`, then later skip Phase 1 altogether via `--phase1-grid-path <joblib>` when you only need to rerun the full GridSearch.

The SLURM script already passes the staged-search flags, so you can submit immediately or adjust the CLI knobs for even quicker sweeps.
````