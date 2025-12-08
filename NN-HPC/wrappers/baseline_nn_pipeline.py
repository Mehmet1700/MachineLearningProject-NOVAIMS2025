# wrappers/baseline_nn_pipeline.py
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import FunctionTransformer
from utils.preprocessing import get_feature_types, build_preprocessor
from wrappers.keras_nn import keras_nn


def _to_dense(X):
    """Convert sparse matrices to dense arrays for Keras."""
    if hasattr(X, "toarray"):
        X = X.toarray()
    elif isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    elif isinstance(X, pd.Series):
        X = X.to_frame().to_numpy()
    else:
        X = np.asarray(X)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def _log_missing_counts(df: pd.DataFrame, columns, label: str, top_n: int = 20):
    if not columns:
        print(f"[IMPUTE] {label}: no columns")
        return
    missing_series = df[columns].isna().sum().sort_values(ascending=False)
    total = len(df)
    missing_series = missing_series[missing_series > 0]
    if missing_series.empty:
        print(f"[IMPUTE] {label}: no missing values (n={total})")
        return
    print(f"[IMPUTE] {label}: top missing counts out of n={total}")
    for col, cnt in missing_series.head(top_n).items():
        pct = (cnt / total) * 100 if total else 0
        print(f"           {col}: {cnt} ({pct:.2f}% )")


def build_flexible_nn_pipeline(
    X_train, random_state: int = 42, debug_imputation: bool = False
):
    """
    Pipeline with a placeholder feature selection step:

        [preprocess] -> [feature_sel] -> [model]

    'feature_sel' will be replaced in GridSearch / CV.
    """
    numeric_features, categorical_features = get_feature_types(X_train)
    if debug_imputation:
        _log_missing_counts(X_train, numeric_features, "Numeric features before impute")
        _log_missing_counts(
            X_train, categorical_features, "Categorical features before impute"
        )
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    nn_regressor = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=100,
        random_state=random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("feature_sel", "passthrough"),  # to be swapped
            ("model", nn_regressor),
        ]
    )

    return pipeline



def build_flexible_keras_pipeline(
    X_train, random_state: int = 42, debug_imputation: bool = False
):
    """
    Pipeline for Keras-based NN:

        [preprocess] -> [feature_sel] -> [KerasRegressor]
    """
    numeric_features, categorical_features = get_feature_types(X_train)
    if debug_imputation:
        _log_missing_counts(X_train, numeric_features, "Numeric features before impute")
        _log_missing_counts(
            X_train, categorical_features, "Categorical features before impute"
        )
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    keras_reg = keras_nn(
        hidden_layer_sizes=(128, 64),
        learning_rate=1e-3,
        batch_size=256,
        epochs=50,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("feature_sel", "passthrough"),
            ("to_dense", FunctionTransformer(_to_dense, accept_sparse=True)),
            ("model", keras_reg),
        ]
    )

    return pipeline

