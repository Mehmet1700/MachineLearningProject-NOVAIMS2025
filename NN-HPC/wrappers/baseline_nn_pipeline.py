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


def build_flexible_nn_pipeline(X_train, random_state: int = 42):
    """
    Pipeline with a placeholder feature selection step:

        [preprocess] -> [feature_sel] -> [model]

    'feature_sel' will be replaced in GridSearch / CV.
    """
    numeric_features, categorical_features = get_feature_types(X_train)
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



def build_flexible_keras_pipeline(X_train, random_state: int = 42):
    """
    Pipeline for Keras-based NN:

        [preprocess] -> [feature_sel] -> [KerasRegressor]
    """
    numeric_features, categorical_features = get_feature_types(X_train)
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

