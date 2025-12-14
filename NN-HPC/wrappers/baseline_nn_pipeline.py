"""
Pipeline Factory for Neural Network Models.

This module constructs Scikit-Learn compatible pipelines for various architectures:
1. Keras MLP (Multi-Layer Perceptron)
2. PyTorch FT-Transformer (Feature Tokenizer Transformer)

The pipelines generally follow this structure:
[Imputation] -> [Feature Engineering] -> [Encoding/Scaling] -> [Feature Selection] -> [Model]
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, RobustScaler
from sklearn.compose import ColumnTransformer, make_column_selector

# Use imblearn Pipeline to support resampling if needed in future
try:
    from imblearn.pipeline import Pipeline
except ImportError:
    from sklearn.pipeline import Pipeline
    print("[WARN] imbalanced-learn not found. Oversampling will be skipped.")

from utils.preprocessing import get_feature_types
from utils.imputation import get_imputation_pipeline, get_encoding_transformer, get_scaler, drop_interval_cols
from utils.feature_engineering import FeatureEngineeringTransformer, load_mappings
from wrappers.keras_nn import keras_nn
from wrappers.pytorch_wrapper import SklearnFTTransformer


def _to_dense(X):
    """Convert sparse matrices to dense arrays (required for Keras/PyTorch)."""
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
    """Helper to log missing values during debugging."""
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


def build_flexible_keras_pipeline(
    X_train, random_state: int = 42, debug_imputation: bool = False
):
    """
    Constructs a pipeline for the Keras MLP model.
    
    Structure:
    1. Imputation (Median/Most Frequent)
    2. Feature Engineering (Ratios, Flags)
    3. Encoding (OneHot) & Scaling (RobustScaler)
    4. Feature Selection (Placeholder, tuned later)
    5. Keras Regressor
    """
    numeric_features, categorical_features = get_feature_types(X_train)
    if debug_imputation:
        _log_missing_counts(X_train, numeric_features, "Numeric features before impute")
        _log_missing_counts(X_train, categorical_features, "Categorical features before impute")
    
    brand_map, model_map = load_mappings()

    # Unpack imputation steps to allow flat pipeline structure
    imputation_pipe = get_imputation_pipeline()
    steps = list(imputation_pipe.steps)
    
    steps.append(('feature_eng', FeatureEngineeringTransformer(brand_mapping=brand_map, model_mapping=model_map)))
    
    steps.extend([
        ('encoding', get_encoding_transformer()),
        ('drop_intervals', FunctionTransformer(drop_interval_cols)),
        ('scaling', get_scaler())
    ])

    # Default Keras parameters (will be overridden by GridSearchCV)
    keras_reg = keras_nn(
        hidden_layer_sizes=(128, 64),
        learning_rate=1e-3,
        batch_size=256,
        epochs=50,
    )

    steps.extend([
        ("feature_sel", "passthrough"),
        ("to_dense", FunctionTransformer(_to_dense, accept_sparse=True)),
        ("model", keras_reg),
    ])

    return Pipeline(steps=steps)


def build_flexible_ft_transformer_pipeline(
    X_train, random_state: int = 42, debug_imputation: bool = False
):
    """
    Constructs a pipeline for the PyTorch FT-Transformer.
    
    Structure:
    1. Imputation
    2. Feature Engineering
    3. Preprocessing Split:
       - Numerical -> RobustScaler
       - Categorical -> OrdinalEncoder (Integers for Embeddings)
    4. FT-Transformer Model
    """
    brand_map, model_map = load_mappings()

    imputation_pipe = get_imputation_pipeline()
    steps = list(imputation_pipe.steps)
    
    steps.append(('feature_eng', FeatureEngineeringTransformer(brand_mapping=brand_map, model_mapping=model_map)))
    
    # Explicitly define categorical columns expected after Feature Engineering
    cat_cols = ['Brand', 'model', 'transmission', 'fuelType', 'brand_segment', 'car_segment']
    
    # Use negative indices to identify categorical columns after concatenation
    # (ColumnTransformer appends them in order)
    n_cats = len(cat_cols)
    cat_indices = [i for i in range(-n_cats, 0)] # [-6, -5, ..., -1]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), make_column_selector(dtype_include=np.number)),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
        ],
        verbose_feature_names_out=False
    )
    
    steps.append(('preprocessing', preprocessor))
    
    ft_transformer = SklearnFTTransformer(
        d_model=192,
        n_layers=3,
        n_heads=8,
        dropout=0.1,
        batch_size=256,
        epochs=100,
        cat_indices=cat_indices,
        random_state=random_state
    )
    
    steps.append(('model', ft_transformer))
    
    return Pipeline(steps)



