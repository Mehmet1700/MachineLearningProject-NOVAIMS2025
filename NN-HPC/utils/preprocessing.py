# utils/preprocessing.py
from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def get_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Infer numeric vs categorical features from dtypes.
    Only uses types (no statistics) → no leakage.
    """
    numeric_features = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = df.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()
    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    ColumnTransformer:
      - numeric: median imputation + standard scaling
      - categorical: most_frequent imputation + one-hot encoding
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor
