# data_loaders/cars_data.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import json
import re

import numpy as np
import pandas as pd
import pandas.api.types as ptypes

from sklearn.model_selection import train_test_split
from utils.preprocessing import clean_dataset

TARGET_COL = "price"   # adjust if your target col is named differently
ID_COL = "carID"       # Our ID column is carID
RANDOM_STATE = 42



def make_sklearn_friendly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert pandas nullable/extension dtypes into plain NumPy dtypes
    that scikit-learn can handle:

      - Nullable ints/floats/bools -> float64
      - String/other extension types -> object
      - Replace pd.NA with np.nan

    No statistics are used -> no leakage, just technical conversion.
    """
    df = df.copy()

    for col in df.columns:
        dtype = df[col].dtype

        # 1) Extension dtypes (Int64, Float64, boolean, string, etc.)
        if ptypes.is_extension_array_dtype(dtype):
            # Numeric-like extension → convert to float64
            if ptypes.is_integer_dtype(dtype) or ptypes.is_float_dtype(dtype) or ptypes.is_bool_dtype(dtype):
                df[col] = df[col].astype("float64")
            else:
                # string[python], StringDtype, etc. → plain object
                df[col] = df[col].astype("object")

        # 2) Just in case something is still "string" but not extension
        elif ptypes.is_string_dtype(dtype):
            df[col] = df[col].astype("object")

    # 3) Now safely replace any remaining pd.NA with np.nan
    df = df.replace({pd.NA: np.nan})

    return df



# -------------------------------------------------------------------
# Full rule-based cleaning for both train + test (no target leakage)
# -------------------------------------------------------------------
def rule_based_cleaning_for_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    mapping_dir: str | Path = "mapping",
    debug: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply ALL your rule-based cleaning steps from the notebook to train & test.
    No target, no learned statistics → safe to apply before splitting.
    """
    # Use the centralized cleaning function from utils.preprocessing
    # Train: drop outliers (year < 2000, etc.)
    train_clean = clean_dataset(train, mapping_dir=mapping_dir, drop_outliers=True)
    
    # Test: DO NOT drop outliers (keep rows, set to NaN)
    test_clean = clean_dataset(test, mapping_dir=mapping_dir, drop_outliers=False)

    return train_clean, test_clean




# -------------------------------------------------------------------
# Full train + test loading (no split)
# -------------------------------------------------------------------
def load_full_train_and_test(
    train_path: str,
    test_path: str,
    mapping_dir: str | Path = "mapping",
    return_test_ids: bool = False,
    return_train_ids: bool = False,
    debug_cleaning: bool = False,
):
    """
    Load Kaggle train + test, apply rule-based cleaning (no imputation, no scaling),
    make them sklearn-friendly, and return:

        X_full, y_full, X_test

    where X_full is the ENTIRE labeled training set (no split).
    """
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    train_clean, test_clean = rule_based_cleaning_for_train_test(
        train_raw,
        test_raw,
        mapping_dir=mapping_dir,
        debug=debug_cleaning,
    )
    

    # convert nullable dtypes/pd.NA -> numpy-friendly
    train_clean = make_sklearn_friendly(train_clean)
    test_clean = make_sklearn_friendly(test_clean)

    if TARGET_COL not in train_clean.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in train data.")

    y_full = train_clean[TARGET_COL]

    train_ids = None
    if ID_COL in train_clean.columns:
        train_ids = train_clean[ID_COL].copy()
        X_full = train_clean.drop(columns=[TARGET_COL, ID_COL])
    else:
        X_full = train_clean.drop(columns=[TARGET_COL])
        if return_train_ids:
            if train_clean.index.name == ID_COL:
                train_ids = train_clean.index.to_series().copy()
            else:
                train_ids = pd.Series(np.arange(len(train_clean)), name=ID_COL)

    if ID_COL in test_clean.columns:
        test_ids = test_clean[ID_COL].copy()
        X_test = test_clean.drop(columns=[ID_COL])
    elif test_clean.index.name == ID_COL:
        test_ids = test_clean.index.to_series().copy()
        X_test = test_clean
    else:
        test_ids = pd.Series(np.arange(len(test_clean)), name=ID_COL)
        X_test = test_clean

    if return_test_ids and return_train_ids:
        return X_full, y_full, X_test, test_ids, train_ids
    if return_test_ids:
        return X_full, y_full, X_test, test_ids
    if return_train_ids:
        return X_full, y_full, X_test, train_ids
    return X_full, y_full, X_test





# -------------------------------------------------------------------
# Public function used by the NN project
# -------------------------------------------------------------------
def load_train_val_test(
    train_path: str,
    test_path: str,
    mapping_dir: str | Path = "../mapping",
    val_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    """
    1) Load raw Kaggle train + test CSV
    2) Apply rule-based cleaning to BOTH (no target leakage)
    3) Split cleaned train into (X_train, X_val, y_train, y_val)
    4) Return also cleaned Kaggle test (X_test_clean)

    Returns:
        X_train, y_train, X_val, y_val, X_test
    """
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    train_clean, test_clean = rule_based_cleaning_for_train_test(
        train_raw, test_raw, mapping_dir=mapping_dir
    )

    # Make data safe for scikit-learn (no pd.NA, no extension dtypes)
    train_clean = make_sklearn_friendly(train_clean)
    test_clean = make_sklearn_friendly(test_clean)

    #debugging print(train_clean.info())
    print(f"Train shape after cleaning: {train_clean.shape}")
    print(f"Test shape after cleaning: {test_clean.shape}")

    if TARGET_COL not in train_clean.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in train data.")

    y = train_clean[TARGET_COL]
    X = train_clean.drop(columns=[TARGET_COL])

    # Drop ID column from features (keep it in test for later if you need it)
    if ID_COL in X.columns:
        X = X.drop(columns=[ID_COL])
    if ID_COL in test_clean.columns:
        test_features = test_clean.drop(columns=[ID_COL])
    else:
        test_features = test_clean

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state
    )

    return X_train, y_train, X_val, y_val, test_features