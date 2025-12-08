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

TARGET_COL = "price"   # adjust if your target col is named differently
ID_COL = "carID"       # Our ID column is carID
RANDOM_STATE = 42



def _log_unique_values(series: pd.Series, label: str, max_values: int = 40):
    """Pretty-print basic stats and unique values for debugging."""
    nunique = series.nunique(dropna=True)
    na_count = series.isna().sum()
    print(f"[CLEANING] {label}: unique={nunique}, missing={na_count}")
    if nunique == 0:
        return
    if nunique <= max_values:
        unique_vals = series.dropna().unique()
        try:
            sorted_vals = sorted(unique_vals, key=lambda v: str(v))
        except Exception:
            sorted_vals = list(unique_vals)
        print(f"           values={sorted_vals}")
    else:
        sample_vals = series.dropna().unique()[:max_values]
        print(
            f"           showing first {max_values} values: {[str(v) for v in sample_vals]}"
        )




# -------------------------------------------------------------------
# Helper: numeric casting + string cleanup (from your notebook)
# -------------------------------------------------------------------
def to_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").round().astype("Int64")


def to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)

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


def apply_basic_type_casting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same type casting & object/string cleaning you had in the notebook.
    This is safe & rule-based (no target, no statistics).
    """
    df = df.copy()

    nan_report = {}

    num_cols_suggest = [
        "price",
        "mileage",
        "engineSize",
        "mpg",
        "tax",
        "year",
        "previousOwners",
    ]
    for col in num_cols_suggest:
        if col in df.columns:
            before = df[col].isna().sum()
            if col in ["year", "previousOwners"]:
                df[col] = to_int_series(df[col])
            else:
                df[col] = to_float_series(df[col])
            after = df[col].isna().sum()
            added = after - before
            if added > 0:
                nan_report[col] = added

    # object → string cleanup
    for c in df.select_dtypes(include="object").columns:
        before = df[c].isna().sum()
        df[c] = (
            df[c]
            .astype("string")
            .str.strip()
            .replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
        )
        after = df[c].isna().sum()
        added = after - before
        if added > 0:
            nan_report[c] = added

    # (Optional) you can log nan_report if you want
    return df


# -------------------------------------------------------------------
# Transmission mapping
# -------------------------------------------------------------------
def normalize_transmission(value: str):
    if pd.isna(value):
        return pd.NA
    value = str(value).strip().lower()
    value = re.sub(r"[.,_]", " ", value)
    value = " ".join(value.split())
    return value


def apply_transmission_mapping(
    df: pd.DataFrame,
    mapping_path: Path,
    col: str = "transmission",
) -> pd.DataFrame:
    df = df.copy()
    if col not in df.columns:
        return df

    with mapping_path.open("r", encoding="utf-8") as f:
        raw_map = json.load(f)

    trans_canon = {normalize_transmission(k): v for k, v in raw_map.items()}

    norm_col = f"{col}_norm"
    df[norm_col] = df[col].apply(normalize_transmission)

    df[col] = df[norm_col].map(trans_canon)

    df.drop(columns=[norm_col], inplace=True)
    return df

# -------------------------------------------------------------------
# Fuel mapping
# -------------------------------------------------------------------
def normalize_fuel(value: str):
    if pd.isna(value):
        return pd.NA
    value = str(value).strip().lower()
    value = re.sub(r"[.,-_]", " ", value)
    value = " ".join(value.split())
    return value


def apply_fuel_mapping(
    df: pd.DataFrame,
    mapping_path: Path,
    col: str = "fuelType",
) -> pd.DataFrame:
    df = df.copy()
    if col not in df.columns:
        return df

    with mapping_path.open("r", encoding="utf-8") as f:
        raw_fuel_map = json.load(f)

    fuel_canon = {normalize_fuel(k): v for k, v in raw_fuel_map.items()}

    norm_col = f"{col}_norm"
    df[norm_col] = df[col].apply(normalize_fuel)
    df[col] = df[norm_col].map(fuel_canon)
    df.drop(columns=[norm_col], inplace=True)

    return df



# -------------------------------------------------------------------
# Brand mapping
# -------------------------------------------------------------------
def normalize_brand(value: str):
    if pd.isna(value):
        return pd.NA
    value = str(value).strip().lower()
    value = re.sub(r"[.,-_]", " ", value)
    value = " ".join(value.split())
    return value


def apply_brand_mapping(
    df: pd.DataFrame,
    mapping_path: Path,
    col: str = "Brand",
) -> pd.DataFrame:
    df = df.copy()
    if col not in df.columns:
        return df

    with mapping_path.open("r", encoding="utf-8") as f:
        raw_brand_map = json.load(f)

    brand_canon = {normalize_brand(k): v for k, v in raw_brand_map.items()}

    was_nan_before = df[col].isna()

    norm_col = f"{col}_norm"
    df[norm_col] = df[col].apply(normalize_brand)

    mapped = df[norm_col].map(brand_canon)

    clean_col = f"{col}_clean"
    df[clean_col] = df[col].where(mapped.isna(), mapped)

    # new NaNs from missing mapping
    new_nans_mask = df[clean_col].isna() & (~was_nan_before)
    # (we don’t need the count here, just keep behaviour)

    unmapped = (
        df.loc[mapped.isna() & (~was_nan_before) & df[norm_col].notna(), norm_col]
        .dropna()
        .unique()
    )

    df[col] = df[clean_col]
    df.drop(columns=[norm_col, clean_col], inplace=True)

    # you can log `unmapped` if you want
    return df

# -------------------------------------------------------------------
# Model mapping
# -------------------------------------------------------------------
def norm_model(s):
    if pd.isna(s):
        return pd.NA
    s = str(s).strip().lower()
    s = re.sub(r"[.,\-_ ]+", "", s)
    return s


def apply_model_mapping(
    df: pd.DataFrame,
    mapping_path: Path,
    col: str = "model",
) -> pd.DataFrame:
    df = df.copy()
    if col not in df.columns:
        return df

    with mapping_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    raw_aliases = cfg["aliases"]
    regex_rules = cfg.get("regex_rules", [])

    aliases = {norm_model(k): v for k, v in raw_aliases.items()}

    def apply_regex_first(val: str) -> str:
        for rule in regex_rules:
            pat = rule["pattern"]
            repl = rule["replace"]
            m = re.fullmatch(pat, val)
            if m:
                return m.expand(repl)
        return val

    df[f"{col}_norm"] = df[col].apply(norm_model)

    def map_model(norm_val):
        if pd.isna(norm_val):
            return pd.NA
        norm_val = apply_regex_first(norm_val)
        return aliases.get(norm_val, pd.NA)

    df[f"{col}_mapped"] = df[f"{col}_norm"].apply(map_model)

    df[col] = df[f"{col}_mapped"]

    # everything that had a value but no mapping -> NA
    mask_unmapped = df[col].isna() & df[f"{col}_norm"].notna()
    df.loc[mask_unmapped, col] = pd.NA

    df.drop(columns=[f"{col}_norm", f"{col}_mapped"], inplace=True)
    return df


# -------------------------------------------------------------------
# Brand from model mapping
# -------------------------------------------------------------------
def fill_brand_from_model(
    df: pd.DataFrame,
    mapping_path: Path,
    brand_col: str = "Brand",
    model_col: str = "model",
) -> pd.DataFrame:
    df = df.copy()
    if brand_col not in df.columns or model_col not in df.columns:
        return df

    with mapping_path.open("r", encoding="utf-8") as f:
        model_to_brand = json.load(f)

    mask = df[brand_col].isna() & df[model_col].notna()
    mapped = df.loc[mask, model_col].map(model_to_brand)

    fill_mask = mask.copy()
    fill_mask.loc[mask] = mapped.notna()

    df.loc[fill_mask, brand_col] = mapped[mapped.notna()]

    return df


# -------------------------------------------------------------------
# Outlier & rule-based NA handling (years, engineSize, mpg, etc.)
# -------------------------------------------------------------------
def apply_outlier_rules(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # engine size bounds
    if "engineSize" in df.columns:
        df.loc[
            (df["engineSize"] > 6.2) | (df["engineSize"] < 0.8), "engineSize"
        ] = np.nan

    # negative mileage / tax / previousOwners
    if "mileage" in df.columns:
        df.loc[df["mileage"] < 0, "mileage"] = np.nan

    if "tax" in df.columns:
        df.loc[df["tax"] < 0, "tax"] = np.nan

    if "previousOwners" in df.columns:
        df.loc[df["previousOwners"] < 0, "previousOwners"] = np.nan
        df.loc[df["previousOwners"] % 1 != 0, "previousOwners"] = np.nan

    # unusual mpg rules
    if "mpg" in df.columns and "fuelType" in df.columns:
        unusual_mpg_mask = (
            ((df["mpg"] > 150) & (df["fuelType"] == "Electric"))
            | ((df["mpg"] < 70) & (df["fuelType"] == "Electric"))
            | ((df["mpg"] > 100) & (df["fuelType"] == "Hybrid"))
            | ((df["mpg"] < 35) & (df["fuelType"] == "Hybrid"))
            | (
                (df["mpg"] > 80)
                & (df["fuelType"] != "Hybrid")
                & (df["fuelType"] != "Electric")
            )
            | (
                (df["mpg"] < 8)
                & (df["fuelType"] != "Hybrid")
                & (df["fuelType"] != "Electric")
            )
        )
        df.loc[unusual_mpg_mask, "mpg"] = np.nan

    # paintQuality%
    if "paintQuality%" in df.columns:
        df.loc[
            (df["paintQuality%"] > 100) | (df["paintQuality%"] < 0), "paintQuality%"
        ] = np.nan

    # transmission "Unknown" -> na
    if "transmission" in df.columns:
        df.loc[df["transmission"] == "Unknown", "transmission"] = np.nan

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
    mapping_dir = Path(mapping_dir)

    train = apply_basic_type_casting(train)
    test = apply_basic_type_casting(test)

    # check duplicates in carID (we don't drop; just keep info)
    if ID_COL in train.columns:
        # just for awareness, no change in data:
        train_dup_count = train.duplicated(subset=[ID_COL]).sum()
        print(f"[TRAIN] duplicate {ID_COL} count: {train_dup_count}")

    if ID_COL in test.columns:
        test_dup_count = test.duplicated(subset=[ID_COL]).sum()
        print(f"[TEST] duplicate {ID_COL} count: {test_dup_count}")

    # make all string columns lowercase (values, not column names)
    for c in train.select_dtypes(include="string").columns:
        train[c] = train[c].str.lower()
    for c in test.select_dtypes(include="string").columns:
        test[c] = test[c].str.lower()

    # mappings (paths mirror your notebook)
    train = apply_transmission_mapping(
        train, mapping_dir / "transmission_mapping.json"
    )
    test = apply_transmission_mapping(test, mapping_dir / "transmission_mapping.json")
    if debug and "transmission" in train.columns:
        _log_unique_values(train["transmission"], "TRAIN transmission after mapping")
    if debug and "transmission" in test.columns:
        _log_unique_values(test["transmission"], "TEST transmission after mapping")

    train = apply_fuel_mapping(train, mapping_dir / "fueltype_mapping.json")
    test = apply_fuel_mapping(test, mapping_dir / "fueltype_mapping.json")
    if debug and "fuelType" in train.columns:
        _log_unique_values(train["fuelType"], "TRAIN fuelType after mapping")
    if debug and "fuelType" in test.columns:
        _log_unique_values(test["fuelType"], "TEST fuelType after mapping")

    train = apply_brand_mapping(train, mapping_dir / "brandname_mapping.json", col="Brand")
    test = apply_brand_mapping(test, mapping_dir / "brandname_mapping.json", col="Brand")
    if debug and "Brand" in train.columns:
        _log_unique_values(train["Brand"], "TRAIN Brand after mapping")
    if debug and "Brand" in test.columns:
        _log_unique_values(test["Brand"], "TEST Brand after mapping")

    train = apply_model_mapping(train, mapping_dir / "modelname_mapping.json", col="model")
    test = apply_model_mapping(test, mapping_dir / "modelname_mapping.json", col="model")
    if debug and "model" in train.columns:
        _log_unique_values(train["model"], "TRAIN model after mapping", max_values=20)
    if debug and "model" in test.columns:
        _log_unique_values(test["model"], "TEST model after mapping", max_values=20)

    train = fill_brand_from_model(train, mapping_dir / "brand_model_mapping.json")
    test = fill_brand_from_model(test, mapping_dir / "brand_model_mapping.json")
    if debug and "Brand" in train.columns:
        _log_unique_values(train["Brand"], "TRAIN Brand after fill-brand-from-model")
    if debug and "Brand" in test.columns:
        _log_unique_values(test["Brand"], "TEST Brand after fill-brand-from-model")

    # outliers / impossible values -> NaN or dropped (same rules as notebook)
    train = apply_outlier_rules(train)
    test = apply_outlier_rules(test)

    return train, test




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
            train_ids = pd.Series(np.arange(len(train_clean)), name=ID_COL)

    if ID_COL in test_clean.columns:
        test_ids = test_clean[ID_COL].copy()
        X_test = test_clean.drop(columns=[ID_COL])
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