# utils/feature_engineering.py
from typing import Tuple
import numpy as np
import pandas as pd


def add_model_engine_rarity_cv(
    X_full: pd.DataFrame,
    X_test: pd.DataFrame | None,
    model_col: str = "model",
    engine_col: str = "engineSize",
    new_col: str = "model_engine_freq",
    log_scale: bool = True,
):
    """
    Compute (model, engineSize) frequency on FULL labeled data X_full,
    then apply mapping to both X_full and X_test.

    NOTE: This is still safe because:
      - we never use y,
      - frequencies are just counts of X,
      - and they are part of the 'feature space', not target statistics.
    """
    X_full_new = X_full.copy()
    X_test_new = X_test.copy() if X_test is not None else None

    freq_series = (
        X_full_new
        .groupby([model_col, engine_col], dropna=False)
        .size()
        .rename("freq")
    )
    freq_map = freq_series.to_dict()

    def _apply_freq(df: pd.DataFrame) -> pd.DataFrame:
        keys = list(zip(df[model_col], df[engine_col]))
        freqs = np.array([freq_map.get(k, 0) for k in keys], dtype=float)
        if log_scale:
            freqs = np.log1p(freqs)
        df[new_col] = freqs
        return df

    X_full_new = _apply_freq(X_full_new)
    if X_test_new is not None:
        X_test_new = _apply_freq(X_test_new)

    return X_full_new, X_test_new