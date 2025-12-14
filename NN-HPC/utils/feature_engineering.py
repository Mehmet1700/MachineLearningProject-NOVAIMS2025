# utils/feature_engineering.py
from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin


def load_mappings(mapping_dir: str = None) -> Tuple[Dict, Dict]:
    """
    Load brand and model segment mappings.
    """
    if mapping_dir is None:
        # Default to ../../mapping relative to this file
        mapping_dir = Path(__file__).resolve().parent.parent.parent / "mapping"
    else:
        mapping_dir = Path(mapping_dir)

    brand_mapping = {}
    model_mapping = {}

    try:
        with open(mapping_dir / 'brandsegment_mapping.json', 'r') as f:
            brand_mapping = json.load(f)
        with open(mapping_dir / 'carsegment_mapping.json', 'r') as f:
            model_mapping = json.load(f)
        print("Mappings loaded successfully.")
    except FileNotFoundError:
        print(f"Mapping files not found in {mapping_dir}. Please check the paths.")
    
    return brand_mapping, model_mapping


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, brand_mapping=None, model_mapping=None):
        self.brand_mapping = brand_mapping
        self.model_mapping = model_mapping

    def fit(self, X, y=None):
        df = X.copy()
        # Calculate medians/quantiles from TRAIN set only
        self.trans_medians_ = df.groupby('transmission')['mpg'].median()
        self.year_median_ = df['year'].median()
        self.mileage_q75_ = df['mileage'].quantile(0.75)
        self.mileage_q90_ = df['mileage'].quantile(0.90)
        return self

    def transform(self, X):
        df = X.copy()
        
        # Apply trans_medians
        # Map and fill missing with global median of the mapping (or 0 if diff)
        mpg_median_trans = df['transmission'].map(self.trans_medians_)
        # Handle cases where transmission category in test wasn't in train
        mpg_median_trans = mpg_median_trans.fillna(self.trans_medians_.median()) 
        
        df['mpg_diff_transmission'] = df['mpg'] - mpg_median_trans
        df['mpg_diff_transmission'] = df['mpg_diff_transmission'].fillna(0)

        # car_age
        df['car_age'] = 2020 - df['year']
        
        # efficiency_ratio
        df['efficiency_ratio'] = df['mpg'] / (df['engineSize'] + 0.1)
        
        # mileage_per_year
        df['mileage_per_year'] = df['mileage'] / (df['car_age'] + 0.1)
        
        # first_owner_flag
        df['first_owner_flag'] = (df['previousOwners'] == 0).astype(int)
        
        # many_owners_flag
        df['many_owners_flag'] = (df['previousOwners'] >= 3).astype(int)
        
        # engine_tax_ratio
        df['engine_tax_ratio'] = df['engineSize'] / (df['tax'] + 0.1)
        
        # age_per_10k_miles
        df['age_per_10k_miles'] = df['car_age'] / (df['mileage']/10000 + 1)
        
        # high_mileage_flag
        df['high_mileage_flag'] = (df['mileage'] > self.mileage_q75_).astype(int)
        df['very_high_mileage_flag'] = (df['mileage'] > self.mileage_q90_).astype(int)
        
        # Ratios
        df['liter_per_mpg'] = df['engineSize'] / (df['mpg'] + 0.1)
        df['engine_per_age'] = df['engineSize'] / (df['car_age'] + 0.1)
        df['mpg_per_age'] = df['mpg'] / (df['car_age'] + 0.1)
        df['tax_per_age'] = df['tax'] / (df['car_age'] + 0.1)
        
        # Interactions
        df['mpg_x_engine'] = df['mpg'] * df['engineSize']
        df['tax_x_engine'] = df['tax'] * df['engineSize']
        df['mpg_x_tax'] = df['mpg'] * df['tax']
        
        df['engine_x_age'] = df['engineSize'] * df['car_age']
        df['mpg_x_age'] = df['mpg'] * df['car_age']
        df['tax_x_age'] = df['tax'] * df['car_age']
        
        # Centered year
        df['year_centered'] = df['year'] - self.year_median_
        
        # More ratios
        df['tax_per_liter'] = df['tax'] / (df['engineSize'] + 0.1)
        df['tax_per_mpg'] = df['tax'] / (df['mpg'] + 0.1)

        # New mapping features
        if self.brand_mapping:
            df['brand_segment'] = df['Brand'].map(self.brand_mapping)
            # Fill unknown brands if any with a placeholder or mode
            df['brand_segment'] = df['brand_segment'].fillna('Unknown')
            
        if self.model_mapping:
            df['car_segment'] = df['model'].map(self.model_mapping)
            # Fill unknown models if any
            df['car_segment'] = df['car_segment'].fillna('Unknown')
        
        return df


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