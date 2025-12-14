# utils/feature_selection.py
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    SelectFromModel,
    RFE,
    RFECV,
)
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin


class CorrelationVarianceSelector(BaseEstimator, TransformerMixin):
    """
    Custom Feature Selector that wraps the drop_low_variance_features and 
    drop_highly_correlated_features logic into a sklearn-compatible transformer.
    """
    def __init__(self, variance_threshold=0.01, correlation_threshold=0.90):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.support_ = None

    def fit(self, X, y=None):
        # Convert to DataFrame if needed to use the existing logic
        if not isinstance(X, pd.DataFrame):
            # Create generic column names if input is numpy array
            cols = [f"feat_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=cols)
        else:
            X_df = X.copy()

        # 1. Variance Threshold
        # We calculate variance manually to determine which columns to keep
        variances = X_df.var()
        keep_var_mask = variances >= self.variance_threshold
        
        # Filter X_df to only high variance columns for the next step
        X_high_var = X_df.loc[:, keep_var_mask]
        
        # 2. Correlation Filter
        # We use the logic from drop_highly_correlated_features
        # Calculate correlation matrices
        corr_p = X_high_var.corr(method="pearson").abs()
        corr_s = X_high_var.corr(method="spearman").abs()
        corr_k = X_high_var.corr(method="kendall").abs()
        
        to_drop = set()
        columns = X_high_var.columns.tolist()
        
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1 = columns[i]
                col2 = columns[j]
                
                if col2 in to_drop:
                    continue
                    
                votes = 0
                if corr_p.loc[col1, col2] > self.correlation_threshold: votes += 1
                if corr_s.loc[col1, col2] > self.correlation_threshold: votes += 1
                if corr_k.loc[col1, col2] > self.correlation_threshold: votes += 1
                
                if votes >= 2:
                    to_drop.add(col2)
        
        # Final support mask
        # Start with all False
        self.support_ = np.zeros(X_df.shape[1], dtype=bool)
        
        # Get indices of columns that survived variance filter
        kept_var_indices = np.where(keep_var_mask)[0]
        
        # Among those, check which ones are NOT in to_drop
        final_kept_indices = []
        high_var_cols = X_high_var.columns.tolist()
        
        for idx, col_name in zip(kept_var_indices, high_var_cols):
            if col_name not in to_drop:
                final_kept_indices.append(idx)
                
        self.support_[final_kept_indices] = True
        
        print(f"[CustomSelector] Kept {self.support_.sum()} features out of {X_df.shape[1]}")
        return self

    def transform(self, X):
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        return X[:, self.support_]
    
    def get_support(self, indices=False):
        if indices:
            return np.where(self.support_)[0]
        return self.support_


def drop_low_variance_features(x: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """
    Remove columns whose variance is below `threshold`.
    Useful to deal with features that are constant or quasi-constant.
    """
    print(f"\n[FILTER] Dropping features with variance < {threshold}...")
    variances = x.var()
    low_variance_features = variances[variances < threshold].index
    
    if len(low_variance_features) > 0:
        print(f"  -> Dropping {len(low_variance_features)} features: {list(low_variance_features)}")
        x = x.drop(columns=low_variance_features)
    else:
        print("  -> No low-variance features found.")
        
    return x


def drop_highly_correlated_features(x: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """
    Drop redundant features based on multi-metric correlation voting.
    If at least two of (Pearson, Spearman, Kendall) exceed `threshold` (absolute value),
    arbitrarily drop the second feature to reduce multicollinearity.
    """
    print(f"\n[FILTER] Dropping highly correlated features (threshold={threshold})...")
    
    # Calculate correlation matrices once
    print("  -> Calculating Pearson correlation...")
    corr_p = x.corr(method="pearson").abs()
    print("  -> Calculating Spearman correlation...")
    corr_s = x.corr(method="spearman").abs()
    print("  -> Calculating Kendall correlation...")
    corr_k = x.corr(method="kendall").abs()
    
    to_drop = set()
    columns = x.columns.tolist()
    
    # Iterate over upper triangle
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1 = columns[i]
            col2 = columns[j]
            
            # Skip if col2 is already marked for dropping
            if col2 in to_drop:
                continue
                
            # Check votes
            votes = 0
            if corr_p.loc[col1, col2] > threshold: votes += 1
            if corr_s.loc[col1, col2] > threshold: votes += 1
            if corr_k.loc[col1, col2] > threshold: votes += 1
            
            if votes >= 2:
                print(f"  -> Dropping '{col2}' (correlated with '{col1}')")
                to_drop.add(col2)
    
    if to_drop:
        print(f"  -> Total dropped due to correlation: {len(to_drop)}")
        x = x.drop(columns=list(to_drop))
    else:
        print("  -> No highly correlated features found.")
        
    return x


def perform_rfe_selection(x: pd.DataFrame, y: pd.Series, n_features_to_select: int = 20, step: int = 1, random_state: int = 42) -> pd.DataFrame:
    """
    Perform Recursive Feature Elimination (RFE) using RandomForestRegressor.
    """
    print(f"\n[WRAPPER] Performing RFE (n_features={n_features_to_select}, step={step})...")
    
    # Use RandomForest as the estimator
    estimator = RandomForestRegressor(n_jobs=-1, random_state=random_state)
    
    selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=step)
    selector.fit(x, y)
    
    selected_features = x.columns[selector.support_]
    print(f"  -> Selected {len(selected_features)} features.")
    
    return x[selected_features]


def perform_rfecv_selection(x: pd.DataFrame, y: pd.Series, min_features_to_select: int = 10, step: int = 1, cv: int = 3, random_state: int = 42) -> pd.DataFrame:
    """
    Perform RFE with Cross-Validation (RFECV) to automatically select the best number of features.
    """
    print(f"\n[WRAPPER] Performing RFECV (min_features={min_features_to_select}, cv={cv})...")
    
    estimator = RandomForestRegressor(n_jobs=-1, random_state=random_state)
    
    selector = RFECV(
        estimator=estimator, 
        step=step, 
        cv=cv, 
        min_features_to_select=min_features_to_select, 
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )
    selector.fit(x, y)
    
    selected_features = x.columns[selector.support_]
    print(f"  -> Optimal number of features: {selector.n_features_}")
    
    return x[selected_features]


def candidate_selectors(random_state: int = 42):
    """
    High-Performance Feature Selection Mode (Option C):
    - Focus on efficient, robust, modern methods
    - No RFE (too slow)
    - No RandomForest (too heavy)
    - LassoCV with cv=3: good compromise between quality & speed
    """

    candidates = {}

    # 1) No Feature Selection (Baseline)
    candidates["none"] = "passthrough"

    # 2) Filter Selection (very efficient, good baseline)
    candidates["filter_freg_k100"] = SelectKBest(
        score_func=f_regression,
        k=100,   # good default choice — stable and efficient
    )

    # 3) Embedded Selection using LassoCV (very strong for regression problems)
    lasso = LassoCV(
        cv=3,  # more efficient than cv=5 / 10
        random_state=random_state,
        n_jobs=-1,
    )
    candidates["lasso_embedded"] = SelectFromModel(
        estimator=lasso,
        threshold="median",   # selects approx. half of all features
    )
    
    # 4) Wrapper Selection (RFE) - Optional, can be slow
    # candidates["rfe_20"] = RFE(estimator=RandomForestRegressor(n_jobs=-1, random_state=random_state), n_features_to_select=20, step=5)

    # 5) Custom Filter (Variance + Correlation)
    candidates["custom_filter"] = CorrelationVarianceSelector()

    return candidates
