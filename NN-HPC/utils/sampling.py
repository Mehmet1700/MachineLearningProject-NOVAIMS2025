import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.base import BaseEstimator

class CarSegmentOversampler(BaseEstimator):
    """
    Oversamples the dataset to balance the 'car_segment' feature.
    Matches the logic from Notebook 12c.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self._sampling_type = "over-sampling"

    def fit_resample(self, X, y):
        """
        Resample X and y based on 'car_segment' distribution.
        """
        # Check if X is a DataFrame and has the required column
        if not isinstance(X, pd.DataFrame):
            # If it's numpy array, we can't check column name easily unless we track it.
            # But in our pipeline, we expect DataFrame here.
            print("[WARN] CarSegmentOversampler received non-DataFrame or missing columns. Skipping.")
            return X, y
            
        if 'car_segment' not in X.columns:
            print("[WARN] 'car_segment' column not found in X. Skipping oversampling.")
            return X, y
        
        print(f"[SAMPLING] Starting Oversampling based on 'car_segment'. Original shape: {X.shape}")

        # Combine for resampling
        df = X.copy()
        # Handle y being Series or array
        if isinstance(y, pd.Series):
            y_name = y.name or 'target'
            df[y_name] = y.values
        else:
            y_name = 'target'
            df[y_name] = y
        
        # Calculate target count (majority class)
        max_count = df['car_segment'].value_counts().max()
        print(f"[SAMPLING] Target count per segment (majority class): {max_count}")
        
        balanced_dfs = []
        unique_segments = df['car_segment'].unique()
        
        for segment in unique_segments:
            df_segment = df[df['car_segment'] == segment]
            
            if len(df_segment) == 0:
                continue
                
            # Resample
            df_resampled = resample(
                df_segment,
                replace=True,
                n_samples=max_count,
                random_state=self.random_state
            )
            balanced_dfs.append(df_resampled)
            
        # Combine and shuffle
        if not balanced_dfs:
            return X, y
            
        df_balanced = pd.concat(balanced_dfs, ignore_index=True)
        df_balanced = df_balanced.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        # Separate back
        X_res = df_balanced.drop(columns=[y_name])
        y_res = df_balanced[y_name]
        
        # Restore index name if possible (though index was reset)
        if hasattr(y, 'name'):
            y_res.name = y.name
            
        print(f"[SAMPLING] Oversampling complete. New shape: {X_res.shape}")
        return X_res, y_res
