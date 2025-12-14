import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

# --- Helper Pipelines ---

# Used for low-cardinality categorical features
low_cardinality_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Used for high-cardinality categorical features like model
high_cardinality_pipeline = Pipeline([
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

def create_imputation_pipeline(low_card_cols, high_card_cols=[]):
    """
    Helper function to create specific pipelines for each imputer.
    This ensures we only try to encode columns that are actually present as features.
    """
    transformers = []
    if low_card_cols:
        transformers.append(('low_card', low_cardinality_pipeline, low_card_cols))
    if high_card_cols:
        transformers.append(('high_card', high_cardinality_pipeline, high_card_cols))
        
    # remainder='passthrough' keeps the numerical columns
    encoder = ColumnTransformer(transformers, remainder='passthrough', verbose_feature_names_out=False)
    
    return Pipeline([
        ('encoder', encoder),
        ("rf_model", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
    ])

# --- Categorical Imputers ---

class TransmissionImputer(BaseEstimator, TransformerMixin):
    def __init__(self, pipeline_template, min_model_count=10):
        self.pipeline_template = pipeline_template
        self.min_model_count = min_model_count

    def fit(self, X, y=None):
        df = X.copy()

        # store lookup tables learned from training data
        counts = df.dropna(subset=['transmission'])['model'].value_counts()
        self.valid_models_ = counts[counts >= self.min_model_count].index

        self.model_modes_ = (
            df.dropna(subset=['model', 'transmission'])
              .groupby('model')['transmission']
              .agg(lambda s: s.mode().iloc[0])
        )

        # Train ML model for remaining missing values
        train_df = df[df['transmission'].notna()].copy()

        features = [
            "Brand", "model", "fuelType",
            "engineSize", "year", "mpg", "tax",
        ]

        pipe = clone(self.pipeline_template)
        pipe.fit(train_df[features], train_df['transmission'])

        self.ml_model_ = pipe
        self.features_ = features
        return self

    def transform(self, X):
        df = X.copy()

        # rule-based fill
        for model in self.valid_models_:
            mask = (df['model'] == model) & (df['transmission'].isna())
            if model in self.model_modes_:
                df.loc[mask, 'transmission'] = self.model_modes_[model]

        # ML fallback
        missing_mask = df['transmission'].isna()
        if missing_mask.any():
            df.loc[missing_mask, 'transmission'] = self.ml_model_.predict(
                df.loc[missing_mask, self.features_]
            )
        return df

class FuelTypeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, pipeline_template, min_model_count=10):
        self.pipeline_template = pipeline_template
        self.min_model_count = min_model_count

    def fit(self, X, y=None):
        df = X.copy()

        counts = df.dropna(subset=['fuelType'])['model'].value_counts()
        self.valid_models_ = counts[counts >= self.min_model_count].index

        self.model_modes_ = (
            df.dropna(subset=['model', 'fuelType'])
              .groupby('model')['fuelType']
              .agg(lambda s: s.mode().iloc[0])
        )

        features = ['Brand', 'model', 'transmission',
                    'engineSize', 'year', 'mpg']
        train_df = df[df['fuelType'].notna()]

        pipe = clone(self.pipeline_template)
        pipe.fit(train_df[features], train_df['fuelType'])

        self.ml_model_ = pipe
        self.features_ = features
        return self

    def transform(self, X):
        df = X.copy()

        # rule-based fill
        for model in self.valid_models_:
            mask = (df['model'] == model) & (df['fuelType'].isna())
            df.loc[mask, 'fuelType'] = self.model_modes_.get(model, np.nan)

        # ML fallback
        missing = df['fuelType'].isna()
        if missing.any():
            df.loc[missing, 'fuelType'] = self.ml_model_.predict(
                df.loc[missing, self.features_]
            )
        return df

class BrandImputer(BaseEstimator, TransformerMixin):
    def __init__(self, pipeline_template, min_model_count=20):
        self.pipeline_template = pipeline_template
        self.min_model_count = min_model_count

    def fit(self, X, y=None):
        df = X.copy()

        # rule-based brand per model
        self.model_to_brand_ = (
            df.dropna(subset=['Brand', 'model'])
              .groupby('model')['Brand']
              .agg(lambda s: s.mode().iloc[0])
        )

        # ML fallback
        features = ['transmission', 'engineSize',
                    'fuelType', 'mpg', 'model']
        train_df = df[df['Brand'].notna()]

        pipe = clone(self.pipeline_template)
        pipe.fit(train_df[features], train_df['Brand'])

        self.ml_model_ = pipe
        self.features_ = features
        return self

    def transform(self, X):
        df = X.copy()

        # rule-based fill
        mask = df['Brand'].isna() & df['model'].notna()
        df.loc[mask, 'Brand'] = df.loc[mask, 'model'].map(self.model_to_brand_)

        # ML fallback
        missing = df['Brand'].isna()
        if missing.any():
            df.loc[missing, 'Brand'] = self.ml_model_.predict(
                df.loc[missing, self.features_]
            )
        return df

class ModelImputer(BaseEstimator, TransformerMixin):
    def __init__(self, pipeline_template, min_brand_count=20):
        self.pipeline_template = pipeline_template
        self.min_brand_count = min_brand_count

    def fit(self, X, y=None):
        df = X.copy()

        # rule-based: most frequent model per (Brand, transmission)
        self.lookup_ = (
            df.dropna(subset=['Brand', 'transmission', 'model'])
              .groupby(['Brand', 'transmission'])['model']
              .agg(lambda s: s.value_counts().idxmax())
        )

        # ML fallback
        features = ['Brand', 'year', 'engineSize', 'mpg',
                    'tax', 'mileage', 'fuelType', 'transmission']
        train_df = df[df['model'].notna()]

        pipe = clone(self.pipeline_template)
        pipe.fit(train_df[features], train_df['model'])

        self.ml_model_ = pipe
        self.features_ = features
        return self

    def transform(self, X):
        df = X.copy()

        # rule-based fill
        for idx, row in df[df['model'].isna()].iterrows():
            key = (row['Brand'], row['transmission'])
            if key in self.lookup_.index:
                df.at[idx, 'model'] = self.lookup_.loc[key]

        # ML fallback
        missing = df['model'].isna()
        if missing.any():
            df.loc[missing, 'model'] = self.ml_model_.predict(
                df.loc[missing, self.features_]
            )
        return df

# --- Numerical Imputers ---

class MileageImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        df = X.copy()

        self.year_medians_ = df.groupby('year')['mileage'].median()
        self.global_median_ = df['mileage'].median()
        return self

    def transform(self, X):
        df = X.copy()

        df['mileage'] = df['mileage'].fillna(
            df['year'].map(self.year_medians_)
        )
        df['mileage'] = df['mileage'].fillna(self.global_median_)
        return df

class YearImputer(BaseEstimator, TransformerMixin):
    def __init__(self, bins=30):
        self.bins = bins

    def fit(self, X, y=None):
        df = X.copy()

        df['mileage_bin'] = pd.cut(df['mileage'], bins=self.bins)
        self.bin_medians_ = df.groupby('mileage_bin')['year'].median()
        self.global_median_ = df['year'].median()
        return self

    def transform(self, X):
        df = X.copy()

        df['mileage_bin'] = pd.cut(df['mileage'], bins=self.bins)

        df['year'] = df['year'].fillna(
            df['mileage_bin'].map(self.bin_medians_)
        )
        df['year'] = df['year'].fillna(self.global_median_)

        df.drop(columns='mileage_bin', inplace=True)
        return df

class MPGImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        df = X.copy()

        self.model_medians_ = df.groupby(['model', 'fuelType'])['mpg'].median()
        self.brand_medians_ = df.groupby(['Brand', 'fuelType'])['mpg'].median()
        self.fuel_medians_ = df.groupby(['fuelType'])['mpg'].median()
        self.global_median_ = df['mpg'].median()
        return self

    def transform(self, X):
        df = X.copy()

        for idx, row in df[df['mpg'].isna()].iterrows():
            km = (row['model'], row['fuelType'])
            kb = (row['Brand'], row['fuelType'])
            if km in self.model_medians_.index:
                df.at[idx, 'mpg'] = self.model_medians_.loc[km]
            elif kb in self.brand_medians_.index:
                df.at[idx, 'mpg'] = self.brand_medians_.loc[kb]

        df['mpg'] = df['mpg'].fillna(df['fuelType'].map(self.fuel_medians_))
        df['mpg'] = df['mpg'].fillna(self.global_median_)
        return df

class PreviousOwnersImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.median_ = X['previousOwners'].median()
        return self

    def transform(self, X):
        df = X.copy()
        df['previousOwners'] = df['previousOwners'].fillna(self.median_)
        return df

class TaxImputer(BaseEstimator, TransformerMixin):
    def __init__(self, bins=15):
        self.bins = bins

    def fit(self, X, y=None):
        df = X.copy()

        df['mpg_bin'] = pd.cut(df['mpg'], bins=self.bins)

        self.medians_one_ = df.groupby(['model', 'fuelType', 'mpg_bin'])['tax'].median()
        self.medians_two_ = df.groupby(['Brand', 'fuelType', 'mpg_bin'])['tax'].median()
        self.medians_three_ = df.groupby(['fuelType', 'mpg_bin'])['tax'].median()
        self.fuel_medians_ = df.groupby('fuelType')['tax'].median()
        self.global_median_ = df['tax'].median()
        return self

    def transform(self, X):
        df = X.copy()

        df['mpg_bin'] = pd.cut(df['mpg'], bins=self.bins)

        for idx, row in df[df['tax'].isna()].iterrows():
            km = (row['model'], row['fuelType'], row['mpg_bin'])
            kb = (row['Brand'], row['fuelType'], row['mpg_bin'])
            ke = (row['fuelType'], row['mpg_bin'])

            if km in self.medians_one_.index:
                df.at[idx, 'tax'] = self.medians_one_.loc[km]
            elif kb in self.medians_two_.index:
                df.at[idx, 'tax'] = self.medians_two_.loc[kb]
            elif ke in self.medians_three_.index:
                df.at[idx, 'tax'] = self.medians_three_.loc[ke]

        df['tax'] = df['tax'].fillna(df['fuelType'].map(self.fuel_medians_))
        df['tax'] = df['tax'].fillna(self.global_median_)
        
        # Drop the temporary bin column
        df.drop(columns='mpg_bin', inplace=True)
        return df

class EngineSizeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        df = X.copy()

        self.model_medians_ = df.groupby(['model', 'fuelType'])['engineSize'].median()
        self.brand_medians_ = df.groupby(['Brand', 'fuelType'])['engineSize'].median()
        self.fuel_medians_ = df.groupby(['model'])['engineSize'].median()
        self.global_median_ = df['engineSize'].median()
        return self

    def transform(self, X):
        df = X.copy()

        for idx, row in df[df['engineSize'].isna()].iterrows():
            km = (row['model'], row['fuelType'])
            kb = (row['Brand'], row['fuelType'])
            if km in self.model_medians_.index:
                df.at[idx, 'engineSize'] = self.model_medians_.loc[km]
            elif kb in self.brand_medians_.index:
                df.at[idx, 'engineSize'] = self.brand_medians_.loc[kb]

        df['engineSize'] = df['engineSize'].fillna(df['model'].map(self.fuel_medians_))
        df['engineSize'] = df['engineSize'].fillna(self.global_median_)
        return df

class HasDamageImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if 'hasDamage' in df.columns:
            # Fill NaNs with 1 (assuming missing means damage/unknown, as per notebook logic)
            df['hasDamage'] = df['hasDamage'].fillna(1)
        return df

# --- Pipeline Construction ---

def get_imputation_pipeline():
    """
    Returns the full imputation pipeline.
    """
    # Create specific pipelines for each imputer based on the features they use
    transmission_template = create_imputation_pipeline(
        low_card_cols=['Brand', 'fuelType'], 
        high_card_cols=['model']
    )

    fuel_template = create_imputation_pipeline(
        low_card_cols=['Brand', 'transmission'], 
        high_card_cols=['model']
    )

    brand_template = create_imputation_pipeline(
        low_card_cols=['transmission', 'fuelType'], 
        high_card_cols=['model']
    )

    model_template = create_imputation_pipeline(
        low_card_cols=['Brand', 'fuelType', 'transmission'], 
        high_card_cols=[]
    )

    rule_imputer = Pipeline([
            ("hasDamage_imp", HasDamageImputer()),
            ("transmission_imp", TransmissionImputer(transmission_template)),
            ("fuelType_imp", FuelTypeImputer(fuel_template)),
            ("brand_imp", BrandImputer(brand_template)),
            ("model_imp", ModelImputer(model_template)),
            ("mileage_imp", MileageImputer()),
            ("year_imp", YearImputer()),
            ("mpg_imp", MPGImputer()),
            ("previousOwners_imp", PreviousOwnersImputer()),
            ("tax_imp", TaxImputer()),
            ("engineSize_imp", EngineSizeImputer()),
        ])
    
    return rule_imputer

# --- Encoding & Scaling ---

def get_encoding_transformer():
    """
    Returns the ColumnTransformer for OneHotEncoding.
    """
    # Define columns for encoding
    onehot_cols = ['Brand', 'fuelType', 'transmission', 'model', 'brand_segment', 'car_segment']

    # Initialize encoders
    onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Create ColumnTransformer
    encoding_transformer = ColumnTransformer(
        transformers=[
            ('onehot', onehot_encoder, onehot_cols)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False 
    )
    return encoding_transformer

def get_scaler():
    """
    Returns the Scaler (RobustScaler).
    """
    from sklearn.preprocessing import RobustScaler
    return RobustScaler()

def drop_interval_cols(df):
    """
    Safety check to remove any leaked interval columns (like 'mpg_bin' or 'mileage_bin').
    """
    if not hasattr(df, "select_dtypes"):
        # If it's a numpy array or sparse matrix, we can't check columns by name/type easily.
        # Assuming the previous step (Encoding) returned a dense/sparse matrix without interval objects.
        return df

    # Identify columns with Interval data type
    interval_cols = df.select_dtypes(include=['interval']).columns.tolist()
    # Also check for specific names just in case
    bin_cols = [c for c in df.columns if c.endswith('_bin')]
    cols_to_drop = list(set(interval_cols + bin_cols))
    
    if cols_to_drop:
        print(f"Dropping leaked columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
        
    return df
