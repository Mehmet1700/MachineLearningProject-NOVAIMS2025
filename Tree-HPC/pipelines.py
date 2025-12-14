from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import QuantileTransformer
import numpy as np

def build_tree_pipeline(model_type, preprocessor, random_state=42):
    """
    Builds a pipeline for a given tree-based model.
    
    Args:
        model_type (str): 'rf', 'et', or 'hgb'
        preprocessor (ColumnTransformer): The preprocessor to use.
        random_state (int): Random state for reproducibility.
        
    Returns:
        Pipeline: The complete pipeline.
    """
    
    if model_type == 'rf':
        regressor = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    elif model_type == 'et':
        regressor = ExtraTreesRegressor(random_state=random_state, n_jobs=-1)
    elif model_type == 'hgb':
        regressor = HistGradientBoostingRegressor(random_state=random_state)
    elif model_type == 'dt':
        regressor = DecisionTreeRegressor(random_state=random_state)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Use TransformedTargetRegressor to handle potential target skewness.
    # We apply Log-Transformation (log1p) to the prices, which typically improves
    # performance for regression tasks with skewed target distributions (like prices).
    model = TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline

def build_rf_pipeline(preprocessor, random_state=42):
    return build_tree_pipeline('rf', preprocessor, random_state)

def build_et_pipeline(preprocessor, random_state=42):
    return build_tree_pipeline('et', preprocessor, random_state)

def build_hgb_pipeline(preprocessor, random_state=42):
    return build_tree_pipeline('hgb', preprocessor, random_state)

def build_dt_pipeline(preprocessor, random_state=42):
    return build_tree_pipeline('dt', preprocessor, random_state)
