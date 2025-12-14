"""
Hyperparameter search spaces for Tree-based models.
"""

from scipy.stats import randint, uniform

def build_random_forest_param_grid():
    """
    Returns the parameter grid for RandomForestRegressor.
    """
    return {
        "model__regressor__n_estimators": [100, 300, 500, 800],
        "model__regressor__max_depth": [None, 10, 20, 30, 50],
        "model__regressor__min_samples_split": [2, 5, 10],
        "model__regressor__min_samples_leaf": [1, 2, 4],
        "model__regressor__max_features": ["sqrt", "log2", None],
        "model__regressor__bootstrap": [True, False]
    }

def build_extra_trees_param_grid():
    """
    Returns the parameter grid for ExtraTreesRegressor.
    Refined based on previous run (Best: MAE ~1248).
    Focus: High n_estimators, deep trees, all features (or close to it).
    """
    return {
        "model__regressor__n_estimators": [800, 1000, 1200, 1500],
        "model__regressor__max_depth": [50, 70, 100, None],
        "model__regressor__min_samples_split": [2, 5, 8],
        "model__regressor__min_samples_leaf": [1, 2],
        "model__regressor__max_features": [None, 0.9, 0.8], # None was best, exploring high fractions too
        "model__regressor__bootstrap": [False] # False was clearly better
    }

def build_hist_gradient_boosting_param_grid():
    """
    Returns the parameter grid for HistGradientBoostingRegressor.
    """
    return {
        "model__regressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__regressor__max_iter": [100, 300, 500, 1000],
        "model__regressor__max_leaf_nodes": [31, 63, 127, 255],
        "model__regressor__min_samples_leaf": [20, 50, 100],
        "model__regressor__l2_regularization": [0.0, 0.1, 1.0, 10.0],
        "model__regressor__max_depth": [None, 10, 20]
    }

def build_decision_tree_param_grid():
    """
    Returns the parameter grid for DecisionTreeRegressor.
    """
    return {
        "model__regressor__max_depth": [None, 10, 20, 30, 50],
        "model__regressor__min_samples_split": [2, 5, 10, 20],
        "model__regressor__min_samples_leaf": [1, 2, 4, 8],
        "model__regressor__max_features": ["sqrt", "log2", None],
        "model__regressor__splitter": ["best", "random"]
    }
