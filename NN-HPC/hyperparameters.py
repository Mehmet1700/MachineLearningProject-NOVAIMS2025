"""
Hyperparameter Configurations for Neural Network Training.

This module defines the search spaces (parameter grids) for RandomizedSearchCV/GridSearchCV
for different model architectures (Keras MLP, PyTorch FT-Transformer).

It includes:
- Feature selection candidates.
- Hyperparameter grids for Phase 1 (Exploration) and Phase 2 (Refinement).
"""

from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
from sklearn.linear_model import Lasso

def get_feature_selectors(random_state: int = 42):
    """
    Returns a dictionary of feature selection methods to be tuned.
    """
    return {
        "none": "passthrough",
        "filter_freg_k100": SelectKBest(score_func=f_regression, k=100),
        "lasso_embedded": SelectFromModel(
            Lasso(alpha=0.01, random_state=random_state), 
            threshold="mean"
        ),
    }

def build_phase_one_search_space(random_state: int = 42):
    """
    Defines the broad search space for the initial 'Warm-up' phase (RandomizedSearch).
    Targeting Keras MLP.
    """
    selectors = get_feature_selectors(random_state)
    
    hidden_layer_options = [
        (512, 256, 128),
        (384, 192, 96),
        (256, 128, 64),
        (256, 128, 64, 32),
        (384, 192, 96, 48),
    ]
    
    return {
        "feature_sel": [
            selectors["filter_freg_k100"],
            selectors["lasso_embedded"],
            selectors["none"],
        ],
        "model__hidden_layer_sizes": hidden_layer_options,
        "model__learning_rate": [1e-3, 5e-4],
        "model__dropout": [0.0, 0.1, 0.2],
        "model__l2_reg": [0.0, 1e-4, 1e-3],
        "model__activation": ["relu", "swish"],
        "model__batch_size": [128, 256],
        "model__epochs": [100, 150],
    }

def build_keras_mlp_param_grid(random_state: int = 42):
    """
    Defines the parameter grid for the Keras MLP (Multi-Layer Perceptron).
    """
    selectors = get_feature_selectors(random_state)
    
    # Example grid - can be adjusted based on Phase 1 results
    return [
        {
            "feature_sel": [selectors["none"]],
            "model__hidden_layer_sizes": [(512, 256, 128)],
            "model__learning_rate": [1e-3],
            "model__dropout": [0.1],
            "model__batch_size": [128, 256],
            "model__epochs": [150],
        },
    ]

def build_ft_transformer_param_grid(random_state: int = 42):
    """
    Defines the parameter grid for the PyTorch FT-Transformer.
    
    Note: Feature selection is handled implicitly or skipped for Transformers 
    as they learn feature importance via attention mechanisms.
    """
    return [
        {
            # Architecture
            "model__d_model": [192],          # Embedding dimension
            "model__n_layers": [3],           # Number of Transformer blocks
            "model__n_heads": [8],            # Number of attention heads
            
            # Regularization
            "model__dropout": [0.1],
            "model__weight_decay": [1e-5],
            
            # Training
            "model__batch_size": [512],       # Large batch size for A100 efficiency
            "model__epochs": [50],            # Sufficient for convergence with OneCycleLR
            "model__learning_rate": [1e-3],   # Peak LR for OneCycleLR
        },
    ]
