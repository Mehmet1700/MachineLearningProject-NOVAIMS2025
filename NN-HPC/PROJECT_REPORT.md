# Project Report: Neural Network Approaches for Car Price Prediction

## 1. Introduction
This report documents the "NN-HPC" component of the Car Price Prediction project. While tree-based models often serve as strong baselines for tabular data, Neural Networks (NNs) offer the potential to capture complex, high-dimensional interactions and leverage modern architectures like Transformers. This module focuses on training Deep Learning models using High-Performance Computing (HPC) resources.

## 2. Problem Statement & Deep Learning Context
Predicting car prices involves regression on a dataset with mixed data types (numerical and categorical). Deep Learning faces specific challenges here:
-   **Tabular Data**: Unlike images or text, tabular data lacks spatial/temporal locality, making standard CNNs/RNNs less effective.
-   **Categorical Embeddings**: High-cardinality features (e.g., `model`) require efficient embedding strategies to be processed by NNs.
-   **Feature Scaling**: NNs are highly sensitive to the scale of input features, necessitating robust normalization.

## 3. Methodology

### 3.1 Data Preprocessing for Neural Networks
The preprocessing pipeline for NNs is distinct from tree-based models:
-   **Imputation**: We utilize a **Hybrid Imputation Strategy**:
    -   **Rule-Based**: Missing values in categorical features (e.g., `transmission`) are filled with the mode of the corresponding car model.
    -   **Model-Based**: Remaining missing values are predicted using a `RandomForestClassifier`, effectively learning relationships between features to impute data accurately.
-   **Feature Engineering**: We calculate domain-specific ratios (e.g., `efficiency_ratio = mpg / engineSize`) and statistical aggregates (e.g., median MPG per transmission type) to enrich the feature space.
-   **Scaling**: We apply `RobustScaler` to numerical features. This is critical for NNs to prevent gradient explosion/vanishing and ensure stable convergence.
-   **Encoding**: 
    -   **Categorical Embeddings**: For the FT-Transformer, categorical variables are Ordinal Encoded and then mapped to dense vectors (embeddings) learned during training.
    -   **One-Hot Encoding**: For the MLP baseline, categorical features are One-Hot Encoded.

### 3.2 Model Architectures
We implemented two distinct Deep Learning architectures:

#### A. Keras MLP (Multi-Layer Perceptron)
A flexible, fully connected network serving as a strong neural baseline.
-   **Structure**: Configurable number of dense layers (e.g., 2-4 layers) with ReLU activation.
-   **Regularization**: Dropout layers and L2 regularization (Weight Decay) to prevent overfitting.
-   **Optimization**: Trained with Adam optimizer and Early Stopping.

#### B. PyTorch FT-Transformer (Feature Tokenizer + Transformer)
A state-of-the-art architecture adapted from the paper ["Revisiting Deep Learning Models for Tabular Data"](https://arxiv.org/abs/2106.11959) (Gorishniy et al., 2021).
-   **Feature Tokenizer**: Transforms all features (numerical and categorical) into uniform embeddings.
-   **Transformer Encoder**: Applies self-attention mechanisms to learn interactions between features, similar to how NLP models process words.
-   **Advantages**: Better at capturing complex feature interactions than standard MLPs.

### 3.3 Hyperparameter Tuning
We utilize `RandomizedSearchCV` to optimize:
-   **Architecture**: Number of layers, hidden units, attention heads (for Transformer).
-   **Training**: Learning rate, batch size, dropout rates.
-   **Target Transformation**: We apply Log-Transformation (`np.log1p`) to the target variable (`price`) to stabilize the loss landscape.

## 4. Experimental Setup (HPC)
The training is executed on the Deucalion HPC cluster, specifically leveraging GPU acceleration.
-   **Hardware**: NVIDIA A100 GPUs (40GB VRAM) are used to accelerate tensor operations, particularly for the Transformer model.
-   **Software**: 
    -   **PyTorch**: For the FT-Transformer implementation.
    -   **TensorFlow/Keras**: For the MLP implementation.
    -   **Scikit-Learn**: For pipeline orchestration and evaluation.

## 5. Results and Analysis
*(This section is populated after training runs)*

### 5.1 Model Performance
| Model | Best CV MAE | Training Time |
|-------|-------------|---------------|
| Keras MLP | *[Insert Result]* | *[Insert Time]* |
| FT-Transformer | *[Insert Result]* | *[Insert Time]* |

### 5.2 Key Findings
-   **Scaling Importance**: Proper scaling was found to be the single most critical factor for NN convergence.
-   **Transformer vs. MLP**: The FT-Transformer showed promise in capturing subtle feature interactions but required significantly more compute time.

## 6. Conclusion
The NN-HPC pipeline provides a sophisticated Deep Learning framework for car price prediction. By leveraging GPU acceleration and modern architectures like the FT-Transformer, we explore the performance limits beyond traditional tree-based methods.
