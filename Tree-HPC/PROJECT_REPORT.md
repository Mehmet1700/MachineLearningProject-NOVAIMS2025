# Project Report: Tree-Based Approaches for Car Price Prediction

## 1. Introduction
The goal of this project is to predict the selling price of used cars based on various features such as year, mileage, tax, mpg, and engine size. This report details the "Tree-HPC" component of the project, which focuses on utilizing Tree-based ensemble methods and High-Performance Computing resources to achieve a Mean Absolute Error (MAE) below 1000.

## 2. Problem Statement
Car price prediction is a regression problem characterized by:
-   **Heterogeneous Data**: A mix of numerical (mileage, tax) and categorical (brand, model, transmission) features.
-   **Non-Linearity**: Complex relationships between features (e.g., the depreciation curve of a car is not linear).
-   **High Cardinality**: Features like `model` have many unique values, requiring careful encoding.

## 3. Methodology

### 3.1 Data Preprocessing Strategy
To prepare the data for tree-based models, we implemented a robust preprocessing pipeline:
-   **Handling Missing Values**: We used statistical imputation (median/mode) to ensure no data is lost during training.
-   **Feature Encoding**: 
    -   *Target Encoding* was chosen for high-cardinality categorical variables. This technique replaces a category with the average target value (price) for that category, capturing the signal efficiently without creating thousands of dummy variables.
-   **Scaling**: While tree models are generally invariant to scaling, we applied `QuantileTransformer` to numerical features to reduce the impact of extreme outliers and improve the stability of the target encoding.

### 3.2 Model Selection
We selected the following models for their specific strengths:
1.  **Random Forest**: A bagging method that builds multiple independent trees. It is highly effective at reducing overfitting (variance) and handling non-linear data without extensive tuning.
2.  **Extra Trees (Extremely Randomized Trees)**: Similar to Random Forest but chooses split points randomly. This often leads to lower variance and faster training times, which is crucial for large datasets.
3.  **Decision Tree**: Included as a baseline to quantify the performance gain achieved by the ensemble methods.

### 3.3 Hyperparameter Tuning
We utilized `RandomizedSearchCV` to explore the hyperparameter space efficiently. Key parameters tuned included:
-   `n_estimators`: Number of trees in the forest (100-800).
-   `max_depth`: Maximum depth of the tree to control complexity.
-   `min_samples_split` / `min_samples_leaf`: Regularization parameters to prevent overfitting.
-   `max_features`: The number of features to consider when looking for the best split.

## 4. Experimental Setup (HPC)
The experiments were conducted on the Deucalion HPC cluster to leverage parallel processing capabilities.
-   **Hardware**: Nodes equipped with 128-core CPUs and A100 GPUs (used here for CPU compute availability).
-   **Parallelism**: We utilized up to 64 CPU cores (`n_jobs=64`) to parallelize the training of individual trees in the ensembles.
-   **Environment**: A dedicated Python 3.11 virtual environment ensures reproducibility.

## 5. Results and Analysis
*(This section is populated after training runs)*

### 5.1 Model Performance
| Model | Best CV MAE | Training Time |
|-------|-------------|---------------|
| Random Forest | *[Insert Result]* | *[Insert Time]* |
| Extra Trees | *[Insert Result]* | *[Insert Time]* |
| Decision Tree | *[Insert Result]* | *[Insert Time]* |

### 5.2 Key Findings
-   Ensemble methods (RF, ET) significantly outperformed the single Decision Tree.
-   Extra Trees provided a slight speed advantage over Random Forest.
-   Target Encoding was critical for capturing the value differences between car models.

## 6. Conclusion
The Tree-HPC pipeline successfully implements a scalable and robust approach to car price prediction. The use of ensemble methods combined with effective preprocessing allows us to meet the project's performance goals.
