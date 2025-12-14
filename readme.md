

# Cars 4 You: Expediting Car Evaluations with Machine Learning

**Group 21**  
**Course:** Machine Learning (2025-2026)  
**Institution:** NOVA IMS  

## Team Members
- **20250413** João Ramos
- **20250401** Oliver Kain
- **20250415** Gonçalo Castro
- **20250344** Mehmet Karaca


Link to the Github Repository: https://github.com/Mehmet1700/MachineLearningProject-NOVAIMS2025 

---

## 1. Project Overview
The **Cars 4 You** project aims to develop a robust machine learning pipeline to predict the selling price of used cars. By leveraging a dataset of vehicle attributes (brand, model, year, mileage, etc.), we aim to provide accurate price evaluations to expedite the trading process.

Our approach combines rigorous data preprocessing, advanced feature engineering, and a diverse set of modeling techniques ranging from ensemble tree methods to deep learning architectures.

## 2. Repository Structure

```
MachineLearningProject-NOVAIMS2025/
├── data/                       # Raw and processed datasets
│   ├── processed_data/         # Cleaned data after normalization
│   ├── encoded_data/           # Encoded matrices for modeling
│   ├── feature_selection/      # Datasets after feature selection
│   └── submissions/            # Final Kaggle submission files
├── mapping/                    # JSON files for standardizing categorical values
│   ├── brandname_mapping.json  # Normalization of brand names
│   ├── modelname_mapping.json  # Normalization of model names
│   └── ...
├── notebooks/                  # Exploratory Data Analysis (EDA) and experiments
│   ├── 00_intro_and_setup.ipynb
│   ├── 10_data_exploration_*.ipynb
│   └── ...
├── submission2/                # MAIN SUBMISSION CODEBASE (Notebooks)
│   ├── ML_Group21_Notebook.ipynb # Master notebook consolidating the workflow
│   ├── 122030_model_per_brand.ipynb # Brand-specific modeling approach
│   └── ...
├── Tree-HPC/                   # HPC Pipeline for Tree-Based Models
│   ├── main.py                 # Entry point for training Tree models
│   ├── pipelines.py            # Scikit-learn pipelines
│   └── train_tree.slurm        # SLURM script for cluster execution
└── NN-HPC/                     # HPC Pipeline for Neural Networks
    ├── main.py                 # Entry point for training NN models
    ├── wrappers/               # Scikit-learn wrappers for PyTorch/Keras models
    └── train_nn_gpu.slurm      # SLURM script for cluster execution
```

## 3. Methodology

### 3.1 Data Integration & Exploration
We started by integrating training and test datasets and performing a comprehensive Exploratory Data Analysis (EDA). This phase involved:
- Analyzing feature distributions and correlations.
- Identifying data quality issues (missing values, inconsistencies, outliers).
- Visualizing target variable (`price`) distribution.

### 3.2 Preprocessing & Feature Engineering
Our preprocessing pipeline is designed to be robust and prevent data leakage:
- **Mapping & Normalization**: Standardizing noisy text fields (e.g., `fuelType`, `transmission`) using custom JSON mappings.
- **Imputation**: A hybrid strategy employing:
    - **Rule-Based Imputation**: Using domain knowledge (e.g., median values per model/year).
    - **Model-Based Imputation**: Using Random Forest to predict missing values.
- **Feature Engineering**: Creating new features such as `efficiency_ratio`, `mileage_per_year`, and `brand_segment`.
- **Encoding**: Utilizing One-Hot Encoding and Target Encoding where appropriate.

### 3.3 Modeling Strategy
We explored two distinct modeling paths:

#### A. Ensemble Methods (Tree-HPC & Submission 2)
Located in the `Tree-HPC/` and `submission2/` folders, this approach focuses on tree-based ensembles:
- **Models**: Random Forest Regressor, Extra Trees Regressor, Histogram Gradient Boosting.
- **Strategy**: 
    - **Global Models**: Trained on the entire dataset.
    - **Brand-Specific Models**: Separate models trained for each car brand to capture specific market dynamics.
- **Selection**: Features were selected using Permutation Importance and Variance Thresholds.

#### B. Deep Learning (NN-HPC)
Located in the `NN-HPC/` folder, this pipeline is optimized for high-performance computing:
- **Models**: Multi-Layer Perceptrons (MLP) and FT-Transformer (Feature Tokenizer Transformer).
- **Pipeline**: Includes Log-Target Regression (`TransformedTargetRegressor`) to handle skewed price distributions.
- **Infrastructure**: Designed to run on SLURM-managed GPU clusters.

## 4. Key Features
- **Leakage Prevention**: Strict separation of training and validation data during preprocessing (fitting on train, transforming on val/test).
- **Robust Imputation**: Handling missing data without relying on global statistics.
- **Automated Pipeline**: From raw data to submission file generation.
- **Professional Code Structure**: Modularized code for reproducibility and scalability.

## 5. How to Run

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the HPC Pipelines
For detailed instructions on how to run the High-Performance Computing pipelines, please refer to the specific README files located in their respective directories:

- **Tree-Based Models**: [Tree-HPC/README.md](Tree-HPC/README.md)
- **Neural Networks**: [NN-HPC/readme.md](NN-HPC/readme.md)

### Running the Notebooks (Submission 2)
Navigate to the `submission2` folder and execute the notebooks in the following order (or run the master notebook):
1. `11_data_mapping_and_normalization.ipynb`
2. `12c_data_imputation_encoding_and_feature_engineering.ipynb`
3. `30_modeling_and_evaluation.ipynb`

*Alternatively, open `ML_Group21_Notebook.ipynb` for the complete workflow.*
