**Data**
- `data/train.csv`: Training dataset with features and target variable.
- `data/test.csv`: Test dataset with the same feature schema as train, no target.
- `data/sample_submission.csv`: Example submission template with expected output format.
- `data/processed_data/`: Cleaned datasets after initial normalization and type fixes
  - `11_processed_train_data.csv`, `11_processed_test_data.csv`.
- `data/encoded_data/`: Splits and encoded matrices used for modeling
  - `12_x_train.csv`, `12_x_val.csv`, `12_x_test.csv`, `12_y_train.csv`, `12_y_val.csv`.
- `data/feature_selection/`: Feature-selected matrices used in later modeling
  - `20_x_train.csv`, `20_x_val.csv`, `20_x_test.csv`, `20_y_train.csv`, `20_y_val.csv`.
- `data/submissions/`: Generated submission files from modeling runs (timestamped `30_submission_*.csv`).

**Mapping**
- `mapping/brandname_mapping.json`: Normalizes noisy brand strings (e.g., abbreviations and misspellings) to canonical brand names.
- `mapping/modelname_mapping.json`: Maps model aliases and variations to standardized model names.
- `mapping/brand_model_mapping.json`: Derives canonical brand from standardized model names (e.g., "A4" → "Audi").
- `mapping/fueltype_mapping.json`: Harmonizes fuel type strings to `Petrol`, `Diesel`, `Hybrid`, `Electric`, or `Other`.
- `mapping/transmission_mapping.json`: Normalizes transmission values to `Manual`, `Automatic`, `Semi-Auto`, `Unknown`, or `Other`.

**submission1_notebooks**
- `00_data_exploration.ipynb`: Initial EDA on train/test, target distribution checks, and data quality review.
- `11_data_mapping_and_normalization.ipynb`: Applies brand/model/fuel/transmission mappings and basic normalization.
- `12_data_imputation_encoding_and_feature_engineering.ipynb`: Imputes missing values, encodes categorical features, and engineers additional features.
- `12b_data_imputation_encoding_and_feature_engineering.ipynb`: Iteration/refinement on imputation and encoding pipeline.
- `20_feature_selection.ipynb`: Evaluates and selects predictive features (filter/wrapper/model-based methods).
- `30_modeling_and_evaluation.ipynb`: Trains baseline and tuned models, evaluates performance, and produces submission files.
- `images/`: Figures referenced by the notebooks.
