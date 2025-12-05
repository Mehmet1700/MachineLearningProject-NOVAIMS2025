# Wall of Shame: Code Errors and Bad Practices

## Data Leakage: Global Imputation
# ERROR: Calculating statistics (mean) on the entire dataset before splitting. 
# This leaks information from the test set into the training process.
def bad_imputation(df):
    # Using global statistics compromises model integrity
    global_mean = df['salary'].mean() # Uses ENTIRE dataset's information
    df['salary_filled'] = df['salary'].fillna(global_mean)

## Improper Scaling: Fitting on Test Data
# ERROR: Using .fit_transform() on the test set. 
# The scaler should only be fitted on the train set, then .transform() applied to the test set.
# Fitting on the test set calculates a new mean/std based on test data (Leakage/Inconsistency).
scaler = StandardScaler()
numerical_cols = train_clean.select_dtypes(include=np.number).columns

train_numerical_scaled = scaler.fit_transform(train_clean[numerical_cols])
test_numerical_scaled = scaler.fit_transform(test_clean[numerical_cols]) # <--- WRONG

## Improper Scaling: Function with no return
# ERROR: The function fits a scaler inside but does not return the fitted scaler object.
# You cannot apply the exact same scaling rules to new data (test set) later.
def correct_scaling(train_ohe):
    num_cols = train_ohe.select_dtypes(include=['number']).columns.tolist()
    # ... code removing price ...
    scaler = MinMaxScaler()
    
    # fit to training data
    scaler.fit(train_ohe[num_cols]) 
    
    # transform the data
    train_scaled_values = scaler.transform(train_ohe[num_cols])
    
    # ... dataframe reconstruction ...
    return train_scaled

## Inefficient One-Hot Encoding
# ERROR: Using pd.get_dummies is not "wrong", but using sklearn's OneHotEncoder is 
# preferred for pipelines to handle unknown categories in the test set automatically.
train_categorical_scaled_df = pd.get_dummies(train_categorical_df, columns=categorical_cols.tolist(), drop_first=True)
test_categorical_scaled_df = pd.get_dummies(test_categorical_df, columns=categorical_cols.tolist(), drop_first=True)
# Align columns between train and test (in case test has different categories)
train_categorical_scaled_df, test_categorical_scaled_df = train_categorical_scaled_df.align(
    test_categorical_df, join='left', axis=1, fill_value=0
)

## Inefficient Encoding Loops (Ordinal)
# ERROR: Looping through columns to apply OrdinalEncoder one by one. 
# OrdinalEncoder can handle 2D DataFrames directly (vectorized).
def apply_ordinal_encoding(car1, car_test1, categorical_columns):
    encoders = {}
    for column in categorical_columns: # <--- Unnecessary loop
        if column not in car1.columns:
            continue
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        # Fit and transform on car (train)
        car1[column] = encoder.fit_transform(car1[[column]])
        # Transform on car_test (test)
        car_test1[column] = encoder.transform(car_test1[[column]])

## Inefficient Encoding Loops (LabelEncoder)
# ERROR: Using LabelEncoder (intended for targets) on features via a loop, 
# and manually handling unseen categories with lambda functions.
for col in categorical_cols:
    le = LabelEncoder()
    # ... fitting logic ...
    # Handle unseen categories by replacing them with the first known class
    df_copy[col] = df_copy[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])

## Evaluation on Training Data (Overfitting)
# ERROR: The model is fitted on `X_val_method` and then evaluated on `X_val_method`.
# This evaluates the model on the data it just memorized (training score), not a holdout set.
# Also: Fitting models inside a loop for feature comparison without proper CV.
for method_name, (X_val_method, features) in all_methods.items():
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_val_method, y_val.values.flatten()) # Fitting on val set?
    y_pred = model.predict(X_val_method)            # Predicting on the same set
    mae = mean_absolute_error(y_val, y_pred)        # This score is meaningless (overfitted)

## Re-fitting Imputers Incorrectly
# ERROR: Creating and fitting a new imputer every time the function is called.
# This prevents consistent imputation across train/test splits in production.
def impute_numeric_features(X_features_list, metric_features):
    imputer = SimpleImputer(strategy='median')
    X_train_numeric_to_fit = X_features_list[0][metric_features]
    imputer.fit(X_train_numeric_to_fit) # <--- Fits new imputer inside function
    # ... transformation logic ...

## Wrong Statistical Test
# ERROR: Using Chi-Square (Categorical vs Categorical) to select features for 
# a Regression problem (Continuous Target).
def TestIndependence(X, y, var, alpha=0.05):
    dfObserved = pd.crosstab(y, X) # Crosstab of Continuous Y vs Categorical X is huge/sparse
    chi2, p, dof, expected = stats.chi2_contingency(dfObserved.values)
    # ... logic deciding to discard feature based on this p-value ...

## Wrong Model for RFE
# ERROR: Using a Classifier (LogisticRegression) for Recursive Feature Elimination (RFE)
# on a Regression task (predicting car price).
model = LogisticRegression() # <--- Classifier
rfe = RFE(estimator=model, n_features_to_select=3)

## Insights without Action
# ERROR: Identifying high p-values (insignificant variables) in OLS regression 
# but failing to remove them or act on the insight.
# (Code output shows 'model_freq' p-value = 0.553, but text says "For the rest... We should not drop them")

## Using R-Squared for Wrapper Method Selection
# ERROR: Using .score() (R-squared) to determine the optimal number of features in RFE.
# R-squared naturally increases as you add more features. It does not penalize complexity.
for n in np.arange(1, len(train_numerical_scaled_df.columns)+1):
    model = LinearRegression()
    rfe = RFE(estimator=model, n_features_to_select=n)
    # ...
    val_score = model.score(X_val_rfe, y_test) # R-squared
    if(val_score >= high_score):
        high_score = val_score
        nof = n # Will almost always pick max features

## Inefficient Cross-Validation
# ERROR: Running cross_val_score 3 separate times (once for MSE, once for MAE, once for R2).
# This triples the computation time unnecessarily.
mse_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
mae_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

## Nested CV / Leakage in GridSearch
# ERROR: Fitting GridSearchCV on `X_train_FINAL` which has likely already been 
# imputed/scaled using global info. This is "Nested CV" leakage.
grid_search_rf = GridSearchCV(estimator=rf_model, cv=5, ...)
grid_search_rf.fit(X_train_FINAL, y_train) 

## Inefficient Manual Mapping
# ERROR: Writing complex manual loops to fuzzy match strings instead of using 
# optimized libraries or cleaner mapping logic.
def learn_transmission_map(series, canon=["transmission"], threshold=80):
    mapping = {}
    for u in s.dropna().unique():
        # ... verbose manual matching logic ...
        # ... loop inside loop ...

## Inconsistent Imputation Logic
# ERROR: Imputing test data using the median OF THE TEST DATA (independent median).
# Test data should always be imputed using statistics derived from the TRAIN data.
if isTest == False:
    imputer = KNNImputer(n_neighbors=5)
    df[features_to_impute] = imputer.fit_transform(df[features_to_impute])
else:
    # For test set: fill missing numerics with train-independent median
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median()) # <--- Wrong, uses test median

## Redundant Evaluation
# ERROR: Fitting a model manually, then running CV on the same training data immediately after.
# Often implies confusion about which metric (CV vs Holdout) to use for decision making.
model.fit(data['X_train'], y_train.values.ravel())
y_pred = model.predict(data['X_val'])
# ... 
cv_scores = cross_val_score(model, data['X_train'], ...) # Why CV now?

## Target Leakage
# ERROR: Using the target variable ('price') as a feature to impute missing values 
# in other features ('mileage'). You won't have 'price' when predicting new data.
features = ['year', 'price', 'fuel_type_encoded'] # <--- Price is the target!
target = 'mileage'
# ...
rf_model.fit(X_train, y_train) # Training a model that relies on Price

## Double Preprocessing
# ERROR: Performing manual preprocessing (imputation/encoding) earlier in the notebook,
# and then feeding that PROCESSED data into a sklearn Pipeline that tries to process it AGAIN.
numerical_features = ['mpg', 'engineSize', ...]
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features), # Scaling data that might already be scaled
        ('cat_oh', OneHotEncoder(...), oh_encoded_features),
    ]
)

## Inconsistent Mode Imputation
# ERROR: Comments say "Use train mode for test", but code fills with string "Unknown".
# Code does not match comments/intent.
for col in categorical_cols:
    train_clean[col].fillna("Unknown", inplace=True)
    test_clean[col].fillna("Unknown", inplace=True) # Comment says "Use train mode"??

## Bad Justifications (LLM Hallucinations)
# ERROR: Using generic text (likely LLM generated) that explains concepts incorrectly 
# or irrelevantly (e.g., claiming StandardScaler winsorizes data).
# "We used StandardScaler... Formula is z=(x-mean)/std" (Generic filler text)
# "scaled our data using the Robust Scaler ... and removing/winsorizing all of them would be inefficient and, by using this technique, we can handle these better." (RobustScaler does not winsorize).

## Acknowledging LLM for Trivial Tasks
# ERROR: Explicitly commenting that ChatGPT was used for basic syntax.
# #here we used chatGPT to help us to construct the following DataFrame
negatives_summary = pd.DataFrame(...)