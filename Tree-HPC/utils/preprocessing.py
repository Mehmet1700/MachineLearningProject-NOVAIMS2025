# utils/preprocessing.py
from typing import List, Tuple
import pandas as pd
import json
import re
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def get_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Infer numeric vs categorical features from dtypes.
    Only uses types (no statistics) → no leakage.
    """
    numeric_features = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = df.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()
    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    ColumnTransformer:
      - numeric: median imputation + standard scaling
      - categorical: most_frequent imputation + one-hot encoding
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def clean_dataset(df: pd.DataFrame, mapping_dir: str | Path = "mapping", drop_outliers: bool = False) -> pd.DataFrame:
    df = df.copy()
    mapping_dir = Path(mapping_dir)

    # --- 0. Initial Cleanup (Type Casting & String Normalization) ---
    # Lowercase all string columns
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.lower().replace({"nan": pd.NA, "none": pd.NA, "": pd.NA})

    # --- 1. Fuel Type ---
    try:
        with open(mapping_dir / "fueltype_mapping.json", "r", encoding="utf-8") as f:
            fuel_map = json.load(f)
        
        def norm_fueltype(fueltype):
            if pd.isna(fueltype): return fueltype
            fueltype = str(fueltype).strip().lower()
            fueltype = re.sub(r'[.,-_]', ' ', fueltype)
            fueltype = ' '.join(fueltype.split())
            return fueltype

        if "fuelType" in df.columns:
            df["fuelType"] = df["fuelType"].apply(norm_fueltype).map(fuel_map)
    except Exception as e:
        print(f"Warning: Could not clean fuelType: {e}")

    # --- 2. Transmission ---
    try:
        with open(mapping_dir / "transmission_mapping.json", "r", encoding="utf-8") as f:
            trans_map = json.load(f)

        def norm_transmission(transmission):
            if pd.isna(transmission): return transmission
            transmission = str(transmission).strip().lower()
            transmission = re.sub(r'[.,_]', ' ', transmission)
            transmission = ' '.join(transmission.split())
            return transmission

        if "transmission" in df.columns:
            df["transmission"] = df["transmission"].apply(norm_transmission).map(trans_map)
            # Remove unknown transmission values
            df.loc[df['transmission'] == 'Unknown', 'transmission'] = np.nan
    except Exception as e:
        print(f"Warning: Could not clean transmission: {e}")

    # --- 3. Brand ---
    try:
        with open(mapping_dir / "brandname_mapping.json", "r", encoding="utf-8") as f:
            brand_map = json.load(f)

        def norm_brand(brand):
            if pd.isna(brand): return pd.NA
            brand = str(brand).strip().lower()
            brand = re.sub(r"[.,-_]", " ", brand)
            brand = " ".join(brand.split())
            return brand

        if "Brand" in df.columns:
            nb = df["Brand"].apply(norm_brand)
            mapped = nb.map(brand_map)
            df["Brand"] = df["Brand"].where(mapped.isna(), mapped)
    except Exception as e:
        print(f"Warning: Could not clean Brand: {e}")

    # --- 4. Model ---
    try:
        with open(mapping_dir / "modelname_mapping.json", "r", encoding="utf-8") as f:
            model_map_data = json.load(f)
        
        ALIASES = model_map_data.get("aliases", {})
        REGEX_RULES = model_map_data.get("regex_rules", [])

        def norm_model(x):
            if pd.isna(x): return np.nan
            s = str(x).lower()
            s = re.sub(r'[\s\-_]+', '', s)
            s = re.sub(r'[^a-z0-9\+]', '', s)
            return s

        def post_canon_model(s):
            if pd.isna(s): return s
            t = str(s)
            t = re.sub(r'\\b([ACEGSV]) Clas\\b', r'\\1-Class', t)
            t = re.sub(r'\\b(GL[ABCES]) Clas\\b', r'\\1-Class', t)
            t = re.sub(r'\\b([A-Z]{2}) Class\\b', r'\\1-Class', t)
            t = re.sub(r'^([a-z])\-Class$', lambda m: m.group(1).upper() + "-Class", t)
            t = re.sub(r'^gl([abcse])\-Class$', lambda m: "GL" + m.group(1).upper() + "-Class", t, flags=re.I)
            t = re.sub(r'^I(10|20|30|40|800)$', r'i\\1', t)
            t = re.sub(r'^IX(1|20|35)$', r'ix\\1', t)
            t = re.sub(r'^tt$', 'TT', t, flags=re.I)
            t = re.sub(r'^r8$', 'R8', t, flags=re.I)
            t = re.sub(r'^sq7$', 'SQ7', t, flags=re.I)
            return t

        def apply_regex_rules(norm_key):
            out = norm_key
            matched = False
            for rule in REGEX_RULES:
                pat = rule.get("pattern", "")
                rep = rule.get("replace", "")
                new = re.sub(pat, rep, out)
                if new != out:
                    matched = True
                out = new
            return out, matched

        if "model" in df.columns:
            original = df["model"].astype("string")
            normed = original.apply(norm_model)
            alias_mapped = normed.map(ALIASES) if ALIASES else pd.Series(pd.NA, index=df.index)
            out = original.where(alias_mapped.isna(), alias_mapped)
            
            need = out.isna() | (out == original)
            if need.any():
                def _rx_or_na(k):
                    if isinstance(k, str):
                        res, ok = apply_regex_rules(k)
                        return res if ok else pd.NA
                    return pd.NA
                rx_res = normed[need].apply(_rx_or_na)
                out.loc[need] = out.loc[need].where(rx_res.isna(), rx_res)
            
            df["model"] = out.apply(post_canon_model)
            
            # Fix specific model issues
            df.loc[df['model'].str.lower() == 'kadjar', 'model'] = np.nan

    except Exception as e:
        print(f"Warning: Could not clean model: {e}")

    # --- 5. Fill Brand from Model ---
    try:
        with open(mapping_dir / "brand_model_mapping.json", "r", encoding="utf-8") as f:
            model_to_brand = json.load(f)
        
        if "Brand" in df.columns and "model" in df.columns:
            mask = df["Brand"].isna() & df["model"].notna()
            mapped = df.loc[mask, "model"].map(model_to_brand)
            df.loc[mask, "Brand"] = mapped
    except Exception as e:
        print(f"Warning: Could not fill Brand from Model: {e}")

    # --- 6. Outliers & Errors ---
    
    # Engine Size
    if "engineSize" in df.columns:
        df.loc[((df['engineSize'] > 6.2) | (df['engineSize'] < 0.8)), 'engineSize'] = np.nan

    # Year
    if "year" in df.columns:
        # Non-integer years
        df.loc[df['year'] % 1 != 0, 'year'] = np.nan
        # Future years
        df.loc[df['year'] > 2020, 'year'] = np.nan
        
        # Negative years -> set mileage to NaN (as per notebook logic)
        if "mileage" in df.columns:
            df.loc[df['year'] < 0, 'mileage'] = np.nan

    # Tax
    if "tax" in df.columns:
        df.loc[df['tax'] < 0, 'tax'] = np.nan

    # Mileage
    if "mileage" in df.columns:
        df.loc[df['mileage'] < 0, 'mileage'] = np.nan

    # Previous Owners
    if "previousOwners" in df.columns:
        df.loc[df['previousOwners'] < 0, 'previousOwners'] = np.nan
        df.loc[df['previousOwners'] % 1 != 0, 'previousOwners'] = np.nan

    # MPG Outliers
    if "mpg" in df.columns and "fuelType" in df.columns:
        unusual_mpg_mask = ((df['mpg'] > 150) & (df["fuelType"] == "Electric")) | \
               ((df['mpg'] < 70) & (df["fuelType"] == "Electric")) | \
               ((df['mpg'] > 100) & (df["fuelType"] == "Hybrid")) | \
               ((df['mpg'] < 35) & (df["fuelType"] == "Hybrid")) | \
               ((df['mpg'] > 80) & (df["fuelType"] != "Hybrid") & (df["fuelType"] != "Electric")) | \
               ((df['mpg'] < 8) & (df["fuelType"] != "Hybrid") & (df["fuelType"] != "Electric"))
        df.loc[unusual_mpg_mask, 'mpg'] = np.nan

    # Drop PaintQuality
    if "paintQuality%" in df.columns:
        df.drop(columns=['paintQuality%'], inplace=True)

    # --- 7. Drop Rows (Only for Train) ---
    if drop_outliers:
        if "year" in df.columns:
            before = len(df)
            df = df[df["year"] >= 2000]
            after = len(df)
            if before != after:
                print(f"  -> Dropped {before - after} rows with year < 2000")

    return df


def remove_price_outliers_per_model(X: pd.DataFrame, y: pd.Series, multiplier: float = 1.5) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Remove rows where price is an outlier within its car model group.
    Outliers are defined as values outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR].
    """
    # Combine X and y temporarily
    df = X.copy()
    y_name = y.name or "price"
    df[y_name] = y.values
    
    initial_len = len(df)
    
    # Calculate bounds per model
    stats = df.groupby('model')[y_name].agg(
        Q1=lambda x: x.quantile(0.25),
        Q3=lambda x: x.quantile(0.75)
    )
    stats['IQR'] = stats['Q3'] - stats['Q1']
    stats['lower'] = stats['Q1'] - multiplier * stats['IQR']
    stats['upper'] = stats['Q3'] + multiplier * stats['IQR']
    
    # Map bounds to original df
    df['lower_bound'] = df['model'].map(stats['lower'])
    df['upper_bound'] = df['model'].map(stats['upper'])
    
    # Filter: Keep rows where price is within bounds OR bounds are NaN (e.g. too few samples)
    # Also keep rows where model was not found in stats (should not happen if grouped by model)
    mask_keep = (
        (df[y_name] >= df['lower_bound']) & 
        (df[y_name] <= df['upper_bound'])
    ) | df['lower_bound'].isna()
    
    df_filtered = df[mask_keep]
    
    dropped_count = initial_len - len(df_filtered)
    print(f"  -> Dropped {dropped_count} price outliers based on model-specific IQR (multiplier={multiplier})")
    
    # Separate back
    X_filtered = df_filtered.drop(columns=[y_name, 'lower_bound', 'upper_bound'])
    y_filtered = df_filtered[y_name]
    
    # Restore index name if needed
    if hasattr(y, 'name'):
        y_filtered.name = y.name
        
    return X_filtered, y_filtered
