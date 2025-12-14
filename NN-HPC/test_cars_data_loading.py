import sys
import os

# Add the current directory to sys.path so we can import from data_loaders
sys.path.append(os.getcwd())

from data_loaders.cars_data import load_train_val_test

import os

# Returns the "Current Working Directory"
aktueller_pfad = os.getcwd()
print(f"You are here: {aktueller_pfad}")

print("Testing load_train_val_test...")
try:
    X_train, y_train, X_val, y_val, X_test = load_train_val_test(
        train_path="../data/train.csv",
        test_path="../data/test.csv",
        mapping_dir="../mapping"
    )

    print("\n--- Results ---")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape:   {X_val.shape}")
    print(f"y_val shape:   {y_val.shape}")
    print(f"X_test shape:  {X_test.shape}")

    # Verification
    expected_test_rows = 32567  # Based on user's previous message
    if X_test.shape[0] == expected_test_rows:
        print(f"\nSUCCESS: X_test has {expected_test_rows} rows.")
    else:
        print(f"\nWARNING: X_test has {X_test.shape[0]} rows, expected {expected_test_rows}.")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
