"""
feature_engineering.py
-----------------------
Module 4: Feature Engineering for Cookie Cats Retention Prediction.

Purpose:
    1. Load the cleaned dataset.
    2. Encode the categorical 'version' column using One-Hot Encoding.
    3. Separate the data into Features (X) and Target (y).
    4. Save the processed, model-ready dataset to outputs/ so the next
       module (Train-Test Split) can load it directly.

Why this module exists:
    Machine Learning models only understand numbers. 'version' is text
    ('gate_30' / 'gate_40'), so it must be converted to a numeric format
    before any model can use it. We also drop columns that are not
    useful as predictive features (like userid, which is just an ID).

Author : (Your Name)
Project: Cookie Cats Retention Prediction - Game District Portfolio Project
"""

# ---------------------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------------------
import pandas as pd
import os


# ---------------------------------------------------------------------------
# 2. PATH CONFIGURATION
# ---------------------------------------------------------------------------
# Same pattern as eda.py: build paths relative to this file's location,
# so the script works regardless of which folder you run it from.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # -> Cookie_Cats_Project/
DATA_PATH = os.path.join(BASE_DIR, "data", "cookie_cats.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Where we will save the final, model-ready dataset.
PROCESSED_DATA_PATH = os.path.join(OUTPUT_DIR, "processed_data.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 3. LOAD DATA
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load the raw (but already cleaned) Cookie Cats dataset."""
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 4. ENCODE THE 'version' COLUMN
# ---------------------------------------------------------------------------
def encode_version(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-Hot Encode the 'version' column.

    Why One-Hot Encoding (and not Label Encoding)?
        'version' has no natural order - gate_30 is not "less than" or
        "greater than" gate_40, they are just two different designs.
        Label Encoding (0/1 on the same column) would falsely imply
        an order/ranking to the model. One-Hot Encoding avoids this by
        creating a separate binary column per category.

    Why drop_first=True?
        With only 2 categories, keeping both columns is redundant -
        if 'version_gate_40' = 0, we already know the player is on
        gate_30. Keeping only one column avoids this redundancy
        (known as the "dummy variable trap") and keeps the dataset
        smaller and cleaner.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset containing the raw 'version' column.

    Returns
    -------
    pd.DataFrame
        The dataset with 'version' replaced by a single numeric column:
        'version_gate_40' (1 = gate_40, 0 = gate_30).
    """
    # pd.get_dummies() is pandas' built-in One-Hot Encoding function.
    # columns=["version"]  -> only encode this column, leave others untouched
    # drop_first=True      -> drop the first category to avoid redundancy
    # dtype=int             -> store as 0/1 integers instead of True/False
    df_encoded = pd.get_dummies(df, columns=["version"], drop_first=True, dtype=int)

    print("Encoded column created: 'version_gate_40' (1 = gate_40, 0 = gate_30)")
    return df_encoded


# ---------------------------------------------------------------------------
# 5. PREPARE FEATURES (X) AND TARGET (y)
# ---------------------------------------------------------------------------
def prepare_features_and_target(df: pd.DataFrame):
    """
    Split the dataset into Features (X) and Target (y).

    Why drop 'userid'?
        It is just a unique identifier for each player - it carries no
        predictive information about behavior. Including it would let
        the model "memorize" individual users instead of learning
        general patterns, which hurts generalization.

    Why drop 'retention_1' from X?
        This is a judgment call worth understanding: 'retention_1' is
        highly correlated with 'retention_7' (our target) and is itself
        a strong predictor. Many real-world portfolio projects DO keep
        it as a feature, since at prediction time (e.g., 1 day after
        install) you would already know whether the player returned on
        Day 1. We keep it as a feature here since it is realistically
        available before Day 7 and is one of the most useful signals
        we found during EDA.

    Parameters
    ----------
    df : pd.DataFrame
        The encoded dataset.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (all predictive columns).
    y : pd.Series
        Target vector (retention_7 - what we want to predict).
    """
    # Target variable: what we want the model to predict.
    y = df["retention_7"]

    # Features: everything EXCEPT userid (not predictive) and retention_7
    # (this is the answer we're trying to predict, so it cannot be an input).
    X = df.drop(columns=["userid", "retention_7"])

    print(f"\nFeature columns (X): {list(X.columns)}")
    print(f"Target column (y): retention_7")
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")

    return X, y


# ---------------------------------------------------------------------------
# 6. MAIN EXECUTION
# ---------------------------------------------------------------------------
def run_feature_engineering():
    """
    Run the full feature engineering pipeline:
    load -> encode -> split into X/y -> save processed dataset.
    """
    # Step 1: Load the cleaned data
    df = load_data()

    # Step 2: Encode the categorical 'version' column
    df_encoded = encode_version(df)

    # Step 3: Separate features and target
    X, y = prepare_features_and_target(df_encoded)

    # Step 4: Save the processed dataset for the next module (Train-Test Split).
    # We recombine X and y into one file so train_model.py can load a single,
    # clean, ready-to-use CSV instead of re-doing this preprocessing.
    processed_df = X.copy()
    processed_df["retention_7"] = y
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"\nProcessed dataset saved to: {PROCESSED_DATA_PATH}")
    print("\nPreview of processed data:")
    print(processed_df.head())

    return X, y


if __name__ == "__main__":
    run_feature_engineering()
    