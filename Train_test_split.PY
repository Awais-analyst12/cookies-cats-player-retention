"""
train_test_split.py
--------------------
Module 5: Train-Test Split for Cookie Cats Retention Prediction.

Purpose:
    Split the processed dataset (X, y) into training and testing sets,
    so the model can be trained on one portion of the data and evaluated
    on a completely separate, unseen portion. This is how we measure
    whether the model actually learned general patterns, instead of
    just memorizing the data it was trained on.

Author : (Your Name)
Project: Cookie Cats Retention Prediction - Game District Portfolio Project
"""

# ---------------------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------------------
import pandas as pd
import os
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# 2. PATH CONFIGURATION
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # -> Cookie_Cats_Project/
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "outputs", "processed_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Where the 4 split files will be saved, so train_model.py can load
# them directly without re-running this split every time.
X_TRAIN_PATH = os.path.join(OUTPUT_DIR, "X_train.csv")
X_TEST_PATH = os.path.join(OUTPUT_DIR, "X_test.csv")
Y_TRAIN_PATH = os.path.join(OUTPUT_DIR, "y_train.csv")
Y_TEST_PATH = os.path.join(OUTPUT_DIR, "y_test.csv")


# ---------------------------------------------------------------------------
# 3. LOAD THE PROCESSED DATASET
# ---------------------------------------------------------------------------
def load_processed_data():
    """
    Load the dataset produced by feature_engineering.py and re-split it
    into X (features) and y (target).
    """
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Processed data loaded. Shape: {df.shape}")

    # Re-separate features and target, same logic as feature_engineering.py
    X = df.drop(columns=["retention_7"])
    y = df["retention_7"]

    return X, y


# ---------------------------------------------------------------------------
# 4. SPLIT THE DATA
# ---------------------------------------------------------------------------
def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Split X and y into training and testing sets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    test_size : float
        Proportion of the data to reserve for testing (0.2 = 20%).
    random_state : int
        Seed for reproducibility - ensures we get the SAME split every
        time we run this script, instead of a different random split
        on each run.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple of pd.DataFrame / pd.Series
    """
    # train_test_split() is scikit-learn's standard splitting function.
    #
    # test_size=0.2    -> 80% of data goes to training, 20% to testing
    # random_state=42  -> makes the split reproducible (same result every run)
    # stratify=y       -> preserves the True/False ratio of retention_7
    #                     in BOTH the training and testing sets, which
    #                     matters because our target is imbalanced
    #                     (~18.6% True, ~81.4% False from our EDA).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"\nTraining set size: {X_train.shape[0]} rows ({(1 - test_size) * 100:.0f}%)")
    print(f"Testing set size:  {X_test.shape[0]} rows ({test_size * 100:.0f}%)")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 5. VERIFY STRATIFICATION WORKED
# ---------------------------------------------------------------------------
def verify_stratification(y_train: pd.Series, y_test: pd.Series):
    """
    Print the True/False ratio in both sets, so we can visually confirm
    that stratify=y preserved the original class balance in both splits.
    """
    train_ratio = y_train.mean()  # mean of a boolean column = proportion of True
    test_ratio = y_test.mean()

    print(f"\nRetention_7 = True ratio in TRAINING set: {train_ratio:.4f} ({train_ratio*100:.2f}%)")
    print(f"Retention_7 = True ratio in TESTING set:  {test_ratio:.4f} ({test_ratio*100:.2f}%)")
    print("(These two percentages should be nearly identical - that's stratify=y working correctly.)")


# ---------------------------------------------------------------------------
# 6. SAVE THE SPLITS
# ---------------------------------------------------------------------------
def save_splits(X_train, X_test, y_train, y_test):
    """Save all 4 split pieces to outputs/, ready for train_model.py to load."""
    X_train.to_csv(X_TRAIN_PATH, index=False)
    X_test.to_csv(X_TEST_PATH, index=False)
    y_train.to_csv(Y_TRAIN_PATH, index=False)
    y_test.to_csv(Y_TEST_PATH, index=False)

    print(f"\nSaved: {X_TRAIN_PATH}")
    print(f"Saved: {X_TEST_PATH}")
    print(f"Saved: {Y_TRAIN_PATH}")
    print(f"Saved: {Y_TEST_PATH}")


# ---------------------------------------------------------------------------
# 7. MAIN EXECUTION
# ---------------------------------------------------------------------------
def run_train_test_split():
    """Run the full train-test split pipeline."""
    X, y = load_processed_data()

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    verify_stratification(y_train, y_test)

    save_splits(X_train, X_test, y_train, y_test)

    print("\nTrain-Test Split completed successfully.")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_train_test_split()