"""
save_model.py
--------------
Module 8: Model Saving for Cookie Cats Retention Prediction.

Purpose:
    Retrain the final, best-tuned Random Forest model (using the best
    hyperparameters found by GridSearchCV in Module 7) on the FULL
    training data, then save it to disk using Joblib. This saved file
    becomes the model the Streamlit app (Module 9) will load directly,
    without needing to retrain every time someone uses the app.

Why retrain here instead of reusing the GridSearchCV model object?
    GridSearchCV's best_estimator_ is already a fully trained model.
    In a real pipeline you could save that directly. Here, we
    re-define it explicitly with the best parameters we found, so this
    script is self-contained and reproducible on its own - you can run
    this file independently at any time to regenerate the saved model.

Author : (Your Name)
Project: Cookie Cats Retention Prediction - Game District Portfolio Project
"""

# ---------------------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------------------
import pandas as pd
import os
import joblib  # the standard library for saving/loading scikit-learn models
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


# ---------------------------------------------------------------------------
# 2. PATH CONFIGURATION
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

X_TRAIN_PATH = os.path.join(OUTPUT_DIR, "X_train.csv")
X_TEST_PATH = os.path.join(OUTPUT_DIR, "X_test.csv")
Y_TRAIN_PATH = os.path.join(OUTPUT_DIR, "y_train.csv")
Y_TEST_PATH = os.path.join(OUTPUT_DIR, "y_test.csv")

# This is the final saved model file - app.py (Module 9) will load this.
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "best_model.joblib")

os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 3. LOAD DATA
# ---------------------------------------------------------------------------
def load_splits():
    """Load the train/test splits saved by Module 5."""
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 4. BUILD AND TRAIN THE FINAL MODEL
# ---------------------------------------------------------------------------
def build_final_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Build the final Random Forest using the best hyperparameters found
    by GridSearchCV in Module 7.

    IMPORTANT: Update the values below to match YOUR GridSearchCV output
    from Module 7 if they differ (these are the values from your run:
    max_depth=10, min_samples_split=10, n_estimators=200).
    """
    final_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        class_weight="balanced",   # keeps the Recall improvement from Module 7
        random_state=42
    )

    final_model.fit(X_train, y_train)
    print("Final model trained using best hyperparameters from GridSearchCV.")

    return final_model


# ---------------------------------------------------------------------------
# 5. SAVE THE MODEL
# ---------------------------------------------------------------------------
def save_model(model: RandomForestClassifier):
    """
    Save the trained model to disk using Joblib.

    Why Joblib (and not pickle directly)?
        Joblib is the standard tool recommended by scikit-learn itself
        for saving models. It is more efficient than plain pickle for
        objects containing large numpy arrays (which Random Forest's
        many decision trees internally are), resulting in faster
        save/load times and smaller file sizes.
    """
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")


# ---------------------------------------------------------------------------
# 6. VERIFY THE SAVED MODEL WORKS
# ---------------------------------------------------------------------------
def verify_saved_model(X_test: pd.DataFrame, y_test: pd.Series):
    """
    Load the model back from disk and confirm it produces sensible
    predictions - this is a sanity check to make sure save/load works
    correctly before we rely on it in the Streamlit app.
    """
    print("\nVerifying saved model...")

    # joblib.load() reads the saved file back into a usable model object,
    # exactly as it was before saving - no retraining needed.
    loaded_model = joblib.load(MODEL_SAVE_PATH)

    y_pred = loaded_model.predict(X_test)
    y_proba = loaded_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"Loaded model -> Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
    print("(These should match the final tuned model's results from Module 7.)")


# ---------------------------------------------------------------------------
# 7. MAIN EXECUTION
# ---------------------------------------------------------------------------
def run_model_saving():
    """Run the full Module 8 pipeline: build -> save -> verify."""
    X_train, X_test, y_train, y_test = load_splits()

    final_model = build_final_model(X_train, y_train)

    save_model(final_model)

    verify_saved_model(X_test, y_test)

    print("\nModule 8 - Model Saving completed successfully.")
    print(f"The file '{os.path.basename(MODEL_SAVE_PATH)}' is now ready to be")
    print("loaded by app.py for the Streamlit deployment (Module 9).")


if __name__ == "__main__":
    run_model_saving()