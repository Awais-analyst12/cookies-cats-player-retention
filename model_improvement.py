"""
model_improvement.py
---------------------
Module 7: Model Improvement for Cookie Cats Retention Prediction.

Purpose:
    Take the best baseline model from Module 6 (Random Forest, which had
    the highest ROC-AUC) and improve it through 4 professional techniques:

    1. Cross Validation       - verify performance is stable, not a lucky split
    2. Class Imbalance Handling - improve Recall on the minority class (True)
    3. GridSearchCV            - automatically find the best hyperparameters
    4. Feature Importance      - understand which features drive predictions

Author : (Your Name)
Project: Cookie Cats Retention Prediction - Game District Portfolio Project
"""

# ---------------------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)


# ---------------------------------------------------------------------------
# 2. PATH CONFIGURATION
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

X_TRAIN_PATH = os.path.join(OUTPUT_DIR, "X_train.csv")
X_TEST_PATH = os.path.join(OUTPUT_DIR, "X_test.csv")
Y_TRAIN_PATH = os.path.join(OUTPUT_DIR, "y_train.csv")
Y_TEST_PATH = os.path.join(OUTPUT_DIR, "y_test.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 3. LOAD DATA
# ---------------------------------------------------------------------------
def load_splits():
    """Load the train/test splits saved by Module 5."""
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 4. CROSS VALIDATION
# ---------------------------------------------------------------------------
def run_cross_validation(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Run 5-Fold Stratified Cross Validation on a baseline Random Forest.

    Why Cross Validation?
        In Module 6, we trained on ONE train-test split and got ONE
        ROC-AUC score. But what if that particular split happened to be
        "easy" or "lucky"? We wouldn't know if the model is genuinely
        good, or if we just got fortunate with that specific 80/20 split.

        Cross Validation solves this by splitting the TRAINING data into
        5 equal parts ("folds"). It then trains 5 separate times, each
        time using 4 folds for training and 1 fold for validation,
        rotating which fold is held out. This gives us 5 ROC-AUC scores
        instead of 1 - if they are all similar, we can trust the model's
        performance is stable and not the result of a lucky split.

    Why StratifiedKFold (not plain KFold)?
        Same reasoning as stratify=y in Module 5 - our target is
        imbalanced (~18.6% True), so each fold needs to preserve that
        ratio, otherwise some folds might barely contain any True cases.
    """
    print(f"\n{'='*60}")
    print("STEP 1: CROSS VALIDATION")
    print(f"{'='*60}")

    # A simple baseline Random Forest, used just to measure stability
    # before any tuning - this is NOT the final model.
    baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # StratifiedKFold: splits data into 5 folds, preserving class ratio in each.
    # shuffle=True + random_state: ensures folds are randomly assigned but reproducible.
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # cross_val_score automatically: splits data into folds -> trains on 4,
    # validates on 1 -> repeats 5 times -> returns all 5 scores.
    cv_scores = cross_val_score(
        baseline_model, X_train, y_train,
        cv=cv_strategy, scoring="roc_auc", n_jobs=-1
    )

    print(f"ROC-AUC scores across 5 folds: {np.round(cv_scores, 4)}")
    print(f"Mean ROC-AUC: {cv_scores.mean():.4f}")
    print(f"Standard Deviation: {cv_scores.std():.4f}")
    print(
        "(A small standard deviation means performance is consistent "
        "across different subsets of data - the model is stable, not lucky.)"
    )

    return cv_scores


# ---------------------------------------------------------------------------
# 5. CLASS IMBALANCE HANDLING
# ---------------------------------------------------------------------------
def compare_class_weight(X_train, X_test, y_train, y_test):
    """
    Compare a standard Random Forest vs one trained with class_weight='balanced'.

    Why this matters:
        In Module 6, every model struggled with Recall (~45-51%) on the
        True class. This happens because the model is implicitly
        rewarded for predicting the MAJORITY class (False, ~81% of data)
        since that alone gives high accuracy. The model has little
        incentive to correctly catch the minority class (True, ~18.6%).

        class_weight='balanced' tells the model to treat mistakes on the
        minority class as MORE costly than mistakes on the majority
        class - proportionally to how rare that class is. This pushes
        the model to pay more attention to correctly identifying
        returning players (True), generally raising Recall at some cost
        to Precision.
    """
    print(f"\n{'='*60}")
    print("STEP 2: CLASS IMBALANCE HANDLING")
    print(f"{'='*60}")

    # --- Standard Random Forest (no class weighting) ---
    standard_model = RandomForestClassifier(n_estimators=100, random_state=42)
    standard_model.fit(X_train, y_train)
    y_pred_standard = standard_model.predict(X_test)

    # --- Random Forest with class_weight='balanced' ---
    # This automatically adjusts weights inversely proportional to class
    # frequency: rare classes (True) get a higher weight than common
    # classes (False), without needing to manually compute the ratio.
    balanced_model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )
    balanced_model.fit(X_train, y_train)
    y_pred_balanced = balanced_model.predict(X_test)

    print("\n--- Standard Random Forest ---")
    print(f"Precision: {precision_score(y_test, y_pred_standard):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_standard):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred_standard):.4f}")

    print("\n--- Random Forest (class_weight='balanced') ---")
    print(f"Precision: {precision_score(y_test, y_pred_balanced):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_balanced):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred_balanced):.4f}")

    print(
        "\n(Expect Recall to go UP and Precision to go slightly DOWN with "
        "class_weight='balanced' - this is the standard trade-off when "
        "handling imbalanced data.)"
    )

    return balanced_model


# ---------------------------------------------------------------------------
# 6. GRIDSEARCHCV - HYPERPARAMETER TUNING
# ---------------------------------------------------------------------------
def run_grid_search(X_train, y_train):
    """
    Use GridSearchCV to automatically find the best combination of
    Random Forest hyperparameters.

    Why GridSearchCV?
        A model like Random Forest has several "knobs" (hyperparameters)
        we can adjust - e.g., how many trees to build (n_estimators), how
        deep each tree can grow (max_depth), how many samples are needed
        to split a node (min_samples_split). Manually trying combinations
        one at a time is slow and unsystematic.

        GridSearchCV automates this: we give it a "grid" of possible
        values for each hyperparameter, and it tries EVERY combination,
        using Cross Validation on each one, then tells us which
        combination scored best. It's exhaustive but reliable.

    Note on grid size:
        We keep the grid relatively small here (2 x 2 x 2 = 8 combinations,
        each evaluated with 3-fold CV = 24 total model fits) to keep
        runtime practical. In a real production setting with more time/
        compute budget, you could expand this grid further.
    """
    print(f"\n{'='*60}")
    print("STEP 3: GRIDSEARCHCV - HYPERPARAMETER TUNING")
    print(f"{'='*60}")

    # The grid: every key is a hyperparameter name, every value is a
    # list of options GridSearchCV will try.
    param_grid = {
        "n_estimators": [100, 200],       # number of trees in the forest
        "max_depth": [10, 20],            # max depth of each tree (None = unlimited, omitted to save time)
        "min_samples_split": [2, 10],     # min samples required to split an internal node
    }

    # We use class_weight='balanced' here too, since we already confirmed
    # in Step 2 that it improves Recall - we want the FINAL tuned model
    # to also handle the imbalance properly.
    base_model = RandomForestClassifier(random_state=42, class_weight="balanced")

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="roc_auc",   # the metric used to pick the best combination
        cv=3,                  # 3-fold CV for each combination (keeps runtime reasonable)
        n_jobs=-1,             # use all available CPU cores to speed up search
        verbose=1              # print progress so we can see it's working
    )

    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters found: {grid_search.best_params_}")
    print(f"Best cross-validated ROC-AUC: {grid_search.best_score_:.4f}")

    # grid_search.best_estimator_ is already a fully trained model using
    # the best parameter combination - ready to use directly.
    return grid_search.best_estimator_


# ---------------------------------------------------------------------------
# 7. EVALUATE THE FINAL TUNED MODEL
# ---------------------------------------------------------------------------
def evaluate_final_model(model, X_test, y_test):
    """Evaluate the final tuned model on the held-out test set."""
    print(f"\n{'='*60}")
    print("FINAL TUNED MODEL - TEST SET PERFORMANCE")
    print(f"{'='*60}")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")

    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))


# ---------------------------------------------------------------------------
# 8. FEATURE IMPORTANCE
# ---------------------------------------------------------------------------
def plot_feature_importance(model, X_train: pd.DataFrame):
    """
    Extract and visualize feature importance from the trained Random Forest.

    What is feature importance?
        Random Forest can tell us, after training, how much each feature
        contributed to reducing prediction error across all its trees.
        Higher importance = that feature was more useful for splitting
        the data into correct classes.

    Why this matters for the business:
        This directly validates (or challenges) the insights we found
        during EDA. If 'sum_gamerounds' comes out as the most important
        feature, it confirms what the violin plot suggested - early
        engagement is the strongest signal of long-term retention.
    """
    print(f"\n{'='*60}")
    print("STEP 4: FEATURE IMPORTANCE")
    print(f"{'='*60}")

    # feature_importances_ is an array, one value per column, in the
    # same order as the columns in X_train.
    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print(importance_df.to_string(index=False))

    # Plot the importances as a horizontal bar chart for easy comparison.
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x="Importance", y="Feature",
                hue="Feature", palette="viridis", legend=False)
    plt.title("Feature Importance - Random Forest", fontsize=14, fontweight="bold")
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "14_feature_importance.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {save_path}")

    return importance_df


# ---------------------------------------------------------------------------
# 9. MAIN EXECUTION
# ---------------------------------------------------------------------------
def run_model_improvement():
    """Run the full Module 7 pipeline."""
    X_train, X_test, y_train, y_test = load_splits()

    # Step 1: Cross Validation (on baseline model, to check stability)
    run_cross_validation(X_train, y_train)

    # Step 2: Class Imbalance Handling (compare standard vs balanced)
    compare_class_weight(X_train, X_test, y_train, y_test)

    # Step 3: GridSearchCV (find best hyperparameters)
    best_model = run_grid_search(X_train, y_train)

    # Step 4: Evaluate the final tuned model on the test set
    evaluate_final_model(best_model, X_test, y_test)

    # Step 5: Feature Importance from the final tuned model
    plot_feature_importance(best_model, X_train)

    print(f"\n{'='*60}")
    print("Module 7 - Model Improvement completed successfully.")
    print(f"{'='*60}")

    return best_model


if __name__ == "__main__":
    run_model_improvement()