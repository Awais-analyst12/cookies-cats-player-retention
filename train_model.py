"""
train_model.py
---------------
Module 6: Baseline Machine Learning Models for Cookie Cats Retention Prediction.

Purpose:
    Train 6 different classification models on the same training data,
    evaluate each one on the SAME unseen test data, and compare their
    performance side by side. This tells us which algorithm is the best
    starting point before we move on to tuning/improving it further.

Models trained:
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest
    4. K-Nearest Neighbors (KNN)
    5. Support Vector Machine (SVM)
    6. Naive Bayes

Author : (Your Name)
Project: Cookie Cats Retention Prediction - Game District Portfolio Project
"""

# ---------------------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os

# Used to create a stratified sample for SVM training (see Model 5 below).
from sklearn.model_selection import train_test_split

# Each model is a different "algorithm family" for learning patterns
# in the data. We import one class per model from scikit-learn.
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Preprocessing: KNN and SVM are distance-based algorithms, so features
# on different scales (e.g., sum_gamerounds going 0-500 vs retention_1
# being just 0/1) can unfairly dominate the distance calculation.
# StandardScaler rescales every feature to have mean=0, std=1.
from sklearn.preprocessing import StandardScaler

# Metrics: the tools we use to measure how good each model's predictions are.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ---------------------------------------------------------------------------
# 2. PATH CONFIGURATION
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

X_TRAIN_PATH = os.path.join(OUTPUT_DIR, "X_train.csv")
X_TEST_PATH = os.path.join(OUTPUT_DIR, "X_test.csv")
Y_TRAIN_PATH = os.path.join(OUTPUT_DIR, "y_train.csv")
Y_TEST_PATH = os.path.join(OUTPUT_DIR, "y_test.csv")

RESULTS_PATH = os.path.join(OUTPUT_DIR, "model_comparison.csv")


# ---------------------------------------------------------------------------
# 3. LOAD THE TRAIN/TEST SPLITS
# ---------------------------------------------------------------------------
def load_splits():
    """Load the 4 files saved by train_test_split.py (Module 5)."""
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)

    # y was saved as a single-column CSV, so we use .squeeze() to convert
    # the resulting 1-column DataFrame into a plain Series (what sklearn expects).
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 4. FEATURE SCALING
# ---------------------------------------------------------------------------
def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Standardize features to mean=0, std=1.

    Why scale?
        KNN and SVM calculate distances between data points. A feature
        like 'sum_gamerounds' (range ~0-500) would completely dominate a
        feature like 'retention_1' (range 0-1) in distance calculations,
        even if retention_1 is actually more predictive. Scaling puts
        every feature on equal footing.

    Why fit_transform on train but only transform on test?
        We learn the scaling parameters (mean, std) ONLY from the
        training data, then apply that same transformation to the test
        data. If we fit on the test set too, information from the
        "unseen" data would leak into our preprocessing - this is called
        data leakage and gives an unrealistically optimistic evaluation.

    Returns
    -------
    X_train_scaled, X_test_scaled : np.ndarray
        Scaled versions of the feature sets, used by KNN and SVM only.
    """
    scaler = StandardScaler()

    # fit_transform: learns mean/std from X_train AND applies the scaling
    X_train_scaled = scaler.fit_transform(X_train)

    # transform only: applies the SAME mean/std learned from training data
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


# ---------------------------------------------------------------------------
# 5. EVALUATE A SINGLE MODEL
# ---------------------------------------------------------------------------
def evaluate_model(model_name: str, y_true, y_pred, y_proba) -> dict:
    """
    Calculate every requested metric for one model's predictions and
    print a full report.

    Metric definitions (explained here since metrics-first was the choice):

    - Accuracy: (Correct Predictions) / (Total Predictions)
        The simplest metric, but MISLEADING on imbalanced data like ours
        (81% of players are False/non-retained). A model that just
        predicts "False" every time would already score ~81% accuracy
        without learning anything useful - that's why we need the metrics below.

    - Precision: Of all players the model PREDICTED would return (True),
        what fraction actually did return?
        High precision = when the model says "this player will come back",
        it's usually right. Important if a marketing team will only spend
        budget on players the model flags as "likely to return".

    - Recall: Of all players who ACTUALLY returned (True), what fraction
        did the model correctly catch?
        High recall = the model successfully finds most of the players
        who will return. Important if missing a retained player is costly
        (e.g., you want to identify and reward ALL loyal players).

    - F1 Score: The harmonic mean of Precision and Recall - a single
        number that balances both. Useful for comparing models when you
        care about both false positives and false negatives.

    - ROC-AUC: Measures how well the model separates the two classes
        (True vs False) across ALL possible decision thresholds, not just
        the default 0.5 cutoff. Ranges 0.5 (random guessing) to 1.0
        (perfect separation). This is often the most reliable single
        metric for imbalanced classification problems.

    Parameters
    ----------
    model_name : str
        Name of the model, used for printing.
    y_true : array-like
        Actual target values from the test set.
    y_pred : array-like
        Model's predicted class labels (True/False).
    y_proba : array-like
        Model's predicted PROBABILITY of the positive class (True),
        needed specifically for ROC-AUC.

    Returns
    -------
    dict
        A dictionary of all computed metrics, used later to build the
        final comparison table.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    # Confusion Matrix layout:
    #                  Predicted False   Predicted True
    # Actual False     True Negative     False Positive
    # Actual True       False Negative   True Positive
    print(f"\nConfusion Matrix:")
    print(cm)

    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred))

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
    }


# ---------------------------------------------------------------------------
# 6. TRAIN AND EVALUATE ALL 6 MODELS
# ---------------------------------------------------------------------------
def train_all_models(X_train, X_test, y_train, y_test):
    """
    Train all 6 baseline models and collect their evaluation metrics.

    Returns
    -------
    pd.DataFrame
        A comparison table with one row per model and one column per metric.
    """
    results = []

    # Scaled versions, needed only for distance-based models (KNN, SVM).
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # -----------------------------------------------------------------
    # Model 1: Logistic Regression
    # A linear model - draws a straight decision boundary between
    # classes. Fast, interpretable, and a great baseline to beat.
    # -----------------------------------------------------------------
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability of class "True"
    results.append(evaluate_model("Logistic Regression", y_test, y_pred, y_proba))

    # -----------------------------------------------------------------
    # Model 2: Decision Tree
    # Splits data into branches based on feature thresholds (e.g.,
    # "sum_gamerounds > 30?"). Easy to interpret but prone to overfitting.
    # -----------------------------------------------------------------
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    results.append(evaluate_model("Decision Tree", y_test, y_pred, y_proba))

    # -----------------------------------------------------------------
    # Model 3: Random Forest
    # An "ensemble" of many Decision Trees, each trained on a random
    # subset of data/features. Averaging their votes reduces overfitting
    # and usually performs better than a single tree.
    # -----------------------------------------------------------------
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    results.append(evaluate_model("Random Forest", y_test, y_pred, y_proba))

    # -----------------------------------------------------------------
    # Model 4: K-Nearest Neighbors (KNN)
    # Classifies a player by looking at the "k" most similar players
    # (nearest neighbors) and taking a majority vote. Uses SCALED data
    # since it relies on distance calculations.
    # -----------------------------------------------------------------
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    results.append(evaluate_model("KNN", y_test, y_pred, y_proba))

    # -----------------------------------------------------------------
    # Model 5: Support Vector Machine (SVM)
    # Finds the optimal boundary (hyperplane) that best separates the
    # two classes with the maximum margin. Also uses SCALED data, and
    # probability=True is required to get probabilities for ROC-AUC.
    #
    # IMPORTANT - why we train on a SAMPLE here:
    # SVM's training time grows roughly QUADRATICALLY with the number of
    # rows. On our full training set (~72,000 rows) it can take an
    # impractically long time to fit. This is a well-known, real-world
    # limitation of SVM - it does not scale well to large datasets.
    # The standard professional workaround is to either (a) train on a
    # representative random sample, or (b) switch to LinearSVC for
    # large data. Here we use option (a): a stratified sample of 8,000
    # rows, which keeps the True/False ratio identical to the full
    # training set while making training time practical.
    # -----------------------------------------------------------------
    SVM_SAMPLE_SIZE = 8000
    X_train_svm, _, y_train_svm, _ = train_test_split(
        X_train_scaled, y_train,
        train_size=SVM_SAMPLE_SIZE,
        random_state=42,
        stratify=y_train
    )

    model = SVC(probability=True, random_state=42)
    model.fit(X_train_svm, y_train_svm)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    results.append(evaluate_model("SVM (trained on 8,000-row sample)", y_test, y_pred, y_proba))

    # -----------------------------------------------------------------
    # Model 6: Naive Bayes
    # A probabilistic model based on Bayes' Theorem, assuming features
    # are independent of each other ("naive" assumption). Very fast,
    # works well as a simple baseline.
    # -----------------------------------------------------------------
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    results.append(evaluate_model("Naive Bayes", y_test, y_pred, y_proba))

    # Convert the list of result dictionaries into a single comparison table.
    comparison_df = pd.DataFrame(results)

    return comparison_df


# ---------------------------------------------------------------------------
# 7. MAIN EXECUTION
# ---------------------------------------------------------------------------
def run_model_training():
    """Run the full Module 6 pipeline: load data -> train 6 models -> compare."""
    X_train, X_test, y_train, y_test = load_splits()

    comparison_df = train_all_models(X_train, X_test, y_train, y_test)

    # Sort by ROC-AUC (a reliable metric for imbalanced classification)
    # so the best-performing model appears at the top.
    comparison_df = comparison_df.sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)

    print(f"\n{'='*60}")
    print("FINAL MODEL COMPARISON (sorted by ROC-AUC)")
    print(f"{'='*60}")
    print(comparison_df.to_string(index=False))

    comparison_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nComparison table saved to: {RESULTS_PATH}")

    return comparison_df


if __name__ == "__main__":
    run_model_training()
    