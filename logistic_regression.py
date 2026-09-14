import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = r"preprocessed_train.csv"
TEST_FILE = r"preprocessed_test.csv"
TARGET_COLUMN = "Response"

RANDOM_STATE = 42
RIDGE_C = 1.0
LASSO_C = 0.1
MAX_ITERATIONS = 2000

PLOT_DIRECTORY = os.path.join(
    "static",
    "plots"
)


# ============================================================
# LOAD PREPROCESSED TRAIN / TEST DATA
# ============================================================

def load_preprocessed_data():

    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_FILE}. "
            "Run preprocessing.py first."
        )

    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(
            f"Testing dataset not found: {TEST_FILE}. "
            "Run preprocessing.py first."
        )

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in training data."
        )

    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in testing data."
        )

    return train_df, test_df


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

def prepare_data(train_df, test_df):

    X_train = train_df.drop(
        columns=[TARGET_COLUMN]
    ).astype("float64")

    y_train = train_df[TARGET_COLUMN].astype("int64")

    X_test = test_df.drop(
        columns=[TARGET_COLUMN]
    ).astype("float64")

    y_test = test_df[TARGET_COLUMN].astype("int64")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Training and testing feature columns do not match."
        )

    return X_train, X_test, y_train, y_test


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": auc,
        "predictions": predictions,
        "probabilities": probabilities
    }


# ============================================================
# CONFUSION MATRIX
# ============================================================

def create_confusion_matrix_plot(
    y_test,
    predictions,
    filename,
    title
):

    os.makedirs(
        PLOT_DIRECTORY,
        exist_ok=True
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix
    )

    display.plot()

    plt.title(title)
    plt.tight_layout()

    path = os.path.join(
        PLOT_DIRECTORY,
        filename
    )

    plt.savefig(
        path,
        dpi=150
    )

    plt.close()

    return filename


# ============================================================
# ROC CURVE
# ============================================================

def create_roc_curve_plot(
    y_test,
    probability,
    filename,
    title
):

    os.makedirs(
        PLOT_DIRECTORY,
        exist_ok=True
    )

    fpr, tpr, _ = roc_curve(
        y_test,
        probability
    )

    auc = roc_auc_score(
        y_test,
        probability
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        fpr,
        tpr,
        label=f"ROC AUC = {auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    path = os.path.join(
        PLOT_DIRECTORY,
        filename
    )

    plt.savefig(
        path,
        dpi=150
    )

    plt.close()

    return filename


# ============================================================
# COMPARISON PLOT
# ============================================================

def create_comparison_plot(results):

    os.makedirs(
        PLOT_DIRECTORY,
        exist_ok=True
    )

    models = list(results.keys())

    accuracy = [
        results[model]["Accuracy"]
        for model in models
    ]

    f1 = [
        results[model]["F1 Score"]
        for model in models
    ]

    auc = [
        results[model]["ROC AUC"]
        for model in models
    ]

    x = range(len(models))
    width = 0.25

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        [i - width for i in x],
        accuracy,
        width=width,
        label="Accuracy"
    )

    plt.bar(
        x,
        f1,
        width=width,
        label="F1 Score"
    )

    plt.bar(
        [i + width for i in x],
        auc,
        width=width,
        label="ROC AUC"
    )

    plt.xticks(
        list(x),
        models,
        rotation=20
    )

    plt.ylabel("Score")
    plt.title("Logistic Regression Model Comparison")
    plt.legend()
    plt.grid(
        axis="y",
        alpha=0.2
    )
    plt.tight_layout()

    path = os.path.join(
        PLOT_DIRECTORY,
        "logistic_regression_comparison.png"
    )

    plt.savefig(
        path,
        dpi=150
    )

    plt.close()

    return "logistic_regression_comparison.png"


# ============================================================
# MAIN LOGISTIC REGRESSION FUNCTION
# ============================================================

def run_logistic_regression():

    print("=" * 80)
    print("POLICYPULSE - LOGISTIC REGRESSION")
    print("=" * 80)

    train_df, test_df = load_preprocessed_data()

    X_train, X_test, y_train, y_test = prepare_data(
        train_df,
        test_df
    )

    print("\nTraining shape:")
    print(X_train.shape)

    print("\nTesting shape:")
    print(X_test.shape)

    print("\nTarget distribution in training data:")
    print(y_train.value_counts().sort_index())

    # ========================================================
    # WITHOUT REGULARIZATION
    # ========================================================

    print("\n" + "=" * 80)
    print("WITHOUT REGULARIZATION")
    print("LOGISTIC REGRESSION")
    print("=" * 80)

    logistic_model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=MAX_ITERATIONS,
        random_state=RANDOM_STATE
    )

    logistic_model.fit(
        X_train,
        y_train
    )

    logistic_results = evaluate_model(
        logistic_model,
        X_test,
        y_test
    )

    # ========================================================
    # RIDGE / L2
    # ========================================================

    print("\n" + "=" * 80)
    print("WITH REGULARIZATION")
    print("RIDGE LOGISTIC REGRESSION - L2")
    print("=" * 80)

    ridge_model = LogisticRegression(
        penalty="l2",
        C=RIDGE_C,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=MAX_ITERATIONS,
        random_state=RANDOM_STATE
    )

    ridge_model.fit(
        X_train,
        y_train
    )

    ridge_results = evaluate_model(
        ridge_model,
        X_test,
        y_test
    )

    # ========================================================
    # LASSO / L1
    # ========================================================

    print("\n" + "=" * 80)
    print("WITH REGULARIZATION")
    print("LASSO LOGISTIC REGRESSION - L1")
    print("=" * 80)

    lasso_model = LogisticRegression(
        penalty="l1",
        C=LASSO_C,
        solver="liblinear",
        class_weight="balanced",
        max_iter=MAX_ITERATIONS,
        random_state=RANDOM_STATE
    )

    lasso_model.fit(
        X_train,
        y_train
    )

    lasso_results = evaluate_model(
        lasso_model,
        X_test,
        y_test
    )

    results = {
        "Logistic Regression": logistic_results,
        "Ridge Logistic Regression": ridge_results,
        "Lasso Logistic Regression": lasso_results
    }

    # ========================================================
    # PLOTS
    # ========================================================

    logistic_cm = create_confusion_matrix_plot(
        y_test,
        logistic_results["predictions"],
        "logistic_regression_confusion_matrix.png",
        "Logistic Regression - Confusion Matrix"
    )

    ridge_cm = create_confusion_matrix_plot(
        y_test,
        ridge_results["predictions"],
        "ridge_logistic_confusion_matrix.png",
        "Ridge Logistic Regression - Confusion Matrix"
    )

    lasso_cm = create_confusion_matrix_plot(
        y_test,
        lasso_results["predictions"],
        "lasso_logistic_confusion_matrix.png",
        "Lasso Logistic Regression - Confusion Matrix"
    )

    logistic_roc = create_roc_curve_plot(
        y_test,
        logistic_results["probabilities"],
        "logistic_regression_roc_curve.png",
        "Logistic Regression - ROC Curve"
    )

    ridge_roc = create_roc_curve_plot(
        y_test,
        ridge_results["probabilities"],
        "ridge_logistic_roc_curve.png",
        "Ridge Logistic Regression - ROC Curve"
    )

    lasso_roc = create_roc_curve_plot(
        y_test,
        lasso_results["probabilities"],
        "lasso_logistic_roc_curve.png",
        "Lasso Logistic Regression - ROC Curve"
    )

    comparison_plot = create_comparison_plot(
        results
    )

    # ========================================================
    # COMPARISON TABLE
    # ========================================================

    comparison = pd.DataFrame({
        "Category": [
            "Without Regularization",
            "With Regularization",
            "With Regularization"
        ],
        "Model": [
            "Logistic Regression",
            "Ridge Logistic Regression (L2)",
            "Lasso Logistic Regression (L1)"
        ],
        "Accuracy": [
            logistic_results["Accuracy"],
            ridge_results["Accuracy"],
            lasso_results["Accuracy"]
        ],
        "Precision": [
            logistic_results["Precision"],
            ridge_results["Precision"],
            lasso_results["Precision"]
        ],
        "Recall": [
            logistic_results["Recall"],
            ridge_results["Recall"],
            lasso_results["Recall"]
        ],
        "F1 Score": [
            logistic_results["F1 Score"],
            ridge_results["F1 Score"],
            lasso_results["F1 Score"]
        ],
        "ROC AUC": [
            logistic_results["ROC AUC"],
            ridge_results["ROC AUC"],
            lasso_results["ROC AUC"]
        ]
    })

    print("\n" + "=" * 80)
    print("LOGISTIC REGRESSION MODEL COMPARISON")
    print("=" * 80)
    print(comparison.to_string(index=False))

    return {
        "dataset_shape": (
            train_df.shape[0] + test_df.shape[0],
            train_df.shape[1]
        ),
        "training_shape": X_train.shape,
        "testing_shape": X_test.shape,
        "comparison": comparison.to_dict(
            orient="records"
        ),
        "plots": {
            "comparison": comparison_plot,
            "logistic_cm": logistic_cm,
            "ridge_cm": ridge_cm,
            "lasso_cm": lasso_cm,
            "logistic_roc": logistic_roc,
            "ridge_roc": ridge_roc,
            "lasso_roc": lasso_roc
        }
    }


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":
    run_logistic_regression()
