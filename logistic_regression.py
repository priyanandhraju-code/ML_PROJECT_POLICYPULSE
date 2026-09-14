import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

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

FILE_PATH = r"preprocessed_data.csv"

TARGET_COLUMN = "Response"

TEST_SIZE = 0.20

RANDOM_STATE = 42


# Logistic Regression regularization parameters

RIDGE_C = 1.0

LASSO_C = 0.1


# Maximum number of optimization iterations

MAX_ITERATIONS = 2000


# Directory for generated plots

PLOT_DIRECTORY = os.path.join(
    "static",
    "plots"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_preprocessed_data():

    if not os.path.exists(FILE_PATH):

        raise FileNotFoundError(
            f"Preprocessed dataset not found: {FILE_PATH}"
        )

    return pd.read_csv(FILE_PATH)


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    # --------------------------------------------------------
    # Separate features from target
    # --------------------------------------------------------

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN]

    # --------------------------------------------------------
    # Convert all features to float64
    # --------------------------------------------------------
    #
    # The preprocessed dataset contains:
    #
    # - standardized numerical features
    # - one-hot encoded categorical features
    # - binary features
    #
    # Using one consistent floating-point type ensures that
    # matrix operations inside Logistic Regression are
    # performed consistently.
    # --------------------------------------------------------

    X = X.astype("float64")

    # Convert target to integer

    y = y.astype("int64")

    return X, y


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y):

    return train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y
    )


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    # --------------------------------------------------------
    # Generate class predictions
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Generate probability predictions
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    # --------------------------------------------------------
    # F1 Score
    # --------------------------------------------------------

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    # --------------------------------------------------------
    # ROC AUC
    # --------------------------------------------------------

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Calculate confusion matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # Create display
    # --------------------------------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix
    )

    display.plot()

    plt.title(title)

    plt.tight_layout()

    # --------------------------------------------------------
    # Save plot
    # --------------------------------------------------------

    path = os.path.join(
        PLOT_DIRECTORY,
        filename
    )

    plt.savefig(path)

    plt.close()

    return path


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

    # --------------------------------------------------------
    # Calculate ROC curve
    # --------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y_test,
        probability
    )

    # --------------------------------------------------------
    # Calculate ROC AUC
    # --------------------------------------------------------

    auc = roc_auc_score(
        y_test,
        probability
    )

    # --------------------------------------------------------
    # Create ROC plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        fpr,
        tpr,
        label=f"ROC AUC = {auc:.4f}"
    )

    # Random classifier reference line

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(title)

    plt.legend()

    plt.grid(
        alpha=0.2
    )

    # --------------------------------------------------------
    # Save plot
    # --------------------------------------------------------

    path = os.path.join(
        PLOT_DIRECTORY,
        filename
    )

    plt.tight_layout()

    plt.savefig(path)

    plt.close()

    return path


# ============================================================
# COMPARISON PLOT
# ============================================================

def create_comparison_plot(results):

    os.makedirs(
        PLOT_DIRECTORY,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Get model names
    # --------------------------------------------------------

    models = list(
        results.keys()
    )

    # --------------------------------------------------------
    # Get Accuracy values
    # --------------------------------------------------------

    accuracy = [

        results[model]["Accuracy"]

        for model in models

    ]

    # --------------------------------------------------------
    # Get F1 Score values
    # --------------------------------------------------------

    f1 = [

        results[model]["F1 Score"]

        for model in models

    ]

    # --------------------------------------------------------
    # Get ROC AUC values
    # --------------------------------------------------------

    auc = [

        results[model]["ROC AUC"]

        for model in models

    ]

    # --------------------------------------------------------
    # X-axis positions
    # --------------------------------------------------------

    x = range(
        len(models)
    )

    width = 0.25

    # --------------------------------------------------------
    # Create comparison chart
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    # Accuracy

    plt.bar(
        [i - width for i in x],
        accuracy,
        width=width,
        label="Accuracy"
    )

    # F1 Score

    plt.bar(
        x,
        f1,
        width=width,
        label="F1 Score"
    )

    # ROC AUC

    plt.bar(
        [i + width for i in x],
        auc,
        width=width,
        label="ROC AUC"
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    plt.xticks(
        list(x),
        models,
        rotation=20
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Logistic Regression Model Comparison"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.2
    )

    # --------------------------------------------------------
    # Save comparison plot
    # --------------------------------------------------------

    path = os.path.join(
        PLOT_DIRECTORY,
        "logistic_regression_comparison.png"
    )

    plt.tight_layout()

    plt.savefig(path)

    plt.close()

    return path


# ============================================================
# MAIN LOGISTIC REGRESSION FUNCTION
# ============================================================

def run_logistic_regression():

    print("=" * 80)

    print(
        "POLICYPULSE - V4 LOGISTIC REGRESSION"
    )

    print("=" * 80)

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = load_preprocessed_data()

    print("\nDataset shape:")

    print(df.shape)

    # ========================================================
    # PREPARE DATA
    # ========================================================

    X, y = prepare_data(df)

    print("\nNumber of features:")

    print(X.shape[1])

    print("\nTarget:")

    print(TARGET_COLUMN)

    print("\nTarget distribution:")

    print(y.value_counts())

    print("\nTarget proportions:")

    print(
        y.value_counts(
            normalize=True
        )
    )

    # --------------------------------------------------------
    # NUMERICAL DATA VALIDATION
    # --------------------------------------------------------
    #
    # These checks verify the exact feature matrix that will
    # be supplied to Logistic Regression.
    # --------------------------------------------------------

    print("\n" + "=" * 80)

    print("FEATURE MATRIX NUMERICAL CHECK")

    print("=" * 80)

    print("\nFeature data types:")

    print(
        X.dtypes.value_counts()
    )

    print("\nNaN values:")

    print(
        X.isna().sum().sum()
    )

    print("\nInfinite values:")

    print(
        X.isin(
            [float("inf"), float("-inf")]
        ).sum().sum()
    )

    print("\nLargest absolute feature value:")

    print(
        X.abs().max().max()
    )

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    print("\nTraining shape:")

    print(X_train.shape)

    print("\nTesting shape:")

    print(X_test.shape)

    # ========================================================
    # WITHOUT REGULARIZATION
    # ========================================================

    print("\n" + "=" * 80)

    print("WITHOUT REGULARIZATION")

    print("LOGISTIC REGRESSION")

    print("=" * 80)

    # --------------------------------------------------------
    # Logistic Regression without regularization
    # --------------------------------------------------------
    #
    # penalty=None means no L1/L2 regularization.
    #
    # lbfgs is explicitly selected as the optimization solver.
    # --------------------------------------------------------

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

    print(
        "\nAccuracy:",
        logistic_results["Accuracy"]
    )

    print(
        "Precision:",
        logistic_results["Precision"]
    )

    print(
        "Recall:",
        logistic_results["Recall"]
    )

    print(
        "F1 Score:",
        logistic_results["F1 Score"]
    )

    print(
        "ROC AUC:",
        logistic_results["ROC AUC"]
    )

    # ========================================================
    # RIDGE LOGISTIC REGRESSION
    # ========================================================

    print("\n" + "=" * 80)

    print("WITH REGULARIZATION")

    print("RIDGE LOGISTIC REGRESSION - L2")

    print("=" * 80)

    # --------------------------------------------------------
    # Ridge Logistic Regression
    # --------------------------------------------------------
    #
    # L2 regularization penalizes large coefficient values.
    #
    # C controls the inverse strength of regularization.
    # Smaller C = stronger regularization.
    # --------------------------------------------------------

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

    print(
        "\nC:",
        RIDGE_C
    )

    print(
        "Accuracy:",
        ridge_results["Accuracy"]
    )

    print(
        "Precision:",
        ridge_results["Precision"]
    )

    print(
        "Recall:",
        ridge_results["Recall"]
    )

    print(
        "F1 Score:",
        ridge_results["F1 Score"]
    )

    print(
        "ROC AUC:",
        ridge_results["ROC AUC"]
    )

    # ========================================================
    # LASSO LOGISTIC REGRESSION
    # ========================================================

    print("\n" + "=" * 80)

    print("WITH REGULARIZATION")

    print("LASSO LOGISTIC REGRESSION - L1")

    print("=" * 80)

    # --------------------------------------------------------
    # Lasso Logistic Regression
    # --------------------------------------------------------
    #
    # L1 regularization encourages some coefficients to become
    # exactly zero, which can perform feature selection.
    #
    # liblinear supports L1 regularization for Logistic
    # Regression.
    # --------------------------------------------------------

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

    print(
        "\nC:",
        LASSO_C
    )

    print(
        "Accuracy:",
        lasso_results["Accuracy"]
    )

    print(
        "Precision:",
        lasso_results["Precision"]
    )

    print(
        "Recall:",
        lasso_results["Recall"]
    )

    print(
        "F1 Score:",
        lasso_results["F1 Score"]
    )

    print(
        "ROC AUC:",
        lasso_results["ROC AUC"]
    )

    # ========================================================
    # STORE RESULTS
    # ========================================================

    results = {

        "Logistic Regression":
            logistic_results,

        "Ridge Logistic Regression":
            ridge_results,

        "Lasso Logistic Regression":
            lasso_results

    }

    # ========================================================
    # CONFUSION MATRICES
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

    # ========================================================
    # ROC CURVES
    # ========================================================

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

    # ========================================================
    # COMPARISON PLOT
    # ========================================================

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

    # ========================================================
    # PRINT COMPARISON
    # ========================================================

    print("\n" + "=" * 80)

    print(
        "LOGISTIC REGRESSION MODEL COMPARISON"
    )

    print("=" * 80)

    print(
        comparison.to_string(
            index=False
        )
    )

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "dataset_shape":
            df.shape,

        "training_shape":
            X_train.shape,

        "testing_shape":
            X_test.shape,

        "comparison":
            comparison.to_dict(
                orient="records"
            ),

        "plots": {

            "comparison":
                comparison_plot,

            "logistic_cm":
                logistic_cm,

            "ridge_cm":
                ridge_cm,

            "lasso_cm":
                lasso_cm,

            "logistic_roc":
                logistic_roc,

            "ridge_roc":
                ridge_roc,

            "lasso_roc":
                lasso_roc

        }

    }


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    run_logistic_regression()

    print(
        "\nV4 Logistic Regression completed successfully."
    )