import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = r"preprocessed_train.csv"
TEST_FILE = r"preprocessed_test.csv"
TARGET_COLUMN = "Response"

RANDOM_STATE = 42
RIDGE_ALPHA = 1.0
LASSO_ALPHA = 0.001

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

    y_train = train_df[TARGET_COLUMN].astype("float64")

    X_test = test_df.drop(
        columns=[TARGET_COLUMN]
    ).astype("float64")

    y_test = test_df[TARGET_COLUMN].astype("float64")

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

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "predictions": predictions
    }


# ============================================================
# ACTUAL VS PREDICTED PLOT
# ============================================================

def create_prediction_plot(
    y_test,
    predictions,
    model_name,
    filename
):

    os.makedirs(
        PLOT_DIRECTORY,
        exist_ok=True
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        y_test,
        predictions,
        alpha=0.25,
        s=10
    )

    plt.xlabel(
        "Actual Response"
    )

    plt.ylabel(
        "Predicted Response"
    )

    plt.title(
        f"{model_name}: Actual vs Predicted"
    )

    plt.grid(
        alpha=0.2
    )

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
# MODEL COMPARISON PLOT
# ============================================================

def create_comparison_plot(results):

    os.makedirs(
        PLOT_DIRECTORY,
        exist_ok=True
    )

    model_names = list(
        results.keys()
    )

    rmse_values = [
        results[name]["RMSE"]
        for name in model_names
    ]

    r2_values = [
        results[name]["R2"]
        for name in model_names
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    axes[0].bar(
        model_names,
        rmse_values
    )

    axes[0].set_title(
        "RMSE Comparison"
    )

    axes[0].set_ylabel(
        "RMSE"
    )

    axes[0].tick_params(
        axis="x",
        rotation=20
    )

    axes[1].bar(
        model_names,
        r2_values
    )

    axes[1].set_title(
        "R² Score Comparison"
    )

    axes[1].set_ylabel(
        "R² Score"
    )

    axes[1].tick_params(
        axis="x",
        rotation=20
    )

    plt.tight_layout()

    path = os.path.join(
        PLOT_DIRECTORY,
        "linear_regression_comparison.png"
    )

    plt.savefig(
        path,
        dpi=150
    )

    plt.close()

    return "linear_regression_comparison.png"


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_linear_regression():

    print("=" * 80)
    print("POLICYPULSE - LINEAR REGRESSION")
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

    # ========================================================
    # MODEL 1 - WITHOUT REGULARIZATION
    # ========================================================

    print("\n" + "=" * 80)
    print("WITHOUT REGULARIZATION")
    print("LINEAR REGRESSION")
    print("=" * 80)

    linear_model = LinearRegression()

    linear_model.fit(
        X_train,
        y_train
    )

    linear_results = evaluate_model(
        linear_model,
        X_test,
        y_test
    )

    # ========================================================
    # MODEL 2 - RIDGE / L2
    # ========================================================

    print("\n" + "=" * 80)
    print("WITH REGULARIZATION")
    print("RIDGE REGRESSION - L2")
    print("=" * 80)

    ridge_model = Ridge(
        alpha=RIDGE_ALPHA
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
    # MODEL 3 - LASSO / L1
    # ========================================================

    print("\n" + "=" * 80)
    print("WITH REGULARIZATION")
    print("LASSO REGRESSION - L1")
    print("=" * 80)

    lasso_model = Lasso(
        alpha=LASSO_ALPHA,
        max_iter=10000
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
        "Linear Regression": linear_results,
        "Ridge Regression": ridge_results,
        "Lasso Regression": lasso_results
    }

    # ========================================================
    # PLOTS
    # ========================================================

    linear_plot = create_prediction_plot(
        y_test,
        linear_results["predictions"],
        "Linear Regression",
        "linear_regression_predictions.png"
    )

    ridge_plot = create_prediction_plot(
        y_test,
        ridge_results["predictions"],
        "Ridge Regression",
        "ridge_regression_predictions.png"
    )

    lasso_plot = create_prediction_plot(
        y_test,
        lasso_results["predictions"],
        "Lasso Regression",
        "lasso_regression_predictions.png"
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
            "Linear Regression",
            "Ridge Regression (L2)",
            "Lasso Regression (L1)"
        ],
        "MAE": [
            linear_results["MAE"],
            ridge_results["MAE"],
            lasso_results["MAE"]
        ],
        "MSE": [
            linear_results["MSE"],
            ridge_results["MSE"],
            lasso_results["MSE"]
        ],
        "RMSE": [
            linear_results["RMSE"],
            ridge_results["RMSE"],
            lasso_results["RMSE"]
        ],
        "R2 Score": [
            linear_results["R2"],
            ridge_results["R2"],
            lasso_results["R2"]
        ]
    })

    print("\n" + "=" * 80)
    print("LINEAR REGRESSION MODEL COMPARISON")
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
            "linear": linear_plot,
            "ridge": ridge_plot,
            "lasso": lasso_plot,
            "comparison": comparison_plot
        }
    }


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":
    run_linear_regression()
