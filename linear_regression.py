"""Linear, Ridge and Lasso regression on unscaled annual insurance premium."""
import numpy as np
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning
from load_data import load_data
from preprocessing import get_model_data
from project_runtime import cached_analysis
from modeling import model_matrix, save_figure

TARGET_COLUMN = "Annual_Premium"
RIDGE_ALPHA = 1.0
LASSO_ALPHA = 1.0


def regression_data():
    train, test, _, _ = get_model_data()
    raw = load_data()
    target = raw[TARGET_COLUMN]
    if target.isna().any() or not np.isfinite(target).all():
        raise ValueError("Annual_Premium must contain finite, non-missing values for regression.")
    # Response is already excluded; premium must not be its own predictor.
    return (train.drop(columns=TARGET_COLUMN), test.drop(columns=TARGET_COLUMN),
            target.loc[train.index], target.loc[test.index])


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = float(mean_squared_error(y_test, predictions))
    return {"MAE": float(mean_absolute_error(y_test, predictions)), "MSE": mse,
            "RMSE": mse ** 0.5, "R2": float(r2_score(y_test, predictions)), "predictions": predictions}


def create_prediction_plot(y_test, predictions, model_name, filename):
    indices = np.random.default_rng(42).choice(len(y_test), min(5000, len(y_test)), replace=False)
    actual = np.asarray(y_test)[indices]
    predicted = np.asarray(predictions)[indices]
    plt.figure(figsize=(8, 6))
    plt.scatter(actual, predicted, s=10, alpha=0.3)
    lower, upper = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    plt.plot([lower, upper], [lower, upper], "--", color="gray", label="Perfect prediction")
    plt.xlabel("Actual Annual Premium")
    plt.ylabel("Predicted Annual Premium")
    plt.title(f"{model_name}: Actual vs Predicted")
    plt.legend()
    return save_figure(filename)


def create_comparison_plot(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ["RMSE", "R2"]):
        ax.bar(list(results), [r[metric] for r in results.values()])
        ax.set_title(f"{metric} Comparison")
        ax.tick_params(axis="x", rotation=20)
    return save_figure("linear_regression_comparison.png")


@cached_analysis
def run_linear_regression():
    train, test, y_train, y_test = regression_data()
    # Premium targets are large; float32 coordinate descent can stall before
    # the Lasso duality-gap criterion is met. Retain float64 for regression.
    X_train, X_test = model_matrix(train, np.float64), model_matrix(test, np.float64)
    models = {"Linear Regression": LinearRegression(), "Ridge Regression": Ridge(alpha=RIDGE_ALPHA, solver="lsqr"),
              "Lasso Regression": Lasso(alpha=LASSO_ALPHA, max_iter=10000)}
    results, plots, comparison = {}, {}, []
    for index, (name, model) in enumerate(models.items()):
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            model.fit(X_train, y_train)
        result = evaluate_model(model, X_test, y_test)
        results[name] = result
        key = ["linear", "ridge", "lasso"][index]
        plots[key] = create_prediction_plot(y_test, result["predictions"], name, f"{key}_regression_predictions.png")
        comparison.append({"Category": "Without Regularization" if index == 0 else "With Regularization",
                           "Model": name + (["", " (L2)", " (L1)"][index]),
                           **{m: result[m] for m in ["MAE", "MSE", "RMSE"]}, "R2 Score": result["R2"]})
    plots["comparison"] = create_comparison_plot(results)
    return {"dataset_shape": (len(train) + len(test), train.shape[1] + 1), "training_shape": train.shape,
            "testing_shape": test.shape, "comparison": comparison, "plots": plots, "target": TARGET_COLUMN,
            "models": models, "baseline_mae": float(mean_absolute_error(y_test, np.full(len(y_test), np.median(y_train))))}


if __name__ == "__main__":
    print(run_linear_regression()["comparison"])
