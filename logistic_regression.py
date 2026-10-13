"""Unregularized, L2 and L1 classifiers evaluated on one held-out split."""
from inspect import signature
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from preprocessing import get_model_data
from project_runtime import cached_analysis
from modeling import model_matrix, classification_metrics, confusion_plot, roc_plot, save_figure, checked_fit

TARGET_COLUMN = "Response"
RIDGE_C = 1.0
LASSO_C = 0.1
MAX_ITERATIONS = 2000
evaluate_model = classification_metrics


def create_confusion_matrix_plot(y_test, predictions, model_name, filename):
    return confusion_plot(y_test, predictions, model_name, filename)


def create_roc_curve_plot(y_test, probabilities, model_name, filename):
    return roc_plot(y_test, probabilities, model_name, filename)


def create_comparison_plot(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["F1 Score", "ROC AUC", "Average Precision"]):
        ax.bar(list(results), [r[metric] for r in results.values()])
        ax.set_ylim(0, 1)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
    return save_figure("logistic_regression_comparison.png")


def build_logistic_models():
    common = dict(class_weight="balanced", max_iter=MAX_ITERATIONS, random_state=42)
    # Recent sklearn uses l1_ratio; older supported versions use penalty.
    l1_options = {"l1_ratio": 1.0} if signature(LogisticRegression).parameters["penalty"].default == "deprecated" else {"penalty": "l1"}
    return {"Logistic Regression": LogisticRegression(C=float("inf"), solver="lbfgs", **common),
              "Ridge Logistic Regression": LogisticRegression(C=RIDGE_C, solver="lbfgs", **common),
              "Lasso Logistic Regression": LogisticRegression(C=LASSO_C, solver="liblinear", **l1_options, **common)}


@cached_analysis
def run_logistic_regression():
    train, test, y_train, y_test = get_model_data()
    X_train, X_test = model_matrix(train), model_matrix(test)
    models = build_logistic_models()
    results, plots, comparison = {}, {}, []
    for index, (name, model) in enumerate(models.items()):
        checked_fit(model, X_train, y_train)
        result = evaluate_model(model, X_test, y_test)
        results[name] = result
        key = ["logistic", "ridge", "lasso"][index]
        slug = "logistic_regression" if index == 0 else f"{key}_logistic"
        plots[f"{key}_cm"] = confusion_plot(y_test, result["predictions"], name, f"{slug}_confusion_matrix.png")
        plots[f"{key}_roc"] = roc_plot(y_test, result["probabilities"], name, f"{slug}_roc_curve.png")
        comparison.append({"Category": "Without Regularization" if index == 0 else "With Regularization",
                           "Model": name + ["", " (L2)", " (L1)"][index],
                           **{metric: result[metric] for metric in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "PR AUC", "Average Precision", "Balanced Accuracy", "Log Loss", "Brier Score"]}})
    plots["comparison"] = create_comparison_plot(results)
    return {"dataset_shape": (len(train) + len(test), train.shape[1] + 1), "training_shape": train.shape,
            "testing_shape": test.shape, "comparison": comparison, "plots": plots,
            "models": models, "majority_baseline_accuracy": float(y_test.value_counts(normalize=True).max())}


if __name__ == "__main__":
    print(run_logistic_regression()["comparison"])
