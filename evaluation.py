"""Classification comparison uses the same untouched test rows for all models."""
import importlib.util
from logistic_regression import run_logistic_regression
from linear_regression import run_linear_regression
from tree_based import ALGORITHMS, run_tree_algorithm
from preprocessing import get_prepared_dataset
from project_runtime import cached_analysis

METRICS = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "PR AUC"]


@cached_analysis
def run_comparison():
    logistic = run_logistic_regression()
    rows = [{"Model": row["Model"], **{metric: row[metric] for metric in METRICS}} for row in logistic["comparison"]]
    unavailable = []
    for algorithm, name in ALGORITHMS.items():
        if algorithm in {"xgboost", "lightgbm"} and importlib.util.find_spec(algorithm) is None:
            unavailable.append(name)
            continue
        result = run_tree_algorithm(algorithm)
        rows.append({"Model": name, **{metric: result[metric] for metric in METRICS}})
    rows.sort(key=lambda row: (row["PR AUC"], row["ROC AUC"]), reverse=True)
    _, _, y_train, y_test, _, summary = get_prepared_dataset()
    return {"comparison": rows, "best": rows[0], "unavailable": unavailable,
            "majority_baseline": float(y_test.value_counts(normalize=True).max()),
            "pr_baseline": float(y_test.mean()), "summary": summary,
            "regression": run_linear_regression()["comparison"],
            "training_positive_rate": float(y_train.mean()), "testing_positive_rate": float(y_test.mean())}
