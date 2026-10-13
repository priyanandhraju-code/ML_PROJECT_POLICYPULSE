"""Tree classifiers including the reference's entropy decision tree."""
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight
from preprocessing import get_model_data
from project_runtime import cached_analysis
from modeling import matrix_for_model, classification_metrics, confusion_plot, roc_plot, checked_fit

ALGORITHMS = {
    "decision_tree": "Decision Tree (Gini)", "id3": "Decision Tree (Entropy / ID3-style)",
    "random_forest": "Random Forest", "extra_trees": "Extra Trees",
    "gradient_boosting": "Gradient Boosting", "adaboost": "AdaBoost",
    "xgboost": "XGBoost", "lightgbm": "LightGBM",
}
evaluate_model = classification_metrics


def build_model(algorithm):
    common = dict(random_state=42)
    if algorithm in {"decision_tree", "id3"}:
        return DecisionTreeClassifier(criterion="entropy" if algorithm == "id3" else "gini", class_weight="balanced", max_depth=12, min_samples_leaf=10, **common)
    if algorithm in {"random_forest", "extra_trees"}:
        model_type = RandomForestClassifier if algorithm == "random_forest" else ExtraTreesClassifier
        return model_type(n_estimators=100, class_weight="balanced", max_depth=15, min_samples_leaf=5, n_jobs=4, **common)
    if algorithm == "gradient_boosting":
        return GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, **common)
    if algorithm == "adaboost":
        return AdaBoostClassifier(n_estimators=100, learning_rate=0.5, **common)
    if algorithm == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ValueError("XGBoost is optional. Install requirements-optional.txt to run it.") from error
        return XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", n_jobs=4, **common)
    if algorithm == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as error:
            raise ValueError("LightGBM is optional. Install requirements-optional.txt to run it.") from error
        return LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31, n_jobs=4, verbosity=-1, **common)
    raise ValueError("Unknown tree algorithm. Choose an available model.")


@cached_analysis
def run_tree_algorithm(algorithm):
    if algorithm not in ALGORITHMS:
        raise ValueError("Unknown tree algorithm. Choose an available model.")
    train, test, y_train, y_test = get_model_data()
    model = build_model(algorithm)
    weights = compute_sample_weight("balanced", y_train) if algorithm in {"gradient_boosting", "adaboost", "xgboost", "lightgbm"} else None
    checked_fit(model, matrix_for_model(model, train), y_train, sample_weight=weights)
    metrics = evaluate_model(model, matrix_for_model(model, test), y_test)
    result = {"algorithm": ALGORITHMS[algorithm], "dataset_shape": (len(train) + len(test), train.shape[1] + 1),
              "feature_count": train.shape[1], "training_shape": train.shape, "testing_shape": test.shape,
              "model": model, "y_test": y_test, **metrics}
    result["confusion_matrix_plot"] = f"{algorithm}_confusion_matrix.png"
    result["roc_curve_plot"] = f"{algorithm}_roc_curve.png"
    create_confusion_matrix_plot(result, result["confusion_matrix_plot"])
    create_roc_curve_plot(result, result["roc_curve_plot"])
    return result


def create_confusion_matrix_plot(result, filename):
    return confusion_plot(result["y_test"], result["predictions"], result["algorithm"], filename)


def create_roc_curve_plot(result, filename):
    return roc_plot(result["y_test"], result["probabilities"], result["algorithm"], filename)
