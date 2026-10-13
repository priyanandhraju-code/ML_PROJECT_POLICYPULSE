"""Model selection and calibrated decisions use training data only."""
import hashlib
import importlib.util
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, TunedThresholdClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
from preprocessing import PolicyPreprocessor, get_model_data, load_dataset
from logistic_regression import build_logistic_models
from tree_based import ALGORITHMS, build_model
from modeling import matrix_for_model, checked_fit, classification_metrics
from project_runtime import cached_analysis

LOGISTIC_NAMES = {"logistic": "Logistic Regression", "ridge_logistic": "Ridge Logistic Regression (L2)",
                  "lasso_logistic": "Lasso Logistic Regression (L1)"}
WEIGHTED_ALGORITHMS = {"gradient_boosting", "adaboost", "xgboost", "lightgbm"}


def available_classifiers():
    algorithms = {**LOGISTIC_NAMES, **ALGORITHMS}
    return {key: name for key, name in algorithms.items()
            if key not in {"xgboost", "lightgbm"} or importlib.util.find_spec(key) is not None}


def make_classifier(algorithm):
    if algorithm in LOGISTIC_NAMES:
        return list(build_logistic_models().values())[list(LOGISTIC_NAMES).index(algorithm)]
    return build_model(algorithm)


def raw_training_data():
    train, test, y_train, y_test = get_model_data()
    raw = load_dataset().drop(columns="Response")
    return raw.loc[train.index], raw.loc[test.index], y_train, y_test


def selection_split():
    raw_train, _, y_train, _ = raw_training_data()
    return train_test_split(raw_train, y_train, test_size=0.25, random_state=137, stratify=y_train)


class PolicyClassifier(ClassifierMixin, BaseEstimator):
    """Cloneable raw-data estimator; every CV fold fits its own preprocessing."""

    def __init__(self, algorithm="lightgbm"):
        self.algorithm = algorithm

    def fit(self, X, y):
        self.preprocessor_ = PolicyPreprocessor().fit(X)
        transformed = self.preprocessor_.transform(X)
        self.model_ = make_classifier(self.algorithm)
        weights = compute_sample_weight("balanced", y) if self.algorithm in WEIGHTED_ALGORITHMS else None
        checked_fit(self.model_, matrix_for_model(self.model_, transformed), y, **({"sample_weight": weights} if weights is not None else {}))
        self.classes_ = self.model_.classes_
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict_proba(self, X):
        transformed = self.preprocessor_.transform(X)
        return self.model_.predict_proba(matrix_for_model(self.model_, transformed))

    def predict(self, X):
        transformed = self.preprocessor_.transform(X)
        return self.model_.predict(matrix_for_model(self.model_, transformed))


@cached_analysis
def select_classifier():
    fit, validation, y_fit, y_validation = selection_split()
    processor = PolicyPreprocessor().fit(fit)
    fit_features, validation_features = processor.transform(fit), processor.transform(validation)
    rows = []
    for algorithm, name in available_classifiers().items():
        model = make_classifier(algorithm)
        weights = compute_sample_weight("balanced", y_fit) if algorithm in WEIGHTED_ALGORITHMS else None
        checked_fit(model, matrix_for_model(model, fit_features), y_fit, **({"sample_weight": weights} if weights is not None else {}))
        metrics = classification_metrics(model, matrix_for_model(model, validation_features), y_validation)
        rows.append({"algorithm": algorithm, "Model": name,
                     **{key: value for key, value in metrics.items() if key not in {"predictions", "probabilities"}}})
        print(f"Validation checked: {name}", flush=True)
    rows.sort(key=lambda row: (row["Average Precision"], row["ROC AUC"]), reverse=True)
    return {"best": rows[0], "comparison": rows, "fit_rows": len(fit), "validation_rows": len(validation),
            "fit_index_hash": hashlib.sha256(np.asarray(fit.index, dtype=np.int64).tobytes()).hexdigest(),
            "validation_index_hash": hashlib.sha256(np.asarray(validation.index, dtype=np.int64).tobytes()).hexdigest()}


@cached_analysis
def calibrated_selected_model(algorithm):
    raw_train, raw_test, y_train, y_test = raw_training_data()
    # Calibration uses out-of-fold training predictions; threshold tuning has
    # its own outer folds. Neither receives any test rows or test labels.
    calibrator = CalibratedClassifierCV(
        estimator=PolicyClassifier(algorithm), method="sigmoid", ensemble=False,
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=137), n_jobs=1,
    )
    model = TunedThresholdClassifierCV(
        estimator=calibrator, scoring="f1", thresholds=200,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=193),
        refit=True, n_jobs=1, store_cv_results=True,
    )
    checked_fit(model, raw_train, y_train)
    metrics = classification_metrics(model, raw_test, y_test)
    return {"algorithm": algorithm, "Model": available_classifiers()[algorithm], "model": model,
            "threshold": float(model.best_threshold_), "training_cv_f1": float(model.best_score_),
            "calibrated": True, **metrics}
