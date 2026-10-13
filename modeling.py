"""Common evaluation and plotting for fair supervised model comparisons."""
import numpy as np
import warnings
from scipy import sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, precision_recall_curve, auc,
    log_loss, brier_score_loss, balanced_accuracy_score,
)
from sklearn.exceptions import ConvergenceWarning
from project_runtime import PLOT_DIR, atomic_output


def model_matrix(frame, dtype=np.float32):
    # Build by column to avoid a large dense float64 copy of one-hot features.
    if frame.empty or frame.columns.has_duplicates:
        raise ValueError("Model inputs need rows and uniquely named predictor columns.")
    matrix = sparse.hstack([sparse.csr_matrix(frame[c].to_numpy(dtype=dtype).reshape(-1, 1)) for c in frame], format="csr")
    if not np.isfinite(matrix.data).all():
        raise ValueError("Model predictors must contain only finite numeric values.")
    return matrix


def matrix_for_model(model, frame):
    # XGBoost tree boosters interpret absent CSR entries as missing. Preserve
    # actual zeros in binary/one-hot features by supplying a dense matrix.
    if model.__class__.__module__.split(".")[0] == "xgboost":
        matrix = frame.to_numpy(dtype=np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError("Model predictors must contain only finite numeric values.")
        return matrix
    return model_matrix(frame)


def checked_fit(model, X, y, **parameters):
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        return model.fit(X, y, **parameters)


def classification_metrics(model, X_test, y_test):
    y_test = np.asarray(y_test)
    predictions = np.asarray(model.predict(X_test))
    distribution = np.asarray(model.predict_proba(X_test), dtype=np.float64)
    classes = list(model.classes_)
    if len(classes) != 2 or set(classes) != {0, 1}:
        raise ValueError("Classification requires model classes 0 and 1.")
    if y_test.ndim != 1 or predictions.shape != y_test.shape or distribution.shape != (len(y_test), 2):
        raise ValueError("Prediction, probability and target row counts must match.")
    if set(np.unique(y_test)) != {0, 1} or not np.isin(predictions, [0, 1]).all():
        raise ValueError("Evaluation requires binary labels and both test classes.")
    if not np.isfinite(distribution).all() or (distribution < 0).any() or (distribution > 1).any() or not np.allclose(distribution.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("Predicted probabilities must be finite, between 0 and 1, and sum to 1.")
    probabilities = distribution[:, classes.index(1)]
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    return {
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "Precision": float(precision_score(y_test, predictions, zero_division=0)),
        "Recall": float(recall_score(y_test, predictions, zero_division=0)),
        "F1 Score": float(f1_score(y_test, predictions, zero_division=0)),
        "ROC AUC": float(roc_auc_score(y_test, probabilities)),
        "Average Precision": float(average_precision_score(y_test, probabilities)),
        "PR AUC": float(auc(recall, precision)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "Log Loss": float(log_loss(y_test, np.column_stack([1 - probabilities, probabilities]), labels=[0, 1])),
        "Brier Score": float(brier_score_loss(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "predictions": predictions, "probabilities": probabilities,
    }


def save_figure(filename):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    try:
        with atomic_output(PLOT_DIR / filename) as temporary:
            plt.savefig(temporary, dpi=120)
    finally:
        plt.close()
    return "plots/" + filename


def confusion_plot(y_test, predictions, name, filename):
    ConfusionMatrixDisplay(confusion_matrix(y_test, predictions, labels=[0, 1]), display_labels=["Not interested", "Interested"]).plot(cmap="Blues")
    plt.title(f"{name} — Confusion Matrix")
    return save_figure(filename)


def roc_plot(y_test, probabilities, name, filename):
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_test, probabilities):.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} — ROC Curve")
    plt.legend()
    return save_figure(filename)
