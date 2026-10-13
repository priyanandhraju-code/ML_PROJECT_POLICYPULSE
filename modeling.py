"""Common evaluation and plotting for fair supervised model comparisons."""
import numpy as np
from scipy import sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve,
)
from project_runtime import PLOT_DIR, atomic_output


def model_matrix(frame, dtype=np.float32):
    # Build by column to avoid a large dense float64 copy of one-hot features.
    return sparse.hstack([sparse.csr_matrix(frame[c].to_numpy(dtype=dtype).reshape(-1, 1)) for c in frame], format="csr")


def classification_metrics(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "Precision": float(precision_score(y_test, predictions, zero_division=0)),
        "Recall": float(recall_score(y_test, predictions, zero_division=0)),
        "F1 Score": float(f1_score(y_test, predictions, zero_division=0)),
        "ROC AUC": float(roc_auc_score(y_test, probabilities)),
        "PR AUC": float(average_precision_score(y_test, probabilities)),
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
