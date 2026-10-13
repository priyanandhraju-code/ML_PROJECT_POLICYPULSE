"""Independent NumPy checks of every fitted model's full-dataset outputs."""
import argparse
import json
import sys
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import precision_recall_curve, auc
from preprocessing import get_model_data
from modeling import model_matrix, matrix_for_model
from linear_regression import regression_data, run_linear_regression
from logistic_regression import run_logistic_regression
from tree_based import run_tree_algorithm, ALGORITHMS
from unsupervised import get_clustering_data, run_kmeans
from unsupervised import run_clustering, SAMPLE_LIMITS
from model_selection import select_classifier, calibrated_selected_model, raw_training_data, selection_split
from scipy.spatial.distance import cdist


def independent_classification(y, predicted, scores):
    y, predicted, scores = np.asarray(y), np.asarray(predicted), np.asarray(scores, dtype=np.float64)
    assert len(y) == len(predicted) == len(scores)
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    tp = int(((y == 1) & (predicted == 1)).sum())
    fp = int(((y == 0) & (predicted == 1)).sum())
    tn = int(((y == 0) & (predicted == 0)).sum())
    fn = int(((y == 1) & (predicted == 0)).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    positives, negatives = int(y.sum()), int((y == 0).sum())
    ranks = rankdata(scores, method="average")
    roc = (ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    order = np.argsort(-scores, kind="stable")
    ordered_y, ordered_scores = y[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered_scores)), len(y) - 1]
    cumulative = np.cumsum(ordered_y)[ends]
    average_precision = float(np.sum(np.diff(np.r_[0, cumulative]) / positives * cumulative / (ends + 1)))
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(scores, epsilon, 1 - epsilon)
    return {"Accuracy": (tp + tn) / len(y), "Precision": precision, "Recall": recall,
            "F1 Score": 2 * precision * recall / max(precision + recall, 1e-30), "ROC AUC": float(roc),
            "Average Precision": average_precision, "Log Loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
            "Brier Score": float(np.mean((scores - y) ** 2)), "confusion_matrix": [[tn, fp], [fn, tp]]}


def independent_silhouette(X, labels):
    mask = np.asarray(labels) != -1
    X, labels = np.asarray(X, dtype=np.float64)[mask], np.asarray(labels)[mask]
    if len(X) > 1500:
        indices = np.random.default_rng(42).choice(len(X), 1500, replace=False)
        X, labels = X[indices], labels[indices]
    if not 2 <= len(set(labels)) < len(labels):
        return None
    distances = cdist(X, X)
    silhouette = np.zeros(len(X))
    for label in set(labels):
        own = labels == label
        if own.sum() == 1:
            continue
        a = distances[np.ix_(own, own)].sum(axis=1) / (own.sum() - 1)
        b = np.min(np.stack([distances[np.ix_(own, labels == other)].mean(axis=1) for other in set(labels) if other != label]), axis=0)
        silhouette[own] = np.divide(b - a, np.maximum(a, b), out=np.zeros_like(a), where=np.maximum(a, b) > 0)
    return float(silhouette.mean())


def independent_cluster_indices(X, labels):
    labels = np.asarray(labels)
    retained = labels != -1
    values, labels = np.asarray(X, dtype=np.float64)[retained], labels[retained]
    groups = [values[labels == label] for label in sorted(set(labels))]
    if not 2 <= len(groups) < len(values):
        return {}
    centers = np.array([group.mean(axis=0) for group in groups])
    within = sum(np.sum((group - center) ** 2) for group, center in zip(groups, centers))
    between = sum(len(group) * np.sum((center - values.mean(axis=0)) ** 2) for group, center in zip(groups, centers))
    calinski = 1.0 if within == 0 else (between / (len(groups) - 1)) / (within / (len(values) - len(groups)))
    scatters = np.array([np.linalg.norm(group - center, axis=1).mean() for group, center in zip(groups, centers)])
    center_distances = cdist(centers, centers)
    if np.allclose(scatters, 0) or np.allclose(center_distances, 0):
        davies = 0.0
    else:
        center_distances[center_distances == 0] = np.inf
        davies = np.max((scatters[:, None] + scatters[None, :]) / center_distances, axis=1).mean()
    return {"davies_bouldin": float(davies), "calinski_harabasz": float(calinski)}


def main(output):
    train, test, y_train, y_test = get_model_data()
    X_test = model_matrix(test)
    report = {"dataset_sha256": hashlib.sha256((ROOT / "Policypulse.csv").read_bytes()).hexdigest(),
              "rows": len(train) + len(test), "training_rows": len(train), "testing_rows": len(test),
              "classification": [], "regression": []}
    raw_health = pd.read_csv(ROOT / "Policypulse.csv")
    report["data_health"] = {"raw_missing_values": int(raw_health.isna().sum().sum()),
                              "duplicate_customer_ids": int(raw_health.id.duplicated().sum()),
                              "class_counts": {str(key): int(value) for key, value in raw_health.Response.value_counts().items()}}
    logistic = run_logistic_regression()
    entries = [(name, model, logistic["comparison"][index]) for index, (name, model) in enumerate(logistic["models"].items())]
    for algorithm in ALGORITHMS:
        result = run_tree_algorithm(algorithm)
        entries.append((result["algorithm"], result["model"], result))
    for name, model, reported in entries:
        inputs = matrix_for_model(model, test)
        predicted = model.predict(inputs)
        scores = model.predict_proba(inputs)[:, list(model.classes_).index(1)]
        values = independent_classification(y_test, predicted, scores)
        for metric in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]:
            np.testing.assert_allclose(reported[metric], values[metric], rtol=1e-10, atol=1e-12)
        key = "Average Precision" if "Average Precision" in reported else "PR AUC"
        np.testing.assert_allclose(reported[key], values["Average Precision"], rtol=1e-10, atol=1e-12)
        precision, recall, _ = precision_recall_curve(y_test, scores)
        values["trapezoidal_pr_auc"] = float(auc(recall, precision))
        for metric in ["Log Loss", "Brier Score"]:
            if metric in reported:
                np.testing.assert_allclose(reported[metric], values[metric], rtol=1e-10, atol=1e-12)
        if "Average Precision" in reported:
            np.testing.assert_allclose(reported["PR AUC"], values["trapezoidal_pr_auc"], rtol=1e-10, atol=1e-12)
        values["mean_score"] = float(scores.mean())
        report["classification"].append({"Model": name, **values})
        print(name, "independent metrics passed", flush=True)
    _, reg_test, reg_y_train, reg_y_test = regression_data()
    regression = run_linear_regression()
    matrix = model_matrix(reg_test, np.float64)
    for index, (name, model) in enumerate(regression["models"].items()):
        predictions = model.predict(matrix)
        residual = np.asarray(reg_y_test, dtype=np.float64) - predictions
        values = {"MAE": float(np.mean(np.abs(residual))), "MSE": float(np.mean(residual ** 2)),
                  "RMSE": float(np.sqrt(np.mean(residual ** 2))), "R2 Score": float(1 - np.sum(residual ** 2) / np.sum((reg_y_test - reg_y_test.mean()) ** 2))}
        for metric, value in values.items():
            np.testing.assert_allclose(value, regression["comparison"][index][metric], rtol=1e-10, atol=1e-8)
        report["regression"].append({"Model": name, **values, "negative_predictions": int((predictions < 0).sum()),
                                     "iterations": np.asarray(getattr(model, "estimator_", model).n_iter_).tolist() if getattr(getattr(model, "estimator_", model), "n_iter_", None) is not None else None})
        print(name, "independent metrics passed", flush=True)
    X = get_clustering_data()
    model, labels = run_kmeans(X, 3)
    residuals = np.asarray(X, dtype=np.float64) - np.asarray(model.cluster_centers_, dtype=np.float64)[labels]
    manual_wcss = float(np.sum(residuals ** 2))
    report["kmeans"] = {"rows": len(X), "features": X.shape[1], "model_features": model.n_features_in_,
                        "reported_wcss": float(model.inertia_), "independent_wcss": manual_wcss,
                        "relative_wcss_error": abs(model.inertia_ - manual_wcss) / max(manual_wcss, 1e-15)}
    assert report["kmeans"]["relative_wcss_error"] < 1e-5
    report["clustering"] = []
    configurations = [("kmeans", method, "ward", 0.8, 10) for method in ["manual", "elbow", "silhouette"]]
    configurations += [("hierarchical", "manual", linkage, 0.8, 10) for linkage in ["ward", "complete", "average"]]
    configurations += [("dbscan", "manual", "ward", 0.8, 10), ("dbscan", "manual", "ward", 0.00001, 100)]
    for algorithm, method, linkage_method, eps, min_samples in configurations:
        result = run_clustering(algorithm=algorithm, method=method, k=3, linkage_method=linkage_method, eps=eps, min_samples=min_samples)
        features = get_clustering_data(SAMPLE_LIMITS[algorithm])
        assert result["sample_indices"] == features.index.tolist()
        assert result["feature_names"] == features.columns.tolist()
        np.testing.assert_array_equal(result["labels"], result["model"].labels_)
        score = independent_silhouette(features, result["labels"])
        if score is None:
            assert result["silhouette"] is None
        else:
            np.testing.assert_allclose(result["silhouette"], score, rtol=1e-10, atol=1e-10)
        additional_indices = independent_cluster_indices(features, result["labels"])
        for metric, value in additional_indices.items():
            np.testing.assert_allclose(result[metric], value, rtol=1e-10, atol=1e-10)
        if algorithm == "kmeans":
            centers = result["model"].cluster_centers_[result["labels"]]
            np.testing.assert_allclose(result["wcss"], np.sum((features.to_numpy() - centers) ** 2), rtol=1e-12)
            if method == "silhouette":
                np.testing.assert_allclose(result["silhouette"], result["highest_silhouette"], rtol=1e-12)
        assert sum(result["distribution"].values()) == len(features)
        raw = pd.read_csv(ROOT / "Policypulse.csv").loc[features.index].assign(Cluster=result["labels"])
        for profile in result["profiles"]:
            group = raw[raw.Cluster == profile["cluster"]]
            assert len(group) == profile["count"]
            np.testing.assert_allclose(profile["response_rate"], group.Response.mean() * 100, rtol=1e-12)
            np.testing.assert_allclose(profile["population_percentage"], len(group) / len(raw) * 100, rtol=1e-12)
            np.testing.assert_allclose(profile["mean_age"], group.Age.mean(), rtol=1e-12)
            np.testing.assert_allclose(profile["mean_premium"], group.Annual_Premium.mean(), rtol=1e-12)
            np.testing.assert_allclose(profile["mean_vintage"], group.Vintage.mean(), rtol=1e-12)
            np.testing.assert_allclose(profile["previously_insured_rate"], group.Previously_Insured.mean() * 100, rtol=1e-12)
            np.testing.assert_allclose(profile["vehicle_damage_rate"], group.Vehicle_Damage.eq("Yes").mean() * 100, rtol=1e-12)
        np.testing.assert_allclose(result["noise_percentage"], np.mean(result["labels"] == -1) * 100)
        variance = np.linalg.svd(features.to_numpy() - features.to_numpy().mean(axis=0), compute_uv=False) ** 2
        np.testing.assert_allclose(result["pca_variance"], variance[:2] / variance.sum(), rtol=1e-10)
        if algorithm == "hierarchical" and not result["cutoff_tied"]:
            from scipy.cluster.hierarchy import linkage, fcluster
            from sklearn.metrics import adjusted_rand_score
            cut_labels = fcluster(linkage(features, method=linkage_method), result["reference_cutoff"], criterion="distance")
            assert len(set(cut_labels)) == result["k"]
            assert adjusted_rand_score(cut_labels, result["labels"]) == 1
        report["clustering"].append({"algorithm": algorithm, "method": method, "linkage": linkage_method if algorithm == "hierarchical" else None,
                                      "eps": eps if algorithm == "dbscan" else None, "features": result["features"], "rows": result["rows"],
                                      "silhouette": score, **additional_indices, "status": "independent checks passed"})
        print(algorithm, method, linkage_method if algorithm == "hierarchical" else eps if algorithm == "dbscan" else "", "independent cluster metrics passed", flush=True)
    selection = select_classifier()
    calibrated = calibrated_selected_model(selection["best"]["algorithm"])
    fit, validation, _, _ = selection_split()
    raw_train, raw_test, y_train, y_test = raw_training_data()
    assert set(fit.index).isdisjoint(validation.index)
    assert set(fit.index).isdisjoint(raw_test.index)
    assert set(validation.index).isdisjoint(raw_test.index)
    model = calibrated["model"]
    probabilities = model.predict_proba(raw_test)[:, list(model.classes_).index(1)]
    predicted = model.predict(raw_test)
    independently_checked = independent_classification(y_test, predicted, probabilities)
    for metric in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "Average Precision", "Log Loss", "Brier Score"]:
        np.testing.assert_allclose(calibrated[metric], independently_checked[metric], rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(predicted, (probabilities >= calibrated["threshold"]).astype(int))
    report["selected_classifier"] = {"Model": calibrated["Model"], "threshold": calibrated["threshold"],
                                       "validation_average_precision": selection["best"]["Average Precision"], **independently_checked}
    report["selection_split"] = {"fit_rows": len(fit), "validation_rows": len(validation), "test_rows": len(raw_test), "disjoint": True}
    prevalence = float(y_train.mean())
    report["baselines"] = independent_classification(y_test, np.zeros(len(y_test)), np.full(len(y_test), prevalence))
    report["regression_baseline"] = {"median_mae": float(np.mean(np.abs(reg_y_test - np.median(reg_y_train)))),
                                     "mean_rmse": float(np.sqrt(np.mean((reg_y_test - reg_y_train.mean()) ** 2)))}
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "docs" / "model_audit_results.json"))
    main(parser.parse_args().output)
