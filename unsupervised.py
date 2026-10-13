"""Bounded, reproducible customer clustering adapted from Project.zip."""
import hashlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, dendrogram
from preprocessing import preprocess_data
from load_data import load_data
from project_runtime import cached_analysis
from modeling import save_figure
import matplotlib.pyplot as plt

SAMPLE_LIMITS = {"kmeans": 3500, "hierarchical": 1200, "dbscan": 2500}


def get_clustering_data(sample_size=3500):
    frame = preprocess_data().drop(columns="Response")
    if len(frame) > sample_size:
        frame = frame.sample(n=sample_size, random_state=42)
    frame = frame.loc[:, frame.nunique() > 1].astype(np.float32)
    if len(frame) < 3 or frame.shape[1] < 2:
        raise ValueError("Clustering needs at least three rows and two varying features.")
    return frame


def _validate_k(X, k):
    if not 2 <= k <= min(10, len(X) - 1, len(X.drop_duplicates())):
        raise ValueError("K must be between 2 and 10 and smaller than the number of available rows; enough distinct records are required.")


def _safe_silhouette(X, labels):
    labels = np.asarray(labels)
    mask = labels != -1
    subset, assigned = np.asarray(X)[mask], labels[mask]
    if not 2 <= len(np.unique(assigned)) < len(assigned):
        return None
    if len(subset) > 1500:
        indices = np.random.default_rng(42).choice(len(subset), 1500, replace=False)
        subset, assigned = subset[indices], assigned[indices]
    if not 2 <= len(np.unique(assigned)) < len(assigned):
        return None
    return float(silhouette_score(subset, assigned))


def calculate_elbow(X):
    k_values = list(range(1, min(10, len(X) - 1, len(X.drop_duplicates())) + 1))
    if len(k_values) < 2:
        raise ValueError("Not enough distinct records for elbow analysis.")
    scores = [float(KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_) for k in k_values]
    # Maximum normalized distance from the end-point chord is an elbow heuristic.
    normalized = (np.asarray(scores) - min(scores)) / max(max(scores) - min(scores), 1e-12)
    chord = np.linspace(1, 0, len(scores))
    recommended = max(2, k_values[int(np.argmax(chord - normalized))])
    plt.figure(figsize=(9, 6))
    plt.plot(k_values, scores, marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    filename = "kmeans_elbow.png"
    save_figure(filename)
    return {"k_values": k_values, "scores": scores, "recommended_k": recommended, "plot": filename}


def calculate_silhouette(X):
    k_values, scores = [], []
    for k in range(2, min(10, len(X) - 1, len(X.drop_duplicates())) + 1):
        model, clusters = run_kmeans(X, k)
        score = _safe_silhouette(X, clusters)
        if score is not None:
            k_values.append(k)
            scores.append(score)
    if not scores:
        raise ValueError("No valid cluster separation could be measured.")
    best_index = int(np.argmax(scores))
    plt.figure(figsize=(9, 6))
    plt.plot(k_values, scores, marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score Method")
    filename = "kmeans_silhouette.png"
    save_figure(filename)
    return {"k_values": k_values, "scores": scores, "recommended_k": k_values[best_index], "highest_score": scores[best_index], "plot": filename}


def run_kmeans(X, k):
    _validate_k(X, k)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    return model, model.fit_predict(X)


def calculate_kmeans_performance(X, clusters, model):
    return float(model.inertia_), _safe_silhouette(X, clusters)


def get_cluster_distribution(clusters):
    return {int(label): int(count) for label, count in pd.Series(clusters).value_counts().sort_index().items()}


def create_cluster_plot(X, clusters, model, x_feature, y_feature, filename="kmeans_clusters.png"):
    if x_feature not in X or y_feature not in X:
        raise ValueError("Choose available visualization features.")
    plt.figure(figsize=(9, 6))
    for label in sorted(set(clusters)):
        mask = clusters == label
        plt.scatter(X.loc[mask, x_feature], X.loc[mask, y_feature], s=12, alpha=0.5,
                    label="Noise" if label == -1 else f"Cluster {label}")
    if hasattr(model, "cluster_centers_"):
        centers = model.cluster_centers_
        plt.scatter(centers[:, X.columns.get_loc(x_feature)], centers[:, X.columns.get_loc(y_feature)], marker="X", s=180, color="black", label="Centers")
    plt.xlabel(x_feature + " (processed)")
    plt.ylabel(y_feature + " (processed)")
    plt.title("Customer Clusters")
    plt.legend()
    save_figure(filename)
    return filename


@cached_analysis
def run_clustering(algorithm="kmeans", method="silhouette", k=3, x_feature="Annual_Premium", y_feature="Vintage", linkage_method="ward", eps=0.8, min_samples=10):
    if algorithm not in SAMPLE_LIMITS:
        raise ValueError("Choose K-Means, hierarchical clustering, or DBSCAN.")
    X = get_clustering_data(SAMPLE_LIMITS[algorithm])
    if x_feature not in X or y_feature not in X:
        raise ValueError("Choose available visualization features.")
    token = hashlib.sha256(repr((algorithm, method, k, x_feature, y_feature, linkage_method, eps, min_samples)).encode()).hexdigest()[:12]
    elbow, silhouette, dendro_plot = None, None, None
    if algorithm == "kmeans":
        if method == "elbow":
            elbow = calculate_elbow(X)
            k = elbow["recommended_k"]
        elif method == "silhouette":
            silhouette = calculate_silhouette(X)
            k = silhouette["recommended_k"]
        elif method != "manual":
            raise ValueError("Invalid K selection method.")
        model, labels = run_kmeans(X, k)
    elif algorithm == "hierarchical":
        _validate_k(X, k)
        if linkage_method not in {"ward", "complete", "average"}:
            raise ValueError("Choose ward, complete, or average linkage.")
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
        labels = model.fit_predict(X)
        plt.figure(figsize=(10, 5))
        dendrogram(linkage(X, method=linkage_method), truncate_mode="lastp", p=25, leaf_rotation=90)
        plt.title(f"Hierarchical Clustering — {linkage_method.title()} Linkage")
        plt.ylabel("Euclidean Distance")
        dendro_plot = f"hierarchical_dendrogram_{token}.png"
        save_figure(dendro_plot)
    else:
        if not np.isfinite(eps) or not 0 < eps <= 20:
            raise ValueError("Epsilon must be a finite number greater than 0 and at most 20.")
        if not 2 <= min_samples <= min(100, len(X)):
            raise ValueError("Minimum samples must be between 2 and 100 and no greater than the sample size.")
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        k = len(set(labels) - {-1})
    cluster_plot = create_cluster_plot(X, labels, model, x_feature, y_feature, f"{algorithm}_clusters_{token}.png")
    pca = PCA(n_components=2, random_state=42)
    projected = pca.fit_transform(X)
    plt.figure(figsize=(9, 6))
    plt.scatter(projected[:, 0], projected[:, 1], c=labels, s=12, alpha=0.5, cmap="tab10")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.title(f"{algorithm.title()} — PCA Visualization (noise label = -1)")
    pca_plot = f"{algorithm}_pca_{token}.png"
    save_figure(pca_plot)
    raw = load_data().loc[X.index].copy()
    raw["Cluster"] = labels
    profiles = []
    for label, group in raw.groupby("Cluster", sort=True):
        profiles.append({"cluster": int(label), "count": len(group), "mean_age": float(group.Age.mean()),
                         "mean_premium": float(group.Annual_Premium.mean()), "response_rate": float(group.Response.mean() * 100)})
    score = _safe_silhouette(X, labels)
    assigned = labels != -1
    extra_metrics = {}
    if 2 <= len(set(labels[assigned])) < int(assigned.sum()):
        extra_metrics = {"davies_bouldin": float(davies_bouldin_score(X.loc[assigned], labels[assigned])),
                         "calinski_harabasz": float(calinski_harabasz_score(X.loc[assigned], labels[assigned]))}
    return {"rows": len(X), "total_rows": len(preprocess_data()), "features": X.shape[1], "k": k,
            "algorithm": algorithm, "method": method.title() if algorithm == "kmeans" else linkage_method.title() if algorithm == "hierarchical" else "Density",
            "wcss": float(model.inertia_) if hasattr(model, "inertia_") else None,
            "silhouette": score, "distribution": get_cluster_distribution(labels),
            "noise": int((labels == -1).sum()), "profiles": profiles, "cluster_plot": cluster_plot,
            "pca_plot": pca_plot, "dendro_plot": dendro_plot,
            "elbow_plot": elbow["plot"] if elbow else None, "silhouette_plot": silhouette["plot"] if silhouette else None,
            "recommended_k": silhouette["recommended_k"] if silhouette else elbow["recommended_k"] if elbow else None,
            "highest_silhouette": silhouette["highest_score"] if silhouette else None, **extra_metrics}
