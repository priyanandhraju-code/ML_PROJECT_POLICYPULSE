"""Bounded, reproducible customer clustering adapted from Project.zip."""
import hashlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, dendrogram
from preprocessing import PolicyPreprocessor
from load_data import load_data
from project_runtime import cached_analysis
from modeling import save_figure
import matplotlib.pyplot as plt

SAMPLE_LIMITS = {"kmeans": 3500, "hierarchical": 1200, "dbscan": 2500}


def get_clustering_data(sample_size=3500):
    # Unsupervised preprocessing must not depend on Response or on the
    # classification split (which is stratified using that label).
    frame = load_data().drop(columns=["Response", "id"], errors="ignore")
    if len(frame) > sample_size:
        frame = frame.sample(n=sample_size, random_state=42)
    frame = PolicyPreprocessor().fit(frame).transform(frame)
    frame = frame.loc[:, frame.nunique() > 1].astype(np.float64)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Clustering inputs must contain only finite numbers.")
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


def calculate_elbow(X, filename="kmeans_elbow.png"):
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
    plt.axvline(recommended, color="crimson", linestyle="--", label=f"Elbow recommendation K = {recommended}")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.grid(alpha=0.2)
    plt.legend()
    save_figure(filename)
    return {"k_values": k_values, "scores": scores, "recommended_k": recommended, "plot": filename}


def calculate_silhouette(X, filename="kmeans_silhouette.png"):
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
    plt.axvline(k_values[best_index], color="crimson", linestyle="--", label=f"Best silhouette K = {k_values[best_index]}")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Evaluation Curve")
    plt.grid(alpha=0.2)
    plt.legend()
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


def dendrogram_cutoff(merge_distances, k):
    """Height between the last within-cluster and next between-cluster merge.

    Equal merge heights cannot express every K with a horizontal cut. In
    that case return the tied height as a reference, and disclose the tie.
    """
    lower, upper = float(merge_distances[-k]), float(merge_distances[-k + 1])
    return (lower + upper) / 2, lower == upper


@cached_analysis
def run_clustering(algorithm="kmeans", method="silhouette", k=3, x_feature="Annual_Premium", y_feature="Vintage", linkage_method="ward", eps=0.8, min_samples=10):
    if algorithm not in SAMPLE_LIMITS:
        raise ValueError("Choose K-Means, hierarchical clustering, or DBSCAN.")
    X = get_clustering_data(SAMPLE_LIMITS[algorithm])
    if x_feature not in X or y_feature not in X:
        raise ValueError("Choose available visualization features.")
    token = hashlib.sha256(repr((algorithm, method, k, x_feature, y_feature, linkage_method, eps, min_samples)).encode()).hexdigest()[:12]
    elbow, silhouette, dendro_plot = None, None, None
    cutoff, cutoff_tied = None, False
    if algorithm == "kmeans":
        if method not in {"manual", "elbow", "silhouette"}:
            raise ValueError("Invalid K selection method.")
        # Evaluate the same all-feature sample for both recommendations.
        # Per-run filenames keep charts for different configurations independent.
        elbow = calculate_elbow(X, f"kmeans_elbow_{token}.png")
        silhouette = calculate_silhouette(X, f"kmeans_silhouette_{token}.png")
        if method == "elbow":
            k = elbow["recommended_k"]
        elif method == "silhouette":
            k = silhouette["recommended_k"]
        model, labels = run_kmeans(X, k)
    elif algorithm == "hierarchical":
        _validate_k(X, k)
        if linkage_method not in {"ward", "complete", "average"}:
            raise ValueError("Choose ward, complete, or average linkage.")
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
        labels = model.fit_predict(X)
        merges = linkage(X, method=linkage_method)
        cutoff, cutoff_tied = dendrogram_cutoff(merges[:, 2], k)
        plt.figure(figsize=(12, 6))
        dendrogram(merges, truncate_mode="lastp", p=25, leaf_rotation=90, color_threshold=cutoff)
        plt.axhline(cutoff, color="crimson", linestyle="--", label=f"Reference cutoff (K = {k})")
        plt.title(f"Hierarchical Clustering — {linkage_method.title()} Linkage")
        plt.ylabel("Euclidean Distance")
        plt.xlabel("Customer grouping / sample index (counts in parentheses)")
        plt.legend()
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
    palette = plt.get_cmap("tab10")
    for label in sorted(set(labels)):
        mask = labels == label
        if label == -1:
            plt.scatter(projected[mask, 0], projected[mask, 1], s=24, marker="x", color="#e34e54",
                        alpha=0.65, label=f"Noise / Outliers ({int(mask.sum()):,})")
        else:
            plt.scatter(projected[mask, 0], projected[mask, 1], s=14, alpha=0.6,
                        color=palette(int(label) % 10), label=f"Cluster {label} ({int(mask.sum()):,})")
    if hasattr(model, "cluster_centers_"):
        centers = pca.transform(pd.DataFrame(model.cluster_centers_, columns=X.columns))
        plt.scatter(centers[:, 0], centers[:, 1], marker="X", s=140, color="black", edgecolor="white", label="Cluster centers")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    title = {"kmeans": f"K-Means Customer Clusters (K = {k})",
             "hierarchical": f"Hierarchical Customer Clusters ({linkage_method.title()}, K = {k})",
             "dbscan": f"DBSCAN Customer Clusters (eps = {eps:g}, min samples = {min_samples})"}[algorithm]
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.legend(fontsize=8, loc="best", ncols=2 if len(set(labels)) > 6 else 1)
    pca_plot = f"{algorithm}_pca_{token}.png"
    save_figure(pca_plot)
    raw = load_data().loc[X.index].copy()
    raw["Cluster"] = labels
    profiles = []
    for label, group in raw.groupby("Cluster", sort=True):
        profiles.append({"cluster": int(label), "count": len(group), "mean_age": float(group.Age.mean()),
                         "population_percentage": float(len(group) / len(raw) * 100),
                         "mean_premium": float(group.Annual_Premium.mean()),
                         "mean_vintage": float(group.Vintage.mean()),
                         "previously_insured_rate": float(group.Previously_Insured.mean() * 100),
                         "vehicle_damage_rate": float(group.Vehicle_Damage.eq("Yes").mean() * 100),
                         "response_rate": float(group.Response.mean() * 100)})
    score = _safe_silhouette(X, labels)
    assigned = labels != -1
    extra_metrics = {}
    if 2 <= len(set(labels[assigned])) < int(assigned.sum()):
        extra_metrics = {"davies_bouldin": float(davies_bouldin_score(X.loc[assigned], labels[assigned])),
                         "calinski_harabasz": float(calinski_harabasz_score(X.loc[assigned], labels[assigned]))}
    return {"rows": len(X), "total_rows": len(load_data()), "features": X.shape[1], "k": k,
            "model": model, "labels": labels, "sample_indices": X.index.tolist(), "feature_names": X.columns.tolist(),
            "algorithm": algorithm, "method": method.title() if algorithm == "kmeans" else linkage_method.title() if algorithm == "hierarchical" else "Density",
            "wcss": float(model.inertia_) if hasattr(model, "inertia_") else None,
            "silhouette": score, "distribution": get_cluster_distribution(labels),
            "noise": int((labels == -1).sum()), "profiles": profiles, "cluster_plot": cluster_plot,
            "noise_percentage": float(np.mean(labels == -1) * 100),
            "eps": eps if algorithm == "dbscan" else None,
            "min_samples": min_samples if algorithm == "dbscan" else None,
            "linkage_method": linkage_method if algorithm == "hierarchical" else None,
            "reference_cutoff": cutoff, "cutoff_tied": cutoff_tied,
            "pca_variance": pca.explained_variance_ratio_.tolist(),
            "pca_plot": pca_plot, "dendro_plot": dendro_plot,
            "elbow_plot": elbow["plot"] if elbow else None, "silhouette_plot": silhouette["plot"] if silhouette else None,
            "elbow_analysis": elbow, "silhouette_analysis": silhouette,
            "recommended_k": (elbow["recommended_k"] if method == "elbow" else silhouette["recommended_k"] if method == "silhouette" else None) if algorithm == "kmeans" else None,
            "best_silhouette_k": silhouette["recommended_k"] if silhouette else None,
            "highest_silhouette": silhouette["highest_score"] if silhouette else None, **extra_metrics}
