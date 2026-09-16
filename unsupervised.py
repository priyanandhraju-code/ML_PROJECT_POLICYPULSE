import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from data_loader import load_data


BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PLOT_DIRECTORY = os.path.join(BASE_DIRECTORY, "static", "plots")
os.makedirs(PLOT_DIRECTORY, exist_ok=True)


def get_clustering_data():
    X = load_data().copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean(numeric_only=True))
    return X.astype(float)


def calculate_elbow(X):
    k_values = list(range(1, 11))
    wcss_values = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        wcss_values.append(model.inertia_)

    plt.figure(figsize=(9, 6))
    plt.plot(k_values, wcss_values, marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.xticks(k_values)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIRECTORY, "kmeans_elbow.png"))
    plt.close()

    return k_values, wcss_values


def calculate_silhouette(X):
    k_values = list(range(2, 11))
    scores = []
    sample_size = min(5000, len(X))

    for k in k_values:
        print(f"Calculating Silhouette Score for K = {k}")
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = model.fit_predict(X)
        score = silhouette_score(
            X,
            clusters,
            sample_size=sample_size,
            random_state=42
        )
        scores.append(score)

    best_index = int(np.argmax(scores))
    recommended_k = k_values[best_index]
    highest_score = scores[best_index]

    plt.figure(figsize=(9, 6))
    plt.plot(k_values, scores, marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score Method")
    plt.xticks(k_values)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIRECTORY, "kmeans_silhouette.png"))
    plt.close()

    return {
        "k_values": k_values,
        "scores": scores,
        "recommended_k": recommended_k,
        "highest_score": highest_score
    }


def run_kmeans(X, k):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = model.fit_predict(X)
    return model, clusters


def calculate_kmeans_performance(X, clusters, model):
    wcss = model.inertia_
    silhouette = silhouette_score(
        X,
        clusters,
        sample_size=min(5000, len(X)),
        random_state=42
    )
    return wcss, silhouette


def get_cluster_distribution(clusters):
    return pd.Series(clusters).value_counts().sort_index().to_dict()


def create_cluster_plot(X, clusters, model, x_feature, y_feature):
    plt.figure(figsize=(9, 6))

    plt.scatter(
        X[x_feature],
        X[y_feature],
        c=clusters,
        s=10,
        alpha=0.5
    )

    x_index = X.columns.get_loc(x_feature)
    y_index = X.columns.get_loc(y_feature)

    plt.scatter(
        model.cluster_centers_[:, x_index],
        model.cluster_centers_[:, y_index],
        marker="X",
        s=200
    )

    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.title("K-Means Clusters")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIRECTORY, "kmeans_clusters.png"))
    plt.close()

    return "kmeans_clusters.png"
