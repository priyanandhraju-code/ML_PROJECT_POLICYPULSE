"""Compatibility helper for the original standalone clustering script."""
from unsupervised import calculate_kmeans_performance


def calculate_performance(X, clusters, kmeans_model):
    wcss, score = calculate_kmeans_performance(X, clusters, kmeans_model)
    print("WCSS:", wcss, "Silhouette:", score)
    return wcss, score
