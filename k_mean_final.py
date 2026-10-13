"""Standalone K-Means demonstration, sharing the bounded web-app workflow."""
from K_Calculation_K_Means import main as calculate_k
from unsupervised import get_clustering_data, run_kmeans, get_cluster_distribution, create_cluster_plot
from k_means_performance import calculate_performance

perform_kmeans = run_kmeans


def display_cluster_information(X, clusters, kmeans):
    print("Cluster distribution:", get_cluster_distribution(clusters))
    print("Cluster centers:", kmeans.cluster_centers_)


def main():
    X = get_clustering_data()
    model, clusters = perform_kmeans(X, calculate_k(X))
    display_cluster_information(X, clusters, model)
    calculate_performance(X, clusters, model)
    return create_cluster_plot(X, clusters, model, "Annual_Premium", "Vintage")


if __name__ == "__main__":
    print("Saved plot:", main())
