import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from data_loader import load_data

from K_Calculation_K_Means import (
    main as calculate_k
)

from k_means_performance import (
    calculate_performance
)


# ============================================================
# Perform K-Means
# ============================================================

def perform_kmeans(
    X,
    k
):

    print(
        "\n--- Applying K-Means ---"
    )

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X)

    return kmeans, clusters


# ============================================================
# Display Cluster Information
# ============================================================

def display_cluster_information(
    X,
    clusters,
    kmeans
):

    # ========================================================
    # Cluster Labels
    # ========================================================

    print(
        "\nCluster Labels:"
    )

    print(
        clusters
    )

    # ========================================================
    # Cluster Centers
    # ========================================================

    print(
        "\nCluster Centers:"
    )

    print(
        kmeans.cluster_centers_
    )

    # ========================================================
    # Cluster Distribution
    # ========================================================

    print(
        "\nCluster Distribution:"
    )

    cluster_counts = (
        pd.Series(clusters)
        .value_counts()
        .sort_index()
    )

    for cluster, count in cluster_counts.items():

        print(
            "Cluster",
            cluster,
            "=",
            count,
            "data points"
        )


# ============================================================
# Visualization
# ============================================================

def plot_clusters(
    X,
    clusters,
    kmeans
):

    print(
        "\nAvailable Features:"
    )

    for i, feature in enumerate(
        X.columns
    ):

        print(
            i,
            ":",
            feature
        )

    # ========================================================
    # Select X Feature
    # ========================================================

    x_index = int(
        input(
            "\nEnter feature number for X-axis: "
        )
    )

    # ========================================================
    # Select Y Feature
    # ========================================================

    y_index = int(
        input(
            "Enter feature number for Y-axis: "
        )
    )

    x_feature = X.columns[
        x_index
    ]

    y_feature = X.columns[
        y_index
    ]

    # ========================================================
    # Plot
    # ========================================================

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        X[x_feature],
        X[y_feature],
        c=clusters,
        s=20
    )

    # ========================================================
    # Plot Cluster Centers
    # ========================================================

    plt.scatter(
        kmeans.cluster_centers_[:, x_index],
        kmeans.cluster_centers_[:, y_index],
        marker="X",
        s=200
    )

    plt.xlabel(
        x_feature
    )

    plt.ylabel(
        y_feature
    )

    plt.title(
        "K-Means Clusters"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()


# ============================================================
# Main Function
# ============================================================

def main():

    print(
        "\n================================"
    )

    print(
        "       K-MEANS CLUSTERING"
    )

    print(
        "================================"
    )

    # ========================================================
    # 1. Load Dataset
    # ========================================================

    X = load_data()

    print(
        "\nDataset Loaded Successfully"
    )

    print(
        "Number of rows =",
        X.shape[0]
    )

    print(
        "Number of features =",
        X.shape[1]
    )

    # ========================================================
    # 2. Select K
    # ========================================================

    k = calculate_k(X)

    if k is None:

        print(
            "\nK-Means clustering stopped."
        )

        return

    # ========================================================
    # 3. Apply K-Means
    # ========================================================

    kmeans, clusters = perform_kmeans(
        X,
        k
    )

    # ========================================================
    # 4. Display Cluster Information
    # ========================================================

    display_cluster_information(
        X,
        clusters,
        kmeans
    )

    # ========================================================
    # 5. Performance Evaluation
    # ========================================================

    calculate_performance(
        X,
        clusters,
        kmeans
    )

    # ========================================================
    # 6. Visualization
    # ========================================================

    plot_clusters(
        X,
        clusters,
        kmeans
    )


# ============================================================
# Program Starts Here
# ============================================================

if __name__ == "__main__":

    main()