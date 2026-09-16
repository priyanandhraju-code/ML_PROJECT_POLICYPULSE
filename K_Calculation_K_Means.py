import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# Manual K Selection
# ============================================================

def manual_K():

    print("\n--- Manual K Selection ---")

    k = int(
        input(
            "Enter the number of clusters: "
        )
    )

    while k < 2:

        print(
            "K must be at least 2"
        )

        k = int(
            input(
                "Enter the number of clusters: "
            )
        )

    return k


# ============================================================
# Elbow Method
# ============================================================

def elbow_method(X):

    print("\n--- Elbow Method ---")

    wcss = []

    k_values = range(1, 11)

    for k in k_values:

        print(
            "Calculating WCSS for K =",
            k
        )

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        kmeans.fit(X)

        wcss.append(
            kmeans.inertia_
        )

    # ========================================================
    # Display WCSS Values
    # ========================================================

    print("\nWCSS Values")

    for k, value in zip(
        k_values,
        wcss
    ):

        print(
            "K =",
            k,
            "WCSS =",
            value
        )

    # ========================================================
    # Elbow Graph
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        k_values,
        wcss,
        marker="o"
    )

    plt.xlabel(
        "Number of Clusters (K)"
    )

    plt.ylabel(
        "WCSS"
    )

    plt.title(
        "Elbow Method"
    )

    plt.xticks(
        list(k_values)
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()

    # ========================================================
    # User Selects K
    # ========================================================

    k = int(
        input(
            "\nEnter K based on elbow graph: "
        )
    )

    while k < 2 or k > 10:

        print(
            "Please enter K between 2 and 10"
        )

        k = int(
            input(
                "\nEnter K based on elbow graph: "
            )
        )

    return k


# ============================================================
# Silhouette Method
# ============================================================

def silhouette_method(X):

    print(
        "\n--- Silhouette Score Method ---"
    )

    k_values = range(2, 11)

    scores = []

    # Limit calculation to 5000 samples
    sample_size = min(
        5000,
        len(X)
    )

    for k in k_values:

        print(
            "Calculating Silhouette Score for K =",
            k
        )

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        # Generate cluster labels
        labels = kmeans.fit_predict(X)

        # Calculate silhouette score
        score = silhouette_score(
            X,
            labels,
            sample_size=sample_size,
            random_state=42
        )

        scores.append(score)

        print(
            "K =",
            k,
            "Silhouette Score =",
            score
        )

    # ========================================================
    # Find Best K
    # ========================================================

    best_index = scores.index(
        max(scores)
    )

    best_k = list(
        k_values
    )[best_index]

    print(
        "\nHighest Silhouette Score =",
        max(scores)
    )

    print(
        "Recommended K =",
        best_k
    )

    # ========================================================
    # Silhouette Graph
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        k_values,
        scores,
        marker="o"
    )

    plt.xlabel(
        "Number of Clusters (K)"
    )

    plt.ylabel(
        "Silhouette Score"
    )

    plt.title(
        "Silhouette Score Method"
    )

    plt.xticks(
        list(k_values)
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()

    return best_k


# ============================================================
# Method Selection
# ============================================================

def select_method():

    print(
        "\n--- Select K Selection Method ---"
    )

    print(
        "1. Manual K Selection"
    )

    print(
        "2. Elbow Method"
    )

    print(
        "3. Silhouette Score"
    )

    choice = int(
        input(
            "Enter your choice: "
        )
    )

    while choice not in [1, 2, 3]:

        print(
            "Invalid choice"
        )

        choice = int(
            input(
                "Enter your choice: "
            )
        )

    return choice


# ============================================================
# Main Function
# ============================================================

def main(X):

    print(
        "\n================================"
    )

    print(
        "       K-MEANS K SELECTION"
    )

    print(
        "================================"
    )

    # Select method
    choice = select_method()

    # Determine K
    if choice == 1:

        k = manual_K()

    elif choice == 2:

        k = elbow_method(X)

    else:

        k = silhouette_method(X)

    print(
        "\nSelected K =",
        k
    )

    return k
if __name__ == "__main__":
    from data_loader import load_data

    X = load_data()
    main(X)