from sklearn.metrics import silhouette_score


# ============================================================
# Calculate Performance
# ============================================================

def calculate_performance(
    X,
    clusters,
    kmeans_model
):

    # ========================================================
    # WCSS
    # ========================================================

    wcss = kmeans_model.inertia_

    # ========================================================
    # Silhouette Score
    # ========================================================

    sample_size = min(
        5000,
        len(X)
    )

    silhouette = silhouette_score(
        X,
        clusters,
        sample_size=sample_size,
        random_state=42
    )

    # ========================================================
    # Display Results
    # ========================================================

    print(
        "\n================================"
    )

    print(
        "     K-MEANS PERFORMANCE"
    )

    print(
        "================================"
    )

    print(
        "WCSS =",
        wcss
    )

    print(
        "Silhouette Score =",
        silhouette
    )

    return (
        wcss,
        silhouette
    )