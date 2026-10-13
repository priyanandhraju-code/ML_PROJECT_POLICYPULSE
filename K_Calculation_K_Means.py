"""Interactive study entry point using the same K-selection logic as the app."""
from unsupervised import calculate_elbow, calculate_silhouette, _validate_k, get_clustering_data


def _read_choice(prompt, minimum, maximum):
    while True:
        try:
            choice = int(input(prompt))
            if minimum <= choice <= maximum:
                return choice
        except ValueError:
            pass
        print(f"Enter an integer from {minimum} to {maximum}.")


def manual_K():
    return _read_choice("Number of clusters (2–10): ", 2, 10)


def elbow_method(X):
    return calculate_elbow(X)["recommended_k"]


def silhouette_method(X):
    return calculate_silhouette(X)["recommended_k"]


def select_method():
    return _read_choice("1. Manual  2. Elbow  3. Silhouette — choose: ", 1, 3)


def main(X):
    choice = select_method()
    k = manual_K() if choice == 1 else elbow_method(X) if choice == 2 else silhouette_method(X)
    _validate_k(X, k)
    print("Selected K:", k)
    return k


if __name__ == "__main__":
    main(get_clustering_data())
