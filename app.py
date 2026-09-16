from flask import Flask, render_template, request

from load_data import load_data, get_data_summary
from policypulse_eda import run_eda
from preprocessing import preprocess_data
from linear_regression import run_linear_regression
from logistic_regression import run_logistic_regression
from tree_based import (
    run_tree_algorithm,
    create_confusion_matrix_plot,
    create_roc_curve_plot
)
from unsupervised import (
    get_clustering_data,
    calculate_elbow,
    calculate_silhouette,
    run_kmeans,
    calculate_kmeans_performance,
    get_cluster_distribution,
    create_cluster_plot
)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", active="none")


@app.route("/data-loading")
def data_loading():
    error = None
    summary = None
    try:
        df = load_data()
        summary = get_data_summary(df)
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        error=error
    )


@app.route("/eda")
def eda_page():
    error = None
    results = None
    try:
        results = run_eda()
        print(results)
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="eda",
        results=results,
        error=error
    )


@app.route("/preprocessing")
def preprocessing_page():
    error = None
    preprocessing_result = None
    try:
        preprocessing_result = preprocess_data()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "preprocessing.html",
        active="preprocessing",
        preprocessing_result=preprocessing_result,
        error=error
    )


@app.route("/linear-regression")
def linear_regression_page():
    error = None
    result = None
    try:
        result = run_linear_regression()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "linear_regression.html",
        active="linear-regression",
        linear_regression_result=result,
        error=error
    )


@app.route("/logistic-regression")
def logistic_regression_page():
    error = None
    result = None
    try:
        result = run_logistic_regression()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "logistic_regression.html",
        active="logistic-regression",
        logistic_regression_result=result,
        error=error
    )


@app.route("/tree-based", methods=["GET", "POST"])
def tree_based_page():
    error = None
    result = None
    selected_algorithm = "decision_tree"

    if request.method == "POST":
        selected_algorithm = request.form.get(
            "algorithm",
            "decision_tree"
        )
        try:
            result = run_tree_algorithm(selected_algorithm)

            confusion_filename = (
                f"{selected_algorithm}_confusion_matrix.png"
            )
            create_confusion_matrix_plot(
                result,
                confusion_filename
            )

            roc_filename = (
                f"{selected_algorithm}_roc_curve.png"
            )
            create_roc_curve_plot(
                result,
                roc_filename
            )

            result["confusion_matrix_plot"] = confusion_filename
            result["roc_curve_plot"] = roc_filename

        except FileNotFoundError as e:
            error = str(e)
        except Exception as e:
            error = f"Unexpected error: {e}"

    return render_template(
        "tree_based.html",
        active="tree-based",
        selected_algorithm=selected_algorithm,
        result=result,
        error=error
    )


@app.route("/unsupervised", methods=["GET", "POST"])
def unsupervised_page():
    error = None
    result = None
    features = []

    method = "silhouette"
    selected_k = 3
    x_feature = "Annual_Premium"
    y_feature = "Vintage"

    try:
        X = get_clustering_data()
        features = X.columns.tolist()

        if x_feature not in features:
            x_feature = features[0]
        if y_feature not in features:
            y_feature = features[1] if len(features) > 1 else features[0]

        if request.method == "POST":
            method = request.form.get("method", "silhouette")
            x_feature = request.form.get("x_feature", x_feature)
            y_feature = request.form.get("y_feature", y_feature)

            if x_feature not in features:
                x_feature = features[0]
            if y_feature not in features:
                y_feature = features[1] if len(features) > 1 else features[0]

            elbow_plot = False
            silhouette_plot = False

            if method == "manual":
                selected_k = int(request.form.get("k", 3))
                if not 2 <= selected_k <= 10:
                    raise ValueError("K must be between 2 and 10.")

            elif method == "elbow":
                elbow = calculate_elbow(X)
                elbow_plot = True
                selected_k = int(request.form.get("k", 4))
                if not 2 <= selected_k <= 10:
                    raise ValueError("K must be between 2 and 10.")

            elif method == "silhouette":
                silhouette = calculate_silhouette(X)
                silhouette_plot = True
                selected_k = silhouette["recommended_k"]

            else:
                raise ValueError("Invalid K selection method.")

            model, clusters = run_kmeans(X, selected_k)
            wcss, silhouette_score_value = calculate_kmeans_performance(
                X, clusters, model
            )
            distribution = get_cluster_distribution(clusters)

            create_cluster_plot(
                X,
                clusters,
                model,
                x_feature,
                y_feature
            )

            result = {
                "rows": X.shape[0],
                "features": X.shape[1],
                "k": selected_k,
                "method": method.title(),
                "wcss": wcss,
                "silhouette": silhouette_score_value,
                "distribution": distribution,
                "elbow_plot": elbow_plot,
                "silhouette_plot": silhouette_plot
            }

            if method == "silhouette":
                result["recommended_k"] = silhouette["recommended_k"]
                result["highest_silhouette"] = silhouette["highest_score"]

    except ValueError as e:
        error = str(e)
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "unsupervised.html",
        active="unsupervised",
        error=error,
        result=result,
        features=features,
        method=method,
        selected_k=selected_k,
        x_feature=x_feature,
        y_feature=y_feature
    )


if __name__ == "__main__":
    app.run(debug=True)
