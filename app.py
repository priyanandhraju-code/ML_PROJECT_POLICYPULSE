"""PolicyPulse Flask application. Run with python app.py."""
from flask import Flask, render_template, request
from load_data import load_data, get_data_summary
from policypulse_eda import run_eda
from preprocessing import preprocess_data_summary, get_model_data
from linear_regression import run_linear_regression
from logistic_regression import run_logistic_regression
from tree_based import run_tree_algorithm
from unsupervised import get_clustering_data, run_clustering, SAMPLE_LIMITS
from evaluation import run_comparison
from project_runtime import ANALYSIS_LOCK

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.jinja_env.globals["model_feature_names"] = lambda: get_model_data()[0].columns.tolist()


@app.after_request
def fresh_plots(response):
    if request.path.startswith("/static/plots/"):
        response.headers["Cache-Control"] = "no-store"
    return response


def _error_message(error):
    if isinstance(error, (ValueError, FileNotFoundError)):
        return str(error)
    app.logger.exception("PolicyPulse analysis failed")
    return "The analysis could not complete. Please check the application log for details."


@app.route("/")
def index():
    error, rows, columns = None, 0, 0
    try:
        data = load_data()
        rows, columns = data.shape
    except Exception as exc:
        error = _error_message(exc)
    return render_template("index.html", active="none", dataset_rows=rows, dataset_columns=columns, error=error)


@app.route("/data-loading")
def data_loading():
    summary, error = None, None
    try:
        summary = get_data_summary(load_data())
    except Exception as exc:
        error = _error_message(exc)
    return render_template("index.html", active="data-loading", summary=summary, error=error)


@app.route("/eda")
def eda_page():
    results, error = None, None
    try:
        results = run_eda()
    except Exception as exc:
        error = _error_message(exc)
    return render_template("index.html", active="eda", results=results, error=error)


@app.route("/preprocessing")
def preprocessing_page():
    result, error, summary = None, None, None
    try:
        result = preprocess_data_summary()
        summary = result.attrs["summary"]
    except Exception as exc:
        error = _error_message(exc)
    return render_template("preprocessing.html", active="preprocessing", preprocessing_result=result, prep=summary, error=error)


def _regression_page(function, template, active, context_key):
    result, error = None, None
    try:
        result = function()
    except Exception as exc:
        error = _error_message(exc)
    return render_template(template, active=active, error=error, **{context_key: result})


@app.route("/linear-regression")
def linear_regression_page():
    return _regression_page(run_linear_regression, "linear_regression.html", "linear-regression", "linear_regression_result")


@app.route("/logistic-regression")
def logistic_regression_page():
    return _regression_page(run_logistic_regression, "logistic_regression.html", "logistic-regression", "logistic_regression_result")


@app.route("/tree-based", methods=["GET", "POST"])
def tree_based_page():
    result, error = None, None
    selected = request.form.get("algorithm", "decision_tree")
    if request.method == "POST":
        try:
            result = run_tree_algorithm(selected)
        except Exception as exc:
            error = _error_message(exc)
    return render_template("tree_based.html", active="tree-based", selected_algorithm=selected, result=result, error=error)


def _clustering_page(algorithm):
    result, error, features = None, None, []
    method, k, eps, min_samples = "silhouette", 3, 0.8, 10
    linkage_method = "ward"
    x_feature, y_feature = "Annual_Premium", "Vintage"
    try:
        # Avoid unnecessary clustering on a form-only visit.
        with ANALYSIS_LOCK:
            features = get_clustering_data(SAMPLE_LIMITS[algorithm]).columns.tolist()
        x_feature = request.form.get("x_feature", x_feature if x_feature in features else features[0])
        y_feature = request.form.get("y_feature", y_feature if y_feature in features else features[1])
        if request.method == "POST":
            method = request.form.get("method", "silhouette")
            k = int(request.form.get("k", "3"))
            linkage_method = request.form.get("linkage", "ward")
            eps = float(request.form.get("eps", "0.8"))
            min_samples = int(request.form.get("min_samples", "10"))
            result = run_clustering(algorithm, method, k, x_feature, y_feature, linkage_method, eps, min_samples)
            k = result["k"]
    except Exception as exc:
        error = _error_message(exc)
    template = "unsupervised.html" if algorithm == "kmeans" else "clustering.html"
    active = "unsupervised" if algorithm == "kmeans" else algorithm
    return render_template(template, active=active, algorithm=algorithm, result=result, error=error, features=features,
                           method=method, selected_k=k, x_feature=x_feature, y_feature=y_feature,
                           linkage_method=linkage_method, eps=eps, min_samples=min_samples)


@app.route("/unsupervised", methods=["GET", "POST"])
@app.route("/kmeans", methods=["GET", "POST"])
def unsupervised_page():
    return _clustering_page("kmeans")


@app.route("/hierarchical", methods=["GET", "POST"])
def hierarchical_page():
    return _clustering_page("hierarchical")


@app.route("/dbscan", methods=["GET", "POST"])
def dbscan_page():
    return _clustering_page("dbscan")


def _evaluation_page(active):
    result, error = None, None
    if request.method == "POST":
        try:
            result = run_comparison()
        except Exception as exc:
            error = _error_message(exc)
    return render_template("evaluation.html", active=active, result=result, error=error)


@app.route("/model-comparison", methods=["GET", "POST"])
def comparison_page():
    return _evaluation_page("comparison")


@app.route("/best-model", methods=["GET", "POST"])
def best_model_page():
    return _evaluation_page("best-model")


@app.route("/final-results", methods=["GET", "POST"])
def final_results_page():
    return _evaluation_page("final-results")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
