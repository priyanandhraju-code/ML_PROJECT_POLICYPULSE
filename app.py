from flask import Flask, render_template, request

from load_data import load_data, get_data_summary
from policypulse_eda import run_eda
from preprocessing import preprocess_data
from linear_regression import run_linear_regression
from logistic_regression import run_logistic_regression
from tree_based import run_tree_algorithm, ALGORITHMS


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():
    return render_template(
        "index.html",
        active="none"
    )


# ============================================================
# DATA LOADING
# ============================================================

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


# ============================================================
# EDA
# ============================================================

@app.route("/eda")
def eda_page():
    error = None
    results = None

    try:
        results = run_eda()

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


# ============================================================
# PREPROCESSING
# ============================================================

@app.route("/preprocessing")
def preprocessing_page():
    error = None
    preprocessing_result = None
    preprocessing_info = None

    try:
        preprocessing_result, preprocessing_info = preprocess_data()

    except FileNotFoundError as e:
        error = str(e)

    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "preprocessing.html",
        active="preprocessing",
        preprocessing_result=preprocessing_result,
        preprocessing_info=preprocessing_info,
        error=error
    )


# ============================================================
# LINEAR REGRESSION
# ============================================================

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


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

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


# ============================================================
# TREE BASED MODELS
# ============================================================

@app.route("/tree-based", methods=["GET", "POST"])
def tree_based_page():
    error = None
    result = None

    selected_algorithm = request.form.get(
        "algorithm",
        "decision_tree"
    )

    try:
        if selected_algorithm not in ALGORITHMS:
            raise ValueError(
                f"Unknown tree algorithm: {selected_algorithm}"
            )

        if request.method == "POST":
            result = run_tree_algorithm(
                selected_algorithm
            )

    except FileNotFoundError as e:
        error = str(e)

    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "tree_based.html",
        active="tree-based",
        algorithms=ALGORITHMS,
        selected_algorithm=selected_algorithm,
        result=result,
        error=error
    )


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
