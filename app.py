from flask import Flask, render_template

from load_data import (
    load_data,
    get_data_summary
)

from policypulse_eda import run_eda


# ============================================================
# CREATE FLASK APPLICATION
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

        # Step 1: Load CSV

        df = load_data()


        # Step 2: Generate summary

        summary = get_data_summary(
            df
        )


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
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )