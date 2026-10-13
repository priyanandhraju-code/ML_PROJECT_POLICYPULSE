# PolicyPulse: Complete Project Guide

> Study guide plus full source code. Part 1 explains the project, Part 2 explains every file and the concepts in it, Part 3 has viva Q&A, and Part 4 contains the full code of every Python source file.

---

# PART 1: THE PROJECT

## 1.1 What is PolicyPulse?
PolicyPulse is a **machine learning web application** built on a real-world **insurance cross-sell dataset** (`Policypulse.csv`, about **381,109 customers**, 12 columns).

Every customer in the data **already holds a health insurance policy** with the company. The company now wants to offer them **vehicle insurance**. The dataset records whether each customer said **Yes (1)** or **No (0)**.

The app walks through the **complete ML lifecycle** on this data, one web page per stage:

`Data Loading → EDA → Preprocessing → Linear Regression → Logistic Regression → Tree-Based Models → Clustering (K-Means)`

## 1.2 Why was it built? (The problem)
- Calling every customer is **expensive and wasteful**. Most people say **No**: 334,399 of 381,109 (about 88%) are 0, and only 46,710 (about 12%) are 1.
- Sales teams need to know **who is most likely to say Yes**.
- **Goal:** build models that predict `Response` from customer features, and segment customers into groups. The company can then target the right people.

## 1.3 How is it useful? (Business value)
| Benefit | Explanation |
|---|---|
| Better targeting | Call only high-probability customers, which saves time and money. |
| Higher conversion | More policies sold per call. |
| Customer insight | EDA shows which factors matter (previous vehicle damage, already insured, age, vehicle age). |
| Segmentation | K-Means groups customers into segments, so each can get a tailored marketing strategy. |
| Model comparison | Seven or more algorithms are compared on the same data, so the best one can be chosen for deployment. |
| Interactive demo | The Flask UI lets non-technical people run each stage and view the plots. |

## 1.4 Dataset columns
| Column | Meaning | Type |
|---|---|---|
| `id` | Unique row id (dropped, no information) | Identifier |
| `Gender` | Male / Female | Categorical |
| `Age` | Customer age | Numeric |
| `Driving_License` | 1 = has a licence | Binary |
| `Region_Code` | Code of the customer's region | Categorical (a code, not a quantity) |
| `Previously_Insured` | 1 = already has vehicle insurance | Binary |
| `Vehicle_Age` | `< 1 Year`, `1-2 Year`, `> 2 Years` | Categorical |
| `Vehicle_Damage` | Whether the vehicle was damaged before (Yes/No) | Categorical |
| `Annual_Premium` | Yearly premium paid | Numeric (has outliers) |
| `Policy_Sales_Channel` | Code of the outreach channel (agent, mail, phone) | Categorical (a code) |
| `Vintage` | Days the customer has been with the company | Numeric |
| `Response` | **TARGET**: 1 = interested, 0 = not interested | Binary |

## 1.5 Technology stack
| Library | Role |
|---|---|
| **Python 3** | Language |
| **pandas** | Load, clean and transform tabular data (DataFrames) |
| **numpy** | Numerical arrays and maths |
| **matplotlib** | Plotting and saving charts as PNG (uses the non-GUI `Agg` backend for the server) |
| **seaborn** | Statistical plots (heatmap, pairplot, countplot) |
| **scikit-learn** | Models, scalers, train/test split, metrics, K-Means, silhouette |
| **XGBoost** | Optimized gradient-boosting classifier |
| **LightGBM** | Fast, leaf-wise gradient-boosting classifier |
| **Flask** | Web framework (routes plus Jinja2 HTML templates) |
| **HTML / CSS / Jinja2** | Front-end pages in `templates/` and `static/style.css` |

---

# PART 2: PROJECT STRUCTURE AND EVERY FILE

## 2.1 Folder layout
```
ML_PROJECT_POLICYPULSE/
├── Policypulse.csv              ← raw dataset
├── app.py                       ← Flask web app (entry point)
├── load_data.py                 ← Stage 1: loading + summary
├── data_loader.py               ← helper for clustering (loads preprocessed data)
├── policypulse_eda.py           ← Stage 2: exploratory data analysis
├── preprocessing.py             ← Stage 3: orchestrates the cleaning pipeline
├── Outlier_Fix.py               ← IQR outlier clipping
├── Standard_Scaler.py           ← standardization
├── Min_Max_Scaling.py           ← standalone min-max scaling script
├── linear_regression.py         ← Stage 4: Linear / Ridge / Lasso
├── logistic_regression.py       ← Stage 5: Logistic / Ridge / Lasso
├── tree_based.py                ← Stage 6: 7 tree-based classifiers
├── unsupervised.py              ← Stage 7: K-Means used by the web app
├── K_Calculation_K_Means.py     ← standalone: choose K (manual/elbow/silhouette)
├── k_mean_final.py              ← standalone: final K-Means run + plots
├── k_means_performance.py       ← WCSS + silhouette calculation
├── templates/                   ← HTML pages (index, preprocessing, linear, logistic, tree, unsupervised)
├── static/style.css             ← styling
└── static/plots/                ← generated PNG charts
```
`preprocessed_data.csv` is **generated** by the preprocessing stage. It is not in git.

## 2.2 Data flow (how the files connect)
```
Policypulse.csv
   │  load_data.py  ─────────────► summary page
   │  policypulse_eda.py ────────► plots (static/plots)
   ▼
preprocessing.py  (uses Outlier_Fix.py + Standard_Scaler.py)
   ▼
preprocessed_data.csv
   ├──► linear_regression.py
   ├──► logistic_regression.py
   ├──► tree_based.py
   └──► data_loader.py ► unsupervised.py / K-Means scripts
```
`app.py` imports and calls all of these, one route per stage.

## 2.3 File-by-file explanation

### `app.py`: the Flask web application
- **Use:** Entry point. Defines URL routes. Each route calls the matching module, then renders an HTML template with the results.
- **Routes:** `/` (home), `/data-loading`, `/eda`, `/preprocessing`, `/linear-regression`, `/logistic-regression`, `/tree-based` (GET and POST, choose an algorithm), `/unsupervised` (GET and POST, choose K method and features).
- **Concepts:**
  - **Flask routing** (`@app.route`) and `render_template` (Jinja2).
  - **GET vs POST**: POST is used when the user submits a form (choose algorithm or K).
  - **Error handling** with `try/except`, so a missing file shows a friendly message instead of crashing.
  - **Separation of concerns**: the app only handles web logic, while the ML lives in other modules.

### `load_data.py`: Stage 1, data loading
- **Use:** Reads `Policypulse.csv` and builds a summary dict (shape, column types, missing values, sample rows).
- **Concepts:** `pd.read_csv`, `df.shape`, `df.dtypes`, `df.isnull().sum()`, `df.describe()`, custom exceptions (`FileNotFoundError`).

### `policypulse_eda.py`: Stage 2, Exploratory Data Analysis
- **Use:** Understands the data visually and statistically before modelling. The largest file, at about 1,550 lines. Saves many PNGs to `static/plots/`.
- **Concepts:**
  - **Univariate analysis**: histograms of Age and Annual_Premium, and the distribution of Response (which shows **class imbalance**).
  - **Bivariate analysis**: each feature against Response (gender, age group, vehicle age, vehicle damage, previously insured, region, sales channel).
  - **Box plots** for outlier detection (5 numeric columns).
  - **Correlation heatmap** (Pearson correlation between numeric features).
  - **Pairplot** (all feature pairs).
  - `plt.savefig`, and `matplotlib.use("Agg")` so plotting works without a GUI.

### `preprocessing.py`: Stage 3, the cleaning pipeline
- **Use:** Runs the whole data preparation and writes `preprocessed_data.csv`.
- **Steps in order:**
  1. Load the dataset.
  2. **Remove `id`**: it is only an identifier.
  3. **Handle missing values**: median for numeric columns, mode for categorical ones.
  4. **Encode categoricals** with `pd.get_dummies(drop_first=True)` (**one-hot encoding**). `drop_first` avoids the **dummy variable trap** (multicollinearity). Region_Code and Policy_Sales_Channel are treated as categories, not numbers.
  5. **Train/test split** (80/20, `random_state=42`).
  6. **Outlier fixing** on Annual_Premium (calls `Outlier_Fix.py`).
  7. **Standard scaling** (calls `Standard_Scaler.py`).
  8. Recombine and save.
- **Concepts:** data leakage prevention, reproducibility (fixed random seed), encoding, imputation.

### `Outlier_Fix.py`: IQR outlier handling
- **Use:** Function `fix_outliers(train_df, test_df, feature="Annual_Premium")`.
- **Concepts:**
  - **IQR method**: `Q1 = 25th percentile`, `Q3 = 75th percentile`, `IQR = Q3 − Q1`. Lower fence `= Q1 − 1.5·IQR`, upper fence `= Q3 + 1.5·IQR`.
  - **Clipping (winsorizing)**: values beyond the fences are set **to** the fence values, not removed. No rows are lost.
  - **No data leakage**: fences are computed on **train only** and applied to test.

### `Standard_Scaler.py`: standardization
- **Use:** Function `standard_scale(train_df, test_df)`.
- **Concepts:** **Z-score standardization**: `z = (x − mean) / std`, giving mean 0 and std 1. `fit` on train, `transform` on both train and test. It is needed because linear/logistic models and K-Means are sensitive to feature scale, so Annual_Premium (values of tens of thousands) would otherwise dominate Age.

### `Min_Max_Scaling.py`: min-max scaling (standalone)
- **Use:** Demonstrates the alternative scaler, `MinMaxScaler`, which maps values to **0–1**: `x' = (x − min) / (max − min)`.
- **Concepts:** stratified `train_test_split(stratify=y)` (keeps the same Yes/No ratio in train and test), scaling the numeric columns only, and fitting on train only. It is sensitive to outliers, which is why standard scaling is the one used in the final pipeline.

### `linear_regression.py`: Stage 4
- **Use:** Trains **Linear**, **Ridge** and **Lasso** regression on the preprocessed data. It predicts `Response` as a continuous score, then evaluates and plots.
- **Concepts:**
  - **Linear regression**: `y = b0 + b1x1 + … + bnxn`, fitted by least squares.
  - **Ridge (L2)**: adds a penalty `α·Σb²`, which shrinks coefficients.
  - **Lasso (L1)**: adds a penalty `α·Σ|b|`, which can shrink some coefficients to exactly 0 (**feature selection**).
  - **Regularization** reduces **overfitting**.
  - **Metrics:** MAE, MSE, RMSE, R².
  - Predicted-vs-actual plots and a model comparison plot.
  - Note: `Response` is binary, so this is a baseline and not the ideal model. Logistic regression fits binary targets better.

### `logistic_regression.py`: Stage 5
- **Use:** Trains three **classifiers**: plain Logistic, Ridge (L2) and Lasso (L1) Logistic Regression. It saves confusion matrices, ROC curves and a comparison plot.
- **Concepts:**
  - **Logistic regression**: the sigmoid function turns a linear score into a probability from 0 to 1.
  - **Regularization** via `penalty` and `C` (C is the inverse of strength, so small C means stronger regularization). Ridge uses C=1.0 and Lasso uses C=0.1.
  - **Solvers**: `lbfgs` (L2 or none) and `liblinear` (supports L1).
  - **`class_weight="balanced"`**: handles **class imbalance** by weighting the rare "Yes" class more.
  - **Metrics:** accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
  - **ROC curve**: true positive rate against false positive rate at every threshold. **AUC** is the area under it, where 0.5 is random and 1.0 is perfect.
  - `max_iter=2000` so the optimizer converges.

### `tree_based.py`: Stage 6
- **Use:** Trains **seven tree-based classifiers** and produces confusion matrix and ROC plots for each. `run_tree_algorithm(name)` is called by the web app.
- **Models and concepts:**
  | Model | Idea |
  |---|---|
  | **Decision Tree** | A single tree of if/else splits (`max_depth=12`, `min_samples_leaf=10` to limit overfitting). |
  | **Random Forest** | Many trees on random samples and features; the average vote is the result (**bagging**). |
  | **Extra Trees** | Like Random Forest, but split thresholds are also random (faster, lower variance). |
  | **Gradient Boosting** | Trees added sequentially; each fits the **residual errors** of the previous ones. |
  | **AdaBoost** | Boosting that re-weights misclassified samples so the next weak learner focuses on them. |
  | **XGBoost** | Optimized gradient boosting with regularization and fast training. |
  | **LightGBM** | Gradient boosting with leaf-wise tree growth and histogram binning (very fast on large data). |
- **Other concepts:** hyperparameters (`n_estimators`, `max_depth`, `learning_rate`), overfitting control, the bias–variance trade-off, feature-based non-linear modelling with no scaling required.

### `unsupervised.py`: Stage 7, K-Means for the web app
- **Use:** Functions used by the `/unsupervised` route:
  `get_clustering_data`, `calculate_elbow`, `calculate_silhouette`, `run_kmeans`, `calculate_kmeans_performance`, `get_cluster_distribution`, `create_cluster_plot`.
- **Concepts:**
  - **Unsupervised learning**: no labels are used (Response is dropped).
  - **K-Means**: pick K centroids, assign each point to the nearest centroid, recompute the centroids, and repeat until stable.
  - **Elbow method**: plot **WCSS** (within-cluster sum of squares, also called inertia) for K=1..10 and pick the "elbow" where improvement flattens.
  - **Silhouette score**: ranges from −1 to 1. Higher means points are close to their own cluster and far from the others. Used for K=2..10.
  - **Cluster visualization** on two chosen features, with centroids drawn.

### `data_loader.py`
- **Use:** Loads `preprocessed_data.csv` and **drops the Response column**, returning only the features `X`. This is what clustering needs, since it is unsupervised.

### `K_Calculation_K_Means.py`: choosing K (standalone CLI)
- **Use:** Interactive script with a menu: **manual K**, **elbow method** or **silhouette method**.
- **Concepts:** the same elbow and silhouette ideas as above, with user input and printed tables and plots.

### `k_mean_final.py`: final K-Means (standalone)
- **Use:** Fits K-Means with the chosen K, prints cluster information (sizes, centroids) and plots the clusters.
- **Concepts:** `KMeans(n_clusters=k, random_state=42, n_init=...)`, `fit_predict`, centroids, cluster profiling.

### `k_means_performance.py`
- **Use:** `calculate_performance(X, clusters, model)` returns **WCSS** (`model.inertia_`) and the **silhouette score**, computed on a **5,000-row sample** (`sample_size=5000`) because silhouette is O(n²) and would be too slow on 381k rows.

### `templates/*.html` and `static/style.css`
- **Use:** Front-end. `index.html` is the base layout and home page, and the other five templates (preprocessing, linear_regression, logistic_regression, tree_based, unsupervised) display each stage's results. `style.css` (about 1,160 lines) styles the pages.
- **Concepts:** **Jinja2 templating** (`{{ variable }}`, `{% for %}`, `{% if %}`), forms (POST), and static file serving (`static/plots/*.png`).

---

# PART 3: KEY CONCEPTS

## 3.1 Machine learning types
- **Supervised**: learns from labelled data (Response). Linear, logistic and tree models belong here.
  - **Regression**: predicts a number.
  - **Classification**: predicts a class.
- **Unsupervised**: finds structure without labels (K-Means).

## 3.2 Preprocessing terms
| Term | Meaning |
|---|---|
| **Data leakage** | Test-set information leaking into training (for example fitting a scaler on all data). We fit on train only. |
| **Imputation** | Filling missing values (median or mode). |
| **One-hot encoding** | Turning categories into 0/1 columns. |
| **Outlier** | An extreme value. Handled by IQR clipping. |
| **Scaling** | Putting features on a comparable scale (standard or min-max). |
| **Stratified split** | Preserves the class ratio in the train and test sets. |
| **random_state=42** | A fixed seed so results repeat exactly. |

## 3.3 Evaluation metrics
| Metric | Meaning |
|---|---|
| **Accuracy** | Fraction correct. Misleading when the classes are imbalanced. |
| **Precision** | Of the predicted Yes, how many were really Yes. |
| **Recall** | Of the real Yes, how many did we find. |
| **F1** | The harmonic mean of precision and recall. |
| **Confusion matrix** | TP, FP, TN and FN counts. |
| **ROC-AUC** | Separation ability across all thresholds. |
| **MAE, MSE, RMSE, R²** | Regression error and fit metrics. |
| **WCSS and silhouette** | Clustering quality metrics. |

## 3.4 Overfitting and regularization
- **Overfitting**: the model memorises the training data and performs badly on new data.
- **Fixes used:** Ridge/Lasso penalties, `max_depth`, `min_samples_leaf`, train/test evaluation, and ensembles.

---

# PART 4: LIKELY VIVA QUESTIONS AND ANSWERS

1. **What is your project about?** Predicting which health-insurance customers will buy vehicle insurance, and segmenting customers, using ML, with a Flask web UI.
2. **Which is the target variable?** `Response` (0 or 1).
3. **Supervised or unsupervised?** Both. Regression, logistic and trees are supervised, and K-Means is unsupervised.
4. **Why drop `id`?** It is a unique identifier and has no predictive value.
5. **Why treat Region_Code as categorical?** It is a label. Region 28 is not "more" than region 3.
6. **Why one-hot encode?** Models need numbers, and one-hot avoids implying a false order.
7. **Why `drop_first=True`?** It avoids the dummy variable trap (perfect multicollinearity).
8. **Why scale?** So large-valued features do not dominate distance-based and gradient-based models.
9. **Why fit the scaler on train only?** To avoid data leakage.
10. **What is IQR and why clip instead of delete?** IQR is `Q3 − Q1`. Clipping keeps all rows and limits the influence of extremes.
11. **Ridge vs Lasso?** Ridge (L2) shrinks the coefficients. Lasso (L1) can zero them out, which acts as feature selection.
12. **Why logistic regression for classification?** The sigmoid gives a probability between 0 and 1, which suits a binary target.
13. **What is `class_weight="balanced"`?** It penalizes mistakes on the minority class more, to handle imbalance.
14. **Why not just use accuracy?** With about 88% "No", a model that always says "No" scores 88% but is useless. Use recall, F1 and AUC.
15. **What does AUC mean?** The probability that the model ranks a random Yes above a random No. 0.5 is random and 1.0 is perfect.
16. **Bagging vs boosting?** Bagging (Random Forest) trains trees in parallel and averages them, which reduces variance. Boosting trains trees sequentially, each fixing the previous errors, which reduces bias.
17. **Random Forest vs Extra Trees?** Extra Trees also randomizes the split thresholds.
18. **XGBoost vs LightGBM?** XGBoost grows trees level-wise and is regularized. LightGBM grows leaf-wise with histograms, so it is faster on big data.
19. **How does K-Means work?** Initialize centroids, assign each point to the nearest centroid, recompute the centroids, and repeat until stable.
20. **How did you choose K?** Elbow (WCSS) and silhouette score.
21. **Why is silhouette computed on a sample?** It is O(n²), and 381k rows would be too slow.
22. **Does K-Means need scaling?** Yes, because it uses Euclidean distance.
23. **Do tree models need scaling?** No, because splits depend only on the ordering of values.
24. **Why is linear regression weak here?** The target is binary. It can give values outside 0–1 and violates the model's assumptions, so it serves as a baseline.
25. **What is the role of Flask?** It serves the pages, runs the ML code per route and renders the results with Jinja2.

---

# PART 5: FULL SOURCE CODE

> The HTML templates and `style.css` (about 6,700 lines of front-end) are described above but not reproduced here. Every Python source file is included in full below.


## 5.1 `app.py`

```python
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

```

## 5.2 `load_data.py`

```python
import os
import pandas as pd


# ============================================================
# DATASET PATH
# ============================================================

DATA_PATH = r"Policypulse.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_data(path: str = DATA_PATH) -> pd.DataFrame:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    if os.path.getsize(path) == 0:
        raise ValueError(
            "The CSV file is empty. Please check the dataset file."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            "The CSV file contains no data."
        )

    return df


# ============================================================
# DATASET SUMMARY
# ============================================================

def get_data_summary(df: pd.DataFrame) -> dict:

    summary = {

        "n_rows": df.shape[0],

        "n_cols": df.shape[1],

        "columns": list(df.columns),

        "dtypes": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },

        "missing_counts": {
            col: int(df[col].isnull().sum())
            for col in df.columns
        },

        "preview": df.head(10).to_dict(
            orient="records"
        )
    }

    return summary
```

## 5.3 `data_loader.py`

```python
import pandas as pd


# ============================================================
# Load Data
# ============================================================

def load_data():

    data = pd.read_csv(
        r"preprocessed_data.csv"
    )

    # Remove target column
    X = data.drop(
        "Response",
        axis=1
    )

    return X


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    X = load_data()

    print("\nDataset Loaded Successfully")

    print(
        "Number of rows =",
        X.shape[0]
    )

    print(
        "Number of features =",
        X.shape[1]
    )

    print("\nAvailable Columns:")

    print(
        X.columns.tolist()
    )
```

## 5.4 `policypulse_eda.py`

```python
import os

# ============================================================
# MATPLOTLIB BACKEND
# ============================================================

# Important when running Matplotlib through Flask on macOS.
# It prevents GUI/window errors.

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ============================================================
# DATASET PATH
# ============================================================

CSV_PATH = r"Policypulse.csv"


# ============================================================
# PLOT DIRECTORY
# ============================================================

PLOT_DIR = os.path.join(
    "static",
    "plots"
)

os.makedirs(
    PLOT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

sns.set(
    style="whitegrid"
)

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.max_rows",
    200
)


# ============================================================
# SAVE PLOT FUNCTION
# ============================================================

def show(
    title="",
    filename="plot.png"
):

    if title:
        plt.title(title)

    plt.tight_layout()

    filepath = os.path.join(
        PLOT_DIR,
        filename
    )

    plt.savefig(
        filepath,
        dpi=120,
        bbox_inches="tight"
    )

    plt.close("all")

    return filename


# ============================================================
# RUN EDA
# ============================================================

def run_eda():

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    if not os.path.exists(CSV_PATH):

        raise FileNotFoundError(
            f"CSV file not found: {CSV_PATH}"
        )

    data = pd.read_csv(
        CSV_PATH
    )


    # ========================================================
    # BASIC CLEANING
    # ========================================================

    # Remove unnecessary unnamed column if present

    if "Unnamed: 0" in data.columns:

        data = data.drop(
            columns=["Unnamed: 0"]
        )


    # ========================================================
    # 1. DATA LOADED
    # ========================================================

    print("=" * 80)

    print("1. DATA LOADED")

    print("=" * 80)

    print(
        "Shape of data:",
        data.shape
    )

    print(
        "\nFirst 5 rows:"
    )

    print(
        data.head()
    )


    # List to store generated plots

    plot_files = []


    # ========================================================
    # 2. BASIC INFO
    # ========================================================

    print("\n" + "=" * 80)

    print("2. BASIC INFO")

    print("=" * 80)


    print(
        "\nColumn Data Types:"
    )

    print(
        data.dtypes
    )


    print(
        "\nNumeric Description:"
    )

    print(
        data.describe()
    )


    print(
        "\nCategorical Description:"
    )

    print(
        data.describe(
            include="object"
        )
    )


    # ========================================================
    # 3. MISSING VALUES
    # ========================================================

    print("\n" + "=" * 80)

    print("3. MISSING VALUES")

    print("=" * 80)


    missing = data.isnull().sum()


    missing_pct = (
        missing / len(data)
    ) * 100


    missing_df = pd.DataFrame({

        "missing_count":
            missing,

        "missing_pct":
            missing_pct

    })


    missing_df = missing_df[
        missing_df["missing_count"] > 0
    ].sort_values(
        by="missing_pct",
        ascending=False
    )


    if missing_df.empty:

        print(
            "No missing values found."
        )

    else:

        print(
            missing_df
        )


        plt.figure(
            figsize=(10, 5)
        )


        sns.barplot(
            x=missing_df.index,
            y=missing_df[
                "missing_count"
            ]
        )


        plt.xticks(
            rotation=45,
            ha="right"
        )


        plt.xlabel(
            "Columns"
        )


        plt.ylabel(
            "Missing Count"
        )


        filename = show(
            title="Missing Values by Column",
            filename="missing_values.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 4. DUPLICATE ROWS
    # ========================================================

    print("\n" + "=" * 80)

    print("4. DUPLICATE ROWS")

    print("=" * 80)


    duplicate_count = int(
        data.duplicated().sum()
    )


    print(
        "Duplicate rows:",
        duplicate_count
    )


    # ========================================================
    # 5. TARGET VARIABLE - RESPONSE
    # ========================================================

    print("\n" + "=" * 80)

    print("5. TARGET VARIABLE - RESPONSE")

    print("=" * 80)


    if "Response" in data.columns:

        response_counts = (
            data["Response"]
            .value_counts()
            .sort_index()
        )


        print(
            "\nResponse Counts:"
        )

        print(
            response_counts
        )


        response_percentage = (
            data["Response"]
            .value_counts(
                normalize=True
            )
            .sort_index()
            * 100
        )


        print(
            "\nResponse Percentage:"
        )

        print(
            response_percentage
        )


        # Response count plot

        plt.figure(
            figsize=(7, 5)
        )


        sns.countplot(
            data=data,
            x="Response"
        )


        plt.xlabel(
            "Response (0 = No, 1 = Yes)"
        )


        plt.ylabel(
            "Number of Customers"
        )


        filename = show(
            title="Customer Response Distribution",
            filename="response_distribution.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 6. AGE DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 80)

    print("6. AGE DISTRIBUTION")

    print("=" * 80)


    if "Age" in data.columns:

        print(
            data["Age"].describe()
        )


        plt.figure(
            figsize=(9, 5)
        )


        sns.histplot(
            data["Age"],
            bins=30,
            kde=True
        )


        plt.xlabel(
            "Age"
        )


        plt.ylabel(
            "Number of Customers"
        )


        filename = show(
            title="Age Distribution",
            filename="age_distribution.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 7. ANNUAL PREMIUM DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 80)

    print("7. ANNUAL PREMIUM DISTRIBUTION")

    print("=" * 80)


    if "Annual_Premium" in data.columns:

        print(
            data["Annual_Premium"].describe()
        )


        plt.figure(
            figsize=(9, 5)
        )


        sns.histplot(
            data["Annual_Premium"],
            bins=40,
            kde=True
        )


        plt.xlabel(
            "Annual Premium"
        )


        plt.ylabel(
            "Number of Customers"
        )


        filename = show(
            title="Annual Premium Distribution",
            filename="annual_premium_distribution.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 8. OUTLIER DETECTION
    # ========================================================

    print("\n" + "=" * 80)

    print("8. OUTLIER DETECTION")

    print("=" * 80)


    # Only continuous numerical variables are appropriate for
    # numerical outlier boxplots. Region_Code and
    # Policy_Sales_Channel are categorical codes, not continuous
    # measurements, so they are intentionally excluded.
    box_cols = [

        "Age",

        "Annual_Premium",

        "Vintage"

    ]


    box_cols = [

        col

        for col in box_cols

        if col in data.columns

    ]


    for index, col in enumerate(
        box_cols
    ):

        plt.figure(
            figsize=(10, 4)
        )


        sns.boxplot(
            x=data[col]
        )


        plt.xlabel(
            col
        )


        filename = show(
            title=f"Box Plot - {col}",
            filename=(
                f"boxplot_{index + 1}_{col}.png"
            )
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 9. CORRELATION ANALYSIS
    # ========================================================

    print("\n" + "=" * 80)

    print("9. CORRELATION ANALYSIS")

    print("=" * 80)


    # Correlation is calculated only for continuous numerical
    # variables. Region_Code and Policy_Sales_Channel are
    # categorical codes and therefore should not be interpreted
    # as continuous quantities. The ID column is also excluded.
    correlation_columns = [
        "Age",
        "Annual_Premium",
        "Vintage",
        "Driving_License",
        "Previously_Insured",
        "Response"
    ]

    correlation_columns = [
        col
        for col in correlation_columns
        if col in data.columns
    ]

    numeric_data = data[correlation_columns]

    corr = numeric_data.corr()


    print(
        "\nCorrelation Matrix:"
    )


    print(
        np.round(
            corr,
            decimals=2
        )
    )


    plt.figure(
        figsize=(12, 9)
    )


    sns.heatmap(
        np.round(
            corr,
            decimals=2
        ),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )


    filename = show(
        title="Correlation Heatmap",
        filename="correlation_heatmap.png"
    )


    plot_files.append(
        filename
    )


    # ========================================================
    # 10. GENDER VS RESPONSE
    # ========================================================

    print("\n" + "=" * 80)

    print("10. GENDER VS RESPONSE")

    print("=" * 80)


    if (
        "Gender" in data.columns
        and
        "Response" in data.columns
    ):


        gender_response = pd.crosstab(
            data["Gender"],
            data["Response"]
        )


        print(
            gender_response
        )


        plt.figure(
            figsize=(7, 5)
        )


        sns.countplot(
            data=data,
            x="Gender",
            hue="Response"
        )


        plt.xlabel(
            "Gender"
        )


        plt.ylabel(
            "Number of Customers"
        )


        filename = show(
            title="Gender vs Response",
            filename="gender_vs_response.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 11. VEHICLE DAMAGE VS RESPONSE
    # ========================================================

    print("\n" + "=" * 80)

    print("11. VEHICLE DAMAGE VS RESPONSE")

    print("=" * 80)


    if (
        "Vehicle_Damage" in data.columns
        and
        "Response" in data.columns
    ):


        damage_response = pd.crosstab(
            data["Vehicle_Damage"],
            data["Response"]
        )


        print(
            damage_response
        )


        plt.figure(
            figsize=(8, 5)
        )


        sns.countplot(
            data=data,
            x="Vehicle_Damage",
            hue="Response"
        )


        plt.xlabel(
            "Vehicle Damage"
        )


        plt.ylabel(
            "Number of Customers"
        )


        filename = show(
            title="Vehicle Damage vs Response",
            filename="vehicle_damage_vs_response.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 12. PREVIOUSLY INSURED VS RESPONSE
    # ========================================================

    print("\n" + "=" * 80)

    print("12. PREVIOUSLY INSURED VS RESPONSE")

    print("=" * 80)


    if (
        "Previously_Insured" in data.columns
        and
        "Response" in data.columns
    ):


        insured_response = pd.crosstab(
            data["Previously_Insured"],
            data["Response"]
        )


        print(
            insured_response
        )


        plt.figure(
            figsize=(8, 5)
        )


        sns.countplot(
            data=data,
            x="Previously_Insured",
            hue="Response"
        )


        plt.xlabel(
            "Previously Insured (0 = No, 1 = Yes)"
        )


        plt.ylabel(
            "Number of Customers"
        )


        filename = show(
            title="Previously Insured vs Response",
            filename="previously_insured_vs_response.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 13. VEHICLE AGE VS RESPONSE
    # ========================================================

    print("\n" + "=" * 80)

    print("13. VEHICLE AGE VS RESPONSE")

    print("=" * 80)


    if (
        "Vehicle_Age" in data.columns
        and
        "Response" in data.columns
    ):


        vehicle_age_response = pd.crosstab(
            data["Vehicle_Age"],
            data["Response"]
        )


        print(
            vehicle_age_response
        )


        plt.figure(
            figsize=(9, 5)
        )


        sns.countplot(
            data=data,
            x="Vehicle_Age",
            hue="Response"
        )


        plt.xlabel(
            "Vehicle Age"
        )


        plt.ylabel(
            "Number of Customers"
        )


        plt.xticks(
            rotation=20
        )


        filename = show(
            title="Vehicle Age vs Response",
            filename="vehicle_age_vs_response.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 14. AGE VS RESPONSE
    # ========================================================

    print("\n" + "=" * 80)

    print("14. AGE VS RESPONSE")

    print("=" * 80)


    if (
        "Age" in data.columns
        and
        "Response" in data.columns
    ):


        plt.figure(
            figsize=(8, 5)
        )


        sns.boxplot(
            data=data,
            x="Response",
            y="Age"
        )


        plt.xlabel(
            "Response"
        )


        plt.ylabel(
            "Age"
        )


        filename = show(
            title="Age vs Response",
            filename="age_vs_response.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 15. ANNUAL PREMIUM VS RESPONSE
    # ========================================================

    print("\n" + "=" * 80)

    print("15. ANNUAL PREMIUM VS RESPONSE")

    print("=" * 80)


    if (
        "Annual_Premium" in data.columns
        and
        "Response" in data.columns
    ):


        # Sample only for visualization

        premium_plot = data.sample(
            min(
                5000,
                len(data)
            ),
            random_state=42
        )


        plt.figure(
            figsize=(8, 5)
        )


        sns.boxplot(
            data=premium_plot,
            x="Response",
            y="Annual_Premium"
        )


        plt.xlabel(
            "Response"
        )


        plt.ylabel(
            "Annual Premium"
        )


        filename = show(
            title="Annual Premium vs Response",
            filename="annual_premium_vs_response.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 16. REGION CODE VS RESPONSE
    # ========================================================

    print("\n" + "=" * 80)

    print("16. REGION CODE VS RESPONSE")

    print("=" * 80)


    if (
        "Region_Code" in data.columns
        and
        "Response" in data.columns
    ):


        region_response = (

            data.groupby(
                "Region_Code"
            )["Response"]

            .mean()

            .sort_values(
                ascending=False
            )

            .head(15)

        )


        print(
            "\nTop Regions by Response Rate:"
        )


        print(
            region_response
        )


        plt.figure(
            figsize=(10, 6)
        )


        sns.barplot(
            x=region_response.index.astype(str),
            y=region_response.values
        )


        plt.xlabel(
            "Region Code"
        )


        plt.ylabel(
            "Response Rate"
        )


        plt.xticks(
            rotation=45
        )


        filename = show(
            title="Top Regions by Response Rate",
            filename="region_response.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 17. POLICY SALES CHANNEL VS RESPONSE
    # ========================================================

    print("\n" + "=" * 80)

    print("17. POLICY SALES CHANNEL VS RESPONSE")

    print("=" * 80)


    if (
        "Policy_Sales_Channel" in data.columns
        and
        "Response" in data.columns
    ):


        channel_response = (

            data.groupby(
                "Policy_Sales_Channel"
            )["Response"]

            .mean()

            .sort_values(
                ascending=False
            )

            .head(15)

        )


        print(
            "\nTop Policy Sales Channels by Response Rate:"
        )


        print(
            channel_response
        )


        plt.figure(
            figsize=(10, 6)
        )


        sns.barplot(
            x=channel_response.index.astype(str),
            y=channel_response.values
        )


        plt.xlabel(
            "Policy Sales Channel"
        )


        plt.ylabel(
            "Response Rate"
        )


        plt.xticks(
            rotation=45
        )


        filename = show(
            title="Top Policy Sales Channels by Response Rate",
            filename="sales_channel_response.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 18. AGE VS ANNUAL PREMIUM
    # ========================================================

    print("\n" + "=" * 80)

    print("18. AGE VS ANNUAL PREMIUM")

    print("=" * 80)


    if (
        "Age" in data.columns
        and
        "Annual_Premium" in data.columns
    ):


        scatter_data = data.sample(
            min(
                5000,
                len(data)
            ),
            random_state=42
        )


        scatter_data = scatter_data[
            [
                "Age",
                "Annual_Premium"
            ]
        ].dropna()


        # Remove non-finite values

        scatter_data = scatter_data[
            np.isfinite(
                scatter_data
            ).all(axis=1)
        ]


        plt.figure(
            figsize=(8, 5)
        )


        sns.scatterplot(
            data=scatter_data,
            x="Age",
            y="Annual_Premium",
            alpha=0.4
        )


        plt.xlabel(
            "Age"
        )


        plt.ylabel(
            "Annual Premium"
        )


        filename = show(
            title="Age vs Annual Premium",
            filename="age_vs_premium.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 19. RESPONSE RATE BY AGE GROUP
    # ========================================================

    print("\n" + "=" * 80)

    print("19. RESPONSE RATE BY AGE GROUP")

    print("=" * 80)


    if (
        "Age" in data.columns
        and
        "Response" in data.columns
    ):


        age_bins = [
            0,
            25,
            35,
            45,
            55,
            65,
            100
        ]


        age_labels = [
            "18-25",
            "26-35",
            "36-45",
            "46-55",
            "56-65",
            "66+"
        ]


        age_groups = pd.cut(
            data["Age"],
            bins=age_bins,
            labels=age_labels,
            include_lowest=True
        )


        age_response = (

            data.groupby(
                age_groups,
                observed=False
            )["Response"]

            .mean()

            * 100

        )


        print(
            "\nResponse Rate by Age Group:"
        )


        print(
            age_response
        )


        plt.figure(
            figsize=(9, 5)
        )


        sns.barplot(
            x=age_response.index,
            y=age_response.values
        )


        plt.xlabel(
            "Age Group"
        )


        plt.ylabel(
            "Response Rate (%)"
        )


        filename = show(
            title="Response Rate by Age Group",
            filename="age_group_response.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 20. PAIRPLOT
    # ========================================================

    print("\n" + "=" * 80)

    print("20. PAIRPLOT")

    print("=" * 80)


    pair_cols = [

        "Age",

        "Annual_Premium",

        "Vintage",

        "Response"

    ]


    pair_cols = [

        col

        for col in pair_cols

        if col in data.columns

    ]


    if len(pair_cols) >= 2:

        pair_data = data[
            pair_cols
        ].sample(

            min(
                2000,
                len(data)
            ),

            random_state=42

        ).dropna()


        pair_data = pair_data[
            np.isfinite(
                pair_data
            ).all(axis=1)
        ]


        pair_grid = sns.pairplot(
            pair_data,
            hue="Response" if "Response"
            in pair_data.columns else None,
            diag_kind="hist"
        )


        pair_grid.savefig(

            os.path.join(
                PLOT_DIR,
                "pairplot.png"
            ),

            dpi=120,

            bbox_inches="tight"

        )


        plt.close("all")


        plot_files.append(
            "pairplot.png"
        )


    # ========================================================
    # RESPONSE RATE
    # ========================================================

    response_rate = None


    if "Response" in data.columns:

        response_rate = round(
            data["Response"].mean() * 100,
            2
        )


    # ========================================================
    # RESULTS
    # ========================================================

    results = {

        "rows": int(
            data.shape[0]
        ),

        "columns": int(
            data.shape[1]
        ),

        "column_names": list(
            data.columns
        ),

        "duplicate_rows":
            duplicate_count,

        "response_rate":
            response_rate,

        "target_distribution": (
            {
                str(status): int(count)
                for status, count
                in data["Response"].value_counts().sort_index().items()
            }
            if "Response" in data.columns
            else {}
        ),

        "missing_values": {

            column: int(count)

            for column, count
            in data.isnull().sum().items()

        },

        "preview":
            data.head(10).to_dict(
                orient="records"
            ),

        "plots":
            plot_files

    }


    return results


# ============================================================
# DIRECT RUN
# ============================================================

if __name__ == "__main__":

    results = run_eda()


    print("\n" + "=" * 80)

    print(
        "POLICYPULSE EDA COMPLETED"
    )

    print("=" * 80)


    print(
        "\nRows:",
        results["rows"]
    )


    print(
        "Columns:",
        results["columns"]
    )


    print(
        "Duplicate Rows:",
        results["duplicate_rows"]
    )


    print(
        "Response Rate:",
        results["response_rate"],
        "%"
    )


    print(
        "\nPlots saved in:",
        PLOT_DIR
    )
```

## 5.5 `preprocessing.py`

```python
import os

import pandas as pd

from sklearn.model_selection import train_test_split

from Outlier_Fix import fix_outliers
from Standard_Scaler import standard_scale


# ============================================================
# CONFIGURATION
# ============================================================

FILE_PATH = r"Policypulse.csv"

OUTPUT_FILE = r"preprocessed_data.csv"

TARGET_COLUMN = "Response"

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    if not os.path.exists(FILE_PATH):

        raise FileNotFoundError(
            f"Dataset not found: {FILE_PATH}"
        )

    df = pd.read_csv(FILE_PATH)

    return df


# ============================================================
# REMOVE ID
# ============================================================

def remove_id(df):

    df = df.copy()

    if "id" in df.columns:

        df = df.drop(columns=["id"])

        print("\nRemoved 'id' column.")

    return df


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(df):

    df = df.copy()

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    # Numerical missing values
    for column in numerical_columns:

        if df[column].isnull().sum() > 0:

            df[column] = df[column].fillna(
                df[column].median()
            )

    # Categorical missing values
    for column in categorical_columns:

        if df[column].isnull().sum() > 0:

            df[column] = df[column].fillna(
                df[column].mode()[0]
            )

    return df


# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

def encode_categorical_features(df):

    df = df.copy()

    # These columns are categorical.
    #
    # Region_Code and Policy_Sales_Channel may look
    # numerical, but they represent categories/codes.
    categorical_columns = [
        "Gender",
        "Vehicle_Age",
        "Vehicle_Damage",
        "Region_Code",
        "Policy_Sales_Channel"
    ]

    existing_columns = [
        column
        for column in categorical_columns
        if column in df.columns
    ]

    print("\nCategorical columns being encoded:")

    for column in existing_columns:
        print("-", column)

    df = pd.get_dummies(
        df,
        columns=existing_columns,
        drop_first=True,
        dtype=int
    )

    return df


# ============================================================
# PREPROCESS DATA
# ============================================================

def preprocess_data():

    print("=" * 80)

    print("POLICYPULSE PREPROCESSING")

    print("=" * 80)


    # --------------------------------------------------------
    # STEP 1: LOAD
    # --------------------------------------------------------

    df = load_dataset()

    print("\nOriginal dataset shape:")

    print(df.shape)


    # --------------------------------------------------------
    # STEP 2: REMOVE ID
    # --------------------------------------------------------

    df = remove_id(df)

    print("\nAfter removing ID:")

    print(df.shape)


    # --------------------------------------------------------
    # STEP 3: HANDLE MISSING VALUES
    # --------------------------------------------------------

    df = handle_missing_values(df)

    print("\nTotal missing values after handling:")

    print(df.isnull().sum().sum())


    # --------------------------------------------------------
    # STEP 4: ENCODE CATEGORICAL FEATURES
    # --------------------------------------------------------

    df = encode_categorical_features(df)

    print("\nAfter categorical encoding:")

    print(df.shape)


    # --------------------------------------------------------
    # STEP 5: TRAIN / TEST SPLIT
    # --------------------------------------------------------

    train_df, test_df = train_test_split(

        df,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=df[TARGET_COLUMN]

    )


    print("\nTraining dataset shape:")

    print(train_df.shape)


    print("\nTesting dataset shape:")

    print(test_df.shape)


    # --------------------------------------------------------
    # STEP 6: OUTLIER HANDLING
    # --------------------------------------------------------

    train_df, test_df = fix_outliers(
        train_df,
        test_df,
        feature="Annual_Premium"
    )


    # --------------------------------------------------------
    # STEP 7: STANDARD SCALING
    # --------------------------------------------------------

    train_df, test_df, scaler = standard_scale(
        train_df,
        test_df
    )


    # --------------------------------------------------------
    # STEP 8: COMBINE DATA
    # --------------------------------------------------------

    processed_df = pd.concat(
        [train_df, test_df]
    )

    # Restore original row order
    processed_df = processed_df.sort_index()


    # --------------------------------------------------------
    # STEP 9: SAVE
    # --------------------------------------------------------

    processed_df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # --------------------------------------------------------
    # FINAL INFORMATION
    # --------------------------------------------------------

    print("\n" + "=" * 80)

    print("PREPROCESSING COMPLETED")

    print("=" * 80)


    print("\nFinal dataset shape:")

    print(processed_df.shape)


    print("\nFinal columns:")

    print(processed_df.columns.tolist())


    print("\nTotal missing values:")

    print(processed_df.isnull().sum().sum())


    print("\nFirst 5 rows:")

    print(processed_df.head())


    print("\nPreprocessed dataset saved as:")

    print(OUTPUT_FILE)


    return processed_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    preprocess_data()
```

## 5.6 `Outlier_Fix.py`

```python
import pandas as pd


# ============================================================
# OUTLIER FIX FUNCTION
# ============================================================

def fix_outliers(
    train_df,
    test_df,
    feature="Annual_Premium"
):

    train_df = train_df.copy()
    test_df = test_df.copy()


    # ========================================================
    # DISPLAY ORIGINAL STATISTICS
    # ========================================================

    print("\n" + "=" * 80)
    print("OUTLIER DETECTION")
    print("=" * 80)


    print("\nFeature:")

    print(feature)


    print("\nOriginal TRAINING statistics:")

    print(
        train_df[feature].describe()
    )


    print("\nOriginal TESTING statistics:")

    print(
        test_df[feature].describe()
    )


    # ========================================================
    # CALCULATE QUARTILES
    # ========================================================

    Q1 = train_df[feature].quantile(
        0.25
    )

    Q3 = train_df[feature].quantile(
        0.75
    )


    # ========================================================
    # CALCULATE IQR
    # ========================================================

    IQR = Q3 - Q1


    # ========================================================
    # CALCULATE FENCES
    # ========================================================

    lower_fence = Q1 - (
        1.5 * IQR
    )

    upper_fence = Q3 + (
        1.5 * IQR
    )


    print("\n" + "=" * 80)
    print("IQR OUTLIER CALCULATION")
    print("=" * 80)


    print("\nQ1:")

    print(Q1)


    print("\nQ3:")

    print(Q3)


    print("\nIQR:")

    print(IQR)


    print("\nLower Fence:")

    print(lower_fence)


    print("\nUpper Fence:")

    print(upper_fence)


    # ========================================================
    # FIND TRAINING OUTLIERS
    # ========================================================

    train_outliers = train_df[
        (train_df[feature] < lower_fence)
        |
        (train_df[feature] > upper_fence)
    ]


    # ========================================================
    # FIND TESTING OUTLIERS
    # ========================================================

    test_outliers = test_df[
        (test_df[feature] < lower_fence)
        |
        (test_df[feature] > upper_fence)
    ]


    print("\n" + "=" * 80)
    print("OUTLIER COUNTS")
    print("=" * 80)


    print("\nTraining outliers:")

    print(
        len(train_outliers)
    )


    print("\nTesting values outside training fences:")

    print(
        len(test_outliers)
    )


    # ========================================================
    # CLIP TRAINING DATA
    # ========================================================

    train_df[feature] = train_df[
        feature
    ].clip(

        lower=lower_fence,

        upper=upper_fence

    )


    # ========================================================
    # CLIP TESTING DATA
    # ========================================================

    # IMPORTANT:
    #
    # The test data is clipped using the fences learned
    # from the training data.
    #
    # We do NOT calculate new Q1/Q3/IQR values for testing.
    #
    # This prevents information from the testing dataset
    # influencing preprocessing.

    test_df[feature] = test_df[
        feature
    ].clip(

        lower=lower_fence,

        upper=upper_fence

    )


    # ========================================================
    # DISPLAY FINAL STATISTICS
    # ========================================================

    print("\n" + "=" * 80)
    print("DATA AFTER OUTLIER HANDLING")
    print("=" * 80)


    print("\nTraining statistics after outlier handling:")

    print(
        train_df[feature].describe()
    )


    print("\nTesting statistics after outlier handling:")

    print(
        test_df[feature].describe()
    )


    # ========================================================
    # RETURN
    # ========================================================

    return (
        train_df,
        test_df
    )
```

## 5.7 `Standard_Scaler.py`

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONTINUOUS NUMERICAL FEATURES
# ============================================================

# Only continuous numerical features are standardized.
#
# Age              -> Continuous numerical feature
# Annual_Premium   -> Continuous numerical feature
# Vintage          -> Continuous numerical feature
#
# Driving_License and Previously_Insured are binary features
# and should remain as 0/1.
#
# Region_Code and Policy_Sales_Channel are categorical codes.
# They are one-hot encoded in preprocessing.py before reaching
# this stage, so they are NOT included here.

num_cols = [
    "Age",
    "Annual_Premium",
    "Vintage"
]


# ============================================================
# STANDARD SCALING FUNCTION
# ============================================================

def standard_scale(train_df, test_df):

    train_df = train_df.copy()
    test_df = test_df.copy()


    # ========================================================
    # DISPLAY FEATURES
    # ========================================================

    print("\n" + "=" * 80)
    print("FEATURES SELECTED FOR STANDARD SCALING")
    print("=" * 80)

    print("\nContinuous numerical features:")

    for column in num_cols:
        print("-", column)


    # ========================================================
    # STANDARD SCALER
    # ========================================================

    scaler = StandardScaler()


    # ========================================================
    # FIT ONLY ON TRAINING DATA
    # ========================================================

    # The scaler learns the mean and standard deviation
    # ONLY from the training data.
    #
    # This prevents data leakage from the testing dataset.

    train_df[num_cols] = scaler.fit_transform(
        train_df[num_cols]
    )


    # ========================================================
    # TRANSFORM TESTING DATA
    # ========================================================

    # The same scaler learned from the training data
    # is applied to the testing data.

    test_df[num_cols] = scaler.transform(
        test_df[num_cols]
    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 80)
    print("DATA AFTER STANDARD SCALING")
    print("=" * 80)


    print("\nTraining data after Standard Scaling:")

    print(
        train_df[num_cols].head()
    )


    print("\nTesting data after Standard Scaling:")

    print(
        test_df[num_cols].head()
    )


    # ========================================================
    # VERIFY STANDARD SCALING
    # ========================================================

    print("\n" + "=" * 80)
    print("TRAINING DATA MEAN AFTER SCALING")
    print("=" * 80)

    print(
        train_df[num_cols].mean()
    )


    print("\n" + "=" * 80)
    print("TRAINING DATA STANDARD DEVIATION AFTER SCALING")
    print("=" * 80)

    print(
        train_df[num_cols].std()
    )


    # ========================================================
    # VERIFY TESTING DATA
    # ========================================================

    print("\n" + "=" * 80)
    print("TESTING DATA MEAN AFTER SCALING")
    print("=" * 80)

    print(
        test_df[num_cols].mean()
    )


    print("\n" + "=" * 80)
    print("TESTING DATA STANDARD DEVIATION AFTER SCALING")
    print("=" * 80)

    print(
        test_df[num_cols].std()
    )


    # ========================================================
    # DISPLAY SCALER PARAMETERS
    # ========================================================

    print("\n" + "=" * 80)
    print("STANDARD SCALER PARAMETERS")
    print("=" * 80)


    print("\nFeatures:")

    print(num_cols)


    print("\nLearned means:")

    print(
        scaler.mean_
    )


    print("\nLearned standard deviations:")

    print(
        scaler.scale_
    )


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return train_df, test_df, scaler
```

## 5.8 `Min_Max_Scaling.py`

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# ============================================================
# DATASET PATH
# ============================================================

file_path = r"Policypulse.csv"


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(file_path)


# ============================================================
# NUMERICAL FEATURES
# ============================================================

num_cols = [
    "Age",
    "Driving_License",
    "Region_Code",
    "Previously_Insured",
    "Annual_Premium",
    "Policy_Sales_Channel",
    "Vintage"
]


# ============================================================
# TARGET VARIABLE
# ============================================================

target_col = "Response"


# ============================================================
# TRAIN-TEST SPLIT
# ============================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df[target_col]
)


# ============================================================
# DISPLAY ORIGINAL DATA
# ============================================================

print("=" * 80)
print("DATASET")
print("=" * 80)

print(df.head())


print("\nDataset Columns:")
print(df.columns)


# ============================================================
# TRAINING DATA BEFORE SCALING
# ============================================================

print("\n" + "=" * 80)
print("TRAINING DATA BEFORE SCALING")
print("=" * 80)

print(train_df[num_cols].head())


# ============================================================
# TESTING DATA BEFORE SCALING
# ============================================================

print("\n" + "=" * 80)
print("TESTING DATA BEFORE SCALING")
print("=" * 80)

print(test_df[num_cols].head())


# ============================================================
# MIN-MAX SCALING
# ============================================================

scaler = MinMaxScaler()


# Fit the scaler ONLY on training data
# and transform the training data

train_df[num_cols] = scaler.fit_transform(
    train_df[num_cols]
)


# Use the same scaler to transform
# the testing data

test_df[num_cols] = scaler.transform(
    test_df[num_cols]
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 80)
print("DATA AFTER MIN-MAX SCALING")
print("=" * 80)


print(
    "Original dataset shape:",
    df.shape
)


print(
    "Training dataset shape:",
    train_df.shape
)


print(
    "Testing dataset shape:",
    test_df.shape
)


print("\nTraining data after scaling:")

print(
    train_df[num_cols].head()
)


print("\nTesting data after scaling:")

print(
    test_df[num_cols].head()
)


```

## 5.9 `linear_regression.py`

```python
import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# CONFIGURATION
# ============================================================

FILE_PATH = r"preprocessed_data.csv"

TARGET_COLUMN = "Response"

TEST_SIZE = 0.20

RANDOM_STATE = 42

RIDGE_ALPHA = 1.0

LASSO_ALPHA = 0.001

PLOT_DIRECTORY = os.path.join(
    "static",
    "plots"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_preprocessed_data():

    if not os.path.exists(FILE_PATH):

        raise FileNotFoundError(
            f"Preprocessed dataset not found: {FILE_PATH}"
        )

    return pd.read_csv(FILE_PATH)


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

def prepare_data(df):

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    X = df.drop(columns=[TARGET_COLUMN])

    y = df[TARGET_COLUMN]

    return X, y


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

def split_data(X, y):

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "predictions": predictions
    }


# ============================================================
# ACTUAL VS PREDICTED PLOT
# ============================================================

def create_prediction_plot(
    y_test,
    predictions,
    model_name,
    filename
):

    os.makedirs(
        PLOT_DIRECTORY,
        exist_ok=True
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        y_test,
        predictions,
        alpha=0.25,
        s=10
    )

    plt.xlabel(
        "Actual Response"
    )

    plt.ylabel(
        "Predicted Response"
    )

    plt.title(
        f"{model_name}: Actual vs Predicted"
    )

    plt.grid(
        alpha=0.2
    )

    path = os.path.join(
        PLOT_DIRECTORY,
        filename
    )

    plt.tight_layout()

    plt.savefig(path)

    plt.close()

    return path


# ============================================================
# MODEL COMPARISON PLOT
# ============================================================

def create_comparison_plot(results):

    os.makedirs(
        PLOT_DIRECTORY,
        exist_ok=True
    )

    model_names = list(
        results.keys()
    )

    rmse_values = [
        results[name]["RMSE"]
        for name in model_names
    ]

    r2_values = [
        results[name]["R2"]
        for name in model_names
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    axes[0].bar(
        model_names,
        rmse_values
    )

    axes[0].set_title(
        "RMSE Comparison"
    )

    axes[0].set_ylabel(
        "RMSE"
    )

    axes[0].tick_params(
        axis="x",
        rotation=20
    )


    axes[1].bar(
        model_names,
        r2_values
    )

    axes[1].set_title(
        "R² Score Comparison"
    )

    axes[1].set_ylabel(
        "R² Score"
    )

    axes[1].tick_params(
        axis="x",
        rotation=20
    )

    plt.tight_layout()

    path = os.path.join(
        PLOT_DIRECTORY,
        "linear_regression_comparison.png"
    )

    plt.savefig(path)

    plt.close()

    return path


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_linear_regression():

    print("=" * 80)
    print("POLICYPULSE - V4 LINEAR REGRESSION")
    print("=" * 80)


    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_preprocessed_data()

    print("\nDataset shape:")
    print(df.shape)


    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------

    X, y = prepare_data(df)

    print("\nNumber of features:")
    print(X.shape[1])


    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    print("\nTraining shape:")
    print(X_train.shape)

    print("\nTesting shape:")
    print(X_test.shape)


    # ========================================================
    # MODEL 1
    # WITHOUT REGULARIZATION
    # ========================================================

    print("\n" + "=" * 80)
    print("WITHOUT REGULARIZATION")
    print("LINEAR REGRESSION")
    print("=" * 80)

    linear_model = LinearRegression()

    linear_model.fit(
        X_train,
        y_train
    )

    linear_results = evaluate_model(
        linear_model,
        X_test,
        y_test
    )

    print("\nMAE:", linear_results["MAE"])
    print("MSE:", linear_results["MSE"])
    print("RMSE:", linear_results["RMSE"])
    print("R²:", linear_results["R2"])


    # ========================================================
    # MODEL 2
    # WITH REGULARIZATION
    # RIDGE - L2
    # ========================================================

    print("\n" + "=" * 80)
    print("WITH REGULARIZATION")
    print("RIDGE REGRESSION - L2")
    print("=" * 80)

    ridge_model = Ridge(
        alpha=RIDGE_ALPHA
    )

    ridge_model.fit(
        X_train,
        y_train
    )

    ridge_results = evaluate_model(
        ridge_model,
        X_test,
        y_test
    )

    print("\nAlpha:", RIDGE_ALPHA)
    print("MAE:", ridge_results["MAE"])
    print("MSE:", ridge_results["MSE"])
    print("RMSE:", ridge_results["RMSE"])
    print("R²:", ridge_results["R2"])


    # ========================================================
    # MODEL 3
    # WITH REGULARIZATION
    # LASSO - L1
    # ========================================================

    print("\n" + "=" * 80)
    print("WITH REGULARIZATION")
    print("LASSO REGRESSION - L1")
    print("=" * 80)

    lasso_model = Lasso(
        alpha=LASSO_ALPHA,
        max_iter=10000
    )

    lasso_model.fit(
        X_train,
        y_train
    )

    lasso_results = evaluate_model(
        lasso_model,
        X_test,
        y_test
    )

    print("\nAlpha:", LASSO_ALPHA)
    print("MAE:", lasso_results["MAE"])
    print("MSE:", lasso_results["MSE"])
    print("RMSE:", lasso_results["RMSE"])
    print("R²:", lasso_results["R2"])


    # ========================================================
    # RESULTS
    # ========================================================

    results = {

        "Linear Regression": linear_results,

        "Ridge Regression": ridge_results,

        "Lasso Regression": lasso_results

    }


    # ========================================================
    # PLOTS
    # ========================================================

    linear_plot = create_prediction_plot(
        y_test,
        linear_results["predictions"],
        "Linear Regression",
        "linear_regression_predictions.png"
    )

    ridge_plot = create_prediction_plot(
        y_test,
        ridge_results["predictions"],
        "Ridge Regression",
        "ridge_regression_predictions.png"
    )

    lasso_plot = create_prediction_plot(
        y_test,
        lasso_results["predictions"],
        "Lasso Regression",
        "lasso_regression_predictions.png"
    )

    comparison_plot = create_comparison_plot(
        results
    )


    # ========================================================
    # COMPARISON TABLE
    # ========================================================

    comparison = pd.DataFrame({

        "Category": [
            "Without Regularization",
            "With Regularization",
            "With Regularization"
        ],

        "Model": [
            "Linear Regression",
            "Ridge Regression (L2)",
            "Lasso Regression (L1)"
        ],

        "MAE": [
            linear_results["MAE"],
            ridge_results["MAE"],
            lasso_results["MAE"]
        ],

        "MSE": [
            linear_results["MSE"],
            ridge_results["MSE"],
            lasso_results["MSE"]
        ],

        "RMSE": [
            linear_results["RMSE"],
            ridge_results["RMSE"],
            lasso_results["RMSE"]
        ],

        "R2 Score": [
            linear_results["R2"],
            ridge_results["R2"],
            lasso_results["R2"]
        ]

    })


    print("\n" + "=" * 80)
    print("LINEAR REGRESSION MODEL COMPARISON")
    print("=" * 80)

    print(
        comparison.to_string(
            index=False
        )
    )


    return {

        "dataset_shape": df.shape,

        "training_shape": X_train.shape,

        "testing_shape": X_test.shape,

        "comparison": comparison.to_dict(
            orient="records"
        ),

        "plots": {

            "linear": linear_plot,

            "ridge": ridge_plot,

            "lasso": lasso_plot,

            "comparison": comparison_plot

        }

    }


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    run_linear_regression()

    print(
        "\nV4 Linear Regression completed successfully."
    )
```

## 5.10 `logistic_regression.py`

```python
import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)


# ============================================================
# CONFIGURATION
# ============================================================

FILE_PATH = r"preprocessed_data.csv"

TARGET_COLUMN = "Response"

TEST_SIZE = 0.20

RANDOM_STATE = 42


# Logistic Regression regularization parameters

RIDGE_C = 1.0

LASSO_C = 0.1


# Maximum number of optimization iterations

MAX_ITERATIONS = 2000


# Directory for generated plots

PLOT_DIRECTORY = os.path.join(
    "static",
    "plots"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_preprocessed_data():

    if not os.path.exists(FILE_PATH):

        raise FileNotFoundError(
            f"Preprocessed dataset not found: {FILE_PATH}"
        )

    return pd.read_csv(FILE_PATH)


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    # --------------------------------------------------------
    # Separate features from target
    # --------------------------------------------------------

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN]

    # --------------------------------------------------------
    # Convert all features to float64
    # --------------------------------------------------------
    #
    # The preprocessed dataset contains:
    #
    # - standardized numerical features
    # - one-hot encoded categorical features
    # - binary features
    #
    # Using one consistent floating-point type ensures that
    # matrix operations inside Logistic Regression are
    # performed consistently.
    # --------------------------------------------------------

    X = X.astype("float64")

    # Convert target to integer

    y = y.astype("int64")

    return X, y


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y):

    return train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y
    )


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    # --------------------------------------------------------
    # Generate class predictions
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Generate probability predictions
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    # --------------------------------------------------------
    # F1 Score
    # --------------------------------------------------------

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    # --------------------------------------------------------
    # ROC AUC
    # --------------------------------------------------------

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC AUC": auc,

        "predictions": predictions,

        "probabilities": probabilities
    }


# ============================================================
# CONFUSION MATRIX
# ============================================================

def create_confusion_matrix_plot(
    y_test,
    predictions,
    filename,
    title
):

    os.makedirs(
        PLOT_DIRECTORY,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Calculate confusion matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # Create display
    # --------------------------------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix
    )

    display.plot()

    plt.title(title)

    plt.tight_layout()

    # --------------------------------------------------------
    # Save plot
    # --------------------------------------------------------

    path = os.path.join(
        PLOT_DIRECTORY,
        filename
    )

    plt.savefig(path)

    plt.close()

    return path


# ============================================================
# ROC CURVE
# ============================================================

def create_roc_curve_plot(
    y_test,
    probability,
    filename,
    title
):

    os.makedirs(
        PLOT_DIRECTORY,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Calculate ROC curve
    # --------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y_test,
        probability
    )

    # --------------------------------------------------------
    # Calculate ROC AUC
    # --------------------------------------------------------

    auc = roc_auc_score(
        y_test,
        probability
    )

    # --------------------------------------------------------
    # Create ROC plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        fpr,
        tpr,
        label=f"ROC AUC = {auc:.4f}"
    )

    # Random classifier reference line

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(title)

    plt.legend()

    plt.grid(
        alpha=0.2
    )

    # --------------------------------------------------------
    # Save plot
    # --------------------------------------------------------

    path = os.path.join(
        PLOT_DIRECTORY,
        filename
    )

    plt.tight_layout()

    plt.savefig(path)

    plt.close()

    return path


# ============================================================
# COMPARISON PLOT
# ============================================================

def create_comparison_plot(results):

    os.makedirs(
        PLOT_DIRECTORY,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Get model names
    # --------------------------------------------------------

    models = list(
        results.keys()
    )

    # --------------------------------------------------------
    # Get Accuracy values
    # --------------------------------------------------------

    accuracy = [

        results[model]["Accuracy"]

        for model in models

    ]

    # --------------------------------------------------------
    # Get F1 Score values
    # --------------------------------------------------------

    f1 = [

        results[model]["F1 Score"]

        for model in models

    ]

    # --------------------------------------------------------
    # Get ROC AUC values
    # --------------------------------------------------------

    auc = [

        results[model]["ROC AUC"]

        for model in models

    ]

    # --------------------------------------------------------
    # X-axis positions
    # --------------------------------------------------------

    x = range(
        len(models)
    )

    width = 0.25

    # --------------------------------------------------------
    # Create comparison chart
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    # Accuracy

    plt.bar(
        [i - width for i in x],
        accuracy,
        width=width,
        label="Accuracy"
    )

    # F1 Score

    plt.bar(
        x,
        f1,
        width=width,
        label="F1 Score"
    )

    # ROC AUC

    plt.bar(
        [i + width for i in x],
        auc,
        width=width,
        label="ROC AUC"
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    plt.xticks(
        list(x),
        models,
        rotation=20
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Logistic Regression Model Comparison"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.2
    )

    # --------------------------------------------------------
    # Save comparison plot
    # --------------------------------------------------------

    path = os.path.join(
        PLOT_DIRECTORY,
        "logistic_regression_comparison.png"
    )

    plt.tight_layout()

    plt.savefig(path)

    plt.close()

    return path


# ============================================================
# MAIN LOGISTIC REGRESSION FUNCTION
# ============================================================

def run_logistic_regression():

    print("=" * 80)

    print(
        "POLICYPULSE - V4 LOGISTIC REGRESSION"
    )

    print("=" * 80)

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = load_preprocessed_data()

    print("\nDataset shape:")

    print(df.shape)

    # ========================================================
    # PREPARE DATA
    # ========================================================

    X, y = prepare_data(df)

    print("\nNumber of features:")

    print(X.shape[1])

    print("\nTarget:")

    print(TARGET_COLUMN)

    print("\nTarget distribution:")

    print(y.value_counts())

    print("\nTarget proportions:")

    print(
        y.value_counts(
            normalize=True
        )
    )

    # --------------------------------------------------------
    # NUMERICAL DATA VALIDATION
    # --------------------------------------------------------
    #
    # These checks verify the exact feature matrix that will
    # be supplied to Logistic Regression.
    # --------------------------------------------------------

    print("\n" + "=" * 80)

    print("FEATURE MATRIX NUMERICAL CHECK")

    print("=" * 80)

    print("\nFeature data types:")

    print(
        X.dtypes.value_counts()
    )

    print("\nNaN values:")

    print(
        X.isna().sum().sum()
    )

    print("\nInfinite values:")

    print(
        X.isin(
            [float("inf"), float("-inf")]
        ).sum().sum()
    )

    print("\nLargest absolute feature value:")

    print(
        X.abs().max().max()
    )

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    print("\nTraining shape:")

    print(X_train.shape)

    print("\nTesting shape:")

    print(X_test.shape)

    # ========================================================
    # WITHOUT REGULARIZATION
    # ========================================================

    print("\n" + "=" * 80)

    print("WITHOUT REGULARIZATION")

    print("LOGISTIC REGRESSION")

    print("=" * 80)

    # --------------------------------------------------------
    # Logistic Regression without regularization
    # --------------------------------------------------------
    #
    # penalty=None means no L1/L2 regularization.
    #
    # lbfgs is explicitly selected as the optimization solver.
    # --------------------------------------------------------

    logistic_model = LogisticRegression(

        penalty=None,

        solver="lbfgs",

        class_weight="balanced",

        max_iter=MAX_ITERATIONS,

        random_state=RANDOM_STATE

    )

    logistic_model.fit(
        X_train,
        y_train
    )

    logistic_results = evaluate_model(
        logistic_model,
        X_test,
        y_test
    )

    print(
        "\nAccuracy:",
        logistic_results["Accuracy"]
    )

    print(
        "Precision:",
        logistic_results["Precision"]
    )

    print(
        "Recall:",
        logistic_results["Recall"]
    )

    print(
        "F1 Score:",
        logistic_results["F1 Score"]
    )

    print(
        "ROC AUC:",
        logistic_results["ROC AUC"]
    )

    # ========================================================
    # RIDGE LOGISTIC REGRESSION
    # ========================================================

    print("\n" + "=" * 80)

    print("WITH REGULARIZATION")

    print("RIDGE LOGISTIC REGRESSION - L2")

    print("=" * 80)

    # --------------------------------------------------------
    # Ridge Logistic Regression
    # --------------------------------------------------------
    #
    # L2 regularization penalizes large coefficient values.
    #
    # C controls the inverse strength of regularization.
    # Smaller C = stronger regularization.
    # --------------------------------------------------------

    ridge_model = LogisticRegression(

        penalty="l2",

        C=RIDGE_C,

        solver="lbfgs",

        class_weight="balanced",

        max_iter=MAX_ITERATIONS,

        random_state=RANDOM_STATE

    )

    ridge_model.fit(
        X_train,
        y_train
    )

    ridge_results = evaluate_model(
        ridge_model,
        X_test,
        y_test
    )

    print(
        "\nC:",
        RIDGE_C
    )

    print(
        "Accuracy:",
        ridge_results["Accuracy"]
    )

    print(
        "Precision:",
        ridge_results["Precision"]
    )

    print(
        "Recall:",
        ridge_results["Recall"]
    )

    print(
        "F1 Score:",
        ridge_results["F1 Score"]
    )

    print(
        "ROC AUC:",
        ridge_results["ROC AUC"]
    )

    # ========================================================
    # LASSO LOGISTIC REGRESSION
    # ========================================================

    print("\n" + "=" * 80)

    print("WITH REGULARIZATION")

    print("LASSO LOGISTIC REGRESSION - L1")

    print("=" * 80)

    # --------------------------------------------------------
    # Lasso Logistic Regression
    # --------------------------------------------------------
    #
    # L1 regularization encourages some coefficients to become
    # exactly zero, which can perform feature selection.
    #
    # liblinear supports L1 regularization for Logistic
    # Regression.
    # --------------------------------------------------------

    lasso_model = LogisticRegression(

        penalty="l1",

        C=LASSO_C,

        solver="liblinear",

        class_weight="balanced",

        max_iter=MAX_ITERATIONS,

        random_state=RANDOM_STATE

    )

    lasso_model.fit(
        X_train,
        y_train
    )

    lasso_results = evaluate_model(
        lasso_model,
        X_test,
        y_test
    )

    print(
        "\nC:",
        LASSO_C
    )

    print(
        "Accuracy:",
        lasso_results["Accuracy"]
    )

    print(
        "Precision:",
        lasso_results["Precision"]
    )

    print(
        "Recall:",
        lasso_results["Recall"]
    )

    print(
        "F1 Score:",
        lasso_results["F1 Score"]
    )

    print(
        "ROC AUC:",
        lasso_results["ROC AUC"]
    )

    # ========================================================
    # STORE RESULTS
    # ========================================================

    results = {

        "Logistic Regression":
            logistic_results,

        "Ridge Logistic Regression":
            ridge_results,

        "Lasso Logistic Regression":
            lasso_results

    }

    # ========================================================
    # CONFUSION MATRICES
    # ========================================================

    logistic_cm = create_confusion_matrix_plot(

        y_test,

        logistic_results["predictions"],

        "logistic_regression_confusion_matrix.png",

        "Logistic Regression - Confusion Matrix"

    )

    ridge_cm = create_confusion_matrix_plot(

        y_test,

        ridge_results["predictions"],

        "ridge_logistic_confusion_matrix.png",

        "Ridge Logistic Regression - Confusion Matrix"

    )

    lasso_cm = create_confusion_matrix_plot(

        y_test,

        lasso_results["predictions"],

        "lasso_logistic_confusion_matrix.png",

        "Lasso Logistic Regression - Confusion Matrix"

    )

    # ========================================================
    # ROC CURVES
    # ========================================================

    logistic_roc = create_roc_curve_plot(

        y_test,

        logistic_results["probabilities"],

        "logistic_regression_roc_curve.png",

        "Logistic Regression - ROC Curve"

    )

    ridge_roc = create_roc_curve_plot(

        y_test,

        ridge_results["probabilities"],

        "ridge_logistic_roc_curve.png",

        "Ridge Logistic Regression - ROC Curve"

    )

    lasso_roc = create_roc_curve_plot(

        y_test,

        lasso_results["probabilities"],

        "lasso_logistic_roc_curve.png",

        "Lasso Logistic Regression - ROC Curve"

    )

    # ========================================================
    # COMPARISON PLOT
    # ========================================================

    comparison_plot = create_comparison_plot(
        results
    )

    # ========================================================
    # COMPARISON TABLE
    # ========================================================

    comparison = pd.DataFrame({

        "Category": [

            "Without Regularization",

            "With Regularization",

            "With Regularization"

        ],

        "Model": [

            "Logistic Regression",

            "Ridge Logistic Regression (L2)",

            "Lasso Logistic Regression (L1)"

        ],

        "Accuracy": [

            logistic_results["Accuracy"],

            ridge_results["Accuracy"],

            lasso_results["Accuracy"]

        ],

        "Precision": [

            logistic_results["Precision"],

            ridge_results["Precision"],

            lasso_results["Precision"]

        ],

        "Recall": [

            logistic_results["Recall"],

            ridge_results["Recall"],

            lasso_results["Recall"]

        ],

        "F1 Score": [

            logistic_results["F1 Score"],

            ridge_results["F1 Score"],

            lasso_results["F1 Score"]

        ],

        "ROC AUC": [

            logistic_results["ROC AUC"],

            ridge_results["ROC AUC"],

            lasso_results["ROC AUC"]

        ]

    })

    # ========================================================
    # PRINT COMPARISON
    # ========================================================

    print("\n" + "=" * 80)

    print(
        "LOGISTIC REGRESSION MODEL COMPARISON"
    )

    print("=" * 80)

    print(
        comparison.to_string(
            index=False
        )
    )

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "dataset_shape":
            df.shape,

        "training_shape":
            X_train.shape,

        "testing_shape":
            X_test.shape,

        "comparison":
            comparison.to_dict(
                orient="records"
            ),

        "plots": {

            "comparison":
                comparison_plot,

            "logistic_cm":
                logistic_cm,

            "ridge_cm":
                ridge_cm,

            "lasso_cm":
                lasso_cm,

            "logistic_roc":
                logistic_roc,

            "ridge_roc":
                ridge_roc,

            "lasso_roc":
                lasso_roc

        }

    }


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    run_logistic_regression()

    print(
        "\nV4 Logistic Regression completed successfully."
    )
```

## 5.11 `tree_based.py`

```python
import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier


# ============================================================
# CONFIGURATION
# ============================================================

FILE_PATH = "preprocessed_data.csv"

TARGET_COLUMN = "Response"

TEST_SIZE = 0.20

RANDOM_STATE = 42

PLOT_DIRECTORY = os.path.join(
    "static",
    "plots"
)


# ============================================================
# LOAD PREPROCESSED DATA
# ============================================================

def load_preprocessed_data():

    if not os.path.exists(FILE_PATH):

        raise FileNotFoundError(

            f"Preprocessed dataset not found: "
            f"{FILE_PATH}"

        )

    df = pd.read_csv(
        FILE_PATH
    )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    if TARGET_COLUMN not in df.columns:

        raise ValueError(

            f"Target column "
            f"'{TARGET_COLUMN}' not found."

        )

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN]

    return X, y


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y):

    return train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y

    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]


    accuracy = accuracy_score(

        y_test,

        predictions

    )


    precision = precision_score(

        y_test,

        predictions,

        zero_division=0

    )


    recall = recall_score(

        y_test,

        predictions,

        zero_division=0

    )


    f1 = f1_score(

        y_test,

        predictions,

        zero_division=0

    )


    roc_auc = roc_auc_score(

        y_test,

        probabilities

    )


    return {

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC AUC": roc_auc,

        "predictions": predictions,

        "probabilities": probabilities

    }


# ============================================================
# 1. DECISION TREE
# ============================================================

def run_decision_tree(

    X_train,
    X_test,
    y_train,
    y_test

):

    print(
        "\nRunning Decision Tree..."
    )


    model = DecisionTreeClassifier(

        random_state=RANDOM_STATE,

        class_weight="balanced",

        max_depth=12,

        min_samples_leaf=10

    )


    model.fit(

        X_train,

        y_train

    )


    metrics = evaluate_model(

        model,

        X_test,

        y_test

    )


    return model, metrics


# ============================================================
# 2. RANDOM FOREST
# ============================================================

def run_random_forest(

    X_train,
    X_test,
    y_train,
    y_test

):

    print(
        "\nRunning Random Forest..."
    )


    model = RandomForestClassifier(

        n_estimators=100,

        random_state=RANDOM_STATE,

        class_weight="balanced",

        n_jobs=-1,

        max_depth=15,

        min_samples_leaf=5

    )


    model.fit(

        X_train,

        y_train

    )


    metrics = evaluate_model(

        model,

        X_test,

        y_test

    )


    return model, metrics


# ============================================================
# 3. EXTRA TREES
# ============================================================

def run_extra_trees(

    X_train,
    X_test,
    y_train,
    y_test

):

    print(
        "\nRunning Extra Trees..."
    )


    model = ExtraTreesClassifier(

        n_estimators=100,

        random_state=RANDOM_STATE,

        class_weight="balanced",

        n_jobs=-1,

        max_depth=15,

        min_samples_leaf=5

    )


    model.fit(

        X_train,

        y_train

    )


    metrics = evaluate_model(

        model,

        X_test,

        y_test

    )


    return model, metrics


# ============================================================
# 4. GRADIENT BOOSTING
# ============================================================

def run_gradient_boosting(

    X_train,
    X_test,
    y_train,
    y_test

):

    print(
        "\nRunning Gradient Boosting..."
    )


    model = GradientBoostingClassifier(

        n_estimators=100,

        learning_rate=0.10,

        max_depth=3,

        random_state=RANDOM_STATE

    )


    model.fit(

        X_train,

        y_train

    )


    metrics = evaluate_model(

        model,

        X_test,

        y_test

    )


    return model, metrics


# ============================================================
# 5. ADABOOST
# ============================================================

def run_adaboost(

    X_train,
    X_test,
    y_train,
    y_test

):

    print(
        "\nRunning AdaBoost..."
    )


    model = AdaBoostClassifier(

        n_estimators=100,

        learning_rate=0.50,

        random_state=RANDOM_STATE

    )


    model.fit(

        X_train,

        y_train

    )


    metrics = evaluate_model(

        model,

        X_test,

        y_test

    )


    return model, metrics


# ============================================================
# 6. XGBOOST
# ============================================================

def run_xgboost(

    X_train,
    X_test,
    y_train,
    y_test

):

    print(
        "\nRunning XGBoost..."
    )


    model = XGBClassifier(

        n_estimators=100,

        learning_rate=0.10,

        max_depth=6,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=RANDOM_STATE,

        eval_metric="logloss",

        n_jobs=-1

    )


    model.fit(

        X_train,

        y_train

    )


    metrics = evaluate_model(

        model,

        X_test,

        y_test

    )


    return model, metrics


# ============================================================
# 7. LIGHTGBM
# ============================================================

def run_lightgbm(

    X_train,
    X_test,
    y_train,
    y_test

):

    print(
        "\nRunning LightGBM..."
    )


    model = LGBMClassifier(

        n_estimators=100,

        learning_rate=0.10,

        max_depth=-1,

        num_leaves=31,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        verbosity=-1

    )


    model.fit(

        X_train,

        y_train

    )


    metrics = evaluate_model(

        model,

        X_test,

        y_test

    )


    return model, metrics


# ============================================================
# ALGORITHM MAP
# ============================================================

ALGORITHMS = {

    "decision_tree": (

        "Decision Tree",

        run_decision_tree

    ),

    "random_forest": (

        "Random Forest",

        run_random_forest

    ),

    "extra_trees": (

        "Extra Trees",

        run_extra_trees

    ),

    "gradient_boosting": (

        "Gradient Boosting",

        run_gradient_boosting

    ),

    "adaboost": (

        "AdaBoost",

        run_adaboost

    ),

    "xgboost": (

        "XGBoost",

        run_xgboost

    ),

    "lightgbm": (

        "LightGBM",

        run_lightgbm

    )

}


# ============================================================
# RUN SELECTED TREE ALGORITHM
# ============================================================

def run_tree_algorithm(
    algorithm
):

    if algorithm not in ALGORITHMS:

        raise ValueError(

            f"Unknown tree algorithm: "
            f"{algorithm}"

        )


    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_preprocessed_data()


    print("\n" + "=" * 80)

    print(
        "POLICYPULSE - V5 TREE BASED MODELS"
    )

    print("=" * 80)


    print(
        "\nSelected Algorithm:"
    )

    print(
        ALGORITHMS[algorithm][0]
    )


    print(
        "\nDataset Shape:"
    )

    print(
        df.shape
    )


    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------

    X, y = prepare_data(
        df
    )


    print(
        "\nNumber of Features:"
    )

    print(
        X.shape[1]
    )


    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = split_data(

        X,

        y

    )


    print(
        "\nTraining Shape:"
    )

    print(
        X_train.shape
    )


    print(
        "\nTesting Shape:"
    )

    print(
        X_test.shape
    )


    # --------------------------------------------------------
    # RUN MODEL
    # --------------------------------------------------------

    display_name, function = ALGORITHMS[algorithm]


    model, metrics = function(

        X_train,

        X_test,

        y_train,

        y_test

    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        display_name.upper()
    )

    print(
        "=" * 80
    )


    print(
        "Accuracy:",
        metrics["Accuracy"]
    )

    print(
        "Precision:",
        metrics["Precision"]
    )

    print(
        "Recall:",
        metrics["Recall"]
    )

    print(
        "F1 Score:",
        metrics["F1 Score"]
    )

    print(
        "ROC AUC:",
        metrics["ROC AUC"]
    )


    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return {

        "algorithm": display_name,

        "dataset_shape": df.shape,

        "feature_count": X.shape[1],

        "training_shape": X_train.shape,

        "testing_shape": X_test.shape,

        "Accuracy": metrics["Accuracy"],

        "Precision": metrics["Precision"],

        "Recall": metrics["Recall"],

        "F1 Score": metrics["F1 Score"],

        "ROC AUC": metrics["ROC AUC"],

        "model": model,

        "X_test": X_test,

        "y_test": y_test,

        "predictions": metrics["predictions"],

        "probabilities": metrics["probabilities"]

    }


# ============================================================
# CONFUSION MATRIX
# ============================================================

def create_confusion_matrix_plot(

    result,

    filename

):

    os.makedirs(

        PLOT_DIRECTORY,

        exist_ok=True

    )


    matrix = confusion_matrix(

        result["y_test"],

        result["predictions"]

    )


    display = ConfusionMatrixDisplay(

        confusion_matrix=matrix

    )


    display.plot()


    plt.title(

        f'{result["algorithm"]} - Confusion Matrix'

    )


    plt.tight_layout()


    path = os.path.join(

        PLOT_DIRECTORY,

        filename

    )


    plt.savefig(

        path,

        dpi=150

    )


    plt.close()


    return path


# ============================================================
# ROC CURVE
# ============================================================

def create_roc_curve_plot(

    result,

    filename

):

    os.makedirs(

        PLOT_DIRECTORY,

        exist_ok=True

    )


    fpr, tpr, _ = roc_curve(

        result["y_test"],

        result["probabilities"]

    )


    plt.figure(
        figsize=(8, 6)
    )


    plt.plot(

        fpr,

        tpr,

        label=(

            f'ROC AUC = '
            f'{result["ROC AUC"]:.4f}'

        )

    )


    plt.plot(

        [0, 1],

        [0, 1],

        linestyle="--"

    )


    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )


    plt.title(

        f'{result["algorithm"]} - ROC Curve'

    )


    plt.legend()


    plt.grid(
        alpha=0.2
    )


    plt.tight_layout()


    path = os.path.join(

        PLOT_DIRECTORY,

        filename

    )


    plt.savefig(

        path,

        dpi=150

    )


    plt.close()


    return path
```

## 5.12 `unsupervised.py`

```python
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from data_loader import load_data


BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PLOT_DIRECTORY = os.path.join(BASE_DIRECTORY, "static", "plots")
os.makedirs(PLOT_DIRECTORY, exist_ok=True)


def get_clustering_data():
    X = load_data().copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean(numeric_only=True))
    return X.astype(float)


def calculate_elbow(X):
    k_values = list(range(1, 11))
    wcss_values = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        wcss_values.append(model.inertia_)

    plt.figure(figsize=(9, 6))
    plt.plot(k_values, wcss_values, marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.xticks(k_values)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIRECTORY, "kmeans_elbow.png"))
    plt.close()

    return k_values, wcss_values


def calculate_silhouette(X):
    k_values = list(range(2, 11))
    scores = []
    sample_size = min(5000, len(X))

    for k in k_values:
        print(f"Calculating Silhouette Score for K = {k}")
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = model.fit_predict(X)
        score = silhouette_score(
            X,
            clusters,
            sample_size=sample_size,
            random_state=42
        )
        scores.append(score)

    best_index = int(np.argmax(scores))
    recommended_k = k_values[best_index]
    highest_score = scores[best_index]

    plt.figure(figsize=(9, 6))
    plt.plot(k_values, scores, marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score Method")
    plt.xticks(k_values)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIRECTORY, "kmeans_silhouette.png"))
    plt.close()

    return {
        "k_values": k_values,
        "scores": scores,
        "recommended_k": recommended_k,
        "highest_score": highest_score
    }


def run_kmeans(X, k):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = model.fit_predict(X)
    return model, clusters


def calculate_kmeans_performance(X, clusters, model):
    wcss = model.inertia_
    silhouette = silhouette_score(
        X,
        clusters,
        sample_size=min(5000, len(X)),
        random_state=42
    )
    return wcss, silhouette


def get_cluster_distribution(clusters):
    return pd.Series(clusters).value_counts().sort_index().to_dict()


def create_cluster_plot(X, clusters, model, x_feature, y_feature):
    plt.figure(figsize=(9, 6))

    plt.scatter(
        X[x_feature],
        X[y_feature],
        c=clusters,
        s=10,
        alpha=0.5
    )

    x_index = X.columns.get_loc(x_feature)
    y_index = X.columns.get_loc(y_feature)

    plt.scatter(
        model.cluster_centers_[:, x_index],
        model.cluster_centers_[:, y_index],
        marker="X",
        s=200
    )

    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.title("K-Means Clusters")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIRECTORY, "kmeans_clusters.png"))
    plt.close()

    return "kmeans_clusters.png"

```

## 5.13 `K_Calculation_K_Means.py`

```python
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
```

## 5.14 `k_mean_final.py`

```python
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
```

## 5.15 `k_means_performance.py`

```python
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
```
