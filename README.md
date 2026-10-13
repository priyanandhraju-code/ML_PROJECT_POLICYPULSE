# PolicyPulse

Insurance-interest classification, annual-premium regression and customer segmentation in a local Flask application.

## Run on Windows

From this project folder, use PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
.\.venv\Scripts\python.exe app.py
```

Open **http://127.0.0.1:5000**. Keep `Policypulse.csv` in the project folder. File paths are resolved relative to the application, so launching from another working directory also works. There is no manual preprocessing prerequisite.

`requirements.txt` installs the core application. `requirements-optional.txt` adds XGBoost and LightGBM; missing optional libraries do not prevent startup. The first model run takes longer because it trains on the full training set. Later visits reuse results in `.analysis-cache/`. Source or dataset changes invalidate disk results. Restart the app after Python source changes.

## Reference and preserved interface

The source of `D:\Project.zip` was compared with `D:\ML_PROJECT_13.zip`. The reference uses an employee-attrition dataset; its algorithms and workflow were adapted to PolicyPulse's insurance dataset. Employee-specific columns, labels and visual styling were not copied.

The interface uses the home page, sidebar, colors and K-Means page from **ML_PROJECT_13.zip**, plus the existing regression, classification and preprocessing page layouts. The older checkout's working Data Loading and EDA result panels were restored inside that interface. Sidebar placeholders now open working analyses.

| Reference capability | PolicyPulse implementation |
|---|---|
| Data loading and EDA | Dataset summary, missing values, distributions, relationships and saved charts |
| Preprocessing | Shared, reproducible split; training-only imputation, one-hot categories, premium IQR clipping and continuous scaling |
| Linear / Ridge / Lasso | Predict unscaled `Annual_Premium`; report MAE, MSE, RMSE and R² |
| Logistic / L2 / L1 | Predict `Response`; report accuracy, precision, recall, F1, ROC AUC and average precision |
| Entropy decision tree | `criterion="entropy"` in sklearn's binary decision tree, labeled ID3-style; not a claim of a literal multiway ID3 implementation |
| Tree ensembles | Random Forest, Extra Trees, Gradient Boosting, AdaBoost, XGBoost and LightGBM |
| K-Means | Manual K, elbow heuristic and silhouette selection; distribution, selected-feature and PCA plots |
| Hierarchical | Ward, complete or average linkage; dendrogram, plots and segment profiles |
| DBSCAN | Validated epsilon/minimum-samples settings; noise counts, plots and segment profiles |
| Evaluation | Classifier comparison, best observed model and final result tables in the existing sidebar sections |

## Modeling decisions

- The full dataset is split **80:20**, using seed **42** and stratification on `Response`. Every supervised model uses the same training/test customer indices.
- Missing or invalid target labels are rejected. Predictors are imputed using training medians/modes. Categories, premium fences and scaling are learned from training rows only. Unknown test categories map to the all-zero dummy representation.
- `id` and exported unnamed index columns are excluded. Region and sales channel are category codes. Only Age, Annual Premium and Vintage are standardized; binary indicators and dummy columns remain numeric indicators.
- Logistic and tree models predict insurance **interest**, not policy eligibility. Class/sample weights are learned from training labels to address imbalance. Metrics use a fixed 0.5 decision threshold; probability ranking is reported separately.
- Linear regression predicts **annual premium** in original units, adapting the reference's continuous income-regression target. Annual premium and Response are excluded from its predictors. Regression target values are not scaled or clipped. Regression metrics cannot be compared directly with classification accuracy/AUC.
- Classifiers are ranked by **average precision**, displayed as PR AUC. The UI includes majority-class accuracy and positive-prevalence baselines. Selecting the best observed result on this test split is exploratory; it is not an independent final deployment estimate.
- Clustering excludes Response and ID. Reproducible sample limits are **3,500** for K-Means, **1,200** for hierarchical clustering and **2,500** for DBSCAN, following the reference's bounded approach. These are displayed as samples in the interface; supervised training is not sampled.
- Silhouette calculation uses at most **1,500** sampled points. DBSCAN noise is excluded from silhouette; a single cluster, all-noise result or invalid sample has no defined silhouette. Segment response rates are calculated after clustering from aligned raw rows, never supplied as clustering features.
- Clustering uses standardized continuous features and unscaled binary/one-hot indicators. PCA is a visualization of the same model inputs. Hierarchical and DBSCAN metrics describe their sample, not the entire dataset.

## Bug fixes

- Restored blank Data Loading and EDA pages in the archived version.
- Removed dependence on a missing `preprocessed_data.csv` and repeated large CSV writes during page visits.
- Replaced fixed preprocessing quartiles, fences, counts and before/after statistics with live calculated values.
- Removed full-dataset imputation/category leakage and reused one training/test split.
- Corrected regression's binary target and removed its target from predictor inputs.
- Avoided dense float64 expansion of one-hot model inputs by using sparse matrices.
- Kept sparse premium regression in float64 so Lasso reaches its convergence criterion; classification uses float32 matrices.
- Loaded optional boosting libraries only when selected.
- Bounded expensive clustering work and handled invalid parameters and undefined silhouette scores.
- Used per-result cluster plot filenames and disabled stale browser plot caching.
- Serialized analytics operations to protect Matplotlib's shared state.
- Saved charts and model caches through unique temporary files and atomic replacement, so readers see completed artifacts.
- Replaced repeated training on refresh with dataset/source-aware local caching.
- Used a common sidebar and activated all existing placeholder analysis links.
- Disabled debug mode by default.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\verify_full_dataset.py
```

The automated suite fits real estimators on a deterministic 1,600-row sample. It checks training-only transformations, unknown categories, target exclusion, consistent splits, all tree models, regression, classification, every clustering workflow, real plot URLs, invalid inputs, page structure, caching and evaluation.

The full-dataset verifier exercises the actual CSV through Flask routes and writes `verification_results.json` with route timings, model metrics and train/test counts. Detailed output is saved under `.analysis-cache/verification.log`.

To explicitly export a preprocessed CSV for study, run `python preprocessing.py`. The web app does not need this export. The older `PolicyPulse_Project_Guide.md` is retained as supplied; its old source listings and regression-target description predate these changes.
