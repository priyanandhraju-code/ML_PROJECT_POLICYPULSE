# PolicyPulse model correctness audit

Completed on 13 October 2026 using the supplied 381,109-customer dataset. The interface theme and layout were preserved. Evaluation navigation still contains only Best Model.

## Verification outcome

- 53 automated tests passed.
- All 20 full-dataset Flask workflows passed.
- Independent checks passed for 11 original classifiers, 3 regressors, 8 clustering configurations and the calibrated selected classifier.
- Classification metrics, average precision, ROC AUC, log loss and Brier score were checked with separate NumPy formulas, including tied scores and confusion-matrix counts.
- Regression errors were recalculated from served predictions. WCSS, silhouette, Davies–Bouldin and Calinski–Harabasz values were independently checked.

## Corrections made

1. Model selection now uses a disjoint validation split within the training pool, rather than picking the winner by test score. Each selection split fits its own preprocessing.
2. Average precision and trapezoidal precision–recall AUC are separate calculations. The interface names the ranking metric Average Precision.
3. XGBoost receives dense input to preserve actual zero values rather than treating absent sparse entries as missing.
4. The selected classifier receives sigmoid probability calibration and F1 threshold tuning using nested training folds. Preprocessing is fitted independently within every fold; test labels are excluded from selection, calibration and threshold fitting.
5. Premium predictions explicitly floor negative raw estimates at zero. Plots and metrics use these served predictions; regression solver tolerances were tightened.
6. Binary imputation uses training modes rather than a median that could invent a 0.5 flag. Invalid binary values, negative raw numeric values, unlearnable all-missing numeric features, duplicate indices and invalid model outputs are rejected.
7. Clustering preprocessing now samples raw predictors and never depends on Response or the label-stratified classification split. All varying encoded predictor columns are supplied to clustering; plot axes only control display.
8. Cached results verify chart content, not merely file existence. Source fingerprints describe the code loaded by the process, preventing an older running server from signing outputs with newer source files. Library versions also enter cache signatures.

## Selected classifier

XGBoost was selected by validation average precision 0.368555. Candidate fitting rows: 228,665; validation rows: 76,222; final held-out test rows: 76,222. All row sets are disjoint.

| Test measure | Original class-balanced model | Calibrated model with training-selected threshold |
|---|---:|---:|
| Accuracy | 0.704206 | 0.781520 |
| Precision | 0.283711 | 0.328934 |
| Recall | 0.926996 | 0.752408 |
| F1 Score | 0.434455 | 0.457751 |
| ROC AUC | 0.857518 | 0.857518 |
| Average Precision | 0.368136 | 0.368136 |
| Log Loss | 0.460814 | 0.267650 |
| Brier Score | 0.171254 | 0.086778 |

The calibrated decision threshold is 0.239911; it was not chosen from test labels. Sigmoid calibration preserves ranking here, while reducing probability losses. The threshold improves F1 and precision with a corresponding reduction in recall.

## All original classifier outputs

These are the original class-balanced models at their default thresholds. Their probability scores are not calibrated; calibration is provided separately for the selected model. All calculations passed.

| Model | ROC AUC | Average precision | F1 | Log loss | Brier score |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8497 | 0.3378 | 0.4300 | 0.4722 | 0.1758 |
| Ridge Logistic Regression | 0.8498 | 0.3374 | 0.4299 | 0.4723 | 0.1758 |
| Lasso Logistic Regression | 0.8498 | 0.3369 | 0.4296 | 0.4724 | 0.1761 |
| Decision Tree (Gini) | 0.8442 | 0.3469 | 0.4319 | 0.5218 | 0.1749 |
| Decision Tree (Entropy / ID3-style) | 0.8460 | 0.3455 | 0.4306 | 0.5266 | 0.1751 |
| Random Forest | 0.8520 | 0.3530 | 0.4321 | 0.4780 | 0.1705 |
| Extra Trees | 0.8498 | 0.3498 | 0.4316 | 0.4816 | 0.1725 |
| Gradient Boosting | 0.8556 | 0.3622 | 0.4299 | 0.4673 | 0.1736 |
| AdaBoost | 0.8412 | 0.3156 | 0.4091 | 0.4861 | 0.1605 |
| XGBoost | 0.8575 | 0.3681 | 0.4345 | 0.4608 | 0.1713 |
| LightGBM | 0.8575 | 0.3686 | 0.4349 | 0.4612 | 0.1713 |

The prevalence-only probability baseline has log loss 0.372001 and Brier score 0.107541. Original weighted scores have larger probability losses than this baseline despite useful ranking; calibrated selected-model probabilities improve on both baselines.

## Premium regression outputs

| Model | MAE | RMSE | R² | Negative served predictions |
|---|---:|---:|---:|---:|
| Linear Regression | 9,852.84 | 14,800.06 | 0.251593 | 0 |
| Ridge Regression | 9,860.06 | 14,803.47 | 0.251248 | 0 |
| Lasso Regression | 9,970.88 | 14,867.36 | 0.244771 | 0 |

Training-median baseline MAE: 11,933.10. Training-mean baseline RMSE: 17,108.34. All three models improve on these baselines, but R² near 0.25 indicates substantial remaining estimation error. R² is not classification accuracy.

## Clustering outputs

| Algorithm / configuration | Sample rows | Encoded predictors | Silhouette | Independent checks |
|---|---:|---:|---:|---|
| kmeans / manual | 3500 | 131 | 0.196572 | Passed |
| kmeans / elbow | 3500 | 131 | 0.176240 | Passed |
| kmeans / silhouette | 3500 | 131 | 0.196572 | Passed |
| hierarchical / ward | 1200 | 108 | 0.189254 | Passed |
| hierarchical / complete | 1200 | 108 | 0.114657 | Passed |
| hierarchical / average | 1200 | 108 | 0.143911 | Passed |
| dbscan / epsilon 0.8 | 2500 | 125 | 0.365596 | Passed |
| dbscan / epsilon 1e-05 | 2500 | 125 | Not defined | Passed |

Silhouette excludes DBSCAN noise and uses at most 1,500 reproducibly sampled assigned points. An undefined silhouette for all-noise or a single-cluster result is handled explicitly, not replaced with a fabricated score. Cluster statistics describe sampled customers, not full-dataset predictive accuracy.

## Practical limits

Always predicting no interest achieves about 87.74% accuracy but zero recall for interested customers. The selected model has about 75.24% positive recall and 32.89% precision at its training-selected F1 threshold. Its lower overall accuracy reflects this precision/recall tradeoff; accuracy alone is misleading for this imbalanced problem.

The audit establishes computational correctness for this dataset and these tested configurations. Prediction error is still present: the selected model has false positives and false negatives, and premium regression has substantial residual error. Small nonzero loss is normal; a calculation error, leakage, mislabeling or invalid prediction requires correction. No zero-error or production-performance guarantee is implied.

Full precision results and the dataset SHA-256 fingerprint are stored in `model_audit_results.json`; the before-correction measurements are retained in `model_audit_before.json`. Displayed table values are rounded. Numerical agreement used relative tolerances near 1e-10, with explicit floating-point tolerances for cluster distances.

## Primary references

- [scikit-learn: separate validation and test data](https://scikit-learn.org/stable/modules/cross_validation.html).
- [scikit-learn: probability calibration](https://scikit-learn.org/stable/modules/calibration.html).
- [XGBoost: sparse versus dense input semantics](https://xgboost.readthedocs.io/en/stable/faq.html#why-do-i-see-different-results-with-sparse-and-dense-data).
