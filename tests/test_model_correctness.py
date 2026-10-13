"""Independent examples and failure cases for model output correctness."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from modeling import classification_metrics, matrix_for_model, model_matrix
from linear_regression import PremiumRegressor
from preprocessing import PolicyPreprocessor, prepare_dataset
from unsupervised import run_kmeans, _safe_silhouette


class FixedClassifier:
    classes_ = np.array([0, 1])

    def predict(self, X):
        return np.array([0, 0, 0, 1])

    def predict_proba(self, X):
        probability = np.array([0.1, 0.4, 0.4, 0.9])
        return np.column_stack([1 - probability, probability])


def test_metrics_match_hand_calculation_with_ties():
    result = classification_metrics(FixedClassifier(), np.zeros((4, 1)), np.array([0, 1, 0, 1]))
    assert result["Accuracy"] == 0.75
    assert result["Precision"] == 1
    assert result["Recall"] == 0.5
    assert result["F1 Score"] == pytest.approx(2 / 3)
    assert result["ROC AUC"] == 0.875
    assert result["Average Precision"] == pytest.approx(5 / 6)
    assert result["PR AUC"] == pytest.approx(11 / 12)
    assert result["Average Precision"] != result["PR AUC"]
    assert result["Log Loss"] == pytest.approx(-np.log([0.9, 0.4, 0.6, 0.9]).mean())
    assert result["Brier Score"] == pytest.approx(np.mean(np.array([0.1, -0.6, 0.4, -0.1]) ** 2))
    assert result["confusion_matrix"] == [[2, 0], [1, 1]]


def test_positive_class_column_is_found_by_label():
    class ReversedClassifier(FixedClassifier):
        classes_ = np.array([1, 0])

        def predict_proba(self, X):
            return super().predict_proba(X)[:, ::-1]

    result = classification_metrics(ReversedClassifier(), np.zeros((4, 1)), [0, 1, 0, 1])
    assert result["ROC AUC"] == 0.875


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -0.1, 1.1])
def test_invalid_probabilities_are_rejected(invalid):
    class InvalidClassifier(FixedClassifier):
        def predict_proba(self, X):
            values = super().predict_proba(X)
            values[0, 1] = invalid
            return values
    with pytest.raises(ValueError, match="probabilities"):
        classification_metrics(InvalidClassifier(), np.zeros((4, 1)), [0, 1, 0, 1])


def test_probability_sum_and_alignment_are_checked():
    class WrongSum(FixedClassifier):
        def predict_proba(self, X):
            return super().predict_proba(X) * 0.5
    with pytest.raises(ValueError, match="sum to 1"):
        classification_metrics(WrongSum(), np.zeros((4, 1)), [0, 1, 0, 1])
    with pytest.raises(ValueError, match="row counts"):
        classification_metrics(FixedClassifier(), np.zeros((3, 1)), [0, 1, 0])


def frame():
    return pd.DataFrame({"Age": [20, 30, 40, 50], "Annual_Premium": [100, 200, 300, 400],
                         "Vintage": [10, 20, 30, 40], "Driving_License": [0, 0, 1, 1],
                         "Previously_Insured": [0, 0, 1, 1], "Gender": ["Male", "Female"] * 2,
                         "Vehicle_Age": ["< 1 Year", "1-2 Year"] * 2, "Vehicle_Damage": ["Yes", "No"] * 2,
                         "Region_Code": [1, 2, 1, 2], "Policy_Sales_Channel": [1, 2, 1, 2]})


def test_binary_imputation_does_not_invent_fractional_flags():
    training = frame()
    processor = PolicyPreprocessor().fit(training)
    probe = training.iloc[[0]].copy()
    probe[["Driving_License", "Previously_Insured"]] = np.nan
    transformed = processor.transform(probe)
    assert transformed.Driving_License.iloc[0] in [0, 1]
    assert transformed.Previously_Insured.iloc[0] in [0, 1]


def test_unlearnable_or_invalid_training_data_is_rejected():
    training = frame()
    training.Age = np.nan
    with pytest.raises(ValueError, match="no finite observations"):
        PolicyPreprocessor().fit(training)
    training = frame()
    training.loc[0, "Driving_License"] = 2
    with pytest.raises(ValueError, match="only 0, 1"):
        PolicyPreprocessor().fit(training)
    training = frame().assign(Response=[0, 1, 0, 1])
    training.index = [0, 0, 1, 1]
    with pytest.raises(ValueError, match="unique indices"):
        prepare_dataset(training)


def test_regression_serves_nonnegative_predictions():
    model = PremiumRegressor(LinearRegression()).fit(np.array([[1], [2], [3]]), np.array([10, 20, 30]))
    assert model.estimator_.predict([[-1]])[0] < 0
    assert model.predict([[-1]])[0] == 0


def test_xgboost_preserves_zeros_as_values():
    xgboost = pytest.importorskip("xgboost")
    features = pd.DataFrame({"flag": [0, 1, 0], "amount": [0, 100, 200]})
    matrix = matrix_for_model(xgboost.XGBClassifier(), features)
    assert isinstance(matrix, np.ndarray)
    np.testing.assert_array_equal(matrix, features.to_numpy())
    assert np.count_nonzero(matrix == 0) == 3


def test_matrix_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        model_matrix(pd.DataFrame({"a": [np.inf]}))


def test_kmeans_inertia_uses_every_feature():
    features = pd.DataFrame({"x": [0, 0, 1, 1], "y": [0, 1, 0, 1], "third": [0, 0, 100, 100]})
    model, labels = run_kmeans(features, 2)
    assert model.n_features_in_ == 3
    independent_wcss = np.sum((features.to_numpy() - model.cluster_centers_[labels]) ** 2)
    assert model.inertia_ == pytest.approx(independent_wcss, rel=1e-12)
    assert labels[0] == labels[1] and labels[2] == labels[3]
    assert _safe_silhouette(features, np.full(4, -1)) is None
    assert _safe_silhouette(features, np.zeros(4)) is None


def test_clustering_preprocessing_is_independent_of_response(monkeypatch):
    import unsupervised
    first = frame().assign(id=[1, 2, 3, 4], Response=[0, 0, 1, 1])
    monkeypatch.setattr(unsupervised, "load_data", lambda: first)
    monkeypatch.setattr(unsupervised, "preprocess_data", lambda: pytest.fail("Clustering used classification preprocessing"), raising=False)
    original = unsupervised.get_clustering_data()
    monkeypatch.setattr(unsupervised, "load_data", lambda: first.assign(Response=[1, 1, 0, 0]))
    permuted = unsupervised.get_clustering_data()
    assert len(original) == 4
    pd.testing.assert_frame_equal(original, permuted)
    assert "Response" not in original and "id" not in original


@pytest.mark.parametrize("method", ["ward", "complete", "average"])
def test_dendrogram_cutoff_represents_requested_cluster_count(method):
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.metrics import adjusted_rand_score
    from sklearn.cluster import AgglomerativeClustering
    from unsupervised import dendrogram_cutoff
    X = np.random.default_rng(193).normal(size=(30, 5))
    merges = linkage(X, method=method)
    for k in [2, 3, 7, 10]:
        cutoff, tied = dendrogram_cutoff(merges[:, 2], k)
        assert not tied
        tree_labels = fcluster(merges, cutoff, criterion="distance")
        model_labels = AgglomerativeClustering(n_clusters=k, linkage=method).fit_predict(X)
        assert len(set(tree_labels)) == k
        assert adjusted_rand_score(tree_labels, model_labels) == 1


def test_tied_dendrogram_cutoff_is_disclosed():
    from unsupervised import dendrogram_cutoff
    cutoff, tied = dendrogram_cutoff(np.array([1, 2, 2, 3]), 3)
    assert cutoff == 2 and tied
