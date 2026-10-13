"""Real fitting and rendering checks, using a deterministic dataset sample."""
import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import app as web
import preprocessing as prep
import project_runtime as runtime
import modeling
import linear_regression as linear
import logistic_regression as logistic
import tree_based as trees
import unsupervised as clustering
import policypulse_eda as eda
import evaluation
import model_selection


@pytest.fixture(scope="module")
def application(tmp_path_factory):
    folder = tmp_path_factory.mktemp("policypulse")
    data = pd.read_csv(runtime.DATA_PATH).sample(n=1600, random_state=42).reset_index(drop=True)
    csv = folder / "sample.csv"
    data.to_csv(csv, index=False)
    patch = pytest.MonkeyPatch()
    patch.setattr(runtime, "DATA_PATH", csv)
    patch.setattr(runtime, "BASE_DIR", folder)
    patch.setattr(prep, "load_dataset", lambda: data.copy())
    patch.setattr(linear, "load_data", lambda: data.copy())
    patch.setattr(clustering, "load_data", lambda: data.copy())
    patch.setattr(eda, "CSV_PATH", csv)
    plots = folder / "static" / "plots"
    plots.mkdir(parents=True)
    patch.setattr(modeling, "PLOT_DIR", plots)
    patch.setattr(runtime, "PLOT_DIR", plots)
    patch.setattr(eda, "PLOT_DIR", plots)
    patch.setattr(web.app, "static_folder", str(folder / "static"))
    prep._prepared_dataset.cache_clear()
    prep._processed_frame.cache_clear()
    for function in [linear.run_linear_regression, logistic.run_logistic_regression, trees.run_tree_algorithm,
                     clustering.run_clustering, eda.run_eda, evaluation.run_comparison,
                     model_selection.select_classifier, model_selection.calibrated_selected_model]:
        function.cache_clear()
    web.app.config.update(TESTING=True)
    yield web.app.test_client(), data
    prep._prepared_dataset.cache_clear()
    prep._processed_frame.cache_clear()
    patch.undo()


def assert_rendered(response, marker):
    assert response.status_code == 200
    assert marker in response.get_data(as_text=True)
    assert '<div class="error">' not in response.get_data(as_text=True)


def test_training_only_transform_and_unseen_categories(application):
    _, data = application
    training = data.iloc[:100].copy()
    processor = prep.PolicyPreprocessor().fit(training)
    probe = data.iloc[[101]].copy()
    probe["Age"] = np.nan
    probe["Gender"] = "New category"
    probe["Annual_Premium"] = 1e10
    before = processor.scaler_.mean_.copy()
    transformed = processor.transform(probe)
    assert transformed.columns.tolist() == processor.transform(training).columns.tolist()
    assert np.isfinite(transformed.to_numpy()).all()
    assert not transformed.filter(like="Gender_").to_numpy().any()
    np.testing.assert_array_equal(before, processor.scaler_.mean_)
    age = processor.scaler_.inverse_transform(transformed[prep.CONTINUOUS_COLUMNS])[0, 0]
    assert age == pytest.approx(training.Age.median())
    premium = processor.scaler_.inverse_transform(transformed[prep.CONTINUOUS_COLUMNS])[0, 1]
    assert premium == pytest.approx(processor.upper_)


def test_split_and_target_validation(application):
    _, data = application
    train, test, y_train, y_test, _, summary = prep.get_prepared_dataset()
    assert set(train.index).isdisjoint(test.index)
    assert len(train) + len(test) == len(data)
    assert train.columns.tolist() == test.columns.tolist()
    assert "id" not in train and "Response" not in train
    assert np.isfinite(train.to_numpy()).all()
    assert y_test.mean() == pytest.approx(y_train.mean(), abs=0.01)
    assert summary["training_rows"] == len(train)
    invalid = data.copy()
    invalid.loc[0, "Response"] = np.nan
    with pytest.raises(ValueError, match="binary labels"):
        prep.prepare_dataset(invalid)
    with pytest.raises(ValueError, match="Missing required"):
        prep.prepare_dataset(data.drop(columns="Age"))


@pytest.mark.parametrize("path,marker", [
    ("/", "Welcome to PolicyPulse"), ("/data-loading", "Column Information"),
    ("/preprocessing", "Preprocessing Completed"), ("/unsupervised", "Select K-Means Parameters"),
    ("/hierarchical", "Clustering Parameters"), ("/dbscan", "Clustering Parameters"),
    ("/model-comparison", "Evaluate Models"), ("/best-model", "Best Model"), ("/final-results", "Final Results"),
])
def test_page_forms(application, path, marker):
    client, _ = application
    response = client.get(path)
    assert_rendered(response, marker)
    html = response.get_data(as_text=True)
    assert html.count('<aside class="sidebar">') == 1
    assert html.count('<main ') == 1
    if path == "/":
        assert html.count("Welcome to PolicyPulse") == 1


def test_regression_predicts_premium_without_target_leakage(application):
    client, _ = application
    train, test, y_train, y_test = linear.regression_data()
    assert "Annual_Premium" not in train and "Response" not in train
    assert y_test.mean() > 1000
    assert_rendered(client.get("/linear-regression"), "Predict Annual_Premium")
    result = linear.run_linear_regression()
    assert len(result["comparison"]) == 3
    assert all(np.isfinite(row["RMSE"]) for row in result["comparison"])


def test_logistic_models_and_cached_refresh(application, monkeypatch):
    client, _ = application
    assert_rendered(client.get("/logistic-regression"), "ROC AUC")
    result = logistic.run_logistic_regression()
    assert len(result["comparison"]) == 3
    for row in result["comparison"]:
        assert 0 <= row["PR AUC"] <= 1
    lasso = result["models"]["Lasso Logistic Regression"]
    assert (lasso.coef_ == 0).any()
    monkeypatch.setattr(logistic.LogisticRegression, "fit", lambda *args, **kwargs: pytest.fail("Refresh retrained the model"))
    assert logistic.run_logistic_regression() is result


@pytest.mark.parametrize("algorithm", list(trees.ALGORITHMS))
def test_all_tree_algorithms(application, algorithm):
    client, _ = application
    if algorithm in {"xgboost", "lightgbm"} and importlib.util.find_spec(algorithm) is None:
        pytest.skip("Optional package not installed")
    response = client.post("/tree-based", data={"algorithm": algorithm})
    assert_rendered(response, "ROC AUC")
    result = trees.run_tree_algorithm(algorithm)
    assert 0 <= result["ROC AUC"] <= 1
    if algorithm == "id3":
        assert result["model"].criterion == "entropy"


@pytest.mark.parametrize("algorithm,extra", [
    ("unsupervised", {"method": "manual", "k": "3"}),
    ("unsupervised", {"method": "elbow"}),
    ("unsupervised", {"method": "silhouette"}),
    ("hierarchical", {"linkage": "average", "k": "3"}),
    ("dbscan", {"eps": "0.8", "min_samples": "10"}),
    ("dbscan", {"eps": "0.00001", "min_samples": "100"}),
])
def test_clustering_and_real_plot_urls(application, algorithm, extra):
    client, _ = application
    response = client.post("/" + algorithm, data={"x_feature": "Annual_Premium", "y_feature": "Vintage", **extra})
    assert_rendered(response, "Customer Segment Profiles")
    html = response.get_data(as_text=True)
    assert 'name="x_feature"' not in html and 'name="y_feature"' not in html
    assert "Customer Clusters in 2D PCA Space" in html
    assert "Population %" in html and "Vehicle Damage %" in html
    for url in re.findall(r'src="([^" ]+\.png(?:\?[^" ]*)?)"', response.get_data(as_text=True)):
        assert client.get(url).status_code == 200, url
    if algorithm == "dbscan" and extra["eps"] == "0.00001":
        assert "Not defined" in response.get_data(as_text=True)


@pytest.mark.parametrize("algorithm,fields", [
    ("kmeans", {"method": "elbow"}),
    ("hierarchical", {"linkage_method": "ward"}),
    ("dbscan", {"eps": 0.8, "min_samples": 10}),
    ("dbscan", {"eps": 0.00001, "min_samples": 100}),
])
def test_clustering_profiles_and_pca_metadata_match_real_rows(application, algorithm, fields):
    _, data = application
    result = clustering.run_clustering(algorithm=algorithm, **fields)
    raw = data.loc[result["sample_indices"]].assign(Cluster=result["labels"])
    assert sum(row["count"] for row in result["profiles"]) == result["rows"]
    assert sum(row["population_percentage"] for row in result["profiles"]) == pytest.approx(100)
    assert result["noise_percentage"] == pytest.approx(100 * np.mean(result["labels"] == -1))
    for row in result["profiles"]:
        group = raw.loc[raw.Cluster == row["cluster"]]
        assert row["mean_vintage"] == pytest.approx(group.Vintage.mean())
        assert row["previously_insured_rate"] == pytest.approx(group.Previously_Insured.mean() * 100)
        assert row["vehicle_damage_rate"] == pytest.approx(group.Vehicle_Damage.eq("Yes").mean() * 100)
    X = clustering.get_clustering_data(clustering.SAMPLE_LIMITS[algorithm])
    variance = np.linalg.svd(X.to_numpy() - X.to_numpy().mean(axis=0), compute_uv=False) ** 2
    np.testing.assert_allclose(result["pca_variance"], variance[:2] / variance.sum(), rtol=1e-10)
    assert result["model"].n_features_in_ == X.shape[1] > 2
    if algorithm == "kmeans":
        analysis = result["silhouette_analysis"]
        assert result["best_silhouette_k"] == analysis["k_values"][np.argmax(analysis["scores"])]
        assert result["k"] == result["recommended_k"] == result["elbow_analysis"]["recommended_k"]


@pytest.mark.parametrize("path,fields,message", [
    ("/unsupervised", {"method": "manual", "k": "99"}, "K must be"),
    ("/unsupervised", {"method": "invalid"}, "Invalid K selection"),
    ("/unsupervised", {"x_feature": "Response"}, "Choose available"),
    ("/hierarchical", {"linkage": "invalid"}, "Choose ward"),
    ("/dbscan", {"eps": "nan"}, "Epsilon must"),
    ("/dbscan", {"min_samples": "1"}, "Minimum samples"),
    ("/tree-based", {"algorithm": "invalid"}, "Unknown tree algorithm"),
])
def test_invalid_input_is_visible_without_server_error(application, path, fields, message):
    client, _ = application
    response = client.post(path, data=fields)
    assert response.status_code == 200
    assert message in response.get_data(as_text=True)


def test_evaluation_and_eda(application):
    client, _ = application
    assert_rendered(client.post("/best-model"), "Validation-Selected Classifier")
    result = evaluation.run_comparison()
    assert result["best"]["Model"] == result["selection"]["best"]["Model"]
    assert result["selection"]["best"]["Average Precision"] == max(row["Average Precision"] for row in result["selection"]["comparison"])
    assert result["best"]["calibrated"]
    assert_rendered(client.get("/eda"), "Response")


def test_missing_dataset_and_working_directory(application, tmp_path, monkeypatch):
    from load_data import load_data
    monkeypatch.chdir(tmp_path)
    assert len(load_data()) > 0
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "missing.csv")


def test_disk_cache_recovers_missing_plots(application):
    calls = []

    @runtime.cached_analysis
    def example_analysis():
        calls.append(1)
        plot = runtime.PLOT_DIR / "cache_recovery.png"
        plot.write_bytes(b"example")
        return {"cluster_plot": plot.name}

    example_analysis()
    assert len(calls) == 1
    example_analysis.cache_clear()
    example_analysis()
    assert len(calls) == 1
    (runtime.PLOT_DIR / "cache_recovery.png").unlink()
    example_analysis.cache_clear()
    example_analysis()
    assert len(calls) == 2


def test_selection_and_test_customers_are_disjoint(application):
    fit, validation, y_fit, y_validation = model_selection.selection_split()
    _, test, _, _ = prep.get_model_data()
    assert set(fit.index).isdisjoint(validation.index)
    assert set(fit.index).isdisjoint(test.index)
    assert set(validation.index).isdisjoint(test.index)
    assert fit.index.equals(y_fit.index)
    assert validation.index.equals(y_validation.index)


def test_changed_plot_invalidates_cached_results(application):
    calls = []
    @runtime.cached_analysis
    def picture_analysis():
        calls.append(1)
        (runtime.PLOT_DIR / "manifest.png").write_bytes(b"correct chart")
        return {"cluster_plot": "manifest.png"}
    picture_analysis()
    (runtime.PLOT_DIR / "manifest.png").write_bytes(b"chart from another result")
    picture_analysis()
    assert len(calls) == 2
    assert (runtime.PLOT_DIR / "manifest.png").read_bytes() == b"correct chart"
