"""Clustering features are prepared on demand, without a manual CSV step."""
from preprocessing import preprocess_data


def load_data():
    return preprocess_data().drop(columns="Response")
