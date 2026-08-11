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