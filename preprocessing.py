"""One training-only preprocessing workflow shared by every model."""
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from load_data import load_data
from project_runtime import BASE_DIR, DATA_PATH, ANALYSIS_LOCK, dataset_version

FILE_PATH = DATA_PATH
OUTPUT_FILE = BASE_DIR / "preprocessed_data.csv"
TARGET_COLUMN = "Response"
TEST_SIZE = 0.20
RANDOM_STATE = 42
NUMERIC_COLUMNS = ["Age", "Annual_Premium", "Vintage", "Driving_License", "Previously_Insured"]
CONTINUOUS_COLUMNS = ["Age", "Annual_Premium", "Vintage"]
BINARY_COLUMNS = ["Driving_License", "Previously_Insured"]
CATEGORICAL_COLUMNS = ["Gender", "Vehicle_Age", "Vehicle_Damage", "Region_Code", "Policy_Sales_Channel"]


def load_dataset():
    return load_data(FILE_PATH)


def remove_id(df):
    return df.drop(columns=["id"], errors="ignore").copy()


class PolicyPreprocessor:
    """Fit medians, categories, premium fences and scaling on training rows."""

    def _clean(self, frame):
        required = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise ValueError("Missing required features: " + ", ".join(missing))
        frame = frame[required].copy()
        for column in NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        for column in BINARY_COLUMNS:
            if not frame[column].dropna().isin([0, 1]).all():
                raise ValueError(f"{column} must contain only 0, 1 or missing values.")
        for column in CONTINUOUS_COLUMNS:
            if (frame[column].dropna() < 0).any():
                raise ValueError(f"{column} must not contain negative raw values.")
        for column in ["Region_Code", "Policy_Sales_Channel"]:
            values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            frame[column] = values.map(lambda value: str(float(value)) if pd.notna(value) else np.nan)
        for column in CATEGORICAL_COLUMNS:
            frame[column] = frame[column].astype(object).where(frame[column].notna(), np.nan)
        return frame

    def _impute(self, frame):
        frame = frame.copy()
        for column, value in self.fill_values_.items():
            frame[column] = frame[column].fillna(value)
        return frame

    def fit(self, frame):
        frame = self._clean(frame)
        self.fill_values_ = {}
        for column in NUMERIC_COLUMNS:
            if not frame[column].notna().any():
                raise ValueError(f"Training feature {column} contains no finite observations.")
            value = frame[column].mode().iloc[0] if column in BINARY_COLUMNS else frame[column].median()
            self.fill_values_[column] = float(value)
        for column in CATEGORICAL_COLUMNS:
            modes = frame[column].mode()
            self.fill_values_[column] = modes.iloc[0] if len(modes) else "Unknown"
        frame = self._impute(frame)
        self.categories_ = {column: sorted(frame[column].astype(str).unique()) for column in CATEGORICAL_COLUMNS}
        self.q1_, self.q3_ = frame["Annual_Premium"].quantile([0.25, 0.75])
        self.iqr_ = self.q3_ - self.q1_
        self.lower_ = self.q1_ - 1.5 * self.iqr_
        self.upper_ = self.q3_ + 1.5 * self.iqr_
        frame["Annual_Premium"] = frame["Annual_Premium"].clip(self.lower_, self.upper_)
        self.scaler_ = StandardScaler().fit(frame[CONTINUOUS_COLUMNS])
        return self

    def transform(self, frame):
        frame = self._impute(self._clean(frame))
        frame["Annual_Premium"] = frame["Annual_Premium"].clip(self.lower_, self.upper_)
        frame[CONTINUOUS_COLUMNS] = self.scaler_.transform(frame[CONTINUOUS_COLUMNS])
        for column in CATEGORICAL_COLUMNS:
            values = frame[column].astype(str)
            # Convert unknown values to missing before constructing a Categorical
            # (also compatible with pandas' stricter future validation).
            values = values.where(values.isin(self.categories_[column]))
            frame[column] = pd.Categorical(values, categories=self.categories_[column])
        return pd.get_dummies(frame, columns=CATEGORICAL_COLUMNS, drop_first=True, dtype=np.uint8)


def _premium_statistics(values):
    return {key: float(value) for key, value in values.describe().items()}


def prepare_dataset(df):
    """Split raw rows first; never impute missing or invalid target labels."""
    if not df.index.is_unique or df.columns.has_duplicates:
        raise ValueError("Dataset rows and column names must have unique indices for label alignment.")
    df = remove_id(df).drop(columns=[c for c in df if c.startswith("Unnamed:")])
    required = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("Response must contain only non-missing binary labels (0 or 1).")
    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("Response needs both classes with at least two rows per class.")
    y = y.astype(np.uint8)
    train, test, y_train, y_test = train_test_split(
        df.drop(columns=TARGET_COLUMN), y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y,
    )
    processor = PolicyPreprocessor().fit(train)
    clean_train = processor._impute(processor._clean(train))["Annual_Premium"]
    clean_test = processor._impute(processor._clean(test))["Annual_Premium"]
    summary = {
        "original_rows": len(df), "training_rows": len(train), "testing_rows": len(test),
        "q1": processor.q1_, "q3": processor.q3_, "iqr": processor.iqr_,
        "lower": processor.lower_, "upper": processor.upper_,
        "training_outliers": int(((clean_train < processor.lower_) | (clean_train > processor.upper_)).sum()),
        "testing_outliers": int(((clean_test < processor.lower_) | (clean_test > processor.upper_)).sum()),
        "statistics": {
            "training": {"before": _premium_statistics(clean_train), "after": _premium_statistics(clean_train.clip(processor.lower_, processor.upper_))},
            "testing": {"before": _premium_statistics(clean_test), "after": _premium_statistics(clean_test.clip(processor.lower_, processor.upper_))},
        },
    }
    return processor.transform(train), processor.transform(test), y_train, y_test, processor, summary


@lru_cache(maxsize=1)
def _prepared_dataset(version):
    return prepare_dataset(load_dataset())


def get_prepared_dataset():
    with ANALYSIS_LOCK:
        return _prepared_dataset(dataset_version())


def get_model_data():
    return get_prepared_dataset()[:4]


def preprocess_data_summary():
    train, test, _, _, _, summary = get_prepared_dataset()
    return SimpleNamespace(shape=(len(train) + len(test), train.shape[1] + 1), attrs={"summary": summary})


@lru_cache(maxsize=1)
def _processed_frame(version):
    train, test, y_train, y_test, _, summary = get_prepared_dataset()
    result = pd.concat([train.assign(Response=y_train), test.assign(Response=y_test)]).sort_index()
    result.attrs["summary"] = summary
    return result


def preprocess_data(export=False):
    with ANALYSIS_LOCK:
        result = _processed_frame(dataset_version())
        if export:
            result.to_csv(OUTPUT_FILE, index=False)
        return result


if __name__ == "__main__":
    result = preprocess_data(export=True)
    print(f"Prepared {result.shape[0]:,} rows; exported to {OUTPUT_FILE}")
