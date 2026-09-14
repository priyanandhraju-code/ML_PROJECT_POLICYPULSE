import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from Outlier_Fix import fix_outliers
from Standard_Scaler import standard_scale


# ============================================================
# CONFIGURATION
# ============================================================

FILE_PATH = r"Policypulse.csv"
OUTPUT_FILE = r"preprocessed_data.csv"
TRAIN_OUTPUT_FILE = r"preprocessed_train.csv"
TEST_OUTPUT_FILE = r"preprocessed_test.csv"

TARGET_COLUMN = "Response"
TEST_SIZE = 0.20
RANDOM_STATE = 42

CATEGORICAL_COLUMNS = [
    "Gender",
    "Vehicle_Age",
    "Vehicle_Damage",
    "Region_Code",
    "Policy_Sales_Channel"
]


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {FILE_PATH}"
        )

    df = pd.read_csv(FILE_PATH)

    if df.empty:
        raise ValueError("The dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

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
# TRAIN / TEST SPLIT
# ============================================================

def split_dataset(df):

    return train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN]
    )


# ============================================================
# HANDLE MISSING VALUES WITHOUT DATA LEAKAGE
# ============================================================

def handle_missing_values(train_df, test_df):
    """
    Learn all imputation values from training data only and
    apply the learned values to both training and testing data.
    """

    train_df = train_df.copy()
    test_df = test_df.copy()

    numerical_columns = train_df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = train_df.select_dtypes(
        include=["object"]
    ).columns

    for column in numerical_columns:

        if column == TARGET_COLUMN:
            continue

        if train_df[column].isnull().any():

            fill_value = train_df[column].median()

            train_df[column] = train_df[column].fillna(
                fill_value
            )

            test_df[column] = test_df[column].fillna(
                fill_value
            )

    for column in categorical_columns:

        if train_df[column].isnull().any():

            mode_values = train_df[column].mode()

            if not mode_values.empty:

                fill_value = mode_values.iloc[0]

                train_df[column] = train_df[column].fillna(
                    fill_value
                )

                test_df[column] = test_df[column].fillna(
                    fill_value
                )

    return train_df, test_df


# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

def encode_categorical_features(train_df, test_df):
    """
    Fit the feature representation from the training data.
    Test data is then aligned to exactly the same feature columns.
    """

    train_df = train_df.copy()
    test_df = test_df.copy()

    existing_columns = [
        column
        for column in CATEGORICAL_COLUMNS
        if column in train_df.columns
    ]

    print("\nCategorical columns being encoded:")

    for column in existing_columns:
        print("-", column)

    train_target = train_df[TARGET_COLUMN].copy()
    test_target = test_df[TARGET_COLUMN].copy()

    train_features = train_df.drop(
        columns=[TARGET_COLUMN]
    )

    test_features = test_df.drop(
        columns=[TARGET_COLUMN]
    )

    train_features = pd.get_dummies(
        train_features,
        columns=existing_columns,
        drop_first=True,
        dtype=int
    )

    test_features = pd.get_dummies(
        test_features,
        columns=existing_columns,
        drop_first=True,
        dtype=int
    )

    # Unknown categories in test are safely represented by the
    # training feature structure. A category seen only in test
    # does not create a new model feature.
    test_features = test_features.reindex(
        columns=train_features.columns,
        fill_value=0
    )

    train_df = train_features.copy()
    test_df = test_features.copy()

    train_df[TARGET_COLUMN] = train_target
    test_df[TARGET_COLUMN] = test_target

    ordered_columns = [
        column
        for column in train_df.columns
        if column != TARGET_COLUMN
    ] + [TARGET_COLUMN]

    train_df = train_df[ordered_columns]
    test_df = test_df[ordered_columns]

    return train_df, test_df


# ============================================================
# PREPROCESS DATA
# ============================================================

def preprocess_data():

    print("=" * 80)
    print("POLICYPULSE PREPROCESSING")
    print("=" * 80)

    # ========================================================
    # STEP 1: LOAD
    # ========================================================

    df = load_dataset()
    original_shape = df.shape

    print("\nOriginal dataset shape:")
    print(original_shape)

    # ========================================================
    # STEP 2: REMOVE ID
    # ========================================================

    df = remove_id(df)

    print("\nAfter removing ID:")
    print(df.shape)

    # ========================================================
    # STEP 3: TRAIN / TEST SPLIT
    # ========================================================
    # Split before learning preprocessing statistics so that
    # model-test data cannot influence preprocessing.
    # ========================================================

    train_df, test_df = split_dataset(df)

    print("\nTraining dataset shape:")
    print(train_df.shape)

    print("\nTesting dataset shape:")
    print(test_df.shape)

    # ========================================================
    # STEP 4: MISSING VALUE HANDLING
    # ========================================================

    train_df, test_df = handle_missing_values(
        train_df,
        test_df
    )

    # ========================================================
    # STEP 5: CATEGORICAL ENCODING
    # ========================================================

    train_df, test_df = encode_categorical_features(
        train_df,
        test_df
    )

    print("\nAfter categorical encoding:")
    print("Training:", train_df.shape)
    print("Testing:", test_df.shape)

    # ========================================================
    # STEP 6: IQR OUTLIER HANDLING
    # ========================================================

    train_df, test_df, iqr_info = fix_outliers(
        train_df,
        test_df,
        feature="Annual_Premium"
    )

    # ========================================================
    # STEP 7: STANDARD SCALING
    # ========================================================

    train_df, test_df, scaler = standard_scale(
        train_df,
        test_df
    )

    # ========================================================
    # STEP 8: FINAL TRAIN/TEST VALIDATION
    # ========================================================

    for name, frame in [
        ("training", train_df),
        ("testing", test_df)
    ]:

        missing = int(
            frame.isnull().sum().sum()
        )

        numeric_values = frame.select_dtypes(
            include=["number"]
        )

        non_finite = int(
            (~np.isfinite(numeric_values)).sum().sum()
        )

        if missing != 0:
            raise ValueError(
                f"Final {name} data contains {missing} missing values."
            )

        if non_finite != 0:
            raise ValueError(
                f"Final {name} data contains {non_finite} non-finite values."
            )

    # ========================================================
    # STEP 9: SAVE MODEL TRAINING / TESTING DATA SEPARATELY
    # ========================================================
    # These two files are the authoritative model inputs.
    # Keeping them separate prevents a second split later from
    # causing preprocessing leakage.
    # ========================================================

    train_df.to_csv(
        TRAIN_OUTPUT_FILE,
        index=False
    )

    test_df.to_csv(
        TEST_OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # STEP 10: CREATE COMBINED DISPLAY DATASET
    # ========================================================

    processed_df = pd.concat(
        [train_df, test_df]
    )

    # At this stage the original row index still identifies the
    # original observations, so restore the original row order.
    processed_df = processed_df.sort_index()

    processed_df = processed_df.reset_index(
        drop=True
    )

    processed_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # FINAL INFORMATION
    # ========================================================

    missing_total = int(
        processed_df.isnull().sum().sum()
    )

    numeric_values = processed_df.select_dtypes(
        include=["number"]
    )

    non_finite_total = int(
        (~np.isfinite(numeric_values)).sum().sum()
    )

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETED")
    print("=" * 80)

    print("\nOriginal dataset shape:")
    print(original_shape)

    print("\nFinal combined dataset shape:")
    print(processed_df.shape)

    print("\nFinal training shape:")
    print(train_df.shape)

    print("\nFinal testing shape:")
    print(test_df.shape)

    print("\nTotal missing values:")
    print(missing_total)

    print("\nTotal non-finite values:")
    print(non_finite_total)

    print("\nModel training data saved as:")
    print(TRAIN_OUTPUT_FILE)

    print("Model testing data saved as:")
    print(TEST_OUTPUT_FILE)

    print("\nCombined display dataset saved as:")
    print(OUTPUT_FILE)

    preprocessing_info = {
        "original_shape": original_shape,
        "after_id_removal_shape": df.shape,
        "training_shape": train_df.shape,
        "testing_shape": test_df.shape,
        "final_shape": processed_df.shape,
        "missing_values": missing_total,
        "non_finite_values": non_finite_total,
        "encoded_columns": [
            column
            for column in CATEGORICAL_COLUMNS
            if column in df.columns
        ],
        "scaled_columns": [
            "Age",
            "Annual_Premium",
            "Vintage"
        ],
        "binary_columns": [
            "Driving_License",
            "Previously_Insured"
        ],
        "target_column": TARGET_COLUMN,
        "iqr": iqr_info,
        "train_output_file": TRAIN_OUTPUT_FILE,
        "test_output_file": TEST_OUTPUT_FILE,
        "combined_output_file": OUTPUT_FILE
    }

    return processed_df, preprocessing_info


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    preprocess_data()
