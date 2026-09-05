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