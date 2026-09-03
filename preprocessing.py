import os
import pandas as pd

from sklearn.model_selection import train_test_split

from Outlier_Fix import fix_outliers
from Standard_Scaler import standard_scale


# ============================================================
# DATASET PATH
# ============================================================

file_path = r"Policypulse.csv"

output_file = r"preprocessed_data.csv"


# ============================================================
# TARGET VARIABLE
# ============================================================

target_col = "Response"


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    return df


# ============================================================
# REMOVE ID COLUMN
# ============================================================

def remove_id(df):

    df = df.copy()

    if "id" in df.columns:

        df = df.drop(
            columns=["id"]
        )

    return df


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(df):

    df = df.copy()

    # Numerical columns
    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    # Categorical columns
    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    # Fill numerical missing values with median
    for column in numerical_columns:

        if df[column].isnull().sum() > 0:

            df[column] = df[column].fillna(
                df[column].median()
            )

    # Fill categorical missing values with mode
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

    categorical_columns = [
        "Gender",
        "Vehicle_Age",
        "Vehicle_Damage"
    ]

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True,
        dtype=int
    )

    return df


# ============================================================
# COMPLETE PREPROCESSING
# ============================================================

def preprocess_data():

    print("=" * 80)
    print("POLICYPULSE PREPROCESSING")
    print("=" * 80)

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    df = load_dataset()

    print("\nOriginal dataset shape:")

    print(df.shape)

    # ========================================================
    # 2. REMOVE ID
    # ========================================================

    df = remove_id(df)

    print("\nAfter removing ID:")

    print(df.shape)

    # ========================================================
    # 3. HANDLE MISSING VALUES
    # ========================================================

    df = handle_missing_values(df)

    print("\nTotal missing values after handling:")

    print(
        df.isnull().sum().sum()
    )

    # ========================================================
    # 4. ENCODE CATEGORICAL FEATURES
    # ========================================================

    df = encode_categorical_features(df)

    print("\nAfter categorical encoding:")

    print(df.shape)

    # ========================================================
    # 5. TRAIN-TEST SPLIT
    # ========================================================

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[target_col]
    )

    print("\nTraining dataset shape:")

    print(train_df.shape)

    print("\nTesting dataset shape:")

    print(test_df.shape)

    # ========================================================
    # 6. OUTLIER FIX
    # ========================================================

    train_df = fix_outliers(
        train_df,
        feature="Annual_Premium"
    )

    # ========================================================
    # 7. STANDARD SCALING
    # ========================================================

    train_df, test_df, scaler = standard_scale(
        train_df,
        test_df
    )

    # ========================================================
    # 8. COMBINE TRAINING AND TESTING DATA
    # ========================================================

    processed_df = pd.concat(
        [train_df, test_df]
    )

    # Restore original row order
    processed_df = processed_df.sort_index()

    # ========================================================
    # 9. SAVE PREPROCESSED DATASET
    # ========================================================

    processed_df.to_csv(
        output_file,
        index=False
    )

    # ========================================================
    # 10. DISPLAY FINAL RESULT
    # ========================================================

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETED")
    print("=" * 80)

    print("\nFinal dataset shape:")

    print(processed_df.shape)

    print("\nFinal columns:")

    print(processed_df.columns.tolist())

    print("\nFirst 5 rows of preprocessed data:")

    print(
        processed_df.head()
    )

    print("\nPreprocessed dataset saved as:")

    print(output_file)

    return processed_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    preprocess_data()