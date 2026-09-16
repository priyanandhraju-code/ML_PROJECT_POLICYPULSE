import pandas as pd


# ============================================================
# Load Data
# ============================================================

def load_data():

    data = pd.read_csv(
        r"preprocessed_data.csv"
    )

    # Remove target column
    X = data.drop(
        "Response",
        axis=1
    )

    return X


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    X = load_data()

    print("\nDataset Loaded Successfully")

    print(
        "Number of rows =",
        X.shape[0]
    )

    print(
        "Number of features =",
        X.shape[1]
    )

    print("\nAvailable Columns:")

    print(
        X.columns.tolist()
    )