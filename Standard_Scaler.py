import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# NUMERICAL FEATURES
# ============================================================

num_cols = [
    "Age",
    "Driving_License",
    "Region_Code",
    "Previously_Insured",
    "Annual_Premium",
    "Policy_Sales_Channel",
    "Vintage"
]


# ============================================================
# STANDARD SCALING FUNCTION
# ============================================================

def standard_scale(train_df, test_df):

    train_df = train_df.copy()
    test_df = test_df.copy()

    # ========================================================
    # STANDARD SCALER
    # ========================================================

    scaler = StandardScaler()

    # ========================================================
    # FIT ONLY ON TRAINING DATA
    # ========================================================

    train_df[num_cols] = scaler.fit_transform(
        train_df[num_cols]
    )

    # ========================================================
    # TRANSFORM TESTING DATA
    # ========================================================

    test_df[num_cols] = scaler.transform(
        test_df[num_cols]
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 80)
    print("DATA AFTER STANDARD SCALING")
    print("=" * 80)

    print("\nTraining data after Standard Scaling:")

    print(
        train_df[num_cols].head()
    )

    print("\nTesting data after Standard Scaling:")

    print(
        test_df[num_cols].head()
    )

    # ========================================================
    # VERIFY STANDARD SCALING
    # ========================================================

    print("\n" + "=" * 80)
    print("TRAINING DATA MEAN AFTER SCALING")
    print("=" * 80)

    print(
        train_df[num_cols].mean()
    )

    print("\n" + "=" * 80)
    print("TRAINING DATA STANDARD DEVIATION AFTER SCALING")
    print("=" * 80)

    print(
        train_df[num_cols].std()
    )

    return train_df, test_df, scaler