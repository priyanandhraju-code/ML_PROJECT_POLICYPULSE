import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONTINUOUS NUMERICAL FEATURES
# ============================================================

# Only continuous numerical features are standardized.
#
# Age              -> Continuous numerical feature
# Annual_Premium   -> Continuous numerical feature
# Vintage          -> Continuous numerical feature
#
# Driving_License and Previously_Insured are binary features
# and should remain as 0/1.
#
# Region_Code and Policy_Sales_Channel are categorical codes.
# They are one-hot encoded in preprocessing.py before reaching
# this stage, so they are NOT included here.

num_cols = [
    "Age",
    "Annual_Premium",
    "Vintage"
]


# ============================================================
# STANDARD SCALING FUNCTION
# ============================================================

def standard_scale(train_df, test_df):

    train_df = train_df.copy()
    test_df = test_df.copy()


    # ========================================================
    # DISPLAY FEATURES
    # ========================================================

    print("\n" + "=" * 80)
    print("FEATURES SELECTED FOR STANDARD SCALING")
    print("=" * 80)

    print("\nContinuous numerical features:")

    for column in num_cols:
        print("-", column)


    # ========================================================
    # STANDARD SCALER
    # ========================================================

    scaler = StandardScaler()


    # ========================================================
    # FIT ONLY ON TRAINING DATA
    # ========================================================

    # The scaler learns the mean and standard deviation
    # ONLY from the training data.
    #
    # This prevents data leakage from the testing dataset.

    train_df[num_cols] = scaler.fit_transform(
        train_df[num_cols]
    )


    # ========================================================
    # TRANSFORM TESTING DATA
    # ========================================================

    # The same scaler learned from the training data
    # is applied to the testing data.

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


    # ========================================================
    # VERIFY TESTING DATA
    # ========================================================

    print("\n" + "=" * 80)
    print("TESTING DATA MEAN AFTER SCALING")
    print("=" * 80)

    print(
        test_df[num_cols].mean()
    )


    print("\n" + "=" * 80)
    print("TESTING DATA STANDARD DEVIATION AFTER SCALING")
    print("=" * 80)

    print(
        test_df[num_cols].std()
    )


    # ========================================================
    # DISPLAY SCALER PARAMETERS
    # ========================================================

    print("\n" + "=" * 80)
    print("STANDARD SCALER PARAMETERS")
    print("=" * 80)


    print("\nFeatures:")

    print(num_cols)


    print("\nLearned means:")

    print(
        scaler.mean_
    )


    print("\nLearned standard deviations:")

    print(
        scaler.scale_
    )


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return train_df, test_df, scaler