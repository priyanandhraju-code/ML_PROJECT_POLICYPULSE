import pandas as pd


# ============================================================
# OUTLIER FIX FUNCTION
# ============================================================

def fix_outliers(
    train_df,
    test_df,
    feature="Annual_Premium"
):
    """
    Detect and treat outliers using the IQR method.

    Important methodology:
    - Q1, Q3 and the IQR fences are learned ONLY from training data.
    - The same training fences are applied to testing data.
    - No test-set statistics are used to learn the treatment limits.
    - Outliers are clipped instead of deleting rows.
    """

    train_df = train_df.copy()
    test_df = test_df.copy()

    if feature not in train_df.columns:
        raise ValueError(
            f"Feature '{feature}' not found in training data."
        )

    if feature not in test_df.columns:
        raise ValueError(
            f"Feature '{feature}' not found in testing data."
        )

    # ========================================================
    # ORIGINAL STATISTICS
    # ========================================================

    train_before = train_df[feature].describe()
    test_before = test_df[feature].describe()

    # ========================================================
    # CALCULATE QUARTILES FROM TRAINING DATA ONLY
    # ========================================================

    q1 = float(
        train_df[feature].quantile(0.25)
    )

    q3 = float(
        train_df[feature].quantile(0.75)
    )

    iqr = float(q3 - q1)

    # ========================================================
    # CALCULATE IQR FENCES
    # ========================================================

    lower_fence = float(
        q1 - (1.5 * iqr)
    )

    upper_fence = float(
        q3 + (1.5 * iqr)
    )

    # ========================================================
    # IDENTIFY OUTLIERS
    # ========================================================

    train_outlier_mask = (
        (train_df[feature] < lower_fence)
        |
        (train_df[feature] > upper_fence)
    )

    test_outlier_mask = (
        (test_df[feature] < lower_fence)
        |
        (test_df[feature] > upper_fence)
    )

    train_outlier_count = int(
        train_outlier_mask.sum()
    )

    test_outlier_count = int(
        test_outlier_mask.sum()
    )

    # ========================================================
    # CLIP TRAINING DATA
    # ========================================================

    train_df[feature] = train_df[feature].clip(
        lower=lower_fence,
        upper=upper_fence
    )

    # ========================================================
    # CLIP TESTING DATA USING TRAINING FENCES
    # ========================================================

    test_df[feature] = test_df[feature].clip(
        lower=lower_fence,
        upper=upper_fence
    )

    # ========================================================
    # AFTER-TREATMENT STATISTICS
    # ========================================================

    train_after = train_df[feature].describe()
    test_after = test_df[feature].describe()

    # ========================================================
    # PRINT INFORMATION
    # ========================================================

    print("\n" + "=" * 80)
    print("OUTLIER DETECTION AND IQR TREATMENT")
    print("=" * 80)

    print("\nFeature:")
    print(feature)

    print("\nTraining Q1:")
    print(q1)

    print("\nTraining Q3:")
    print(q3)

    print("\nIQR:")
    print(iqr)

    print("\nLower Fence:")
    print(lower_fence)

    print("\nUpper Fence:")
    print(upper_fence)

    print("\nTraining outliers:")
    print(train_outlier_count)

    print("\nTesting values outside training fences:")
    print(test_outlier_count)

    print("\nTraining maximum before clipping:")
    print(train_before["max"])

    print("Training maximum after clipping:")
    print(train_after["max"])

    print("\nTesting maximum before clipping:")
    print(test_before["max"])

    print("Testing maximum after clipping:")
    print(test_after["max"])

    # ========================================================
    # INFORMATION FOR FLASK / HTML
    # ========================================================

    iqr_info = {
        "feature": feature,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "train_outliers": train_outlier_count,
        "test_outliers": test_outlier_count,
        "train_before": {
            "min": float(train_before["min"]),
            "q1": float(train_before["25%"]),
            "median": float(train_before["50%"]),
            "q3": float(train_before["75%"]),
            "max": float(train_before["max"])
        },
        "train_after": {
            "min": float(train_after["min"]),
            "q1": float(train_after["25%"]),
            "median": float(train_after["50%"]),
            "q3": float(train_after["75%"]),
            "max": float(train_after["max"])
        },
        "test_before": {
            "min": float(test_before["min"]),
            "q1": float(test_before["25%"]),
            "median": float(test_before["50%"]),
            "q3": float(test_before["75%"]),
            "max": float(test_before["max"])
        },
        "test_after": {
            "min": float(test_after["min"]),
            "q1": float(test_after["25%"]),
            "median": float(test_after["50%"]),
            "q3": float(test_after["75%"]),
            "max": float(test_after["max"])
        }
    }

    return train_df, test_df, iqr_info
