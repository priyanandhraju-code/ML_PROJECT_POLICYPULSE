import pandas as pd


# ============================================================
# OUTLIER FIX FUNCTION
# ============================================================

def fix_outliers(
    train_df,
    test_df,
    feature="Annual_Premium"
):

    train_df = train_df.copy()
    test_df = test_df.copy()


    # ========================================================
    # DISPLAY ORIGINAL STATISTICS
    # ========================================================

    print("\n" + "=" * 80)
    print("OUTLIER DETECTION")
    print("=" * 80)


    print("\nFeature:")

    print(feature)


    print("\nOriginal TRAINING statistics:")

    print(
        train_df[feature].describe()
    )


    print("\nOriginal TESTING statistics:")

    print(
        test_df[feature].describe()
    )


    # ========================================================
    # CALCULATE QUARTILES
    # ========================================================

    Q1 = train_df[feature].quantile(
        0.25
    )

    Q3 = train_df[feature].quantile(
        0.75
    )


    # ========================================================
    # CALCULATE IQR
    # ========================================================

    IQR = Q3 - Q1


    # ========================================================
    # CALCULATE FENCES
    # ========================================================

    lower_fence = Q1 - (
        1.5 * IQR
    )

    upper_fence = Q3 + (
        1.5 * IQR
    )


    print("\n" + "=" * 80)
    print("IQR OUTLIER CALCULATION")
    print("=" * 80)


    print("\nQ1:")

    print(Q1)


    print("\nQ3:")

    print(Q3)


    print("\nIQR:")

    print(IQR)


    print("\nLower Fence:")

    print(lower_fence)


    print("\nUpper Fence:")

    print(upper_fence)


    # ========================================================
    # FIND TRAINING OUTLIERS
    # ========================================================

    train_outliers = train_df[
        (train_df[feature] < lower_fence)
        |
        (train_df[feature] > upper_fence)
    ]


    # ========================================================
    # FIND TESTING OUTLIERS
    # ========================================================

    test_outliers = test_df[
        (test_df[feature] < lower_fence)
        |
        (test_df[feature] > upper_fence)
    ]


    print("\n" + "=" * 80)
    print("OUTLIER COUNTS")
    print("=" * 80)


    print("\nTraining outliers:")

    print(
        len(train_outliers)
    )


    print("\nTesting values outside training fences:")

    print(
        len(test_outliers)
    )


    # ========================================================
    # CLIP TRAINING DATA
    # ========================================================

    train_df[feature] = train_df[
        feature
    ].clip(

        lower=lower_fence,

        upper=upper_fence

    )


    # ========================================================
    # CLIP TESTING DATA
    # ========================================================

    # IMPORTANT:
    #
    # The test data is clipped using the fences learned
    # from the training data.
    #
    # We do NOT calculate new Q1/Q3/IQR values for testing.
    #
    # This prevents information from the testing dataset
    # influencing preprocessing.

    test_df[feature] = test_df[
        feature
    ].clip(

        lower=lower_fence,

        upper=upper_fence

    )


    # ========================================================
    # DISPLAY FINAL STATISTICS
    # ========================================================

    print("\n" + "=" * 80)
    print("DATA AFTER OUTLIER HANDLING")
    print("=" * 80)


    print("\nTraining statistics after outlier handling:")

    print(
        train_df[feature].describe()
    )


    print("\nTesting statistics after outlier handling:")

    print(
        test_df[feature].describe()
    )


    # ========================================================
    # RETURN
    # ========================================================

    return (
        train_df,
        test_df
    )