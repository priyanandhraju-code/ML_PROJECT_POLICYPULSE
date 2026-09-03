import pandas as pd


# ============================================================
# OUTLIER FIX USING IQR
# ============================================================

def fix_outliers(df, feature="Annual_Premium"):

    df = df.copy()

    # ========================================================
    # ORIGINAL STATISTICS
    # ========================================================

    print("\n" + "=" * 80)
    print("OUTLIER DETECTION")
    print("=" * 80)

    print("\nOriginal statistics:")

    print(
        df[feature].describe()
    )

    # ========================================================
    # CALCULATE Q1 AND Q3
    # ========================================================

    Q1 = df[feature].quantile(0.25)

    Q3 = df[feature].quantile(0.75)

    # ========================================================
    # CALCULATE IQR
    # ========================================================

    IQR = Q3 - Q1

    # ========================================================
    # CALCULATE FENCES
    # ========================================================

    lower_fence = Q1 - 1.5 * IQR

    upper_fence = Q3 + 1.5 * IQR

    print("\nQ1 =", Q1)

    print("\nQ3 =", Q3)

    print("\nIQR =", IQR)

    print("\nLower Fence =", lower_fence)

    print("\nUpper Fence =", upper_fence)

    # ========================================================
    # FIND OUTLIERS
    # ========================================================

    outliers = df[
        (df[feature] < lower_fence) |
        (df[feature] > upper_fence)
    ]

    print(
        "\nNumber of outliers:",
        len(outliers)
    )

    # ========================================================
    # CLIP OUTLIERS
    # ========================================================

    df[feature] = df[feature].clip(
        lower=lower_fence,
        upper=upper_fence
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\nMinimum BEFORE clipping:")

    print(
        df[feature].min()
    )

    print("\nMaximum AFTER clipping:")

    print(
        df[feature].max()
    )

    return df