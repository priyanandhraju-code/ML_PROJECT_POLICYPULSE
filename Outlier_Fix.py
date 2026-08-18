import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# LOAD DATASET
# ============================================================

file_path = r"Policypulse.csv"

df = pd.read_csv(file_path)


# ============================================================
# SELECT FEATURE
# ============================================================

feature = "Annual_Premium"


# ============================================================
# ORIGINAL STATISTICS
# ============================================================

print("Original statistics:")

print(
    df[feature].describe()
)


# ============================================================
# CALCULATE Q1 AND Q3
# ============================================================

Q1 = df[feature].quantile(0.25)

Q3 = df[feature].quantile(0.75)


# ============================================================
# CALCULATE IQR
# ============================================================

IQR = Q3 - Q1


# ============================================================
# CALCULATE FENCES
# ============================================================

lower_fence = Q1 - 1.5 * IQR

upper_fence = Q3 + 1.5 * IQR


print("\nQ1 =", Q1)

print("\nQ3 =", Q3)

print("\nIQR =", IQR)

print("\nlower_fence =", lower_fence)

print("\nupper_fence =", upper_fence)


# ============================================================
# FIND OUTLIERS
# ============================================================

outliers = df[
    (df[feature] < lower_fence) |
    (df[feature] > upper_fence)
]


print(
    "\nNumber of outliers:",
    len(outliers)
)


# ============================================================
# CLIP OUTLIERS
# ============================================================

df["Annual_Premium_Clipped"] = df[feature].clip(
    lower=lower_fence,
    upper=upper_fence
)


# ============================================================
# MINIMUM BEFORE CLIPPING
# ============================================================

print("\nMinimum BEFORE clipping:")

print(
    df[feature].min()
)


# ============================================================
# MINIMUM AFTER CLIPPING
# ============================================================

print("\nMinimum AFTER clipping:")

print(
    df["Annual_Premium_Clipped"].min()
)


# ============================================================
# MAXIMUM BEFORE CLIPPING
# ============================================================

print("\nMaximum BEFORE clipping:")

print(
    df[feature].max()
)


# ============================================================
# MAXIMUM AFTER CLIPPING
# ============================================================

print("\nMaximum AFTER clipping:")

print(
    df["Annual_Premium_Clipped"].max()
)


# ============================================================
# MIN-MAX SCALING
# ============================================================

scaler = MinMaxScaler()


df["Annual_Premium_Scaled"] = scaler.fit_transform(
    df[["Annual_Premium_Clipped"]]
)


# ============================================================
# DISPLAY FINAL RESULT
# ============================================================

print("\n" + "=" * 80)

print("DATA AFTER CLIPPING AND MIN-MAX SCALING")

print("=" * 80)


print(
    df[
        [
            "Annual_Premium",
            "Annual_Premium_Clipped",
            "Annual_Premium_Scaled"
        ]
    ].head(10)
)


print(
    "\nScaled minimum:",
    df["Annual_Premium_Scaled"].min()
)


print(
    "Scaled maximum:",
    df["Annual_Premium_Scaled"].max()
)