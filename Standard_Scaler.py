import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ============================================================
# DATASET PATH
# ============================================================

file_path = r"Policypulse.csv"


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(file_path)


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
# TARGET VARIABLE
# ============================================================

target_col = "Response"


# ============================================================
# TRAIN-TEST SPLIT
# ============================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df[target_col]
)


# ============================================================
# DISPLAY ORIGINAL DATA
# ============================================================

print("=" * 80)
print("DATASET")
print("=" * 80)

print(df.head())


print("\nDataset Columns:")
print(df.columns)


# ============================================================
# TRAINING DATA BEFORE SCALING
# ============================================================

print("\n" + "=" * 80)
print("TRAINING DATA BEFORE STANDARD SCALING")
print("=" * 80)

print(train_df[num_cols].head())


# ============================================================
# TESTING DATA BEFORE SCALING
# ============================================================

print("\n" + "=" * 80)
print("TESTING DATA BEFORE STANDARD SCALING")
print("=" * 80)

print(test_df[num_cols].head())


# ============================================================
# STANDARD SCALING
# ============================================================

scaler = StandardScaler()


# Fit ONLY on training data
# and transform training data

train_df[num_cols] = scaler.fit_transform(
    train_df[num_cols]
)


# Transform testing data using
# the scaler fitted on training data

test_df[num_cols] = scaler.transform(
    test_df[num_cols]
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 80)
print("DATA AFTER STANDARD SCALING")
print("=" * 80)


print(
    "Original dataset shape:",
    df.shape
)


print(
    "Training dataset shape:",
    train_df.shape
)


print(
    "Testing dataset shape:",
    test_df.shape
)


print("\nTraining data after Standard Scaling:")

print(
    train_df[num_cols].head()
)


print("\nTesting data after Standard Scaling:")

print(
    test_df[num_cols].head()
)


# ============================================================
# VERIFY STANDARD SCALING
# ============================================================

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

