"""Optional study helper: min-max scale continuous features using training only."""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from load_data import load_data
from preprocessing import CONTINUOUS_COLUMNS


def min_max_scale(train_df, test_df):
    train_df, test_df = train_df.copy(), test_df.copy()
    scaler = MinMaxScaler()
    train_df[CONTINUOUS_COLUMNS] = scaler.fit_transform(train_df[CONTINUOUS_COLUMNS])
    test_df[CONTINUOUS_COLUMNS] = scaler.transform(test_df[CONTINUOUS_COLUMNS])
    return train_df, test_df, scaler


if __name__ == "__main__":
    data = load_data()
    train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data.Response)
    train, test, _ = min_max_scale(train, test)
    print(train[CONTINUOUS_COLUMNS].head())
    print(test[CONTINUOUS_COLUMNS].head())
