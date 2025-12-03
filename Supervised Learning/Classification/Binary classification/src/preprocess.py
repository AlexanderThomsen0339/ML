import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path: str) -> pd.DataFrame:
    """Load raw raisin dataset."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning and type fixes."""

    # Ensure column names are clean and consistent
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # Drop duplicates if any
    df = df.drop_duplicates()

    # Handle missing values if dataset had any
    df = df.dropna()

    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical class column to numeric."""
    df["Class"] = df["Class"].map({"Kecimen": 0, "Besni": 1})
    return df


def split_features(df: pd.DataFrame):
    """Split into X and y."""
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y


def scale_features(X_train, X_test):
    """Scale using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def prepare(path: str):
    """Full pipeline for data preparation."""
    df = load_data(path)
    df = clean_data(df)
    df = encode_labels(df)
    X, y = split_features(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_splits_and_scaler(X_train, X_test, y_train, y_test, scaler):
    # Gem træning og test som CSV
    train_df = pd.concat([pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])]),
                          y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])]),
                         y_test.reset_index(drop=True)], axis=1)

    train_df.to_csv("../data/split/train.csv", index=False)
    test_df.to_csv("../data/split/test.csv", index=False)

    # Gem scaler
    joblib.dump(scaler, "../models/scaler.pkl")