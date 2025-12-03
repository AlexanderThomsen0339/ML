# src/predict.py

import pandas as pd
import joblib
from preprocess import clean_data, encode_labels, split_features, scale_features

def load_model_and_scaler(model_path: str = "../models/trained_model.pkl",
                          scaler_path: str = "../models/scaler.pkl"):
    """Load trained model and scaler from disk."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def prepare_new_data(df: pd.DataFrame, scaler):
    """Clean, encode, and scale new data for prediction."""
    df = clean_data(df)
    # Assuming new data doesn't have labels
    X = df.copy()
    X_scaled = scaler.transform(X)
    return X_scaled

def predict(df: pd.DataFrame):
    """Make predictions on new data."""
    model, scaler = load_model_and_scaler()
    X_scaled = prepare_new_data(df, scaler)
    predictions = model.predict(X_scaled)
    return predictions

if __name__ == "__main__":
    # Example usage with a CSV file
    new_data_path = "../data/raw/new_raisins.csv"
    new_df = pd.read_csv(new_data_path)
    
    preds = predict(new_df)
    print("Predictions:")
    print(preds)
