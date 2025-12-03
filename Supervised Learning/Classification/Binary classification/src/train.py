# src/train.py

import pandas as pd
import json
import joblib
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from preprocess import prepare, save_splits_and_scaler

def main():
    # --- 1. Forbered data ---
    raw_path = "../data/raw/Raisin_Dataset.csv"
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare(raw_path)
    
    # Gem splits og scaler
    save_splits_and_scaler(X_train_scaled, X_test_scaled, y_train, y_test, scaler)

    # --- 2. Definér og træn modellen ---
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # --- 3. Evaluer på test data ---
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

    # --- 4. Gem modellen ---
    model_path = "../models/trained_model.pkl"
    joblib.dump(model, model_path)
    print(f"Trained model saved to {model_path}")

    # --- 5. Gem metadata ---
    metadata = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": None,
        "train_size": X_train_scaled.shape[0],
        "test_size": X_test_scaled.shape[0],
        "accuracy": acc,
        "train_timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    }

    metadata_path = "../models/model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Model metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
