# src/predict.py
import os
import joblib
import pandas as pd
from data_prep import preprocess_new_data, ARTIFACT_DIR

# ---- CONSTANT THRESHOLD ----
THRESHOLD = 0.54

MODEL_PATH = os.path.join(ARTIFACT_DIR, "trained_fraud_model.pkl")

def predict_csv(input_path, output_path):
    """
    Predict fraud probabilities for transactions using a fixed threshold (0.54).
    Adds two columns:
        Fraud_Probability
        isFraud_Predicted
    """
    print(f"Loading trained model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    print(f"Reading new data from: {input_path}")
    df = pd.read_csv(input_path)

    # Preprocess input
    X_new = preprocess_new_data(df)

    # Predict probabilities
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_new)[:, 1]
    else:
        proba = model.predict(X_new)

    # Apply threshold
    preds = (proba >= THRESHOLD).astype(int)

    # Add to dataframe
    df["Fraud_Probability"] = proba.round(4)
    df["isFraud_Predicted"] = preds

    # Save
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    print(f"Fixed threshold: {THRESHOLD}")
    print(f"Total frauds detected: {df['isFraud_Predicted'].sum()} / {len(df)}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fraud detection prediction script")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to save predictions CSV")
    args = parser.parse_args()

    predict_csv(args.input, args.output)
