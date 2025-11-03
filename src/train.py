# src/train.py
import os
import joblib
from data_prep import load_and_prepare_data, ARTIFACT_DIR
from models import FraudDetectionModel  # your model class; must implement train, predict, evaluate

MODEL_PATH = os.path.join(ARTIFACT_DIR, "trained_fraud_model.pkl")

def main():
    # Provide the full path relative to src/ or absolute path if needed
    X_train, X_test, y_train, y_test = load_and_prepare_data(path="C:/Users/91843/Documents/Gen_AI Project/project_root/data/dataset.csv", save_artifacts=True)

    model = FraudDetectionModel()
    print("Training model...")
    model.train(X_train, y_train)
    print("Evaluating model...")
    model.evaluate(X_test, y_test)

    # Save model with joblib
    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
