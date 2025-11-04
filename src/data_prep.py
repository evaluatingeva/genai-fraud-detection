# src/data_prep.py
import os
import gdown
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Any

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def _save_artifact(obj: Any, name: str):
    path = os.path.join(ARTIFACT_DIR, name)
    joblib.dump(obj, path)

def _load_artifact(name: str):
    path = os.path.join(ARTIFACT_DIR, name)
    return joblib.load(path)
    
def download_artifacts_if_missing():
    os.makedirs("src/artifacts", exist_ok=True)

    files = {
        "trained_fraud_model.pkl": "1iIR8Vwde_fK9p_F8SBk5xPtZdfVllgA2",  
        "scaler.pkl": "1M6_M2YAHGY2-tQje7QfLAcOHxc8fEJz_", 
        "freq_maps.pkl": "1iUJaCcKcN-mFv_WMp01u9HMO3w26IT9B",
    }

    for name, file_id in files.items():
        path = os.path.join("src/artifacts", name)
        if not os.path.exists(path):
            print(f"Downloading {name} from Google Drive...")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)
            
def load_and_prepare_data(path="C:/Users/91843/Documents/Gen_AI Project/project_root/data/dataset.csv",
                          save_artifacts=True,
                          verbose=True) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Loads PaySim dataset, balances classes, encodes features, scales numeric columns.
    Saves scaler, freq maps, and feature columns as artifacts.
    Returns: X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(path)
    if verbose:
        print("Initial shape:", df.shape)

    df = df.dropna()

    # --- Frequency encoding for high-cardinality columns ---
    high_card_cols = [c for c in ['nameOrig', 'nameDest'] if c in df.columns]
    freq_maps = {}
    for col in high_card_cols:
        vc = df[col].value_counts(normalize=True)
        freq_maps[col] = vc.to_dict()
        df[col] = df[col].map(freq_maps[col]).fillna(0.0)

    # --- One-hot encode low-cardinality categorical column (type) ---
    if 'type' in df.columns:
        df = pd.get_dummies(df, columns=['type'], drop_first=True)

    # --- Ensure target exists ---
    if 'isFraud' not in df.columns:
        raise ValueError("Dataset must contain 'isFraud' column.")

    # --- Balance the dataset ---
    fraud = df[df['isFraud'] == 1]
    non_fraud = df[df['isFraud'] == 0]
    if len(fraud) == 0:
        raise ValueError("No fraud samples found in dataset.")
    ratio = 3  # keep 3x more non-fraud samples
    non_fraud_down = non_fraud.sample(n=min(len(non_fraud), len(fraud)*ratio), random_state=42)
    df_balanced = pd.concat([fraud, non_fraud_down]).sample(frac=1, random_state=42).reset_index(drop=True)

    if verbose:
        print(f"After balancing: Fraud = {len(fraud)}, Non-Fraud = {len(non_fraud_down)}")

    # --- Split into X and y ---
    X = df_balanced.drop('isFraud', axis=1)
    y = df_balanced['isFraud']

    # Convert to float32 to save memory
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float32)

    # --- Scale numeric features ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # --- Save preprocessing artifacts ---
    if save_artifacts:
        _save_artifact(scaler, "scaler.pkl")
        _save_artifact(freq_maps, "freq_maps.pkl")
        _save_artifact(list(X.columns), "feature_columns.pkl")
        if verbose:
            print(f"Saved artifacts to {ARTIFACT_DIR}")

    # --- Train-test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    if verbose:
        print("Processed shapes:", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


def preprocess_new_data(df: pd.DataFrame,
                        artifacts_dir: str = ARTIFACT_DIR) -> np.ndarray:
    """
    Preprocess new transactions for prediction using saved artifacts.
    """
    scaler = _load_artifact("scaler.pkl")
    freq_maps = _load_artifact("freq_maps.pkl")
    feature_columns = _load_artifact("feature_columns.pkl")

    df_proc = df.copy()

    # Frequency encode
    for col, fmap in freq_maps.items():
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].map(fmap).fillna(0.0)
        else:
            df_proc[col] = 0.0

    # One-hot encode 'type'
    if 'type' in df_proc.columns:
        df_proc = pd.get_dummies(df_proc, columns=['type'], drop_first=True)

    # Align columns
    for col in feature_columns:
        if col not in df_proc.columns:
            df_proc[col] = 0.0
    df_proc = df_proc[feature_columns]
    df_proc = df_proc.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float32)

    # Scale
    X_scaled = scaler.transform(df_proc).astype(np.float32)
    return X_scaled
