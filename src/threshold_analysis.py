import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
from data_prep import load_and_prepare_data

# ---- 1️⃣ Load model & data ----
model_path = "C:/Users/91843/Documents/Gen_AI Project/project_root/src/artifacts/trained_fraud_model.pkl"
data_path = "C:/Users/91843/Documents/Gen_AI Project/project_root/data/dataset.csv"

print("Loading model and test data...")
model = joblib.load(model_path)
X_train, X_test, y_train, y_test = load_and_prepare_data(path=data_path, save_artifacts=False, verbose=False)

# ---- 2️⃣ Predict probabilities ----
y_proba = model.model.predict_proba(X_test)[:, 1]

# ---- 3️⃣ Compute precision, recall, and F1-score ----
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

# ---- 4️⃣ Find best threshold ----
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"Best threshold (max F1): {best_threshold:.3f}")

# ---- 5️⃣ Plot ----
plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision[:-1], label="Precision", linewidth=2)
plt.plot(thresholds, recall[:-1], label="Recall", linewidth=2)
plt.plot(thresholds, f1_scores[:-1], label="F1-score", linestyle="--", linewidth=2)
plt.axvline(best_threshold, color="red", linestyle=":", label=f"Best Threshold = {best_threshold:.2f}")
plt.title("Precision, Recall, and F1-Score vs. Threshold", fontsize=14)
plt.xlabel("Threshold", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
