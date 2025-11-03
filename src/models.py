# src/models.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

class FraudDetectionModel:
    def __init__(self):
        # RandomForest with class_weight='balanced' to handle imbalance
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=14,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )

    def train(self, X_train, y_train):
        print("Training RandomForest with class balancing...")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc:.4f}")
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))
        print("\nClassification Report:\n", classification_report(y_test, preds, digits=4))

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)
        print(f"Model saved at {path}")

    def load(self, path):
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
