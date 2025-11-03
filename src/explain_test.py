import os
import pandas as pd
from dotenv import load_dotenv
from explainer import FraudExplainer
# ✅ Always load .env from project root
from pathlib import Path
from dotenv import load_dotenv
import importlib; importlib.reload(__import__('explainer'))


ROOT_DIR = Path(__file__).resolve().parents[1]  # goes one level up from src/
load_dotenv(ROOT_DIR / ".env")

expl = FraudExplainer()

df = pd.read_csv("C:/Users/91843/Documents/Gen_AI Project/project_root/data/new_transactions.csv")
preds, probs, explanations = expl.explain(df.head(3))

for p, pr, ex in zip(preds, probs, explanations):
    print(f"Prediction: {p} | Probability: {pr:.3f}")
    print("Explanation:", ex)
    print("-" * 70)
