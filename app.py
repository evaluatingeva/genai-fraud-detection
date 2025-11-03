# app.py
import sys
import os

# ✅ Ensure src folder is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from explainer import FraudExplainer
from data_prep import preprocess_new_data

# ✅ Streamlit setup
st.set_page_config(page_title="AI Fraud Detection", page_icon="💸", layout="wide")

st.title("💳 AI-Powered Fraud Detection System")
st.markdown("""
This app predicts whether a transaction is **fraudulent or not**, and provides a short, 
AI-generated explanation using a local instruction-tuned LLM.
""")

st.info("⚙️ The model may download once (~1–2 GB) during first run (Phi-2 or Mistral). After that, it works offline.")


# ✅ Load explainer once
@st.cache_resource
def load_explainer():
    return FraudExplainer()

explainer = load_explainer()

# 🧠 Show which model is active (Phi-2 or Mistral)
try:
    llm_name = explainer.client.model_name
except Exception:
    llm_name = "Unknown"

st.markdown(f"**🧩 Active LLM Model:** `{llm_name}`")

# -------------------- SECTION 1: FILE UPLOAD --------------------
st.header("📂 Upload Transactions File")

uploaded_file = st.file_uploader("Upload a CSV file (PaySim format)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        df = None

    if df is not None:
        required_cols = [
            "nameOrig", "nameDest", "type", "amount",
            "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest"
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            if st.button("🔍 Predict Fraud for Uploaded Data"):
                with st.spinner("Analyzing transactions..."):
                    preds, probs, explanations = explainer.explain(df)
                df_out = df.copy()
                df_out["Fraud_Probability"] = [round(float(p), 4) for p in probs]
                df_out["Fraud_Prediction"] = preds
                df_out["Explanation"] = explanations
                st.success("✅ Analysis complete!")
                st.dataframe(df_out.head(50))
                csv = df_out.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Results as CSV", csv, "fraud_predictions.csv")


st.divider()


# -------------------- SECTION 2: MANUAL ENTRY --------------------
st.header("✍️ Enter Transaction Manually")

col1, col2, col3 = st.columns(3)

with col1:
    nameOrig = st.text_input("Sender Account (nameOrig)", value="C123456789")
    nameDest = st.text_input("Receiver Account (nameDest)", value="M987654321")
    tx_type = st.selectbox("Transaction Type (type)", ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"])

with col2:
    amount = st.number_input("Amount", min_value=0.0, value=10000.0, step=100.0)
    oldbalanceOrg = st.number_input("Old Balance (Sender) - oldbalanceOrg", min_value=0.0, value=10000.0, step=100.0)
    newbalanceOrig = st.number_input("New Balance (Sender) - newbalanceOrig", min_value=0.0, value=0.0, step=100.0)

with col3:
    oldbalanceDest = st.number_input("Old Balance (Receiver) - oldbalanceDest", min_value=0.0, value=0.0, step=100.0)
    newbalanceDest = st.number_input("New Balance (Receiver) - newbalanceDest", min_value=0.0, value=10000.0, step=100.0)

st.caption("Ensure that values match realistic transaction behavior for correct fraud detection.")


if st.button("🔮 Predict Fraud for Manual Entry"):
    df_manual = pd.DataFrame([{
        "nameOrig": nameOrig,
        "nameDest": nameDest,
        "type": tx_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    with st.spinner("Running fraud detection and generating explanation..."):
        try:
            preds, probs, explanations = explainer.explain(df_manual)
            prob = float(probs[0])
            pred = int(preds[0])

            if pred == 1:
                st.error(f"🚨 Fraudulent Transaction Detected! (Probability: {prob:.3f})")
            else:
                st.success(f"✅ Legitimate Transaction (Probability: {prob:.3f})")

            st.subheader("🧠 AI Explanation")
            st.info(explanations[0])
        except Exception as e:
            st.error(f"Error during prediction: {e}")


st.divider()
st.caption("🔹 This app uses RandomForest for fraud detection and Phi-2 (fallback: Mistral) for explanations.")
st.caption("🔹 Works entirely offline after initial download and caching.")
