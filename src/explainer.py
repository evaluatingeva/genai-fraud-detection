# src/explainer.py
import os
import json
import time
import joblib
import pandas as pd
from typing import Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from data_prep import preprocess_new_data, ARTIFACT_DIR

# ✅ Store Hugging Face models locally inside your project
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.path.dirname(__file__), "..", "artifacts", "hf_cache")

MODEL_PATH = os.path.join(ARTIFACT_DIR, "trained_fraud_model.pkl")
CACHE_FILE = os.path.join(ARTIFACT_DIR, "llm_cache.json")
DEFAULT_THRESHOLD = 0.54


# ------------------- CACHE HELPERS -------------------

def _load_cache():
    """Load saved explanations."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_cache(cache):
    """Save explanation cache."""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


# ------------------- LLM WRAPPER -------------------

class LocalLLMClient:
    """
    Loads microsoft/phi-2 as primary model.
    Falls back to mistralai/Mistral-7B-Instruct-v0.3 if Phi-2 fails.
    """

    def __init__(self, primary_model="microsoft/phi-2", fallback_model="mistralai/Mistral-7B-Instruct-v0.3", token=None):
        self.token = token
        self.primary_model = primary_model
        self.fallback_model = fallback_model

        print(f"[INFO] Using local Hugging Face cache: {os.environ['TRANSFORMERS_CACHE']}")

        self.model_name = None
        self.tokenizer = None
        self.model = None

        # Try to load Phi-2 first
        try:
            print(f"[INFO] Attempting to load {primary_model}...")
            self.tokenizer = AutoTokenizer.from_pretrained(primary_model, token=token)
            self.model = AutoModelForCausalLM.from_pretrained(
                primary_model,
                token=token,
                device_map="cpu",
                torch_dtype="auto"
            )
            self.model_name = primary_model
            print(f"[SUCCESS] Loaded {primary_model} ✅")
        except Exception as e:
            print(f"[WARNING] Could not load {primary_model}: {e}")
            print(f"[INFO] Falling back to {fallback_model}...")
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            self.model = AutoModelForCausalLM.from_pretrained(fallback_model, device_map="cpu", torch_dtype="auto")
            self.model_name = fallback_model
            print(f"[SUCCESS] Loaded fallback model: {fallback_model} ✅")

        # Initialize pipeline for text generation
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            return_full_text=False,
            repetition_penalty=1.2,   # discourage repetitive outputs like "What is it?"
            top_p=0.9,                # nucleus sampling for diversity
        )

    def generate(self, prompt: str, max_length: int = 180, temperature: float = 0.4) -> str:
        """Generate a concise explanation using instruction-tuned formatting."""
        try:
            # Convert to instruction-style prompt that Phi-2 understands
            formatted_prompt = f"<|system|>You are a financial fraud analyst.<|user|>{prompt}<|assistant|>"

            result = self.pipe(
                formatted_prompt,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                
            )

            text = result[0]["generated_text"]

            # Remove prompt part if echoed
            if "<|assistant|>" in text:
                text = text.split("<|assistant|>", 1)[-1]

            # Clean repetitive “Answer:” patterns
            text = text.replace("Answer:", "").strip()
            text = text.split("\n")[0]
            return text
        except Exception as e:
           return f"[LLM error: {e}]"



# ------------------- FRAUD EXPLAINER -------------------

class FraudExplainer:
    """Uses a local Phi-2 model (fallback: Mistral) to generate text explanations for fraud predictions."""

    def __init__(self):
        self.threshold = DEFAULT_THRESHOLD
        self.model_wrapper = joblib.load(MODEL_PATH)
        self.cache = _load_cache()

        # Load token if set in environment
        self.hf_token = os.getenv("HUGGINGFACE_API_TOKEN", None)

        # Initialize LLM (Phi-2 → Mistral fallback)
        self.client = LocalLLMClient(primary_model="microsoft/phi-2", fallback_model="mistralai/Mistral-7B-Instruct-v0.3", token=self.hf_token)


    def _build_prompt(self, features: dict, prob: float, label: int) -> str:
        """Build structured natural prompt."""
        return (
            "You are a financial fraud detection assistant.\n"
            f"Transaction details: {json.dumps(features)}\n"
            f"Fraud probability: {prob:.4f}, Label: {label}\n"
            "If label=1, explain briefly (1–2 sentences) why this looks fraudulent.\n"
            "If label=0, explain briefly why it looks normal.\n"
            "Answer:"
        )

    def explain(self, df: pd.DataFrame, max_items: int = 10) -> Tuple[list, list, list]:
        """Predict fraud and generate explanations."""
        if len(df) > max_items:
            df = df.head(max_items)

        X = preprocess_new_data(df)
        model_obj = getattr(self.model_wrapper, "model", self.model_wrapper)
        probs = (
            model_obj.predict_proba(X)[:, 1]
            if hasattr(model_obj, "predict_proba")
            else model_obj.predict(X)
        )
        preds = (probs >= self.threshold).astype(int)
        explanations = []

        for i, row in df.reset_index(drop=True).iterrows():
            features = row.to_dict()
            cache_key = json.dumps(features, sort_keys=True)

            if cache_key in self.cache:
                explanations.append(self.cache[cache_key])
                continue

            prob = float(probs[i])
            label = int(preds[i])
            prompt = self._build_prompt(features, prob, label)
            text = self.client.generate(prompt)
            explanations.append(text)
            self.cache[cache_key] = text
            _save_cache(self.cache)
            time.sleep(0.2)

        return preds.tolist(), probs.tolist(), explanations
