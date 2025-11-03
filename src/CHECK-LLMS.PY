from dotenv import load_dotenv
import os

load_dotenv()

print("HUGGINGFACE_API_TOKEN:", os.getenv("HUGGINGFACE_API_TOKEN"))
print("HF_MODEL_NAME:", os.getenv("HF_MODEL_NAME"))
print("THRESHOLD:", os.getenv("THRESHOLD"))
