from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import time
from datetime import datetime
import numpy as np
import os

app = FastAPI(
    title="Fake Job Detection API",
    version="1.0"
)

# ---------------- LOAD PIPELINE MODEL ----------------
MODEL_PATH = os.path.join("models", "logistic_regression_pipeline.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print("✓ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", e)
    raise RuntimeError("Model could not be loaded")

# ---------------- REQUEST SCHEMA ----------------
class JobInput(BaseModel):
    job_description: str

# ---------------- ROUTES ----------------
@app.get("/")
def root():
    return {"status": "API is running"}

@app.get("/health")
def health():
    return {"status": "Healthy"}

@app.post("/predict")
def predict(data: JobInput):
    start_time = time.time()

    text = data.job_description.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Job description is empty")

    if len(text) < 30:
        raise HTTPException(status_code=400, detail="Job description too short")

    try:
        # PIPELINE handles TF-IDF internally
        pred = model.predict([text])[0]
        prob = model.predict_proba([text])[0]
    except Exception as e:
        print("❌ Prediction error:", e)
        raise HTTPException(status_code=500, detail="Prediction failed")

    label = "Fake" if pred == 1 else "Real"
    confidence = round(float(np.max(prob)) * 100, 2)

    return {
        "prediction": label,
        "confidence": f"{confidence}%",
        "processing_time_ms": round((time.time() - start_time) * 1000, 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
