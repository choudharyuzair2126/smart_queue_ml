import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
import joblib, numpy as np

ML_API_KEY = os.environ.get("ML_API_KEY", "")
MODEL_PATH = "wait_time_model.pkl"

# ─── Auto-train if model missing (fixes Render crash) ────────────────
if not os.path.exists(MODEL_PATH):
    print(f"[startup] {MODEL_PATH} not found — training now...")
    import subprocess, sys
    result = subprocess.run([sys.executable, "train.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Failed to train model on startup")
    print(f"[startup] Model trained and saved to {MODEL_PATH}")

app = FastAPI(title="Smart Queue ML", version="1.0.0")
model = joblib.load(MODEL_PATH)


class QueueFeatures(BaseModel):
    tokens_ahead: int = Field(..., ge=0)
    active_counters: int = Field(..., ge=1)
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    avg_service_time: float = Field(..., ge=0)
    recent_throughput: int = Field(..., ge=0)


@app.get("/health")
def health():
    return {"ok": True, "model_loaded": True}


@app.post("/predict")
def predict(f: QueueFeatures, x_api_key: str | None = Header(default=None)):
    if ML_API_KEY and x_api_key != ML_API_KEY:
        raise HTTPException(status_code=401)
    X = np.array([[f.tokens_ahead, f.active_counters, f.hour_of_day,
                   f.day_of_week, f.avg_service_time, f.recent_throughput]])
    minutes = float(model.predict(X)[0])
    return {"predicted_wait_minutes": round(max(0.0, minutes), 1)}
