# src/serving/api.py
# pip install fastapi uvicorn pydantic joblib numpy pandas
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from pathlib import Path
import joblib, numpy as np, pandas as pd
import os

app = FastAPI(title="Finance-AI API", version="0.1")

ART = Path("artifacts")
SCALER_PATH = ART / "scaler.joblib"
FEATS_PATH  = ART / "feature_names.joblib"
XGB_PATH    = ART / "model_xgb.joblib"
KERAS_PATH  = ART / "model_transformer.h5"

assert FEATS_PATH.exists(), "feature_names.joblib not found in artifacts/"
assert SCALER_PATH.exists(), "scaler.joblib not found in artifacts/"

FEATURES: List[str] = joblib.load(FEATS_PATH)
SCALER = joblib.load(SCALER_PATH)

# เลือกโมเดลอัตโนมัติ: XGB (sklearn) ก่อน ถ้าไม่มีค่อยใช้ Keras
MODEL_KIND = None
MODEL = None

if XGB_PATH.exists():
    MODEL_KIND = "xgb"
    MODEL = joblib.load(XGB_PATH)
elif KERAS_PATH.exists():
    import tensorflow as tf
    MODEL_KIND = "keras"
    MODEL = tf.keras.models.load_model(KERAS_PATH)
else:
    raise RuntimeError("No model found in artifacts/ (expect model_xgb.joblib or model_transformer.h5)")

class Features(BaseModel):
    data: Dict[str, float] = Field(..., description="mapping feature_name -> value")

def _vectorize(payload: Dict[str, float]) -> np.ndarray:
    # จัดลำดับคอลัมน์ตาม FEATURES ถ้าขาดให้ใส่ 0.0
    return np.array([[payload.get(f, 0.0) for f in FEATURES]], dtype=float)

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_KIND, "n_features": len(FEATURES)}

@app.post("/predict")
def predict(body: Features):
    x = _vectorize(body.data)
    x_s = SCALER.transform(x)

    if MODEL_KIND == "xgb":
        yhat = float(MODEL.predict(x_s)[0])
    else:
        # Keras expects sequence shape – ถ้าโมเดลเป็น time-series (WIN, n_feat) ต้องปรับเองตามที่ฝึก
        # ตัวอย่างนี้ใช้เวคเตอร์เดียว (เหมาะกับ XGB/Linear). ถ้าใช้ LSTM/Transformer ให้จัดรูป x_s -> (1,WIN,n_feat)
        yhat = float(MODEL.predict(x_s).ravel()[0])

    signal = 1 if yhat > 0 else -1
    return {"prediction": yhat, "signal": signal, "features_order": FEATURES}

@app.post("/predict-batch")
def predict_batch(rows: List[Features]):
    if not rows:
        return {"predictions": []}
    X = np.vstack([_vectorize(r.data) for r in rows])
    Xs = SCALER.transform(X)
    if MODEL_KIND == "xgb":
        yh = MODEL.predict(Xs).reshape(-1)
    else:
        yh = MODEL.predict(Xs).reshape(-1)
    sig = (yh > 0).astype(int)*2 - 1
    return {"predictions": yh.tolist(), "signals": sig.tolist()}

OUT = Path("outputs")   # <— เพิ่มบรรทัดนี้

@app.get("/signal")
def latest_signal():
    p = OUT / "backtest_transformer.csv"   # <— แก้มาอ่านจาก outputs/
    if not p.exists():
        return {"error": "outputs/backtest_transformer.csv not found"}
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    last_date = str(df.index[-1].date())
    return {"date": last_date, "row": df.iloc[-1].to_dict()}
