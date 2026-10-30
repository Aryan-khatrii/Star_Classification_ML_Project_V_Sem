import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "saved_models" / "best_star_classifier.joblib"
META_PATH = BASE_DIR / "saved_models" / "model_metadata.json"
COMPARISON_PATH = BASE_DIR / "saved_models" / "model_comparison.json"
RF_ANALYSIS_PATH = BASE_DIR / "saved_models" / "random_forest_analysis.json"
HTML_PATH = BASE_DIR / "index.html"
CSS_PATH = BASE_DIR / "style.css"
FEATURES = ["u", "g", "r", "i", "z"]

app = FastAPI(title="Cosmic Classifier API")

@app.get("/style.css")
def stylesheet():
    if not CSS_PATH.exists():
        raise HTTPException(status_code=404, detail="style.css not found")
    return FileResponse(CSS_PATH, media_type="text/css")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run the notebook and save the trained model first.")
model = joblib.load(MODEL_PATH)

metadata = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
saved_features = metadata.get("features", FEATURES)
if saved_features != FEATURES:
    raise RuntimeError(f"Expected model features {FEATURES}, but metadata contains {saved_features}.")

class Observation(BaseModel):
    u: float = Field(allow_inf_nan=False)
    g: float = Field(allow_inf_nan=False)
    r: float = Field(allow_inf_nan=False)
    i: float = Field(allow_inf_nan=False)
    z: float = Field(allow_inf_nan=False)

@app.get("/")
def home():
    if not HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(HTML_PATH)

@app.get("/health")
def health():
    return {"status": "running", "model_loaded": True}

@app.get("/api/model-comparison")
def model_comparison():
    if not COMPARISON_PATH.exists():
        raise HTTPException(status_code=404, detail="Export model_comparison.json from the notebook first.")
    try:
        data = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise HTTPException(status_code=500, detail="Comparison data must be a JSON list.")
        return data
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="model_comparison.json contains invalid JSON.")

@app.get("/api/random-forest-analysis")
def random_forest_analysis():
    if not RF_ANALYSIS_PATH.exists():
        raise HTTPException(status_code=404, detail="Random Forest analysis data not found. Run the export cell in your notebook.")
    try:
        return json.loads(RF_ANALYSIS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="random_forest_analysis.json contains invalid JSON.")

@app.post("/predict")
def predict(observation: Observation):
    try:
        sample = pd.DataFrame([[observation.u, observation.g, observation.r, observation.i, observation.z]], columns=FEATURES)
        prediction = str(model.predict(sample)[0])
        details_map = {
            "STAR": {"title": "Star", "emoji": "⭐", "description": "A luminous astronomical object."},
            "GALAXY": {"title": "Galaxy", "emoji": "🌌", "description": "A system containing stars, gas, dust, and dark matter."},
            "QSO": {"title": "Quasar (QSO)", "emoji": "💫", "description": "An extremely luminous active galactic nucleus."}
        }
        details = details_map.get(prediction.upper(), {"title": prediction, "emoji": "🔭", "description": "Predicted astronomical class."})
        response = {"prediction": prediction, **details}
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(sample)[0]
            response["probabilities"] = {str(label): round(float(prob) * 100, 2) for label, prob in zip(model.classes_, probabilities)}
        return response
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}")
