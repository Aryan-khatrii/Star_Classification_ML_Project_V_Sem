import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "saved_models" / "best_star_classifier.joblib"
META_PATH = BASE_DIR / "saved_models" / "model_metadata.json"
COMPARISON_PATH = BASE_DIR / "saved_models" / "model_comparison.json"

HTML_PATH = BASE_DIR / "index.html"
CSS_PATH = BASE_DIR / "style.css"

FEATURES = ["u", "g", "r", "i", "z"]

app = FastAPI(title="Cosmic Classifier API")


# --------------------------------------------------
# 2. CSS ROUTE
# --------------------------------------------------

@app.get("/style.css")
def stylesheet():
    if not CSS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="style.css not found"
        )

    return FileResponse(
        CSS_PATH,
        media_type="text/css"
    )


# --------------------------------------------------
# 3. LOAD TRAINED MODEL
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Run the notebook and save the trained model first."
    )

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# 4. LOAD MODEL METADATA
# --------------------------------------------------

metadata = {}

if META_PATH.exists():
    metadata = json.loads(
        META_PATH.read_text(encoding="utf-8")
    )

saved_features = metadata.get("features", FEATURES)

if saved_features != FEATURES:
    raise RuntimeError(
        f"Expected model features {FEATURES}, "
        f"but metadata contains {saved_features}. "
        "Retrain and save the model using only u, g, r, i, z."
    )


# --------------------------------------------------
# 5. INPUT DATA VALIDATION
# --------------------------------------------------

class Observation(BaseModel):
    u: float = Field(allow_inf_nan=False)
    g: float = Field(allow_inf_nan=False)
    r: float = Field(allow_inf_nan=False)
    i: float = Field(allow_inf_nan=False)
    z: float = Field(allow_inf_nan=False)


# --------------------------------------------------
# 6. HOMEPAGE ROUTE
# --------------------------------------------------

@app.get("/")
def home():
    if not HTML_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="index.html not found"
        )

    return FileResponse(HTML_PATH)


# --------------------------------------------------
# 7. HEALTH CHECK ROUTE
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "running",
        "model_loaded": True
    }


# --------------------------------------------------
# 8. MODEL COMPARISON ROUTE
# --------------------------------------------------

@app.get("/api/model-comparison")
def model_comparison():
    if not COMPARISON_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Comparison data not found. "
                "Export the latest comparison table "
                "from the Jupyter Notebook."
            )
        )

    try:
        with COMPARISON_PATH.open(
            "r",
            encoding="utf-8"
        ) as file:
            comparison_data = json.load(file)

        if not isinstance(comparison_data, list):
            raise HTTPException(
                status_code=500,
                detail="Comparison data must be a JSON list."
            )

        return comparison_data

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="model_comparison.json contains invalid JSON."
        )

    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read comparison data: {error}"
        )


# --------------------------------------------------
# 9. PREDICTION ROUTE
# --------------------------------------------------

@app.post("/predict")
def predict(observation: Observation):
    try:
        sample = pd.DataFrame(
            [[
                observation.u,
                observation.g,
                observation.r,
                observation.i,
                observation.z
            ]],
            columns=FEATURES
        )

        prediction = str(model.predict(sample)[0])

        result = {
            "STAR": {
                "title": "Star",
                "emoji": "⭐",
                "description": "A stellar object."
            },
            "GALAXY": {
                "title": "Galaxy",
                "emoji": "🌌",
                "description": (
                    "A system containing stars, gas, "
                    "dust, and dark matter."
                )
            },
            "QSO": {
                "title": "Quasar (QSO)",
                "emoji": "💫",
                "description": (
                    "An extremely luminous active "
                    "galactic nucleus."
                )
            }
        }

        details = result.get(
            prediction.upper(),
            {
                "title": prediction,
                "emoji": "🔭",
                "description": "Predicted astronomical class."
            }
        )

        response = {
            "prediction": prediction,
            **details
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(sample)[0]

            response["probabilities"] = {
                str(label): round(
                    float(probability) * 100,
                    2
                )
                for label, probability in zip(
                    model.classes_,
                    probabilities
                )
            }

        return response

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
