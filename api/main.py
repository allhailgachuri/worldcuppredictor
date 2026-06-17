"""
The API layer. This is the only thing your frontend (Lovable, Antigravity-built
UI, or the Streamlit quickstart) ever talks to, it never touches the model
file or the training data directly.

Run from the project root:
    uvicorn api.main:app --reload

Then open http://localhost:8000/docs for an interactive, auto-generated
test page (this comes free with FastAPI, no extra setup required).
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import PredictionRequest, PredictionResponse
from core.features import FEATURE_COLUMNS, build_feature_row

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
TEAM_STATES_PATH = ROOT / "models" / "team_states.json"

ml_artifacts: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loaded once at startup, not on every request, this is the difference
    # between a fast API and one that re-deserializes a model 50 times a
    # second under load.
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"No trained model found at {MODEL_PATH}. Run "
            "'python -m ml.train' first (Phase 3) before starting the API."
        )
    ml_artifacts["model"] = joblib.load(MODEL_PATH)
    with open(TEAM_STATES_PATH) as f:
        ml_artifacts["team_states"] = json.load(f)
    yield
    ml_artifacts.clear()


app = FastAPI(
    title="World Cup Match Outcome Predictor",
    version="1.0.0",
    lifespan=lifespan,
)

# Security note: in development this defaults to localhost only. When you
# deploy, set ALLOWED_ORIGINS in your environment to your actual frontend
# URL(s), comma-separated. Never leave this as "*" in production, an open
# CORS policy lets any website on the internet call your API from a user's
# browser using that user's session.
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8501"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Used by your hosting platform (and you) to confirm the API is alive."""
    return {"status": "ok"}


@app.get("/teams")
def get_teams():
    """
    The frontend should call this once and use it to populate two dropdowns,
    rather than letting users free-type team names. This is what actually
    prevents 'garbage team names' at the source, the Pydantic validation
    below is the second line of defense, not the first.
    """
    return {"teams": sorted(ml_artifacts["team_states"].keys())}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    states = ml_artifacts["team_states"]

    if payload.home_team not in states:
        raise HTTPException(status_code=404, detail=f"Unknown team: {payload.home_team}")
    if payload.away_team not in states:
        raise HTTPException(status_code=404, detail=f"Unknown team: {payload.away_team}")
    if payload.home_team == payload.away_team:
        raise HTTPException(status_code=400, detail="home_team and away_team must differ")

    home_state = states[payload.home_team]
    away_state = states[payload.away_team]

    row = build_feature_row(
        home_elo=home_state["elo"], home_form=home_state["form"],
        away_elo=away_state["elo"], away_form=away_state["form"],
        neutral=payload.neutral_venue,
    )
    X = [[row[col] for col in FEATURE_COLUMNS]]
    probs = ml_artifacts["model"].predict_proba(X)[0]

    # Class order 0/1/2 = away_win/draw/home_win, fixed at training time
    # in ml/train.py's label_result(). If you ever change that ordering,
    # change it here too, or every prediction silently mislabels its output.
    away_p, draw_p, home_p = float(probs[0]), float(probs[1]), float(probs[2])
    outcome = ["away_win", "draw", "home_win"][int(probs.argmax())]

    return PredictionResponse(
        home_team=payload.home_team,
        away_team=payload.away_team,
        home_win_probability=round(home_p, 4),
        draw_probability=round(draw_p, 4),
        away_win_probability=round(away_p, 4),
        predicted_outcome=outcome,
        home_team_elo=round(home_state["elo"], 1),
        away_team_elo=round(away_state["elo"], 1),
    )
