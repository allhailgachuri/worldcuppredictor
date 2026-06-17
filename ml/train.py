"""
Trains the match-outcome predictor.

Run from the project root (after activating your virtual environment):
    python -m ml.train

Reads:  data/results.csv   (download from the martj42 Kaggle dataset,
                             "International football results from 1872 to 2026")
Writes: models/model.joblib
        models/team_states.json
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.features import FEATURE_COLUMNS, build_feature_row, update_elo

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "results.csv"
MODELS_DIR = ROOT / "models"

STARTING_ELO = 1500.0
FORM_WINDOW = 5
FORM_DEFAULT = 1.0  # ~1 point per game: neutral assumption before any history exists


def label_result(home_score: int, away_score: int) -> int:
    """0 = away win, 1 = draw, 2 = home win."""
    if home_score > away_score:
        return 2
    if home_score == away_score:
        return 1
    return 0


def load_matches() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Expected match data at {DATA_PATH}.\n"
            "Download 'results.csv' from the Kaggle dataset "
            "'International football results from 1872 to 2026' (martj42) "
            "and place it in the data/ folder before running this script."
        )
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["neutral"] = df["neutral"].astype(bool)
    return df


def engineer_features(df: pd.DataFrame):
    """
    Walks through every match in chronological order, recording each team's
    Elo rating and recent form *before* that match is used to update them.

    This ordering is the most important detail in this whole file: using
    post-match state as a feature would leak the outcome into the very
    feature meant to predict it, and your offline accuracy would look great
    while the model would be useless on a real upcoming match.
    """
    elo = defaultdict(lambda: STARTING_ELO)
    form = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    matches_played = defaultdict(int)

    rows = []
    for _, m in df.iterrows():
        home, away = m["home_team"], m["away_team"]

        home_form = sum(form[home]) / len(form[home]) if form[home] else FORM_DEFAULT
        away_form = sum(form[away]) / len(form[away]) if form[away] else FORM_DEFAULT

        row = build_feature_row(
            home_elo=elo[home], home_form=home_form,
            away_elo=elo[away], away_form=away_form,
            neutral=m["neutral"],
        )
        row["result"] = label_result(m["home_score"], m["away_score"])
        rows.append(row)

        # Update form (3/1/0 points) for the next time we see these teams.
        if m["home_score"] > m["away_score"]:
            form[home].append(3); form[away].append(0)
        elif m["home_score"] == m["away_score"]:
            form[home].append(1); form[away].append(1)
        else:
            form[home].append(0); form[away].append(3)

        # Update Elo for the next time we see these teams.
        elo[home], elo[away] = update_elo(
            elo[home], elo[away], m["home_score"], m["away_score"]
        )
        matches_played[home] += 1
        matches_played[away] += 1

    features_df = pd.DataFrame(rows)
    final_state = {
        team: {
            "elo": elo[team],
            "form": (sum(form[team]) / len(form[team])) if form[team] else FORM_DEFAULT,
            "matches_played": matches_played[team],
        }
        for team in elo.keys()
    }
    return features_df, final_state


def train_model(features_df: pd.DataFrame):
    # Time-based split: the dataframe is already chronologically sorted, so
    # a positional split IS a date-based split. Never use a random
    # train_test_split here, it would leak future form/Elo into training
    # and inflate your accuracy in a way that won't hold up on real matches.
    split_idx = int(len(features_df) * 0.85)
    train_df = features_df.iloc[:split_idx]
    test_df = features_df.iloc[split_idx:]

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["result"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["result"]

    base_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    # Calibration matters here: a raw classifier's confidence score is often
    # not a trustworthy probability. 5-fold sigmoid calibration fixes that,
    # so a displayed "62% home win" actually means something statistically,
    # rather than just being whichever class scored highest.
    model = CalibratedClassifierCV(estimator=base_pipeline, method="sigmoid", cv=5)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    print("\n--- Hold-out evaluation (most recent ~15% of matches by date) ---")
    print(classification_report(
        y_test, preds, target_names=["Away Win", "Draw", "Home Win"]
    ))
    print(f"Log loss: {log_loss(y_test, probs):.4f}")
    print("(Lower is better. A model that always guesses the average class "
          "distribution scores meaningfully worse than this on log loss.)")

    return model


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    print("Loading match data...")
    df = load_matches()
    print(f"Loaded {len(df):,} matches from {df['date'].min().date()} "
          f"to {df['date'].max().date()}")

    print("Engineering Elo + form features (this walks the full history, "
          "give it a moment)...")
    features_df, final_state = engineer_features(df)

    print("Training calibrated classifier...")
    model = train_model(features_df)

    joblib.dump(model, MODELS_DIR / "model.joblib")
    with open(MODELS_DIR / "team_states.json", "w") as f:
        json.dump(final_state, f, indent=2)

    print(f"\nSaved model to {MODELS_DIR / 'model.joblib'}")
    print(f"Saved {len(final_state)} team states to {MODELS_DIR / 'team_states.json'}")


if __name__ == "__main__":
    main()
