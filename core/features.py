"""
Shared feature-engineering logic.

This module is imported by BOTH the training script (ml/train.py) and the
live API (api/main.py). That's intentional, not just tidiness: the exact
same Elo formula and the exact same feature order must be used at training
time and at prediction time, or the model gets fed something subtly
different from what it learned on. This mismatch is commonly called
"train/serve skew," and it's one of the most common ways ML projects quietly
ship a broken model without any error ever being raised.
"""

from __future__ import annotations

# The order here is the contract between training and inference.
# Never reorder or rename these without retraining the model.
FEATURE_COLUMNS = ["elo_diff", "home_form", "away_form", "neutral"]


def expected_score(rating_a: float, rating_b: float) -> float:
    """Standard Elo expected-score formula: probability A 'beats' B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def margin_multiplier(goal_diff: int) -> float:
    """
    Scales how much a result moves the Elo ratings based on how convincing
    it was. Mirrors the margin-of-victory adjustment used in well-known
    international football Elo models: a 1-0 win barely moves the needle,
    a 4-0 win moves it a lot.
    """
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return 1.75 + (goal_diff - 3) / 8.0


def update_elo(
    rating_home: float,
    rating_away: float,
    home_score: int,
    away_score: int,
    k: float = 20.0,
    home_advantage: float = 50.0,
) -> tuple[float, float]:
    """Returns (new_home_rating, new_away_rating) after one match."""
    adj_home = rating_home + home_advantage
    expected_home = expected_score(adj_home, rating_away)

    if home_score > away_score:
        actual_home = 1.0
    elif home_score == away_score:
        actual_home = 0.5
    else:
        actual_home = 0.0

    multiplier = margin_multiplier(abs(home_score - away_score))
    delta = k * multiplier * (actual_home - expected_home)

    return rating_home + delta, rating_away - delta


def build_feature_row(
    home_elo: float,
    home_form: float,
    away_elo: float,
    away_form: float,
    neutral: bool,
) -> dict:
    """Builds one feature row in the exact column order FEATURE_COLUMNS expects."""
    return {
        "elo_diff": home_elo - away_elo,
        "home_form": home_form,
        "away_form": away_form,
        "neutral": int(neutral),
    }
