"""
Pydantic request/response contracts for the prediction API.

Keeping these in their own module (rather than inline in main.py) makes
them trivial to reuse in tests, and keeps main.py focused purely on
routing logic. This is also exactly what stops a frontend from sending
garbage team names or malformed payloads, FastAPI validates against these
models automatically and returns a clean 422 error before your code ever
runs if something doesn't match.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    home_team: str = Field(..., min_length=1, examples=["Brazil"])
    away_team: str = Field(..., min_length=1, examples=["Argentina"])
    neutral_venue: bool = Field(
        default=True,
        description=(
            "True for most World Cup matches; set False if the home team "
            "is playing in its own country (relevant for the USA, Mexico, "
            "and Canada at the 2026 tournament)."
        ),
    )

    @field_validator("home_team", "away_team")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class PredictionResponse(BaseModel):
    home_team: str
    away_team: str
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    predicted_outcome: Literal["home_win", "draw", "away_win"]
    home_team_elo: float
    away_team_elo: float
