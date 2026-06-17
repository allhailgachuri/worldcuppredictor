"""
Run with: pytest

These tests assume models/model.joblib and models/team_states.json already
exist, i.e. you've run `python -m ml.train` at least once before testing
the API layer. That's intentional: we're testing the API's behavior, not
re-training a model on every test run.
"""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_teams_list_not_empty():
    response = client.get("/teams")
    assert response.status_code == 200
    assert len(response.json()["teams"]) > 0


def test_predict_valid_teams_probabilities_sum_to_one():
    teams = client.get("/teams").json()["teams"]
    response = client.post("/predict", json={
        "home_team": teams[0],
        "away_team": teams[1],
        "neutral_venue": True,
    })
    assert response.status_code == 200
    body = response.json()
    total = (
        body["home_win_probability"]
        + body["draw_probability"]
        + body["away_win_probability"]
    )
    assert abs(total - 1.0) < 0.01


def test_predict_rejects_unknown_team():
    response = client.post("/predict", json={
        "home_team": "Atlantis",
        "away_team": "Brazil",
    })
    assert response.status_code == 404


def test_predict_rejects_same_team_twice():
    teams = client.get("/teams").json()["teams"]
    response = client.post("/predict", json={
        "home_team": teams[0],
        "away_team": teams[0],
    })
    assert response.status_code == 400


def test_predict_rejects_missing_fields():
    response = client.post("/predict", json={"home_team": "Brazil"})
    assert response.status_code == 422  # FastAPI's automatic validation error
