"""
Fast v1 frontend, no AI builder needed. Good for proving the whole stack
works end to end before you invest time in Lovable or Antigravity.

Run locally with:
    streamlit run frontend_quickstart/streamlit_app.py

Set the API_URL environment variable to your deployed FastAPI URL once
Phase 6 is done. Until then it defaults to your local server.
"""

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="World Cup match predictor", page_icon=":soccer:")
st.title("World Cup match outcome predictor")


@st.cache_data(ttl=3600)
def get_teams():
    response = requests.get(f"{API_URL}/teams", timeout=10)
    response.raise_for_status()
    return sorted(response.json()["teams"])


try:
    teams = get_teams()
except requests.RequestException:
    st.error(f"Couldn't reach the API at {API_URL}. Is it running?")
    st.stop()

col1, col2 = st.columns(2)
home_team = col1.selectbox("Home team", teams, index=0)
away_team = col2.selectbox("Away team", teams, index=1)
neutral = st.checkbox(
    "Neutral venue (true for almost all World Cup matches, except host nations)",
    value=True,
)

if st.button("Predict", type="primary"):
    if home_team == away_team:
        st.warning("Pick two different teams.")
    else:
        response = requests.post(
            f"{API_URL}/predict",
            json={
                "home_team": home_team,
                "away_team": away_team,
                "neutral_venue": neutral,
            },
            timeout=10,
        )
        if response.status_code != 200:
            st.error(response.json().get("detail", "Prediction failed."))
        else:
            result = response.json()
            chart_df = pd.DataFrame({
                "Outcome": [f"{away_team} win", "Draw", f"{home_team} win"],
                "Probability": [
                    result["away_win_probability"],
                    result["draw_probability"],
                    result["home_win_probability"],
                ],
            }).set_index("Outcome")
            st.bar_chart(chart_df)
            st.caption(
                f"Elo ratings: {home_team} {result['home_team_elo']}, "
                f"{away_team} {result['away_team_elo']}"
            )
