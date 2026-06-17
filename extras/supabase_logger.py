"""
Optional Phase 7 add-on: log every prediction to Supabase so you can later
analyze things like "how often does the model favor the underdog" or "which
matchups get queried most."

SECURITY NOTE: this uses the Supabase SERVICE key, which must only ever
live on the backend, read from an environment variable, never committed to
git, and never shipped to a frontend. If you instead let Lovable or
Antigravity write to Supabase directly from the browser, use the ANON key
with restrictive Row Level Security policies instead, otherwise a stranger
with your anon key could flood your logs table.

Run this SQL once in the Supabase SQL editor to create the table:

    create table predictions_log (
        id bigint generated always as identity primary key,
        created_at timestamptz default now(),
        home_team text not null,
        away_team text not null,
        home_win_probability float not null,
        draw_probability float not null,
        away_win_probability float not null,
        predicted_outcome text not null
    );

Wire it into api/main.py by importing log_prediction and calling it right
before the predict() function returns its response.
"""

import os

from supabase import create_client

_client = None


def get_client():
    global _client
    if _client is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SERVICE_KEY"]
        _client = create_client(url, key)
    return _client


def log_prediction(home_team, away_team, home_p, draw_p, away_p, outcome):
    try:
        get_client().table("predictions_log").insert({
            "home_team": home_team,
            "away_team": away_team,
            "home_win_probability": home_p,
            "draw_probability": draw_p,
            "away_win_probability": away_p,
            "predicted_outcome": outcome,
        }).execute()
    except Exception as exc:
        # A logging failure should never break a real prediction response.
        print(f"[supabase_logger] failed to log prediction: {exc}")
