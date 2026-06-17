# World Cup match outcome predictor

A full pipeline: historical match data → Elo ratings + a calibrated
classifier → a FastAPI service → a frontend that anyone can open and get
a probability for any matchup.

## Phase 0: environment setup

1. Open this folder as a project in **PyCharm** (your primary IDE for this
   build, the debugger and built-in virtual environment manager will save
   you real time over the next few phases).
2. Create a virtual environment: in PyCharm, `File > Settings > Project >
   Python Interpreter > Add Interpreter > Add Local Interpreter > Virtualenv`.
   If you'd rather do it from a terminal:

       python -m venv venv
       source venv/bin/activate        # on Windows: venv\Scripts\activate
       pip install -r requirements.txt

3. Copy `.env.example` to `.env` and leave the values as-is for now, you'll
   fill in the Supabase ones only if you reach Phase 7.

## Phase 1: get the data

Follow `data/README.md`, download `results.csv` from Kaggle, drop it in
`data/`. This is the only manual step in the whole pipeline, everything
after this is one command.

## Phase 2: understand what you're about to build

Two things are happening underneath this project, and they're genuinely
different skills:

**Elo ratings** are a single number per team that goes up when they beat a
team they "should" lose to, and barely moves when they beat a team they
were expected to beat. It's the same rating system used in chess, adapted
here with a margin-of-victory adjustment, beating someone 4-0 moves the
rating more than scraping past them 1-0.

**Recent form** is a short-term signal Elo misses on its own, a team's
average points across their last 5 matches, since a strong team in a
slump and a weak team on a hot streak both deserve to be modeled
differently than their season-long rating alone would suggest.

Both get computed once across the entire historical dataset, then frozen
as a snapshot, in `models/team_states.json`, that the live API reads from.

## Phase 3: train the model

       python -m ml.train

This reads `data/results.csv`, walks every match in chronological order
computing Elo and form before updating them (never after, that would leak
the outcome into its own prediction), trains a calibrated logistic
regression classifier, evaluates it on the most recent ~15% of matches,
and writes two files:

- `models/model.joblib`, the trained, calibrated classifier
- `models/team_states.json`, every team's current Elo + form, used at
  prediction time

Watch the console output. The classification report and log loss tell you
whether the model is meaningfully better than guessing. Don't expect huge
accuracy numbers, international football has a lot of irreducible
randomness, even professional bookmakers rarely exceed ~55% home-favorite
accuracy with real money on the line.

## Phase 4: run the API

       uvicorn api.main:app --reload

Open `http://localhost:8000/docs`, FastAPI generates this interactive test
page for free. Try `/teams` first, then `/predict` with two team names
from that list.

## Phase 5: test it properly

       pytest

These tests check the API's behavior (valid predictions sum to 1.0,
unknown teams get rejected, duplicate teams get rejected) without
retraining anything. Run this after any change to `api/` or `core/`
before you trust the API again.

## Phase 6: build the frontend

Three options, not mutually exclusive:

**Fastest path**: `streamlit run frontend_quickstart/streamlit_app.py`.
Working UI in under a minute, good for proving the whole stack end to end
before investing in a polished frontend.

**Antigravity**: point it at this prompt: "Build a single-page web app
that calls GET {API_URL}/teams to populate two team dropdowns, then POSTs
to {API_URL}/predict and displays the three returned probabilities as a
horizontal bar chart. Avoid generic SaaS dashboard styling, give it real
typographic personality and a deliberate color choice, not a default
purple gradient." Antigravity can test the result in a live browser as it
builds, which catches integration bugs Lovable can't.

**Lovable**: same prompt works. Lovable tends to produce a more
polished-looking result faster, but be specific about styling in your
prompt, vague prompts produce the same generic look every other Lovable
project has.

Either way, the frontend only ever talks to your FastAPI URL, it never
touches the model file directly.

## Phase 7 (optional): log predictions to Supabase

See `extras/supabase_logger.py` for the full setup, including the SQL to
create the table. Wire it in by importing `log_prediction` in `api/main.py`
and calling it just before `predict()` returns. Keep the service key on
the backend only, see the security note in that file.

## Phase 8: deploy

- **Backend**: Render's free web service tier is the most reliable
  genuinely-free option for FastAPI right now (Railway and Fly.io have
  both killed their free tiers). Expect a ~1 minute cold start after
  inactivity on the free tier, that's normal, not a bug.
- **Frontend**: Lovable and Antigravity both offer one-click hosting for
  what they build. If you go the Streamlit route, Streamlit Community
  Cloud is free and built for exactly this.
- Set `ALLOWED_ORIGINS` on Render to your actual deployed frontend URL,
  not `*`. Set `API_URL` on your frontend host to your actual deployed
  Render URL.

## Security checklist before you call this done

- [ ] `.env` is in `.gitignore` and was never committed (check `git log`
      if you're not sure, a leaked key needs rotating, not just deleting)
- [ ] `ALLOWED_ORIGINS` is your real frontend URL, not `*`, in production
- [ ] Supabase service key (if used) only exists server-side
- [ ] `/predict` rejects unknown teams and mismatched payloads (covered by
      `tests/test_api.py`, but re-check after any schema change)
- [ ] Dependencies are pinned (already done in `requirements.txt`, but
      re-pin after any `pip install` of something new)
