# Data folder

This folder is intentionally empty in the repo, the dataset is too large
and too easy to re-download to commit to git.

## What goes here

Download `results.csv` from the Kaggle dataset **"International football
results from 1872 to 2026"** by user `martj42`, and place it directly in
this folder so the path `data/results.csv` exists.

Expected columns: `date`, `home_team`, `away_team`, `home_score`,
`away_score`, `tournament`, `city`, `country`, `neutral`.

If you don't have a Kaggle account yet, sign up free at kaggle.com, then
either click "Download" on the dataset page, or use the Kaggle CLI:

    pip install kaggle --break-system-packages
    kaggle datasets download -d martj42/international-football-results-from-1872-to-2017
    unzip international-football-results-from-1872-to-2017.zip -d data/
