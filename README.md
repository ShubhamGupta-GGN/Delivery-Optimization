
# FastTrack Dashboard (Release)

This repo contains a Streamlit dashboard that analyses a synthetic logistics survey.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Push all files (including the **data/** folder) to a GitHub repo.
2. On https://share.streamlit.io → New app → select `app.py`.
3. Deploy. The app auto-loads **data/fasttrack_survey_10k.csv**.
