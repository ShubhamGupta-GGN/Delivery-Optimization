
# FastTrack Logistics – Streamlit Feasibility Dashboard

This repository contains an interactive Streamlit application that explores the 10 000‑row synthetic survey
for **FastTrack Logistics** across five analytical perspectives:

1. **Data Visualisation** – 10+ descriptive insights with city/sector filters  
2. **Classification** – KNN, Decision Tree, Random Forest, GBRT  
   * Metrics table, ROC curves, confusion‑matrix toggle, batch‑prediction upload/download  
3. **Clustering** – Interactive K‑Means with dynamic k, elbow chart, personas, downloadable clusters  
4. **Association Rules** – Apriori with adjustable support/confidence and top‑10 rules table  
5. **Regression** – Linear, Ridge, Lasso, Decision Tree (RMSE & R²) with Actual vs Predicted plot

## Quick start (local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Create a new public GitHub repository and upload the files/folders.  
2. On Streamlit Cloud → *New app* → point to `app.py` on your repo.  
3. Set Python ≥ 3.11.  
4. Click **Deploy** – the dashboard is live!

## File structure
```
.
├── app.py                         # Streamlit dashboard
├── data
│   └── fasttrack_survey_10k.csv   # Synthetic dataset
├── requirements.txt               # Python dependencies
└── README.md
```

## License
*Code*: MIT License  
*Data*: Synthetic – freely reusable.
