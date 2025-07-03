
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import seaborn as sns

# ----------------- Helper Paths -----------------
APP_DIR = pathlib.Path(__file__).parent
DEFAULT_PATHS = [
    APP_DIR / "data" / "fasttrack_survey_10k.csv",
    APP_DIR / "fasttrack_survey_10k.csv"
]

def find_default_csv():
    for p in DEFAULT_PATHS:
        if p.exists():
            return p
    return None

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

# ----------------- UI CONFIG -----------------
st.set_page_config(page_title="FastTrack Logistics Dashboard", layout="wide")

# ----------------- Data Loading -----------------
st.sidebar.header("Dataset & Global Controls")
upload = st.sidebar.file_uploader("Upload another CSV (optional)", type=["csv"])
if upload:
    df = load_data(upload)
    st.sidebar.success("Using uploaded file")
else:
    default_csv = find_default_csv()
    if default_csv is None:
        st.error("No default CSV found. Please upload one.")
        st.stop()
    df = load_data(default_csv)
    st.sidebar.info(f"Loaded default dataset: {default_csv.name}")

# ----------------- Descriptive Insights Function -----------------
def descriptive_insights(data):
    st.subheader("Key Descriptive Insights")
    st.markdown("**Sector distribution**")
    st.plotly_chart(px.histogram(data, x="sector", color="sector"), use_container_width=True)
    st.markdown("**City distribution**")
    st.plotly_chart(px.histogram(data, x="city", color="city"), use_container_width=True)
    st.markdown("**Orders per week (log scale)**")
    st.plotly_chart(px.histogram(data, x="orders_per_week", log_y=True), use_container_width=True)
    st.markdown("**Average urgency composition**")
    avg = data[["pct_ultra_urgent","pct_same_day","pct_next_day"]].mean().reset_index()
    st.plotly_chart(px.pie(avg, values=0, names="index"), use_container_width=True)
    st.markdown("**Distance vs Cost**")
    st.plotly_chart(px.scatter(data, x="avg_distance_km", y="curr_cost_aed", trendline="ols"), use_container_width=True)

# ----------------- Tabs -----------------
tabs = st.tabs(["Visualisation","Classification","Clustering","Association Rules","Regression"])

# --- Visualisation ---
with tabs[0]:
    st.header("Exploratory Data Visualisation")
    descriptive_insights(df)

# --- Classification ---
with tabs[1]:
    st.header("Binary Classification")
    target = st.selectbox("Choose target", ["switch_past","beta_opt_in"])
    X=df.drop(columns=[target]); y=df[target]
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    num_cols = X.select_dtypes(include='number').columns.tolist()
    pre = ColumnTransformer([("cat",OneHotEncoder(handle_unknown='ignore'),cat_cols),
                             ("num",StandardScaler(),num_cols)])
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
    models = {
        "KNN": KNeighborsClassifier(5),
        "Decision Tree": DecisionTreeClassifier(max_depth=6),
        "Random Forest": RandomForestClassifier(n_estimators=200,max_depth=8),
        "GBRT": GradientBoostingClassifier(n_estimators=200)
    }
    metrics=[]
    for name,model in models.items():
        pipe=Pipeline([("pre",pre),("model",model)])
        pipe.fit(X_train,y_train)
        y_pred=pipe.predict(X_test)
        metrics.append({
            "Model":name,
            "Accuracy":accuracy_score(y_test,y_pred),
            "Precision":precision_score(y_test,y_pred,average='weighted',zero_division=0),
            "Recall":recall_score(y_test,y_pred,average='weighted',zero_division=0),
            "F1":f1_score(y_test,y_pred,average='weighted',zero_division=0)
        })
    st.dataframe(pd.DataFrame(metrics).set_index("Model").round(3))

# --- Clustering ---
with tabs[2]:
    st.header("KMeans Clustering")
    features = st.multiselect("Numeric features", df.select_dtypes("number").columns.tolist(),
                              default=["avg_distance_km","curr_cost_aed","orders_per_week"])
    k = st.slider("k", 2, 10, 4)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    kmeans = KMeans(n_clusters=k,random_state=42,n_init='auto').fit(X_scaled)
    df["cluster"]=kmeans.labels_
    st.dataframe(df.groupby("cluster")[features].mean().round(2))

# --- Association Rules ---
with tabs[3]:
    st.header("Association Rule Mining")
    trans_cols = st.multiselect("Transaction-like categorical columns",
                                ["sector","city","fuel_strategy"])
    if st.button("Run Apriori") and len(trans_cols)>=2:
        baskets = df[trans_cols].astype(str).apply(lambda row: ",".join(row.values),axis=1)                    .str.split(",").tolist()
        te=TransactionEncoder(); te_arr=te.fit_transform(baskets)
        trans_df=pd.DataFrame(te_arr, columns=te.columns_)
        freq = apriori(trans_df, min_support=0.05, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=0.4)
        st.dataframe(rules.head(10))

# --- Regression ---
with tabs[4]:
    st.header("Regression")
    target_reg = st.selectbox("Numeric target", ["curr_cost_aed","orders_per_week"])
    features = st.multiselect("Features", [c for c in df.columns if c!=target_reg],
                              default=["avg_distance_km","pct_ultra_urgent","fuel_price"])
    X=df[features]; y=df[target_reg]
    cat_cols = X.select_dtypes('object').columns.tolist()
    num_cols = X.select_dtypes('number').columns.tolist()
    pre = ColumnTransformer([("cat",OneHotEncoder(handle_unknown='ignore'),cat_cols),
                             ("num",StandardScaler(),num_cols)])
    models = {
        "Linear":LinearRegression(),
        "Ridge":Ridge(),
        "Lasso":Lasso(alpha=0.01),
        "DecisionTree":DecisionTreeRegressor(max_depth=6)
    }
    perf=[]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    for n,m in models.items():
        pipe=Pipeline([("pre",pre),("m",m)]).fit(X_train,y_train)
        pred=pipe.predict(X_test)
        rmse=np.sqrt(((y_test-pred)**2).mean()); r2=pipe.score(X_test,y_test)
        perf.append({"Model":n,"RMSE":rmse,"R2":r2})
    st.dataframe(pd.DataFrame(perf).set_index("Model").round(3))
