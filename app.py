import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Error Predictor", layout="wide")

EXPECTED_FEATURES = [
    'AvgRM',
    'AvgVM',
    'memory',
    'Class_gb',
    'cpu_pressure',
    'memory_pressure',
    'resource_ratio',
    'has Segmentation fault',
    'has Timeout',
    'has License',
    'has core dumped',
    'has internal error',
    'has crash',
]

LABEL_MAPPING = {
    0: "no_error",
    1: "sporadic_error",
    2: "real_error",
}

CLASS_INDEX = 1  # change if needed

# ==============================
# LOAD MODEL (cached)
# ==============================
@st.cache_resource
def load_model():
    return joblib.load("xgboost_pipeline.joblib")

pipeline = load_model()

# ==============================
# UI
# ==============================
st.title("🔍 Error Prediction & Explanation")

st.write(
    "Upload a CSV file containing error features and get predictions "
    "with detailed explanations."
)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    # ==============================
    # LOAD DATA
    # ==============================
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data")
    st.dataframe(df)

    # ==============================
    # VALIDATION
    # ==============================
    st.subheader("✅ Validation")

    csv_cols = set(df.columns)
    expected_cols = set(EXPECTED_FEATURES)

    missing = expected_cols - csv_cols
    extra = csv_cols - expected_cols

    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    if extra:
        st.error(f"Unexpected columns: {extra}")
        st.stop()

    if df.isnull().any().any():
        st.error("CSV contains missing values.")
        st.stop()

    X = df[EXPECTED_FEATURES].astype(float)

    st.success("Validation passed!")

    # ==============================
    # PREDICTION
    # ==============================
    st.subheader("📊 Prediction")

    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)

    df_out = df.copy()
    df_out["prediction"] = [LABEL_MAPPING[int(p)] for p in y_pred]
    df_out["probability"] = y_proba[:, CLASS_INDEX]

    st.dataframe(df_out)

    # ==============================
    # SHAP explanation
    # ==============================

    st.subheader("🧠 Model Explanation (SHAP)")

    with st.spinner("Computing SHAP explanations..."):
        explainer = shap.Explainer(pipeline.predict_proba, X)
        shap_values = explainer(X)

    # ---- Global summary ----
    st.markdown("### Global Feature Importance")

    plt.clf()
    shap.summary_plot(
        shap_values[..., CLASS_INDEX],
        X,
        show=False
    )
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)

    # ---- Local explanation ----
    st.markdown("### Explanation for First Row")

    fig, ax = plt.subplots()
    shap.plots.waterfall(
        shap_values[0, :, CLASS_INDEX],
        show=False
    )
    st.pyplot(fig)
    plt.close(fig)
