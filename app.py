import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import re

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Error Predictor", layout="wide")

LABEL_MAPPING = {
    0: "sporadic_error",
    1: "real_error",
}

CLASS_INDEX = 1  # class to explain


# ==============================
# HELPER FUNCTIONS
# ==============================

def extract_memory_from_class(class_str):
    if pd.isna(class_str):
        return 0
    match = re.search(r'(\d+)\s*gb', str(class_str).lower())
    return int(match.group(1)) if match else 0


MODEL_FEATURES = [
    'AvgRM', 'AvgVM', 'memory', 'Class_gb',
    'cpu_pressure', 'memory_pressure', 'resource_ratio',
    'has Segmentation fault', 'has Timeout', 'has License',
    'has core dumped', 'has internal error', 'has crash',
    'error_len',
    'sporadic_failures_same_hour',
    'sporadic_failures_machine',
    'clean_AvgRM', 'clean_AvgVM', 'clean_memory',
    'task_failure_count'
]

DEFAULT_ZERO_FEATURES = {
    'sporadic_failures_same_hour',
    'sporadic_failures_machine',
    'clean_AvgRM',
    'clean_AvgVM',
    'clean_memory',
    'task_failure_count',
}

ERROR_KEYWORDS = {
    "has Segmentation fault": "segmentation fault",
    "has Timeout": "timeout",
    "has License": "license",
    "has core dumped": "core dumped",
    "has internal error": "internal error",
    "has crash": "crash",
}


# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    return joblib.load("xgboost_pipeline.joblib")


pipeline = load_model()


# ==============================
# FEATURE ENGINEERING
# ==============================

def derive_error_features(df):
    if "Class_gb" not in df.columns and "Class" in df.columns:
        df["Class_gb"] = df["Class"].apply(extract_memory_from_class)

    if "normalized_error" not in df.columns:
        return df

    err = df["normalized_error"].fillna("").str.lower()

    for col, keyword in ERROR_KEYWORDS.items():
        if col not in df.columns:
            df[col] = err.str.contains(keyword).astype(int)

    if "error_len" not in df.columns:
        df["error_len"] = err.str.len()

    return df


def derive_resource_features(df):
    if "cpu_pressure" not in df.columns:
        if {"CoresConsumption", "cores"}.issubset(df.columns):
            df["cpu_pressure"] = np.where(
                df["cores"].fillna(0) == 0,
                0,
                df["CoresConsumption"].fillna(0) / df["cores"]
            )

    if "memory_pressure" not in df.columns and "AvgVM" in df.columns:
        df["memory_pressure"] = df["AvgVM"]

    if "resource_ratio" not in df.columns:
        if {"AvgRM", "AvgVM"}.issubset(df.columns):
            df["resource_ratio"] = np.where(
                df["AvgVM"].fillna(0) == 0,
                0,
                df["AvgRM"].fillna(0) / df["AvgVM"]
            )

    return df


def add_default_zero_features(df):
    for col in DEFAULT_ZERO_FEATURES:
        if col not in df.columns:
            df[col] = 0
    return df


# ==============================
# UI
# ==============================

st.title("🔍 Error Prediction & Explanation")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data")
    st.dataframe(df)

    # Feature engineering
    with st.spinner("Deriving missing features..."):
        df = derive_error_features(df)
        df = derive_resource_features(df)
        df = add_default_zero_features(df)

    # Validation
    st.subheader("✅ Validation")

    missing = set(MODEL_FEATURES) - set(df.columns)
    if missing:
        st.error(f"Missing required features: {sorted(missing)}")
        st.stop()

    X = df[MODEL_FEATURES].astype(float).fillna(0)

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
    # SHAP EXPLANATION
    # ==============================
    st.subheader("🧠 Model Explanation (SHAP)")

    with st.spinner("Computing SHAP explanations..."):

        # Separate preprocessing and model
        preprocess = pipeline.named_steps[list(pipeline.named_steps.keys())[0]]
        model = pipeline.named_steps[list(pipeline.named_steps.keys())[-1]]

        # Transform input exactly like training
        X_transformed = preprocess.transform(X)

        # Get feature names after preprocessing
        try:
            feature_names = preprocess.get_feature_names_out()
        except:
            feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

        # Convert to DataFrame so SHAP keeps column names
        X_transformed_df = pd.DataFrame(
            X_transformed,
            columns=feature_names,
            index=X.index
        )

        # Create TreeExplainer
        explainer = shap.TreeExplainer(model)

        # Use modern SHAP API (returns Explanation object)
        shap_exp = explainer(X_transformed_df)

        # Select class to explain (binary classification)
        shap_exp_class = shap_exp[..., CLASS_INDEX]

    # ---- Global SHAP Summary Plot ----
    st.markdown("### Global Feature Importance")

    plt.clf()
    shap.summary_plot(
        shap_exp_class,
        X_transformed_df,
        show=False
    )

    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)

    # ---- Local Waterfall Plot ----
    st.markdown("### Explanation for First Row")

    fig, ax = plt.subplots()
    shap.plots.waterfall(
        shap_exp_class[0],
        show=False
    )

    st.pyplot(fig)
    plt.close(fig)
