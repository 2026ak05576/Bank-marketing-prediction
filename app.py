import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Bank Marketing Prediction", layout="wide")
st.title("Bank Marketing Prediction - BITS Assignment 2")

# a. Dataset Upload Option
st.subheader("a. Dataset Upload Option (Test Data CSV)")
uploaded = st.file_uploader("Upload test_data.csv", type=["csv"], label_visibility="collapsed")
st.caption("Upload test_data.csv - 200MB per file - CSV")

if uploaded is not None:
    test_df = pd.read_csv(uploaded)
    st.success("Using uploaded file")
else:
    if os.path.exists("test_data.csv"):
        st.info("Using default test_data.csv from repo")
        test_df = pd.read_csv("test_data.csv")
    else:
        test_df = None
        st.warning("Please upload test_data.csv")

# b. Model Selection
if os.path.exists("model_comparison.csv"):
    st.subheader("d. Models Used - Comparison Table")
    st.dataframe(pd.read_csv("model_comparison.csv"))

st.subheader("b. Model Selection Dropdown")
pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl') and 'scaler' not in f.lower()]
model_choice = st.selectbox("Select Model", pkl_files if pkl_files else ["No model found"], label_visibility="collapsed")
st.write(f"Selected: {model_choice}")

if test_df is not None:
    target_col = test_df.columns[-1]
    X_test = test_df.drop(columns=[target_col])
    y_test = pd.to_numeric(test_df[target_col], errors='coerce')
    y_test = y_test.replace([np.inf, -np.inf], np.nan)
    if y_test.min() < 0 or y_test.max() > 1 or y_test.nunique() > 2:
        y_test = (y_test > y_test.median()).astype(int)
    
    X_test = X_test.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    combined = pd.concat([X_test, y_test], axis=1).dropna()
    y_test = combined[target_col]
    X_test = combined.drop(columns=[target_col])

    st.write("Preview (first 5 rows):", test_df.head())
    st.write(f"Data Shape: {X_test.shape} | Target: {target_col}")

    model = joblib.load(model_choice)
    if os.path.exists("scaler.pkl"):
        try:
            scaler = joblib.load("scaler.pkl")
            if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ == X_test.shape[1]:
                X_test = scaler.transform(X_test)
        except:
            pass

    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:,1]
    except:
        y_proba = y_pred

    st.subheader("c. Display of Evaluation Metrics")
    c1,c2,c3 = st.columns(3)
    c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
    try:
        c2.metric("AUC", f"{roc_auc_score(y_test, y_proba):.4f}")
    except:
        c2.metric("AUC", "N/A")
    c3.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
    c1.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.4f}")
    c2.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.4f}")
    c3.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.4f}")

    st.subheader("d. Confusion Matrix & Classification Report")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    st.text(classification_report(y_test, y_pred, zero_division=0))
