import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Bank Marketing Prediction", layout="wide")
st.title("Bank Marketing Prediction - BITS Assignment 2")

# Show comparison table if exists
if os.path.exists("model_comparison.csv"):
    st.subheader("d. Models Used - Comparison Table")
    df_comp = pd.read_csv("model_comparison.csv")
    st.dataframe(df_comp)

# b. Model selection dropdown [1 mark]
st.subheader("b. Model Selection Dropdown")
pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl') and 'scaler' not in f]
model_choice = st.selectbox("Select Model", pkl_files)

# a. Dataset upload [1 mark]
st.subheader("a. Dataset Upload Option (Test Data CSV)")
uploaded = st.file_uploader("Upload test_data.csv", type=["csv"])

# Use uploaded file or default test_data.csv
if uploaded is not None:
    test_df = pd.read_csv(uploaded)
else:
    if os.path.exists("test_data.csv"):
        st.info("Using default test_data.csv from repo. You can also upload your own.")
        test_df = pd.read_csv("test_data.csv")
    else:
        test_df = None

if test_df is not None:
    st.write("Preview:", test_df.head())
    # Find target column
    target_col = None
    for col in ['target','y','Target','label']:
        if col in test_df.columns:
            target_col = col
            break
    
    if target_col:
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        model = joblib.load(model_choice)
        
        # Handle scaling if needed
        if os.path.exists("scaler.pkl"):
            try:
                scaler = joblib.load("scaler.pkl")
                X_test = scaler.transform(X_test)
            except:
                pass
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1] if hasattr(model,"predict_proba") else y_pred

        # c. Display metrics [1 mark]
        st.subheader("c. Display of Evaluation Metrics")
        c1,c2,c3 = st.columns(3)
        c1.metric("Accuracy", f"{accuracy_score(y_test,y_pred):.4f}")
        c2.metric("AUC", f"{roc_auc_score(y_test,y_proba):.4f}")
        c3.metric("Precision", f"{precision_score(y_test,y_pred):.4f}")
        c1.metric("Recall", f"{recall_score(y_test,y_pred):.4f}")
        c2.metric("F1 Score", f"{f1_score(y_test,y_pred):.4f}")
        c3.metric("MCC", f"{matthews_corrcoef(y_test,y_pred):.4f}")

        # d. Confusion matrix [1 mark]
        st.subheader("d. Confusion Matrix & Classification Report")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        st.code(classification_report(y_test, y_pred))
