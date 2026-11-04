import streamlit as st
import joblib
import pandas as pd
import os

st.title("Bank Marketing Prediction - BITS Assignment")
st.write("Predict if client will subscribe to term deposit")

# Load model comparison
if os.path.exists("model_comparison.csv"):
    df_comp = pd.read_csv("model_comparison.csv")
    st.subheader("Model Comparison (6 Metrics)")
    st.dataframe(df_comp)
    
    best_model_name = df_comp.loc[df_comp['Accuracy'].idxmax(), 'Model']
    st.success(f"Best Model: {best_model_name}")
else:
    best_model_name = "logistic_regression"

# Model selector
model_files = [f for f in os.listdir('.') if f.endswith('.pkl') and f != 'scaler.pkl']
model_choice = st.selectbox("Choose Model", model_files)

# Load scaler
scaler = joblib.load("scaler.pkl") if os.path.exists("scaler.pkl") else joblib.load("model/scaler.pkl")

# Prediction UI
st.subheader("Make Prediction")
st.write("Enter scaled values (0 for demo) or use default")

if st.button("Predict with Test Data Sample"):
    try:
        if os.path.exists(f"model/{model_choice}"):
            model = joblib.load(f"model/{model_choice}")
        else:
            model = joblib.load(model_choice)
        
        test = pd.read_csv("test_data.csv")
        sample = test.drop('target', axis=1).iloc[0:1]
        pred = model.predict(sample)
        proba = model.predict_proba(sample)[0][1]
        st.write(f"Prediction: {'Subscribed (Yes)' if pred[0]==1 else 'Not Subscribed (No)'}")
        st.write(f"Probability: {proba:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
