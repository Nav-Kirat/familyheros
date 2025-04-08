# XAI_Insights.py
import streamlit as st
import os
from PIL import Image

st.set_page_config(page_title="XAI Visuals", page_icon="🧠", layout="wide")
st.title("🧠 Explainable AI Visualizations for Hamper Demand Forecasting")

# --- Helper to show image safely ---
def show_image(filename, caption):
    img_path = os.path.join("xai", filename)
    if os.path.exists(img_path):
        st.image(Image.open(img_path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Image not found: {filename}")

# --- Section 1: Model Evaluation ---
st.subheader("📈 Model Evaluation Metrics")

col1, col2 = st.columns(2)

with col1:
    show_image("actual vs predicted.png", "Actual vs Predicted Daily Hamper Demand")

with col2:
    show_image("residual vs predicted.png", "Residuals vs Predicted Demand")

# --- Section 2: SHAP Summary ---
st.subheader("📊 SHAP Feature Importance Summary")
show_image("shap 1.png", "SHAP Summary Plot")
show_image("shap 2.png", "Mean Impact on Model Output Magnitude")

# --- Section 3: SHAP Dependence ---
st.subheader("🔍 SHAP Dependence Plots")
show_image("shap 3.png", "SHAP Force Plot for Most Recent Prediction")
show_image("shap 4.png", "Dependence: rolling_mean_7d")
show_image("shap 5.png", "Dependence: lag_1d")
show_image("shap 6.png", "Dependence: total_dependents")
