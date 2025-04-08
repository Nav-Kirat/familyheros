import streamlit as st
import os
from PIL import Image

st.set_page_config(page_title="XAI Insights", page_icon="🧠", layout="wide")

st.title("🧠 Explainable AI Visualizations for Hamper Demand Forecasting")

# Helper to load and show image
def show_image(filename, caption):
    img_path = os.path.join("xai", filename)
    if os.path.exists(img_path):
        st.image(Image.open(img_path), caption=caption, use_column_width=True)
    else:
        st.warning(f"Image not found: {filename}")

# Display static visualizations
st.subheader("📉 Model Performance")
col1, col2 = st.columns(2)
with col1:
    show_image("actual vs predicted.png", "Actual vs Predicted Daily Hamper Demand")
with col2:
    show_image("residual vs predicted.png", "Residuals vs Predicted Demand")

st.subheader("📊 SHAP Feature Importance Summary")
show_image("shap 1.png", "SHAP Summary Plot")
show_image("shap 2.png", "Mean Impact on Model Output Magnitude")

st.subheader("🔍 SHAP Dependence Plots")
cols = st.columns(3)
with cols[0]:
    show_image("shap 3.png", "SHAP Force Plot for Most Recent Prediction")
with cols[1]:
    show_image("shap 4.png", "Dependence: rolling_mean_7d")
with cols[2]:
    show_image("shap 5.png", "Dependence: lag_1d")

show_image("shap 6.png", "Dependence: total_dependents")
