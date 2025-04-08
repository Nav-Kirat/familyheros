import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="XAI Insights", layout="wide")
st.title("🧠 Explainable AI Visualizations for Hamper Demand Forecasting")

# --- Load Model ---
@st.cache_data
def load_model():
    try:
        return joblib.load("daily_hamper_demand_forecast_model.joblib")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

if model:
    st.success("✅ Model loaded successfully!")
    st.write(model)

    # OPTIONAL: Display expected input features if available
    if hasattr(model, "feature_names_in_"):
        st.subheader("📌 Model Features")
        st.write(model.feature_names_in_)
    else:
        st.info("No feature name info stored in model.")

    # --- Sample Visualization Placeholder ---
    st.subheader("📊 Historical Distance Distribution")
    try:
        df = pd.read_csv("updated_merged_data_with_distance.csv")

        if "distance_km" in df.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(df["distance_km"], bins=30, kde=True, ax=ax)
            ax.set_title("Distribution of Distance to Pickup")
            ax.set_xlabel("Distance (km)")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.warning("distance_km column not found in CSV.")
    except Exception as e:
        st.error(f"Error reading distance CSV: {e}")
