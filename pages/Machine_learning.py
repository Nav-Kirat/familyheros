import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os

# Import utility functions
from utils import (
    load_model, 
    create_demo_data, 
    predict_future_daily_demand
)

# Set page configuration
st.set_page_config(
    page_title="Hamper Demand Forecast",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Hamper Demand Forecast")
st.markdown("""
This application predicts daily hamper demand based on a trained ElasticNet model.
Select a date range and confidence interval band to generate forecasts.
""")

# Display environment information in debug section
with st.expander("Debug Information"):
    st.write(f"Python version: {sys.version}")
    st.write(f"NumPy version: {np.__version__}")
    st.write(f"Pandas version: {pd.__version__}")
    st.write(f"Matplotlib version: {matplotlib.__version__}")

# Try to load the model
model_data = load_model()

# Sidebar for inputs
st.sidebar.header("Forecast Settings")

# Date range selection
st.sidebar.subheader("Date Range")

# Default dates
today = datetime.now().date()
default_start = today
default_end = today + timedelta(days=30)

# Use date_input with proper format
start_date = st.sidebar.date_input("Start Date", default_start, format="YYYY-MM-DD")
end_date = st.sidebar.date_input("End Date", default_end, format="YYYY-MM-DD")

# Confidence interval selection
st.sidebar.subheader("Confidence Interval")
ci_band = st.sidebar.selectbox(
    "Select Confidence Band",
    options=["Middle", "Upper", "Lower"],
    index=0
)

# Generate forecast button
if st.sidebar.button("Generate Forecast"):
    if start_date > end_date:
        st.error("Error: Start date must be before end date")
    else:
        # Show a spinner while generating forecast
        with st.spinner("Generating forecast..."):
            if model_data:
                # Use the actual model - explicitly convert dates to strings first
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')

                predictions = predict_future_daily_demand(
                    model_data,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    ci_band=ci_band.lower()
                )
            else:
                # If model failed to load, use demo data
                st.warning("Using demo data since the model couldn't be loaded.")
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')
                predictions = create_demo_data(start_date_str, end_date_str, ci_band.lower())

            if predictions is not None:
                # Add day of week
                predictions['day_of_week'] = predictions['date'].dt.day_name()

                # Round the demand predictions
                predictions['predicted_demand_rounded'] = predictions['final_prediction'].round(1)

                # Create two columns for visualization
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader("Forecast Visualization")

                    # Create the plot
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # Plot the predictions
                    ax.plot(predictions['date'], predictions['predicted_demand'],
                            marker='o', linestyle='-', color='blue', label='Middle Prediction')

                    # Add confidence interval
                    ax.fill_between(predictions['date'],
                                    predictions['lower_bound'],
                                    predictions['upper_bound'],
                                    color='lightblue', alpha=0.3,
                                    label='95% Confidence Interval')

                    # Highlight the selected band
                    ax.plot(predictions['date'], predictions['final_prediction'],
                            marker='*', linestyle='-', color='red', linewidth=2,
                            label=f'Selected Band ({ci_band})')

                    ax.set_title(f'Hamper Demand Forecast ({start_date_str} to {end_date_str})')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Predicted Daily Demand')
                    ax.grid(True)
                    ax.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    st.pyplot(fig)

                with col2:
                    st.subheader("Forecast Summary")

                    # Calculate summary statistics
                    total_demand = predictions['final_prediction'].sum().round(0)
                    avg_demand = predictions['final_prediction'].mean().round(1)
                    max_demand = predictions['final_prediction'].max().round(1)
                    min_demand = predictions['final_prediction'].min().round(1)

                    # Display summary metrics
                    st.metric("Total Hampers Needed", f"{total_demand:.0f}")
                    st.metric("Average Daily Demand", f"{avg_demand:.1f}")
                    st.metric("Maximum Daily Demand", f"{max_demand:.1f}")
                    st.metric("Minimum Daily Demand", f"{min_demand:.1f}")

                # Display the prediction table
                st.subheader("Daily Forecast Details")

                # Create a table for display
                table_df = predictions[['date', 'day_of_week', 'predicted_demand_rounded']].copy()
                table_df.columns = ['Date', 'Day of Week', f'Expected Demand ({ci_band} Band)']

                # Format the date column
                table_df['Date'] = table_df['Date'].dt.strftime('%Y-%m-%d')

                # Display the table
                st.dataframe(table_df, use_container_width=True)

                # Download button for CSV
                csv = table_df.to_csv(index=False)
                st.download_button(
                    label="Download Forecast as CSV",
                    data=csv,
                    file_name=f"hamper_demand_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

# Add model info and explanations to sidebar
if model_data:
    st.sidebar.subheader("Model Information")
    st.sidebar.info(f"Model Type: ElasticNet\nLast Training Date: {model_data['last_date'].strftime('%Y-%m-%d')}\nAccuracy (R²): 0.9999")

st.sidebar.subheader("About the Model")
st.sidebar.markdown("""
This forecast uses an ElasticNet regression model trained on historical hamper demand data.

Key features include:
- Cyclical time encodings (day, month)
- Recent demand history (lag values)
- Day of week patterns

The model achieved 99.99% accuracy (R² score) on historical data.
""")

# Add explanation of confidence bands
st.sidebar.subheader("Confidence Bands")
st.sidebar.markdown("""
- **Middle**: Most likely prediction
- **Upper**: Higher estimate (95% confidence)
- **Lower**: Lower estimate (95% confidence)
""")

# Add a section to create hyperparameters file if needed
with st.expander("Create Model Hyperparameters File"):
    st.markdown("""
    If you're experiencing compatibility issues, you can create a hyperparameters file for better compatibility.
    This will extract the core model parameters without the numpy dependencies.
    """)

    if st.button("Generate Hyperparameters File"):
        try:
            # Create sample hyperparameters based on notebook values
            hyperparams = {
                'model_type': 'ElasticNet',
                'alpha': 0.0020691388111478,  # From notebook
                'l1_ratio': 0.06157894736842105,  # From notebook
                'features': [
                    'day_sin', 'day_cos', 'month_sin', 'month_cos', 'week_sin', 'week_cos',
                    'lag_1d', 'lag_7d', 'lag_30d', 'rolling_mean_7d', 'rolling_mean_30d',
                    'is_weekend', 'unique_clients', 'total_dependents', 'returning_proportion'
                ],
                'random_state': 42
            }

            # Save hyperparameters with protocol 3
            hyperparams_path = 'models/hamper_model_hyperparameters.pkl'
            os.makedirs(os.path.dirname(hyperparams_path), exist_ok=True)
            
            import pickle
            with open(hyperparams_path, 'wb') as f:
                pickle.dump(hyperparams, f, protocol=3)

            st.success("Hyperparameters file created successfully! You can now restart the app.")
        except Exception as e:
            st.error(f"Error creating hyperparameters file: {e}")

# Add footer
st.markdown("---")
st.markdown("© 2025 Go Family Heroes | Hamper Demand Forecasting Tool")