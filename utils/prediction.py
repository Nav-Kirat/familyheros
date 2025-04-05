import pandas as pd
import numpy as np
import streamlit as st
import random

def predict_future_daily_demand(model_data, start_date=None, end_date=None, ci_band='middle'):
    """
    Generate predictions for a specific date range using the trained model
    with options for confidence interval selection
    """
    # Get last date from model data
    last_date = pd.Timestamp(model_data['last_date'])

    # Convert dates safely - using direct string formatting to avoid conversion issues
    if isinstance(start_date, pd.Timestamp):
        start_date = pd.Timestamp(start_date)
    elif isinstance(start_date, str):
        try:
            start_date = pd.Timestamp(start_date)
        except:
            st.error(f"Invalid start date format: {start_date}")
            return None

    if isinstance(end_date, pd.Timestamp):
        end_date = pd.Timestamp(end_date)
    elif isinstance(end_date, str):
        try:
            end_date = pd.Timestamp(end_date)
        except:
            st.error(f"Invalid end date format: {end_date}")
            return None

    # If no dates provided, default to 30 days from last date
    if start_date is None:
        start_date = last_date + pd.DateOffset(days=1)
    if end_date is None:
        end_date = last_date + pd.DateOffset(days=30)

    # Validate date range
    if start_date > end_date:
        st.error("Start date must be before end date")
        return None

    # Calculate days between last date in data and end date
    days_ahead = (end_date - last_date).days
    if days_ahead <= 0:
        st.error("End date must be after the last date in the training data")
        return None

    # Access model components
    model = model_data['model']
    features = model_data['features']

    # Create future dates from day after last date to end date
    future_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), end=end_date)

    # Initialize dataframe for predictions
    future_df = pd.DataFrame({'date': future_dates})

    # Add date features
    future_df['year'] = future_df['date'].dt.year
    future_df['month'] = future_df['date'].dt.month
    future_df['day'] = future_df['date'].dt.day
    future_df['day_of_year'] = future_df['date'].dt.dayofyear
    future_df['day_of_week'] = future_df['date'].dt.dayofweek
    future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)

    # Create cyclical features
    future_df['day_sin'] = np.sin(2 * np.pi * future_df['day_of_year']/365)
    future_df['day_cos'] = np.cos(2 * np.pi * future_df['day_of_year']/365)
    future_df['month_sin'] = np.sin(2 * np.pi * future_df['month']/12)
    future_df['month_cos'] = np.cos(2 * np.pi * future_df['month']/12)
    future_df['week_sin'] = np.sin(2 * np.pi * future_df['day_of_week']/7)
    future_df['week_cos'] = np.cos(2 * np.pi * future_df['day_of_week']/7)

    # Initialize with last known values
    future_df['unique_clients'] = model_data['last_values']['unique_clients']
    future_df['total_dependents'] = model_data['last_values']['total_dependents']
    future_df['returning_proportion'] = model_data['last_values']['returning_proportion']

    # Get last known demand values
    last_values = model_data['last_values']['daily_hamper_demand']

    # Define day-of-week patterns
    dow_factors = {
        0: 1.15,  # Monday
        1: 1.05,  # Tuesday
        2: 1.0,   # Wednesday
        3: 0.95,  # Thursday
        4: 0.9,   # Friday
        5: 0.8,   # Saturday
        6: 0.75   # Sunday
    }

    # Make predictions one day at a time
    predictions = []

    for i in range(len(future_df)):
        # Set lag features based on previous predictions or known values
        if i == 0:
            lag_1d = last_values[-1]
            lag_7d = last_values[-7] if len(last_values) >= 7 else np.mean(last_values)
            lag_30d = last_values[-30] if len(last_values) >= 30 else np.mean(last_values)
            rolling_mean_7d = np.mean(last_values[-7:]) if len(last_values) >= 7 else np.mean(last_values)
            rolling_mean_30d = np.mean(last_values[-30:]) if len(last_values) >= 30 else np.mean(last_values)
        else:
            # Use predictions for recent days
            recent_values = list(predictions[:i]) + list(last_values)
            lag_1d = recent_values[i-1]
            lag_7d = recent_values[i-7] if i >= 7 else recent_values[0]
            lag_30d = recent_values[i-30] if i >= 30 else recent_values[0]

            # Calculate rolling means
            rolling_window_7d = recent_values[max(0, i-7):i]
            rolling_window_30d = recent_values[max(0, i-30):i]
            rolling_mean_7d = np.mean(rolling_window_7d)
            rolling_mean_30d = np.mean(rolling_window_30d)

        # Add lag features to the dataframe
        future_df.loc[i, 'lag_1d'] = lag_1d
        future_df.loc[i, 'lag_7d'] = lag_7d
        future_df.loc[i, 'lag_30d'] = lag_30d
        future_df.loc[i, 'rolling_mean_7d'] = rolling_mean_7d
        future_df.loc[i, 'rolling_mean_30d'] = rolling_mean_30d
        future_df.loc[i, 'rolling_std_7d'] = 0.1 * rolling_mean_7d  # Add some variability

        # Make prediction - add try/except to handle missing features
        try:
            # Check if all required features are present
            missing_features = [f for f in features if f not in future_df.columns]
            if missing_features:
                for feature in missing_features:
                    future_df[feature] = 0  # Add missing features with default values
                st.warning(f"Added missing features with default values: {missing_features}")

            X_future = future_df.loc[i:i, features]
            pred = model.predict(X_future)[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            # Fall back to using mean of last values if prediction fails
            pred = np.mean(last_values)

        # Apply day-of-week adjustment
        day_of_week = future_df.loc[i, 'day_of_week']
        dow_factor = dow_factors.get(day_of_week, 1.0)

        # Add some random noise (±5%)
        random_factor = 1 + (random.random() * 0.1 - 0.05)

        # Apply adjustments
        adjusted_pred = pred * dow_factor * random_factor
        predictions.append(adjusted_pred)

    # Add predictions to the dataframe
    future_df['predicted_demand'] = predictions

    # Calculate confidence intervals
    if 'residuals_std' in model_data:
        std_residuals = model_data['residuals_std']
    else:
        # Estimate as a percentage of the predicted value
        std_residuals = future_df['predicted_demand'].mean() * 0.15

    # 95% confidence interval
    confidence_interval = 1.96 * std_residuals
    future_df['lower_bound'] = future_df['predicted_demand'] - confidence_interval
    future_df['upper_bound'] = future_df['predicted_demand'] + confidence_interval

    # Ensure lower bound is not negative
    future_df['lower_bound'] = future_df['lower_bound'].clip(lower=0)

    # Apply the selected confidence interval band
    if ci_band.lower() == 'upper':
        future_df['final_prediction'] = future_df['upper_bound']
    elif ci_band.lower() == 'lower':
        future_df['final_prediction'] = future_df['lower_bound']
    else:  # middle or any other value
        future_df['final_prediction'] = future_df['predicted_demand']

    # Filter to only include dates from start_date onwards
    if start_date > last_date + pd.DateOffset(days=1):
        future_df = future_df[future_df['date'] >= start_date].reset_index(drop=True)

    return future_df