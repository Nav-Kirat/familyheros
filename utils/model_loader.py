import pickle
import os
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

def create_model_from_hyperparams(hyperparams):
    """Create a new model using saved hyperparameters"""
    try:
        from sklearn.linear_model import ElasticNet
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        if hyperparams['model_type'] == 'ElasticNet':
            model = ElasticNet(
                alpha=hyperparams.get('alpha', 0.0020691388111478),
                l1_ratio=hyperparams.get('l1_ratio', 0.06157894736842105),
                fit_intercept=hyperparams.get('fit_intercept', True),
                max_iter=hyperparams.get('max_iter', 1000),
                random_state=hyperparams.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unsupported model type: {hyperparams['model_type']}")

        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        # Create mock model_data structure
        model_data = {
            'model': pipeline,
            'features': hyperparams.get('features', []),
            'last_date': pd.to_datetime('2024-09-01'),
            'last_values': {
                'daily_hamper_demand': [30.0] * 30,
                'unique_clients': 100,
                'total_dependents': 300,
                'returning_proportion': 0.8
            },
            'residuals_std': 3.0
        }

        st.success("Model reconstructed from hyperparameters successfully!")
        return model_data
    except Exception as e:
        st.error(f"Error creating model from hyperparameters: {e}")
        return None

@st.cache_resource
def load_model(model_path='models/daily_hamper_demand_forecast_model.pkl'):
    """Load the saved model with compatibility handling"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        st.success("Model loaded successfully!")
        return model_data
    except (ImportError, ModuleNotFoundError) as e:
        st.error(f"Compatibility error: {e}")
        st.info("Attempting to rebuild model from hyperparameters...")
        try:
            # Try to load hyperparameters instead
            hyperparams_path = 'models/hamper_model_hyperparameters.pkl'
            if os.path.exists(hyperparams_path):
                with open(hyperparams_path, 'rb') as f:
                    hyperparams = pickle.load(f)
                st.info("Hyperparameters loaded. Reconstructing model...")
                return create_model_from_hyperparams(hyperparams)
            else:
                st.error("Hyperparameters file not found.")
                return None
        except Exception as e2:
            st.error(f"Failed to rebuild model: {e2}")
            return None
    except FileNotFoundError:
        st.error(f"Model file '{model_path}' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def save_model_with_protocol(model_data, filename, protocol=3):
    """Save model using a specific pickle protocol for better compatibility"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f, protocol=protocol)
        st.success(f"Model saved to {filename} using protocol {protocol}")
        return True
    except Exception as e:
        st.error(f"Error saving model: {e}")
        return False

def create_demo_data(start_date, end_date, ci_band='middle'):
    """Create synthetic data for demonstration when model fails to load"""
    # Convert date objects to string format and then to pandas timestamp to avoid issues
    if isinstance(start_date, datetime) or isinstance(start_date, pd.Timestamp):
        start_date_str = start_date.strftime('%Y-%m-%d')
    else:
        start_date_str = start_date

    if isinstance(end_date, datetime) or isinstance(end_date, pd.Timestamp):
        end_date_str = end_date.strftime('%Y-%m-%d')
    else:
        end_date_str = end_date

    try:
        start_ts = pd.Timestamp(start_date_str)
        end_ts = pd.Timestamp(end_date_str)
        date_range = pd.date_range(start=start_ts, end=end_ts)
    except Exception as e:
        st.error(f"Error creating date range: {e}")
        # Fallback to using current date and next 7 days
        today = pd.Timestamp.now().floor('D')
        date_range = pd.date_range(start=today, periods=7)

    # Create a dataframe with dates
    df = pd.DataFrame({'date': date_range})
    df['day_of_week'] = df['date'].dt.day_name()

    # Generate synthetic demand data
    base_demand = 30
    df['predicted_demand'] = [
        base_demand +
        (5 if i % 7 == 0 else 0) -  # Higher on Mondays
        (5 if i % 7 in [5, 6] else 0) +  # Lower on weekends
        np.random.normal(0, 3)  # Random noise
        for i in range(len(df))
    ]

    # Add confidence intervals
    std_dev = 3.0
    df['lower_bound'] = df['predicted_demand'] - (1.96 * std_dev)
    df['upper_bound'] = df['predicted_demand'] + (1.96 * std_dev)

    # Ensure lower bound is not negative
    df['lower_bound'] = df['lower_bound'].clip(lower=0)

    # Apply the selected confidence interval band
    if ci_band.lower() == 'upper':
        df['final_prediction'] = df['upper_bound']
    elif ci_band.lower() == 'lower':
        df['final_prediction'] = df['lower_bound']
    else:
        df['final_prediction'] = df['predicted_demand']

    df['predicted_demand_rounded'] = df['final_prediction'].round(1)

    return df