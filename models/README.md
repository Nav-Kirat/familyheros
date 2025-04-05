# Model Files

This directory contains the trained machine learning models and their hyperparameters.

## Files

- `daily_hamper_demand_forecast_model.pkl`: The main ElasticNet model for predicting hamper demand
- `hamper_model_hyperparameters.pkl`: Hyperparameters for the model (used for compatibility)

## Model Details

The hamper demand forecasting model uses ElasticNet regression with the following hyperparameters:
- alpha: 0.0020691388111478
- l1_ratio: 0.06157894736842105
- random_state: 42

## Usage

Models are loaded automatically by the Streamlit application. If you encounter compatibility issues, 
the application will attempt to rebuild the model using the hyperparameters file.