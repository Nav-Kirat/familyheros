# convert_model.py
import pickle
import joblib

with open("daily_hamper_demand_forecast_model.pkl", "rb") as f:
    model = pickle.load(f)

joblib.dump(model, "daily_hamper_demand_forecast_model.joblib")
print("✅ Model converted successfully.")
