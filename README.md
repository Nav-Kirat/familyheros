# Food Hamper Demand Forecasting

![Food Hamper Demand Forecasting](https://img.shields.io/badge/Project-Food%20Hamper%20Demand%20Forecasting-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0+-red)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![Machine Learning](https://img.shields.io/badge/ML-ElasticNet-yellow)

**Live Application:** [gofamilyhero.streamlit.app](https://gofamilyhero.streamlit.app)

## 📋 Overview

This application helps Islamic Family in Edmonton plan resources effectively by forecasting the demand for food hampers. Using machine learning and time series analysis, the system predicts daily hamper requirements which can be aggregated into monthly forecasts to optimize resource allocation and ensure timely assistance to families in need.

## 🔍 Problem Statement

Food insecurity remains a significant challenge for many families. Predicting the demand for food hampers can help optimize resource allocation and ensure that those in need receive timely assistance.

## 🎯 Objectives

- Forecast **monthly** food hamper demand to help **Islamic Family** plan resources effectively
- Aggregate **daily** predictions into monthly forecasts
- Analyze key factors such as client visit frequency and travel distance to improve prediction accuracy
- Provide an intuitive interface for staff to generate and visualize demand forecasts

## 🧠 Machine Learning Approach

This project uses an **ElasticNet regression model** for time series forecasting with the following features:
- Cyclical time encodings (day, month, week)
- Recent demand history (lag values)
- Rolling averages (7-day and 30-day)
- Weekend indicators
- Client metrics (unique clients, total dependents, returning proportion)

The model achieves high accuracy (R² score of 0.9999) on historical data and provides confidence intervals for predictions.

## 🛠️ Key Features

- **Interactive Forecast Generation**: Select date ranges and confidence intervals to generate custom forecasts
- **Visual Analytics**: View forecast visualizations with confidence bands
- **Summary Statistics**: Get key metrics like total, average, maximum, and minimum demand
- **Downloadable Reports**: Export forecast data as CSV files
- **Explainable AI Insights**: Understand the factors driving demand predictions
- **Chat Assistant**: Get answers to questions about the data and forecasts

## 📊 Application Pages

1. **Overview**: Introduction to the project, problem statement, and objectives
2. **Hamper Demand Forecast**: Generate and visualize demand predictions
3. **Exploratory Data Analysis**: Explore historical patterns and trends
4. **Machine Learning**: Details about the forecasting model and its performance
5. **XAI Insights**: Explainable AI to understand prediction factors
6. **Chat Assistant**: Interactive Q&A about the data and forecasts
7. **About Us**: Information about the team and Islamic Family

## 🔧 Technical Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (ElasticNet)
- **Visualization**: Matplotlib, Seaborn
- **NLP & Chat**: Hugging Face Transformers, Sentence Transformers
- **Model Serialization**: Joblib, Pickle

## 📦 Project Structure

```
├── overview.py                                # Main application entry point
├── daily_hamper_demand_forecast_model.joblib  # Serialized ML model
├── hamper_model_hyperparameters.pkl          # Model hyperparameters
├── requirements.txt                          # Project dependencies
├── data/                                     # Data directory
├── models/                                   # Model storage
├── pages/                                    # Application pages
│   ├── About us.py
│   ├── Chat_Assistant.py
│   ├── Exploratory Data Analysis.py
│   ├── Machine_learning.py
│   └── XAI_Insights.py
├── utils/                                    # Utility functions
│   ├── date_utils.py
│   ├── model_loader.py
│   ├── prediction.py
│   └── __init__.py
└── xai/                                      # Explainable AI components
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/food-hamper-demand-forecasting.git
   cd food-hamper-demand-forecasting
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run overview.py
   ```

## 📈 Usage

1. Navigate to the "Hamper Demand Forecast" page
2. Select a date range for your forecast
3. Choose a confidence band (Middle, Upper, or Lower)
4. Click "Generate Forecast" to view predictions
5. Download the forecast as a CSV file if needed

## 🌟 About Islamic Family

Islamic Family in Edmonton is dedicated to fostering the well-being and advancement of every individual within its reach. Guided by compassion, unwavering support, and proactive advocacy, the organization's core mission is to ensure the flourishing of the whole person within the community.

Their services include:
- Essential needs (food, shelter)
- Family support
- Counselling & mental health services
- Newcomer support
- Youth programs
- Prison programming & re-entry
- Community engagement
- Refugee sponsorship

For more information, visit [Islamic Family's website](https://www.islamicfamily.ca/).

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Islamic Family in Edmonton for their collaboration and data
- The Streamlit team for their excellent framework
- All contributors to the open-source libraries used in this project

---

© 2025 Go Family Heroes | Hamper Demand Forecasting Tool