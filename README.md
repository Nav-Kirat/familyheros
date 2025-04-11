<p align="center" draggable="false">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR8HNB-ex4xb4H3-PXRcywP5zKC_3U8VzQTPA&usqp=CAU" 
     width="200px"
     height="auto"/>
</p>

# <h1 align="center" id="heading">Food Hamper Demand Forecasting</h1>

![Food Hamper Demand Forecasting](https://img.shields.io/badge/Project-Food%20Hamper%20Demand%20Forecasting-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0+-red)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![Machine Learning](https://img.shields.io/badge/ML-ElasticNet-yellow)

**Live Application:** [gofamilyhero.streamlit.app](https://gofamilyhero.streamlit.app)

## 📋 Overview

Welcome to the repository for our Capstone project at NorQuest College. This application helps Islamic Family in Edmonton plan resources effectively by forecasting the demand for food hampers. Using machine learning and time series analysis, the system predicts daily hamper requirements which can be aggregated into monthly forecasts to optimize resource allocation and ensure timely assistance to families in need.

## 🔍 Problem Statement

Food insecurity remains a significant challenge for many families in Edmonton. Islamic Family, a local organization dedicated to community support, needs to accurately predict the demand for food hampers to optimize resource allocation, reduce waste, and ensure that those in need receive timely assistance. Without accurate forecasting, they face challenges in planning volunteer schedules, managing inventory, and securing adequate funding and donations.

## 💡 Solution

Our solution implements an ElasticNet regression model for time series forecasting that analyzes historical hamper distribution data along with key factors such as client visit frequency and travel distance. The model provides daily predictions that can be aggregated into monthly forecasts, helping Islamic Family plan their resources more effectively. We've developed an intuitive Streamlit application that allows staff to generate forecasts, visualize demand patterns, and download reports for operational planning.

## 🎯 Objectives

- Forecast **monthly** food hamper demand to help **Islamic Family** plan resources effectively
- Aggregate **daily** predictions into monthly forecasts
- Analyze key factors such as client visit frequency and travel distance to improve prediction accuracy
- Provide an intuitive interface for staff to generate and visualize demand forecasts
- Enable data-driven decision making for volunteer scheduling and inventory management
- Reduce food waste through more accurate demand predictions

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

### Repository Structure

The repository contains the following files and directories:

- `overview.py`: Main application entry point and landing page
- `daily_hamper_demand_forecast_model.joblib`: Serialized machine learning model
- `hamper_model_hyperparameters.pkl`: Model hyperparameters for compatibility
- `requirements.txt`: List of project dependencies
- `data/`: Directory containing datasets and processed data
- `models/`: Directory for model storage and versioning
- `pages/`: Application pages for different functionalities
- `utils/`: Utility functions for data processing and model operations
- `xai/`: Explainable AI components for model interpretability

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
- NorQuest College for their support and guidance

## 👥 Team Members

Our team consists of the following members:

- [Dionathan Santos ](https://www.linkedin.com/in/dionathanadiel)
- [Navkirat singh](https://linkedin.com/in/navkirat)
- [Mayank khera](https://www.linkedin.com/in/mayank-khera-915b12252/)
- [Harshdeep Kaur](https://www.linkedin.com/in/harshdeep-kaur-714b62118/)
- [Vinit Kataria](https://www.linkedin.com/in/vinit-kataria-46b13b222/)

---

© 2025 Go Family Heroes | Hamper Demand Forecasting Tool
