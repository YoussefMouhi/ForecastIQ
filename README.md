# ForecastIQ

### AI-Powered Retail Demand Intelligence

ForecastIQ is a retail demand forecasting application built with Python, machine learning, deep learning, and Streamlit.

The project explores how forecasting techniques can support retail sales analysis and inventory planning.

## Features

- Interactive Streamlit dashboard
- Historical sales analysis
- Retail demand forecasting
- Store and product exploration
- Model performance visualization
- Inventory planning insights
- Dark-themed user interface

## Machine Learning Models

ForecastIQ includes four forecasting approaches:

| Model | Technique |
|---|---|
| SARIMA | Statistical time-series forecasting |
| Prophet | Trend and seasonality forecasting |
| XGBoost | Gradient-boosted decision trees |
| LSTM | Recurrent neural network |

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- PyTorch
- Prophet
- Statsmodels
- Streamlit
- Plotly

## Dataset

The project currently uses synthetically generated retail sales data.

The dataset contains historical sales observations for multiple stores and products.

Synthetic data allows the forecasting pipeline to be demonstrated without exposing confidential business information.

## Model Evaluation

The forecasting models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

Note: The current XGBoost evaluation covers multiple store-product series, while the other models are evaluated on a single series. These results should not be interpreted as a controlled comparison.

## Installation

### 1. Clone the repository

After publication, clone the ForecastIQ repository.

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate

```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch ForecastIQ

```bash
python -m streamlit run app.py
```

## Model Training

ForecastIQ uses four forecasting models:

- SARIMA
- Prophet
- XGBoost
- LSTM

The main training script is:

```bash
python models/train_models.py
```

The independently tested LSTM trainer is:

```bash
python models/train_lstm.py
```

The combined training script still requires final verification.

## Model Performance

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| SARIMA | 40.94 | 50.11 | 31.33% |
| Prophet | 46.16 | 62.75 | 30.78% |
| XGBoost | 13.24 | 18.89 | 10.41% |
| LSTM | 45.52 | 65.59 | 26.37% |

These results are not directly comparable because XGBoost was
evaluated across multiple store-product series, while the other
models were evaluated on a single series.

## Features

- Interactive sales analytics dashboard
- Demand forecasting
- Model performance analysis
- Inventory planning insights
- Store and product exploration
- Dark-themed user interface

## Development Contributions

ForecastIQ includes:

- Custom dashboard branding
- Dark-theme styling and readability improvements
- Independent LSTM training workflow
- LSTM training performance troubleshooting
- Training-only data normalization
- Forecasting model evaluation

## Original Project Attribution

ForecastIQ was developed using the Retail Sales Forecasting System
by kwerlr as its starting point.

Original repository:
https://github.com/kwerlr/retail-sales-forecasting

Redistribution permissions for the original source code have not
yet been confirmed. Public redistribution requires an applicable
license or permission from the original author.

## Author

**Youssef MOUHI**

Data Science & Artificial Intelligence

© 2026 Youssef MOUHI. All rights reserved for original contributions.
