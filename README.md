<div align="center">

<img src="assets/forecastiq_banner.png" alt="ForecastIQ Banner" width="100%">

<br><br>

<h1>
  <img src="assets/forecastiq_logo.png" width="55" alt="ForecastIQ Logo">
  ForecastIQ
</h1>

<hr>

<h2>Intelligent Retail Sales Forecasting & Demand Analytics</h2>

<h3>Predict future demand · Compare forecasting models · Optimize inventory decisions</h3>

<br>

<img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white">
<img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white">
<img src="https://img.shields.io/badge/XGBoost-Machine%20Learning-F28C28">
<img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white">
<img src="https://img.shields.io/badge/LSTM%20V1-Recursive%20Forecasting-8A2BE2">
<img src="https://img.shields.io/badge/LSTM%20V2-Multi--Horizon%20Forecasting-007ACC">

<img src="https://img.shields.io/badge/Prophet-Time%20Series-785EF0">
<img src="https://img.shields.io/badge/SARIMA-Statistical%20Forecasting-008B8B">

<br><br>

<p>
<i>
An end-to-end machine learning and deep learning platform for forecasting
retail sales, analyzing demand patterns, comparing predictive models,
and supporting data-driven inventory decisions.
</i>
</p>

<br>

<a href="#installation">Quick Start</a> ·
<a href="#project-architecture">Architecture</a> ·
<a href="#model-performance">Model Results</a> ·
<a href="#dashboard">Dashboard</a> ·
<a href="#limitations">Limitations</a>

</div>

---

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
