<div align="center">

<img src="assets/forecastiq_banner.png" alt="ForecastIQ Banner" width="100%">

<br><br>

<img src="assets/forecastiq_logo.png" alt="ForecastIQ Logo" width="90">

<h1>ForecastIQ</h1>

<h3>Intelligent Retail Sales Forecasting & Demand Analytics</h3>

**Predict Demand · Analyze Trends · Compare Models · Optimize Inventory**

<br>

<img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white">
<img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white">
<img src="https://img.shields.io/badge/XGBoost-Machine%20Learning-F28C28">
<img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white">
<img src="https://img.shields.io/badge/LSTM%20V1-Recursive-8A2BE2">
<img src="https://img.shields.io/badge/LSTM%20V2-Multi--Horizon-007ACC">
<img src="https://img.shields.io/badge/Prophet-Time%20Series-785EF0">
<img src="https://img.shields.io/badge/SARIMA-Statistical-008B8B">

<br><br>

*An end-to-end retail forecasting platform combining machine learning, statistical modeling, deep learning, and interactive business intelligence.*

**[Overview](#overview) · [Architecture](#system-architecture) · [Models](#forecasting-models) · [Results](#model-performance) · [Dashboard](#interactive-dashboard) · [Installation](#installation)**

</div>

---

## Overview

ForecastIQ is a retail demand forecasting and analytics platform designed to transform historical sales data into actionable business insights.

Retail demand forecasting is challenging because sales patterns depend on multiple factors, including historical demand, seasonality, promotional activity, and differences between products and stores.

ForecastIQ addresses these challenges by integrating five forecasting approaches into a unified interactive application.

The platform allows users to explore historical sales, generate future demand predictions, compare forecasting methods, and support inventory planning.

### Project Objectives

The primary objectives are to:

- Develop a forecasting system for multiple stores and products.
- Compare machine learning, statistical, and deep-learning approaches.
- Generate demand forecasts for horizons of 7 to 90 days.
- Analyze historical sales trends and promotional effects.
- Support inventory planning through demand estimates.
- Visualize forecasting performance and uncertainty.
- Provide an accessible interface for exploring model predictions.

---

## Key Features

### Multi-Series Demand Forecasting

ForecastIQ supports forecasting across 50 store-item combinations.

Users can select an individual store and product to explore historical sales and generate predictions.

### Multiple Forecasting Architectures

The system integrates:

- XGBoost
- SARIMA
- Prophet
- LSTM V1
- LSTM V2

Each model represents a different approach to time-series forecasting.

### Flexible Forecast Horizons

Users can generate forecasts for periods ranging from 7 to 90 days.

This supports short-term operational planning and longer-term demand analysis.

### Interactive Analytics

The dashboard provides visualizations of historical demand, forecasted sales, model comparisons, and business metrics.

### Inventory Decision Support

Demand forecasts are incorporated into inventory-oriented analysis to help users evaluate expected stock requirements.

### Forecast Uncertainty

LSTM V2 forecasts include an approximate uncertainty band derived from validation residuals.

These bands provide an indication of forecasting variability but should not be interpreted as formally calibrated prediction intervals.

---

## Dataset

ForecastIQ uses a structured retail sales dataset containing daily observations across multiple stores and products.

### Dataset Characteristics

| Attribute | Description |
|---|---|
| Total observations | 73,050 |
| Stores | 5 |
| Products | 10 |
| Time series | 50 |
| Frequency | Daily |
| Start date | January 1, 2017 |
| End date | December 31, 2020 |
| Observations per series | 1,461 |
| Forecast horizons | 7–90 days |

### Main Variables

| Variable | Description |
|---|---|
| `date` | Observation date |
| `store_nbr` | Store identifier |
| `item_nbr` | Product identifier |
| `family` | Product category |
| `sales` | Recorded daily sales |
| `onpromotion` | Promotional activity information |

The dataset supports analysis of temporal demand patterns and differences between individual store-item combinations.

### Forecasting Challenges

Retail sales data may exhibit:

- Weekly and longer-term seasonal patterns.
- Changing demand levels.
- Promotional effects.
- Product-specific behavior.
- Differences between stores.
- Demand fluctuations that become harder to predict at longer horizons.

These characteristics motivate the comparison of multiple forecasting techniques.

---

## System Architecture

ForecastIQ follows a modular architecture separating data processing, model development, forecasting inference, evaluation, and visualization.

### Processing Pipeline

```text
                   RETAIL SALES DATA
                          |
                          v
                  DATA PREPROCESSING
                          |
                          v
                  FEATURE ENGINEERING
                          |
           +--------------+--------------+
           |              |              |
           v              v              v
     MACHINE LEARNING  STATISTICAL   DEEP LEARNING
           |           FORECASTING         |
           v              |              v
        XGBoost           |           LSTM V1
                          |           LSTM V2
                     +----+----+
                     |         |
                     v         v
                   SARIMA    Prophet
                     |         |
           +---------+---------+
           |
           v
       FORECAST GENERATION
           |
           v
       MODEL EVALUATION
           |
           v
      STREAMLIT DASHBOARD
           |
     +-----+-----+----------+-----------+
     |           |          |           |
     v           v          v           v
  Forecast    Inventory   Analytics   Comparison
```

### Architecture Components

**Data Layer**

Stores and processes historical retail sales observations.

**Forecasting Layer**

Contains statistical, machine-learning, and deep-learning forecasting implementations.

**Inference Layer**

Loads trained models and generates predictions for selected store-item combinations.

**Evaluation Layer**

Calculates forecasting error metrics and supports model comparisons.

**Presentation Layer**

Provides an interactive Streamlit interface for forecasts, analytics, and inventory-oriented insights.

---

## Forecasting Models

ForecastIQ integrates five forecasting approaches.

### 1. XGBoost — Gradient-Boosted Forecasting

XGBoost is a tree-based machine-learning algorithm that combines multiple decision trees to learn relationships between input features and target values.

Within ForecastIQ, XGBoost is used as a global forecasting model.

**Advantages**

- Learns nonlinear relationships.
- Supports engineered temporal features.
- Can incorporate promotional information.
- Provides strong predictive performance on structured data.

**Considerations**

Prediction quality depends on feature engineering and the availability of future explanatory variables.

---

### 2. SARIMA — Seasonal Statistical Forecasting

SARIMA extends the ARIMA framework by incorporating seasonal components.

It models time-series behavior through autoregressive relationships, differencing, moving averages, and seasonal terms.

ForecastIQ uses SARIMA as a statistical forecasting baseline across individual store-item series.

**Advantages**

- Explicitly models temporal dependencies.
- Provides an interpretable statistical framework.
- Useful for analyzing seasonal behavior.

**Considerations**

SARIMA can be less flexible when sales exhibit complex nonlinear relationships or rapidly changing patterns.

---

### 3. Prophet — Trend and Seasonality Modeling

Prophet is a forecasting framework designed to model time-series trends and recurring seasonal patterns.

ForecastIQ uses Prophet to provide an alternative statistical forecasting approach.

**Advantages**

- Flexible trend modeling.
- Built-in seasonality handling.
- Convenient time-series forecasting workflow.

**Considerations**

Performance depends on whether the model's assumptions adequately represent the underlying demand patterns.

---

### 4. LSTM V1 — Recursive Deep-Learning Forecasting

LSTM V1 uses a recurrent neural-network architecture designed to learn sequential relationships in historical observations.

The model generates forecasts recursively.

In recursive forecasting, previously generated predictions can become inputs for subsequent forecasting steps.

**Advantages**

- Learns temporal dependencies.
- Can model nonlinear sequential relationships.
- Provides a deep-learning alternative to statistical models.

**Limitations**

Recursive forecasting can accumulate prediction errors as the forecast horizon increases.

This motivated the development of a second LSTM architecture.

---

### 5. LSTM V2 — Direct Multi-Horizon Forecasting

LSTM V2 is an enhanced forecasting architecture implemented using PyTorch.

Unlike the recursive approach, LSTM V2 generates multi-step forecasts directly.

The model is designed to produce predictions for horizons extending up to 90 days.

### Key Characteristics

- Global neural-network architecture.
- Direct multi-horizon prediction.
- PyTorch implementation.
- Support for multiple store-item series.
- Integration with the Streamlit dashboard.
- Approximate forecast uncertainty visualization.

### Recursive vs. Direct Forecasting

**LSTM V1**

```text
Historical Data
      |
      v
   LSTM Model
      |
      v
 Prediction t+1
      |
      v
 Prediction t+2
      |
      v
      ...
```

**LSTM V2**

```text
Historical Data
      |
      v
   LSTM Model
      |
      v
 Multi-Horizon Output
      |
      +--> Day 1
      +--> Day 2
      +--> Day 3
      |
      ...
      |
      +--> Day 90
```

The direct approach avoids recursively feeding every generated forecast into the next prediction step.

### Production Inference

The production LSTM V2 model was retrained using historical observations through December 31, 2020.

The inference pipeline supports forecasts for all 50 store-item combinations.

Validation tests successfully generated predictions for 7-day, 30-day, and 90-day horizons across all combinations.

---

## Model Performance

ForecastIQ evaluates forecasting models using error metrics that quantify the differences between actual and predicted sales.

### Evaluation Metrics

**Mean Absolute Error (MAE)**

MAE measures the average absolute difference between predicted and actual sales.

Lower values indicate smaller average prediction errors.

**Weighted Absolute Percentage Error (WAPE)**

WAPE expresses the total absolute forecasting error relative to total actual demand.

Lower values indicate better relative forecasting accuracy.

### 30-Day Forecasting Results

| Rank | Model | MAE | WAPE |
|---:|---|---:|---:|
| 1 | **XGBoost** | **10.35** | **9.49%** |
| 2 | LSTM V2 | 12.83 | 11.76% |
| 3 | Prophet | 13.44 | 12.32% |
| 4 | SARIMA | 16.67 | 15.28% |
| 5 | LSTM V1 | 17.05 | 15.63% |

### Key Observations

**XGBoost achieved the lowest reported forecasting error** in the preliminary 30-day comparison.

LSTM V2 outperformed LSTM V1, indicating improved performance from the direct multi-horizon architecture under the evaluated conditions.

Prophet achieved intermediate performance, while SARIMA and LSTM V1 recorded higher errors in this comparison.

### Evaluation Limitations

These results should be interpreted as preliminary.

The models were not all evaluated under identical training and information-availability conditions.

In particular:

- Training cutoffs differed between some models.
- The XGBoost evaluation used observed future promotion information.
- Forecasting strategies differed between architectures.

Consequently, the reported ranking is useful for exploratory comparison but does not constitute a fully controlled benchmark.

A future evaluation should use consistent rolling-origin validation, aligned training windows, and equivalent future-information assumptions.

---

## Interactive Dashboard

ForecastIQ provides a dark-themed Streamlit application organized into five main pages.

### 1. Overview

The Overview page provides a high-level summary of historical retail sales.

It helps users explore general demand behavior and key business indicators.

### 2. Forecast

The Forecast page is the central forecasting interface.

Users can:

- Select a store.
- Select a product.
- Choose a forecasting model.
- Configure the prediction horizon.
- Explore historical and predicted demand.
- Inspect available forecast uncertainty visualizations.

### 3. Inventory

The Inventory page connects demand forecasting with inventory-oriented decision support.

It helps users interpret expected future demand in an operational context.

### 4. Model Comparison

The Model Comparison page displays forecasting evaluation results.

It allows users to compare different approaches using quantitative error metrics.

### 5. Exploratory Data Analysis

The EDA page provides visual analysis of historical sales behavior.

It supports exploration of demand patterns, temporal changes, and relationships within the retail dataset.

---

## Project Structure

The principal project components are organized as follows:

```text
ForecastIQ-Base/
|
|-- app.py
|
|-- assets/
|   |-- forecastiq_banner.png
|   |-- forecastiq_logo.png
|
|-- data/
|   |-- train.csv
|   |-- stores.csv
|
|-- models/
|   |-- lstm_v2_inference.py
|   |-- multiseries/
|   |   |-- evaluation/
|   |   |-- lstm/
|   |       |-- v2/
|   |           |-- production/
|
|-- utils/
|
|-- requirements.txt
|-- README.md
|-- .gitignore
```

The repository may omit large trained model artifacts, which are required for some forecasting operations.

---

## Installation

### Prerequisites

- Python
- Git
- A supported Python environment
- Dependencies listed in `requirements.txt`
- Required dataset and trained model artifacts

### 1. Clone the Repository

```bash
git clone https://github.com/YoussefMouhi/ForecastIQ.git
cd ForecastIQ
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

For Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the Data and Models

Ensure that the required dataset files are available in the `data/` directory.

Some trained forecasting artifacts are excluded from Git because of their size.

Those models must be supplied or regenerated using the corresponding training pipeline before their forecasts can be generated.

### 5. Launch the Dashboard

```bash
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python -m streamlit run app.py
```

Open the local address displayed by Streamlit.

The application provides interactive access to its forecasting and analytics pages.

---

## Technical Challenges

### Multi-Series Forecasting

Supporting multiple stores and products requires a consistent approach to data processing, model training, and inference.

### Long-Horizon Prediction

Forecasting accuracy can deteriorate as the prediction horizon increases.

LSTM V2 explores direct multi-horizon forecasting as an alternative to recursive prediction.

### Model Integration

Combining statistical models, gradient-boosted trees, and neural networks requires compatible forecasting interfaces and consistent output handling.

### Evaluation Consistency

Different models may use different assumptions and training procedures.

A key future objective is to establish a unified evaluation protocol.

### Model Deployment

Large trained artifacts create challenges for repository distribution and deployment.

Separating source code from model checkpoints helps keep the Git repository manageable.

---

## Limitations

ForecastIQ is a forecasting and analytics prototype rather than a production-ready inventory management system.

Current limitations include:

- Preliminary rather than fully controlled model comparisons.
- Approximate uncertainty bands.
- Dependence on historical retail sales patterns.
- Limited availability of external explanatory variables.
- Large model artifacts that may require separate distribution.
- No automated production retraining pipeline.

These limitations provide opportunities for further engineering and experimentation.

---

## Future Improvements

Potential improvements include:

- Rolling-origin time-series cross-validation.
- Better-calibrated forecast prediction intervals.
- Holiday and calendar feature integration.
- Additional external demand signals.
- Automated model retraining.
- Model monitoring and drift detection.
- Enhanced inventory optimization.
- Cloud deployment.
- More comprehensive forecasting benchmarks.

---

## Author

**Youssef MOUHI**

Computer and Network Engineering  
Artificial Intelligence & Data Science

Project development areas include forecasting, deep learning, model evaluation, interactive analytics, and application integration.

---

<div align="center">

### ForecastIQ

**Turning Historical Sales into Smarter Decisions**

© 2026 Youssef MOUHI. All rights reserved for original contributions.

</div>