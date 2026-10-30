"""ForecastIQ model evaluation metrics."""

import numpy as np


def evaluate_forecast(actual, predicted, model_name="Model"):
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)

    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted arrays must have equal length.")

    if actual.size == 0:
        raise ValueError("Cannot evaluate empty arrays.")

    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Evaluation inputs must contain only finite values.")

    errors = actual - predicted

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(np.square(errors)))

    # Keep the original ForecastIQ convention for reproducibility.
    denominator = np.abs(actual) + 1e-8
    mape = np.mean(np.abs(errors) / denominator) * 100

    return {
        "model": model_name,
        "MAE": round(float(mae), 3),
        "RMSE": round(float(rmse), 3),
        "MAPE": round(float(mape), 3),
    }
