"""Train and evaluate independent SARIMA and Prophet models."""

from pathlib import Path

import numpy as np
import pandas as pd

from utils.metrics_v2 import evaluate_forecast

OUTPUT = Path("validation/models_v2")
OUTPUT.mkdir(parents=True, exist_ok=True)

TRAIN_END = pd.Timestamp("2020-07-01")
TEST_START = pd.Timestamp("2020-10-01")


def load_series():
    data = pd.read_csv("data/train.csv", parse_dates=["date"])

    series = data.loc[
        (data["store_nbr"] == 1)
        & (data["item_nbr"] == "ITEM_001"),
        ["date", "sales"],
    ].sort_values("date").reset_index(drop=True)

    if series.empty:
        raise ValueError("No matching store/item data.")

    return series


def train_sarima(train, test):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        train["sales"].to_numpy(),
        order=(1, 1, 1),
        seasonal_order=(1, 1, 0, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted = model.fit(disp=False)

    start = train["date"].max() + pd.Timedelta(days=1)
    end = test["date"].max()
    dates = pd.date_range(start, end, freq="D")

    predictions = fitted.forecast(steps=len(dates))

    forecast = pd.DataFrame({
        "date": dates,
        "prediction": np.maximum(
            np.asarray(predictions), 0
        ),
    })

    aligned = test[["date"]].merge(
        forecast,
        on="date",
        how="left",
        validate="one_to_one",
    )

    if aligned["prediction"].isna().any():
        raise ValueError("Missing SARIMA forecast dates.")

    return aligned["prediction"].to_numpy()


def train_prophet(train, test):
    from prophet import Prophet

    training = train.rename(
        columns={"date": "ds", "sales": "y"}
    )

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
    )

    model.fit(training)

    future = pd.DataFrame({
        "ds": test["date"].to_numpy()
    })

    prediction = model.predict(future)["yhat"].to_numpy()

    return np.maximum(prediction, 0)


def main():
    series = load_series()

    train = series[series["date"] < TRAIN_END].copy()
    test = series[series["date"] >= TEST_START].copy()

    results = []

    for name, trainer in [
        ("SARIMA", train_sarima),
        ("Prophet", train_prophet),
    ]:
        print(f"\nTraining {name}...", flush=True)

        predictions = trainer(train, test)

        if len(predictions) != len(test):
            raise ValueError(f"{name}: prediction length mismatch")

        metrics = evaluate_forecast(
            test["sales"].to_numpy(),
            predictions,
            name,
        )

        results.append(metrics)
        print(metrics, flush=True)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT / "statistical_results.csv",
        index=False,
    )

    print("\nFORECASTIQ STATISTICAL RESULTS")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
