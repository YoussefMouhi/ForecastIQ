"""Retail forecasting features for the ForecastIQ independent implementation."""

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "day_of_week", "day_of_month", "week_of_year",
    "month", "quarter", "year", "is_weekend",
    "sin_year", "cos_year", "sin_week", "cos_week",
    "is_holiday", "is_christmas", "is_new_year",
    "onpromotion",
    "sales_lag_1", "sales_lag_7",
    "sales_lag_14", "sales_lag_28",
    "sales_roll_mean_7", "sales_roll_std_7",
    "sales_roll_mean_14", "sales_roll_std_14",
    "sales_roll_mean_30", "sales_roll_std_30",
]


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build calendar and historical-demand predictors per store/item."""

    required = {
        "date", "store_nbr", "item_nbr", "sales", "onpromotion"
    }
    missing = required.difference(data.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"])

    frame = frame.sort_values(
        ["date", "store_nbr", "item_nbr"]
    ).copy()

    calendar = frame["date"].dt

    frame["day_of_week"] = calendar.dayofweek
    frame["day_of_month"] = calendar.day
    frame["week_of_year"] = calendar.isocalendar().week.astype(int)
    frame["month"] = calendar.month
    frame["quarter"] = calendar.quarter
    frame["year"] = calendar.year

    frame["is_weekend"] = (
        frame["day_of_week"] >= 5
    ).astype(int)

    year_angle = 2 * np.pi * calendar.dayofyear / 365
    week_angle = 2 * np.pi * calendar.dayofweek / 7

    frame["sin_year"] = np.sin(year_angle)
    frame["cos_year"] = np.cos(year_angle)
    frame["sin_week"] = np.sin(week_angle)
    frame["cos_week"] = np.cos(week_angle)

    frame["is_christmas"] = (
        (calendar.month == 12)
        & (calendar.day >= 20)
        & (calendar.day <= 31)
    ).astype(int)

    frame["is_new_year"] = (
        (calendar.month == 1)
        & (calendar.day <= 3)
    ).astype(int)

    frame["is_holiday"] = (
        frame["is_christmas"] | frame["is_new_year"]
    ).astype(int)

    keys = ["store_nbr", "item_nbr"]
    groups = frame.groupby(keys, sort=False)

    for lag in (1, 7, 14, 28):
        frame[f"sales_lag_{lag}"] = groups["sales"].shift(lag)

    previous_sales = groups["sales"].shift(1)

    for window in (7, 14, 30):
        history = previous_sales.groupby(
            [frame[key] for key in keys],
            sort=False
        )

        frame[f"sales_roll_mean_{window}"] = history.transform(
            lambda series: series.rolling(
                window, min_periods=1
            ).mean()
        )

        frame[f"sales_roll_std_{window}"] = history.transform(
            lambda series: series.rolling(
                window, min_periods=1
            ).std()
        )

    return frame
