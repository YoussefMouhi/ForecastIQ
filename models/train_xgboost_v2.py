"""Independent XGBoost training for ForecastIQ."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from utils.features_v2 import build_features, FEATURE_COLS
from utils.metrics_v2 import evaluate_forecast


OUTPUT = Path("validation/models_v2")
OUTPUT.mkdir(parents=True, exist_ok=True)

TRAIN_END = pd.Timestamp("2020-07-01")
TEST_START = pd.Timestamp("2020-10-01")


def load_dataset():
    sales = pd.read_csv(
        "data/train.csv",
        parse_dates=["date"],
    )

    stores = pd.read_csv("data/stores.csv")

    return sales.merge(
        stores[["store_nbr", "type"]],
        on="store_nbr",
        how="left",
        validate="many_to_one",
    )


def main():
    print("Loading ForecastIQ dataset...")

    data = load_dataset()

    print("Building independent forecasting features...")
    data = build_features(data)

    data = data.dropna(
        subset=FEATURE_COLS
    ).copy()

    encoders = {}

    for column, encoded_name in [
        ("store_nbr", "store_enc"),
        ("item_nbr", "item_enc"),
    ]:
        encoder = LabelEncoder()
        data[encoded_name] = encoder.fit_transform(
            data[column]
        )
        encoders[column] = encoder

    predictors = FEATURE_COLS + [
        "store_enc",
        "item_enc",
    ]

    training = data.loc[
        data["date"] < TRAIN_END
    ].copy()

    testing = data.loc[
        data["date"] >= TEST_START
    ].copy()

    print(f"Training samples: {len(training):,}")
    print(f"Testing samples: {len(testing):,}")

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    print("Training XGBoost...")

    model.fit(
        training[predictors],
        training["sales"],
    )

    predictions = np.maximum(
        model.predict(testing[predictors]),
        0,
    )

    metrics = evaluate_forecast(
        testing["sales"].to_numpy(),
        predictions,
        "XGBoost",
    )

    print("\nINDEPENDENT XGBOOST RESULTS")
    print(pd.DataFrame([metrics]).to_string(index=False))

    joblib.dump(
        model,
        OUTPUT / "xgboost_model.joblib",
    )

    joblib.dump(
        encoders,
        OUTPUT / "encoders.joblib",
    )

    joblib.dump(
        predictors,
        OUTPUT / "features.joblib",
    )

    pd.DataFrame([metrics]).to_csv(
        OUTPUT / "xgboost_results.csv",
        index=False,
    )

    print("\nModel saved under validation/models_v2/")


if __name__ == "__main__":
    main()
