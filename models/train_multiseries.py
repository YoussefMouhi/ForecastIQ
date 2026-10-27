import argparse
import time
import warnings
from pathlib import Path

import joblib
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "train.csv"
OUTPUT_DIR = ROOT / "models" / "multiseries"


def train_series(store, item, series):
    series = series.sort_values("date").reset_index(drop=True)

    if len(series) < 365:
        print(f"SKIPPED: Store {store}, {item}: insufficient history")
        return

    print(f"\nTraining Store {store} / {item}", flush=True)

    # SARIMA
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        start = time.time()

        model = SARIMAX(
            series["sales"].to_numpy(dtype=float),
            order=(1, 1, 1),
            seasonal_order=(1, 1, 0, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=50)

        path = OUTPUT_DIR / "sarima" / f"store_{store}_{item}.pkl"
        model.save(str(path))

        print(
            f"  SARIMA saved: {path.name} "
            f"({time.time() - start:.1f}s)",
            flush=True,
        )

    except Exception as exc:
        print(f"  SARIMA FAILED: {exc}", flush=True)

    # Prophet
    try:
        from prophet import Prophet

        start = time.time()

        prophet_data = series[["date", "sales"]].rename(
            columns={"date": "ds", "sales": "y"}
        )

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
        )

        model.fit(prophet_data)

        path = OUTPUT_DIR / "prophet" / f"store_{store}_{item}.pkl"
        joblib.dump(model, path)

        print(
            f"  Prophet saved: {path.name} "
            f"({time.time() - start:.1f}s)",
            flush=True,
        )

    except Exception as exc:
        print(f"  Prophet FAILED: {exc}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=int, default=None)
    parser.add_argument("--item", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    (OUTPUT_DIR / "sarima").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "prophet").mkdir(parents=True, exist_ok=True)

    combinations = df.groupby(["store_nbr", "item_nbr"], sort=True)

    total = 0

    for (store, item), series in combinations:
        if args.store is not None and store != args.store:
            continue

        if args.item is not None and item != args.item:
            continue

        train_series(store, item, series)
        total += 1

    print(f"\nCompleted training attempts for {total} combinations.")


if __name__ == "__main__":
    main()
