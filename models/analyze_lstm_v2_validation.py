from pathlib import Path
import sys
import importlib.util

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path.cwd()
OUT = ROOT / "models/multiseries/lstm/v2/validation_analysis"
OUT.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2020-07-01")
HORIZONS = [7, 14, 30, 60, 90]
DAYS = 90

torch.set_num_threads(2)


class LSTMV1(nn.Module):
    def __init__(self, n_stores, n_items):
        super().__init__()
        self.store_embedding = nn.Embedding(n_stores, 4)
        self.item_embedding = nn.Embedding(n_items, 6)
        self.lstm = nn.LSTM(1, 64, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(74, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, store, item):
        output, _ = self.lstm(x)
        combined = torch.cat([
            output[:, -1],
            self.store_embedding(store),
            self.item_embedding(item)
        ], dim=1)
        return self.fc(combined)


def load_models():
    base = ROOT / "models/multiseries/lstm"
    v2_dir = base / "v2"

    config1 = joblib.load(base / "global_lstm_config.pkl")
    model1 = LSTMV1(
        config1["n_stores"], config1["n_items"]
    )
    model1.load_state_dict(torch.load(
        base / "global_lstm.pt",
        map_location="cpu",
        weights_only=True
    ))
    model1.eval()

    path = ROOT / "models/train_global_lstm_v2.py"
    spec = importlib.util.spec_from_file_location("lstm_v2", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    config2 = joblib.load(
        v2_dir / "global_lstm_v2_config.pkl"
    )
    model2 = module.DirectGlobalLSTM(
        config2["n_stores"], config2["n_items"]
    )
    model2.load_state_dict(torch.load(
        v2_dir / "global_lstm_v2.pt",
        map_location="cpu",
        weights_only=True
    ))
    model2.eval()

    assert config2["horizon"] == DAYS

    return model1, config1, model2, config2, module.calendar_features


def predict_v1(model, config, history, store, item):
    scale = config["scales"][(int(store), str(item))]
    values = (history["sales"].to_numpy() / scale).tolist()
    length = config["seq_len"]

    s = torch.tensor([config["store_map"][int(store)]])
    i = torch.tensor([config["item_map"][str(item)]])

    predictions = []

    with torch.no_grad():
        for _ in range(DAYS):
            x = torch.tensor(
                values[-length:], dtype=torch.float32
            ).reshape(1, length, 1)

            prediction = max(
                0.0, model(x, s, i).item()
            )

            predictions.append(prediction * scale)
            values.append(prediction)

    return np.array(predictions)


def predict_v2(model, config, calendar_fn, history, store, item):
    scale = config["scales"][(int(store), str(item))]
    length = config["seq_len"]
    recent = history.tail(length)

    features = np.column_stack([
        recent["sales"].to_numpy(dtype=np.float32) / scale,
        calendar_fn(recent["date"])
    ])

    future_dates = pd.date_range(
        START, periods=DAYS, freq="D"
    )

    x = torch.tensor(features[None], dtype=torch.float32)
    future = torch.tensor(
        calendar_fn(future_dates)[None],
        dtype=torch.float32
    )

    s = torch.tensor([config["store_map"][int(store)]])
    i = torch.tensor([config["item_map"][str(item)]])

    with torch.no_grad():
        predictions = model(
            x, s, i, future
        ).squeeze().numpy()

    return np.maximum(predictions * scale, 0)


def metrics(data):
    actual = data["actual"].to_numpy()
    predicted = data["predicted"].to_numpy()

    errors = actual - predicted

    return {
        "MAE": np.mean(np.abs(errors)),
        "RMSE": np.sqrt(np.mean(errors ** 2)),
        "WAPE": (
            100 * np.sum(np.abs(errors)) /
            np.sum(np.abs(actual))
        ),
        "n_predictions": len(data)
    }


def main():
    print("Loading data and LSTM checkpoints...", flush=True)

    df = pd.read_csv(
        ROOT / "data/train.csv",
        parse_dates=["date"]
    )

    model1, config1, model2, config2, calendar_fn = load_models()

    rows = []
    groups = list(df.groupby(["store_nbr", "item_nbr"]))

    for index, ((store, item), group) in enumerate(groups, 1):
        group = group.sort_values("date")

        history = group[group["date"] < START]
        actual = group[
            (group["date"] >= START) &
            (group["date"] < START + pd.Timedelta(days=DAYS))
        ].sort_values("date")

        if len(actual) != DAYS:
            raise ValueError(f"Incomplete test data: {store}/{item}")

        predictions = {
            "LSTM V1": predict_v1(
                model1, config1, history, store, item
            ),
            "LSTM V2": predict_v2(
                model2, config2, calendar_fn,
                history, store, item
            )
        }

        for name, forecast in predictions.items():
            for step, (true, predicted) in enumerate(
                zip(actual["sales"], forecast), 1
            ):
                rows.append({
                    "model": name,
                    "store_nbr": store,
                    "item_nbr": item,
                    "step": step,
                    "actual": float(true),
                    "predicted": float(predicted)
                })

        print(
            f"[{index:02d}/{len(groups)}] "
            f"Store {store} / {item}: completed",
            flush=True
        )

    predictions = pd.DataFrame(rows)
    predictions.to_csv(
        OUT / "validation_predictions.csv", index=False
    )

    results = []

    for horizon in HORIZONS:
        for name in ["LSTM V1", "LSTM V2"]:
            subset = predictions[
                (predictions["model"] == name) &
                (predictions["step"] <= horizon)
            ]

            results.append({
                "model": name,
                "horizon": horizon,
                "n_series": len(groups),
                **metrics(subset)
            })

    summary = pd.DataFrame(results)
    summary = summary.sort_values(["horizon", "MAE"])

    summary.to_csv(
        OUT / "validation_horizon_comparison.csv",
        index=False
    )

    print("\nLSTM V1 vs V2 RESULTS:\n")
    print(summary.round(3).to_string(index=False))
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
