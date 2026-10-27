from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / 'models' / 'multiseries' / 'lstm' / 'v2' / 'production'


class DirectGlobalLSTM(nn.Module):
    def __init__(self, n_stores, n_items, horizon=90):
        super().__init__()
        self.store_embedding = nn.Embedding(n_stores, 4)
        self.item_embedding = nn.Embedding(n_items, 6)
        self.lstm = nn.LSTM(5, 64, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(64 + 4 + 6 + 4 * horizon, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, horizon),
        )

    def forward(self, history, store, item, future_calendar):
        encoded, _ = self.lstm(history)
        context = torch.cat([
            encoded[:, -1],
            self.store_embedding(store),
            self.item_embedding(item),
            future_calendar.flatten(start_dim=1),
        ], dim=1)
        return self.head(context)


def calendar_features(dates):
    dates = pd.DatetimeIndex(dates)
    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy()
    return np.column_stack([
        np.sin(2 * np.pi * dow / 7),
        np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * doy / 365.25),
        np.cos(2 * np.pi * doy / 365.25),
    ]).astype(np.float32)


def load_v2():
    config = joblib.load(MODEL_DIR / "global_lstm_v2_config.pkl")
    model = DirectGlobalLSTM(
        config["n_stores"],
        config["n_items"],
        config["horizon"],
    )
    state = torch.load(
        MODEL_DIR / "global_lstm_v2.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    return model, config


def forecast_v2(df_ser, horizon, store, item, loaded=None):
    model, config = loaded if loaded is not None else load_v2()

    store, item = int(store), str(item)
    key = (store, item)
    if key not in config["scales"]:
        raise ValueError(f"Unknown store/item: {key}")

    seq_len = config["seq_len"]
    max_horizon = config["horizon"]
    if not 1 <= horizon <= max_horizon:
        raise ValueError(f"Horizon must be 1–{max_horizon}")

    history = df_ser.sort_values("date").tail(seq_len)
    if len(history) != seq_len:
        raise ValueError(f"At least {seq_len} observations required")

    scale = config["scales"][key]
    features = np.column_stack([
        history["sales"].to_numpy(dtype=np.float32) / scale,
        calendar_features(history["date"]),
    ])

    dates = pd.date_range(
        history["date"].max() + pd.Timedelta(days=1),
        periods=max_horizon,
        freq="D",
    )

    with torch.no_grad():
        predictions = model(
            torch.tensor(features[None], dtype=torch.float32),
            torch.tensor([config["store_map"][store]], dtype=torch.long),
            torch.tensor([config["item_map"][item]], dtype=torch.long),
            torch.tensor(calendar_features(dates)[None], dtype=torch.float32),
        ).squeeze().numpy()

    predictions = np.maximum(predictions[:horizon] * scale, 0)

    calibration_path = (
        MODEL_DIR.parent / "validation_analysis" / "uncertainty_calibration.csv"
    )

    if not calibration_path.exists():
        raise FileNotFoundError(
            f"Uncertainty calibration not found: {calibration_path}"
        )

    calibration = pd.read_csv(calibration_path)

    # Use the next available calibrated horizon.
    eligible = calibration[
        calibration["horizon"] >= horizon
    ].sort_values("horizon")

    if eligible.empty:
        raise ValueError("No uncertainty estimate for this horizon.")

    margin = float(
        eligible.iloc[0]["q90_absolute_error"]
    )

    lower = np.maximum(predictions - margin, 0)
    upper = predictions + margin

    return pd.DataFrame({
        "date": dates[:horizon],
        "forecast": predictions,
        "lower": lower,
        "upper": upper,
    })
