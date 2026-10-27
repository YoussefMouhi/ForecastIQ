"""ForecastIQ experimental global direct multi-horizon LSTM V2.

Train through 2020-06-30, validate on forecast origins in July-September,
and reserve October-December 2020 for final evaluation.
"""
import copy
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'models' / 'multiseries' / 'lstm' / 'v2'
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
SEQ_LEN = 60
HORIZON = 90
EPOCHS = 35
PATIENCE = 6
BATCH_SIZE = 128
LR = 0.001
TRAIN_END = pd.Timestamp('2020-06-30')
VALID_START = pd.Timestamp('2020-07-01')
VALID_END = pd.Timestamp('2020-09-30')
TEST_START = pd.Timestamp('2020-10-01')

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(2)


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


class DirectGlobalLSTM(nn.Module):
    def __init__(self, n_stores, n_items):
        super().__init__()
        self.store_embedding = nn.Embedding(n_stores, 4)
        self.item_embedding = nn.Embedding(n_items, 6)
        self.lstm = nn.LSTM(input_size=5, hidden_size=64, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(64 + 4 + 6 + 4 * HORIZON, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, HORIZON),
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


def main():
    df = pd.read_csv(ROOT / 'data' / 'train.csv', parse_dates=['date'])
    stores = sorted(int(x) for x in df.store_nbr.unique())
    items = sorted(str(x) for x in df.item_nbr.unique())
    store_map = {v: i for i, v in enumerate(stores)}
    item_map = {v: i for i, v in enumerate(items)}
    scales = {}
    train_arrays = [[] for _ in range(5)]
    valid_arrays = [[] for _ in range(5)]

    for (store, item), group in df.groupby(['store_nbr', 'item_nbr']):
        group = group.sort_values('date').reset_index(drop=True)
        dates = pd.DatetimeIndex(group.date)
        sales = group.sales.to_numpy(dtype=np.float32)
        scale = max(float(sales[dates <= TRAIN_END].mean()), 1.0)
        scales[(int(store), str(item))] = scale
        normalized = sales / scale
        cal = calendar_features(dates)
        # Each example forecasts 90 consecutive days. Never let training
        # labels cross June 30 or validation labels cross September 30.
        for origin in range(SEQ_LEN, len(group) - HORIZON + 1, 7):
            label_start = dates[origin]
            label_end = dates[origin + HORIZON - 1]
            if label_end <= TRAIN_END:
                target = train_arrays
            elif VALID_START <= label_start and label_end <= VALID_END:
                target = valid_arrays
            else:
                continue
            hist = np.column_stack([normalized[origin-SEQ_LEN:origin], cal[origin-SEQ_LEN:origin]])
            future_cal = cal[origin:origin+HORIZON]
            future_sales = normalized[origin:origin+HORIZON]
            target[0].append(hist)
            target[1].append(store_map[int(store)])
            target[2].append(item_map[str(item)])
            target[3].append(future_cal)
            target[4].append(future_sales)

    def to_loader(arrays, shuffle):
        if not arrays[0]:
            raise RuntimeError('No samples available for this split. Check date boundaries.')
        dataset = TensorDataset(
            torch.tensor(np.stack(arrays[0]), dtype=torch.float32),
            torch.tensor(arrays[1], dtype=torch.long),
            torch.tensor(arrays[2], dtype=torch.long),
            torch.tensor(np.stack(arrays[3]), dtype=torch.float32),
            torch.tensor(np.stack(arrays[4]), dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = to_loader(train_arrays, True)
    valid_loader = to_loader(valid_arrays, False)
    print(f'Train samples: {len(train_loader.dataset)} | Validation samples: {len(valid_loader.dataset)}', flush=True)
    model = DirectGlobalLSTM(len(stores), len(items))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    best_loss = float('inf')
    best_weights = None
    wait = 0
    history_rows = []

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        model.train()
        train_sum = 0.0
        for x, s, i, future, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x, s, i, future), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_sum += loss.item() * len(y)
        train_loss = train_sum / len(train_loader.dataset)
        model.eval()
        valid_sum = 0.0
        with torch.no_grad():
            for x, s, i, future, y in valid_loader:
                loss = loss_fn(model(x, s, i, future), y)
                valid_sum += loss.item() * len(y)
        valid_loss = valid_sum / len(valid_loader.dataset)
        history_rows.append({'epoch': epoch, 'train_loss': train_loss, 'valid_loss': valid_loss})
        print(f'Epoch {epoch:02d} | train={train_loss:.5f} | valid={valid_loss:.5f} | {time.time()-start:.1f}s', flush=True)
        if valid_loss < best_loss - 1e-4:
            best_loss = valid_loss
            best_weights = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print('Early stopping.', flush=True)
                break

    torch.save(best_weights, OUT / 'global_lstm_v2.pt')
    joblib.dump({
        'store_map': store_map, 'item_map': item_map, 'scales': scales,
        'seq_len': SEQ_LEN, 'horizon': HORIZON,
        'n_stores': len(stores), 'n_items': len(items),
        'train_end': str(TRAIN_END.date()),
        'validation_period': ['2020-07-01', '2020-09-30'],
        'test_start': str(TEST_START.date()),
        'architecture': 'direct_lstm_v2_calendar_no_future_promotions',
    }, OUT / 'global_lstm_v2_config.pkl')
    pd.DataFrame(history_rows).to_csv(OUT / 'training_history.csv', index=False)
    print(f'\nSaved V2 checkpoint to {OUT}', flush=True)
    print('Existing V1 deployment files were not modified.', flush=True)


if __name__ == '__main__':
    main()
