import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(2)
torch.manual_seed(42)
np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models" / "multiseries" / "lstm"
OUT.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 30
EPOCHS = 15
BATCH_SIZE = 256

print("Loading data...", flush=True)
df = pd.read_csv(ROOT / "data" / "train.csv", parse_dates=["date"])

stores = sorted(df["store_nbr"].unique().tolist())
items = sorted(df["item_nbr"].unique().tolist())

store_map = {int(v): i for i, v in enumerate(stores)}
item_map = {str(v): i for i, v in enumerate(items)}

# Training uses only observations before the validation cutoff.
CUTOFF = pd.Timestamp("2020-07-01")

scales = {}
X, S, I, Y = [], [], [], []

for (store, item), group in df.groupby(["store_nbr", "item_nbr"]):
    group = group.sort_values("date")
    train = group[group["date"] < CUTOFF]
    values = train["sales"].to_numpy(dtype=np.float32)

    scale = max(float(np.mean(values)), 1.0)
    key = (int(store), str(item))
    scales[key] = scale

    normalized = values / scale

    for idx in range(SEQ_LEN, len(normalized)):
        X.append(normalized[idx - SEQ_LEN:idx])
        S.append(store_map[int(store)])
        I.append(item_map[str(item)])
        Y.append(normalized[idx])

X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
S = torch.tensor(S, dtype=torch.long)
I = torch.tensor(I, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)

loader = DataLoader(
    TensorDataset(X, S, I, Y),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

print(f"Training sequences: {len(X)}", flush=True)


class GlobalLSTM(nn.Module):
    def __init__(self, n_stores, n_items):
        super().__init__()
        self.store_embedding = nn.Embedding(n_stores, 4)
        self.item_embedding = nn.Embedding(n_items, 6)
        self.lstm = nn.LSTM(1, 64, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(64 + 4 + 6, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, store, item):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        store_emb = self.store_embedding(store)
        item_emb = self.item_embedding(item)
        combined = torch.cat([last, store_emb, item_emb], dim=1)
        return self.fc(combined)


model = GlobalLSTM(len(stores), len(items))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

print("Starting training...", flush=True)

for epoch in range(EPOCHS):
    started = time.time()
    model.train()
    total = 0.0

    for xb, sb, ib, yb in loader:
        optimizer.zero_grad()
        prediction = model(xb, sb, ib)
        loss = loss_fn(prediction, yb)
        loss.backward()
        optimizer.step()
        total += loss.item()

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"| loss={total / len(loader):.6f} "
        f"| time={time.time() - started:.1f}s",
        flush=True,
    )

torch.save(model.state_dict(), OUT / "global_lstm.pt")

joblib.dump(
    {
        "store_map": store_map,
        "item_map": item_map,
        "scales": scales,
        "seq_len": SEQ_LEN,
        "n_stores": len(stores),
        "n_items": len(items),
    },
    OUT / "global_lstm_config.pkl",
)

print("\nGlobal LSTM training completed!", flush=True)
print("Model saved:", OUT / "global_lstm.pt")
print("Config saved:", OUT / "global_lstm_config.pkl")
