import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

torch.set_num_threads(2)
torch.manual_seed(42)
np.random.seed(42)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

SEQ_LEN = 30
EPOCHS = 10
BATCH_SIZE = 64

print("Loading dataset...", flush=True)

df = pd.read_csv("data/train.csv", parse_dates=["date"])

ser = (
    df[(df.store_nbr == 1) & (df.item_nbr == "ITEM_001")]
    .sort_values("date")
    .reset_index(drop=True)
)

train = ser[ser.date < "2020-07-01"]
test = ser[ser.date >= "2020-10-01"]

print(f"Training rows: {len(train)}", flush=True)
print(f"Testing rows: {len(test)}", flush=True)

if len(train) <= SEQ_LEN or len(test) == 0:
    raise ValueError("Not enough data for training or evaluation.")

values = ser["sales"].to_numpy(dtype=np.float32).reshape(-1, 1)

scaler = MinMaxScaler()
scaler.fit(train["sales"].to_numpy().reshape(-1, 1))

scaled = scaler.transform(values).flatten()

train_scaled = scaled[:len(train)]

X = []
y = []

for i in range(SEQ_LEN, len(train_scaled)):
    X.append(train_scaled[i-SEQ_LEN:i])
    y.append(train_scaled[i])

X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)

loader = DataLoader(
    TensorDataset(X, y),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print(f"Training sequences: {len(X)}", flush=True)
print(f"Batches per epoch: {len(loader)}", flush=True)


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(1, 64, batch_first=True)
        self.drop1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.drop2 = nn.Dropout(0.2)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        return self.fc(self.drop2(out[:, -1, :]))


model = LSTMModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("\nStarting LSTM training...", flush=True)

for epoch in range(EPOCHS):
    start = time.perf_counter()
    model.train()
    total_loss = 0

    for batch_idx, (xb, yb) in enumerate(loader):
        optimizer.zero_grad()

        prediction = model(xb)
        loss = criterion(prediction, yb)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx == 0:
            print(
                f"Epoch {epoch+1}: first batch completed",
                flush=True
            )

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {total_loss/len(loader):.6f} | "
        f"Time: {time.perf_counter()-start:.2f}s",
        flush=True
    )

print("\nEvaluating LSTM...", flush=True)

model.eval()

# Start forecasting at the test boundary, using only
# observations available before the test period.
test_start = int(ser.index[ser.date >= "2020-10-01"][0])
context = scaled[test_start-SEQ_LEN:test_start].tolist()

predictions_scaled = []

with torch.no_grad():
    for _ in range(len(test)):
        inp = torch.tensor(
            context[-SEQ_LEN:],
            dtype=torch.float32
        ).reshape(1, SEQ_LEN, 1)

        pred = model(inp).item()
        predictions_scaled.append(pred)
        context.append(pred)

predictions = scaler.inverse_transform(
    np.array(predictions_scaled).reshape(-1, 1)
).flatten()

predictions = np.maximum(predictions, 0)

actual = test["sales"].to_numpy()

mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))
mape = np.mean(
    np.abs((actual - predictions) / np.maximum(np.abs(actual), 1e-8))
) * 100

print(f"LSTM MAE: {mae:.2f}")
print(f"LSTM RMSE: {rmse:.2f}")
print(f"LSTM MAPE: {mape:.2f}%")

torch.save(model.state_dict(), MODELS_DIR / "lstm_model.pt")
joblib.dump(scaler, MODELS_DIR / "lstm_scaler.pkl")

print("\nLSTM training completed successfully!")
print("Model saved to models/lstm_model.pt")
