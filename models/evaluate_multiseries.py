"""Leakage-aware multi-horizon benchmark for ForecastIQ.

Run from ForecastIQ-Base:
  python models/evaluate_multiseries.py

Does not modify deployed models or models/results.csv.
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=FutureWarning)
START = pd.Timestamp('2020-10-01')
HORIZONS = [7, 14, 30, 60, 90]
N_DAYS = max(HORIZONS)
OUT = ROOT / 'models' / 'multiseries' / 'evaluation'
OUT.mkdir(parents=True, exist_ok=True)


def recursive_xgb(history, dates, promo, artifacts):
    from utils.features import build_features
    model, le_store, le_item, feature_cols = artifacts
    history = history.copy().reset_index(drop=True)
    preds = []
    store = history['store_nbr'].iloc[0]
    item = history['item_nbr'].iloc[0]
    for date, p in zip(dates, promo):
        row = pd.DataFrame({'date': [date], 'store_nbr': [store], 'item_nbr': [item],
                            'family': [history['family'].iloc[0]], 'sales': [np.nan],
                            'onpromotion': [int(p)]})
        feat = build_features(pd.concat([history, row], ignore_index=True))
        feat['store_enc'] = le_store.transform(feat['store_nbr'])
        feat['item_enc'] = le_item.transform(feat['item_nbr'])
        X = feat.iloc[[-1]][feature_cols]
        if X.isna().any().any():
            raise ValueError('XGBoost feature NaNs')
        pred = max(float(model.predict(X)[0]), 0.0)
        preds.append(pred)
        row['sales'] = pred
        history = pd.concat([history, row], ignore_index=True)
    return np.asarray(preds)


def fit_sarima(history, n):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model = SARIMAX(history['sales'].to_numpy(dtype=float), order=(1, 1, 1),
                    seasonal_order=(1, 1, 0, 7), enforce_stationarity=False,
                    enforce_invertibility=False).fit(disp=False, maxiter=50)
    return np.maximum(np.asarray(model.forecast(steps=n)), 0)


def fit_prophet(history, dates):
    from prophet import Prophet
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                    daily_seasonality=False, seasonality_mode='multiplicative')
    model.fit(history[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'}))
    pred = model.predict(pd.DataFrame({'ds': dates}))
    return np.maximum(pred['yhat'].to_numpy(dtype=float), 0)


def load_lstm():
    import torch
    from torch import nn
    torch.set_num_threads(2)

    class GlobalLSTM(nn.Module):
        def __init__(self, n_stores, n_items):
            super().__init__()
            self.store_embedding = nn.Embedding(n_stores, 4)
            self.item_embedding = nn.Embedding(n_items, 6)
            self.lstm = nn.LSTM(1, 64, batch_first=True)
            self.fc = nn.Sequential(nn.Linear(74, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, x, store, item):
            out, _ = self.lstm(x)
            return self.fc(torch.cat([out[:, -1, :], self.store_embedding(store),
                                      self.item_embedding(item)], dim=1))

    folder = ROOT / 'models' / 'multiseries' / 'lstm'
    config = joblib.load(folder / 'global_lstm_config.pkl')
    model = GlobalLSTM(config['n_stores'], config['n_items'])
    model.load_state_dict(torch.load(folder / 'global_lstm.pt', map_location='cpu', weights_only=True))
    model.eval()
    return model, config


def recursive_lstm(history, store, item, n, artifact):
    import torch
    model, cfg = artifact
    scale = cfg['scales'][(int(store), str(item))]
    context = (history['sales'].to_numpy(dtype=float) / scale).tolist()
    seq_len = int(cfg['seq_len'])
    s = torch.tensor([cfg['store_map'][int(store)]], dtype=torch.long)
    i = torch.tensor([cfg['item_map'][str(item)]], dtype=torch.long)
    preds = []
    with torch.no_grad():
        for _ in range(n):
            x = torch.tensor(context[-seq_len:], dtype=torch.float32).reshape(1, seq_len, 1)
            p = max(float(model(x, s, i).item()), 0.0)
            preds.append(p * scale)
            context.append(p)
    return np.asarray(preds)


def metrics(df):
    actual = df['actual'].to_numpy(dtype=float)
    predicted = df['predicted'].to_numpy(dtype=float)
    ae = np.abs(actual - predicted)
    denom = np.sum(np.abs(actual))
    return {'MAE': float(ae.mean()), 'RMSE': float(np.sqrt(np.mean((actual-predicted)**2))),
            'WAPE': float(100 * ae.sum() / denom) if denom else np.nan,
            'n_predictions': len(df)}


def main():
    df = pd.read_csv(ROOT / 'data' / 'train.csv', parse_dates=['date'])
    xgb = (joblib.load(ROOT / 'models/xgb_model.pkl'),
           joblib.load(ROOT / 'models/le_store.pkl'),
           joblib.load(ROOT / 'models/le_item.pkl'),
           joblib.load(ROOT / 'models/xgb_features.pkl'))
    lstm = load_lstm()
    records = []
    failures = []
    groups = list(df.groupby(['store_nbr', 'item_nbr'], sort=True))
    for k, ((store, item), group) in enumerate(groups, 1):
        group = group.sort_values('date').reset_index(drop=True)
        history = group[group.date < START].copy()
        actual = group[(group.date >= START) & (group.date < START + pd.Timedelta(days=N_DAYS))].copy()
        if len(actual) != N_DAYS:
            raise ValueError(f'Expected {N_DAYS} days for {store}/{item}; got {len(actual)}')
        dates = actual['date'].tolist()
        runners = {
            'XGBoost': lambda: recursive_xgb(history, dates, actual['onpromotion'].to_numpy(), xgb),
            'SARIMA': lambda: fit_sarima(history, N_DAYS),
            'Prophet': lambda: fit_prophet(history, dates),
            'LSTM': lambda: recursive_lstm(history, store, item, N_DAYS, lstm),
        }
        for name, run in runners.items():
            try:
                preds = run()
                if len(preds) != N_DAYS or not np.isfinite(preds).all():
                    raise ValueError('Invalid prediction output')
                for date, y, p in zip(dates, actual['sales'], preds):
                    records.append({'model': name, 'store_nbr': int(store), 'item_nbr': str(item),
                                    'date': date, 'actual': float(y), 'predicted': float(p)})
                print(f'[{k:02d}/{len(groups)}] Store {store} / {item}: {name} OK', flush=True)
            except Exception as exc:
                failures.append({'model': name, 'store_nbr': int(store), 'item_nbr': str(item),
                                 'error': str(exc)})
                print(f'[{k:02d}/{len(groups)}] Store {store} / {item}: {name} FAILED: {exc}', flush=True)
        # Save incremental progress for long runs.
        pd.DataFrame(records).to_csv(OUT / 'all_predictions.csv', index=False)
        pd.DataFrame(failures, columns=['model', 'store_nbr', 'item_nbr', 'error']).to_csv(OUT / 'failures.csv', index=False)

    results = pd.DataFrame(records)
    summaries = []
    for (name, store, item), subset in results.groupby(['model', 'store_nbr', 'item_nbr']):
        subset = subset.sort_values('date')
        for h in HORIZONS:
            summaries.append({'model': name, 'store_nbr': store, 'item_nbr': item, 'horizon': h,
                              **metrics(subset.head(h))})
    per_series = pd.DataFrame(summaries)
    per_series.to_csv(OUT / 'per_series_metrics.csv', index=False)

    aggregate = []
    for name, subset in results.groupby('model'):
        for h in HORIZONS:
            selected = subset.sort_values('date').groupby(['store_nbr', 'item_nbr']).head(h)
            aggregate.append({'model': name, 'horizon': h, 'n_series': selected.groupby(['store_nbr', 'item_nbr']).ngroups,
                              **metrics(selected)})
    table = pd.DataFrame(aggregate).sort_values(['horizon', 'MAE'])
    table.to_csv(OUT / 'horizon_comparison.csv', index=False)
    print('\nRESULTS (pooled metrics):\n', table.round(3).to_string(index=False), flush=True)
    print(f'\nSaved evaluation outputs to {OUT}', flush=True)
    if failures:
        print(f'WARNING: {len(failures)} series-model failures; inspect failures.csv. Do not compare incomplete models directly.')
    print('CAVEATS: XGBoost uses actual future promotion flags; XGBoost/LSTM trained through June 2020, SARIMA/Prophet through September 2020.')


if __name__ == '__main__':
    main()
