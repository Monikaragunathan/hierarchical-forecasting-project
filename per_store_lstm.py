# per_store_lstm.py
"""
Per-store LSTM forecasting (single-file).
- Expects train.csv in same folder (Walmart train.csv)
- Aggregates weekly_sales per store (sums dept)
- Trains one simple LSTM per store (input_size=1)
- Evaluates SMAPE and MASE on test set
- Saves models as model_store_<store>.pth and results_summary.json
"""

import os
import json
import warnings
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------
# User params (change if needed)
# -----------------------
DATA_FILE = "train.csv"
SEQ_LEN = 52        # history length (weeks)
PRED_STEPS = 1      # predict 1 step ahead
BATCH_SIZE = 32
EPOCHS = 10         # keep small for quick runs; increase if you want better models
LR = 1e-3
MIN_SERIES_LEN = SEQ_LEN + PRED_STEPS + 10  # skip very short series

# -----------------------
# Utilities
# -----------------------
def safe_load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Put train.csv in the same folder as this script.")
    df = pd.read_csv(path)
    # normalize column names
    df.columns = df.columns.str.lower().str.strip()
    # rename common variants
    df = df.rename(columns={
        "weekly_sales": "weekly_sales",
        "weeklysales": "weekly_sales",
        "weekly sales": "weekly_sales",
        "date": "date",
        "store": "store",
        "dept": "dept",
        "is_holiday": "isholiday",
        "is holiday": "isholiday"
    })
    required = {"date", "store", "weekly_sales"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns at least: {required}. Found: {set(df.columns)}")
    df["date"] = pd.to_datetime(df["date"])
    return df

# -----------------------
# Dataset for one series
# -----------------------
class SeriesDataset(Dataset):
    def __init__(self, series: np.ndarray, seq_len: int):
        # series: 1D numpy array of values
        X, Y = [], []
        for i in range(len(series) - seq_len):
            X.append(series[i:i+seq_len])
            Y.append(series[i+seq_len])
        self.X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)  # (N, seq_len, 1)
        self.Y = torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(-1)  # (N, 1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# -----------------------
# Simple LSTM model (input_size=1)
# -----------------------
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden_size, 1)
    def forward(self, x):
        # x: (batch, seq_len, 1)
        o, _ = self.lstm(x)            # (batch, seq_len, hidden)
        last = o[:, -1, :]             # (batch, hidden)
        r = self.out(last)             # (batch, 1)
        return r

# -----------------------
# Metrics: SMAPE and MASE
# -----------------------
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    num = np.abs(y_pred - y_true)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    den[den == 0] = 1e-8
    return 100.0 * np.mean(num / den)

def mase(y_true: np.ndarray, y_pred: np.ndarray, training_series: np.ndarray, m:int=1) -> float:
    # training_series shape (T_train,)
    # scale = mean abs diff of seasonal naive (m-step)
    if len(training_series) <= m:
        scale = 1.0
    else:
        scale = np.mean(np.abs(training_series[m:] - training_series[:-m]))
        if scale == 0:
            scale = 1e-8
    return np.mean(np.abs(y_true - y_pred)) / scale

# -----------------------
# Train + eval for one store
# -----------------------
def train_store_model(values: np.ndarray, seq_len=SEQ_LEN, epochs=EPOCHS, lr=LR, device=DEVICE) -> Tuple[SimpleLSTM, Dict]:
    """
    values: 1D numpy array (time-ordered)
    returns trained model and eval dict with metrics on test set
    """
    n = len(values)
    # split by time: train 70%, val 15%, test 15% (but ensure at least seq_len)
    test_size = max(PRED_STEPS, int(0.15 * n))
    val_size = max(PRED_STEPS, int(0.15 * n))
    train_end = n - (val_size + test_size)
    val_end = n - test_size

    train_series = values[:train_end]
    val_series = values[train_end:val_end]
    test_series = values[val_end:]

    # scale using training series
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_series.reshape(-1,1)).flatten()
    # for creating windows we need concatenated series (so model sees continuity in sliding windows)
    full_scaled = np.concatenate([
        train_scaled,
        scaler.transform(val_series.reshape(-1,1)).flatten(),
        scaler.transform(test_series.reshape(-1,1)).flatten()
    ])

    # build dataset windows from full_scaled but label always next step; to avoid leakage we will index accordingly:
    # easier approach: build dataset on full_scaled and then use index ranges
    dataset = SeriesDataset(full_scaled, seq_len)
    N = len(dataset)
    # compute indices corresponding to splits:
    # position mapping: full_scaled indices 0..n-1 correspond to sample windows starting 0..n-seq_len-1
    # train windows count = train_end - seq_len
    train_count = max(0, train_end - seq_len)
    val_count = max(0, val_end - seq_len) - train_count
    test_count = max(0, N - train_count - val_count)

    if train_count <= 0 or test_count <= 0:
        return None, {"error":"insufficient_series_length"}

    # create samplers by slicing tensors
    X_all = dataset.X
    Y_all = dataset.Y
    X_train = X_all[:train_count]
    Y_train = Y_all[:train_count]
    X_val = X_all[train_count:train_count+val_count]
    Y_val = Y_all[train_count:train_count+val_count]
    X_test = X_all[train_count+val_count:train_count+val_count+test_count]
    Y_test = Y_all[train_count+val_count:train_count+val_count+test_count]

    train_loader = DataLoader(torch.utils.data.TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(torch.utils.data.TensorDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleLSTM(input_size=1, hidden_size=64, num_layers=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float('inf')
    best_state = None
    for ep in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            train_losses.append(loss.item())
        # val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())
        avg_train = float(np.mean(train_losses)) if train_losses else 0.0
        avg_val = float(np.mean(val_losses)) if val_losses else 0.0
        if avg_val < best_val:
            best_val = avg_val
            best_state = model.state_dict()
        # optional print
        # print(f"  ep {ep+1}/{epochs} train={avg_train:.6f} val={avg_val:.6f}")
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on test set (inverse transform)
    model.eval()
    with torch.no_grad():
        X_test_device = X_test.to(device)
        preds_scaled = model(X_test_device).cpu().numpy().reshape(-1)
        trues_scaled = Y_test.numpy().reshape(-1)

    # inverse scale
    preds = scaler.inverse_transform(preds_scaled.reshape(-1,1)).reshape(-1)
    trues = scaler.inverse_transform(trues_scaled.reshape(-1,1)).reshape(-1)

    # compute metrics: SMAPE, MASE (use train_series for MASE scaling)
    sm = smape(trues, preds)
    ma = mase(trues, preds, train_series, m=1)

    return model, {"smape":float(sm), "mase":float(ma), "n_test": int(len(trues))}

# -----------------------
# MAIN pipeline
# -----------------------
def main():
    print("Per-store LSTM training script")
    df = safe_load_csv(DATA_FILE)
    # aggregate weekly_sales per store per date
    pivot = df.groupby(["date","store"], as_index=False)["weekly_sales"].sum()
    # pivot to wide is optional; we will iterate stores
    stores = sorted(pivot["store"].unique())
    print(f"Found {len(stores)} stores in data. Training per-store models (skipping short series).")

    results = {}
    summary = []
    os.makedirs("models", exist_ok=True)

    for st in stores:
        sub = pivot[pivot["store"] == st].sort_values("date")
        values = sub["weekly_sales"].values.astype(float)
        if len(values) < MIN_SERIES_LEN:
            print(f"Store {st}: skipped (series too short: {len(values)} rows)")
            continue
        print(f"Store {st}: series length {len(values)} -> training...")
        model, info = train_store_model(values)
        if model is None:
            print(f"Store {st}: insufficient data, skipped.")
            continue
        if "error" in info:
            print(f"Store {st}: error {info['error']}")
            continue
        # save model
        model_path = os.path.join("models", f"model_store_{st}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Store {st}: done. SMAPE={info['smape']:.3f}  MASE={info['mase']:.3f}  test_len={info['n_test']}")
        results[f"store_{st}"] = {"model_path": model_path, **info}
        summary.append({"store": int(st), "smape": info["smape"], "mase": info["mase"], "n_test": info["n_test"]})

    # overall stats
    if summary:
        avg_smape = float(np.mean([s["smape"] for s in summary]))
        avg_mase = float(np.mean([s["mase"] for s in summary]))
    else:
        avg_smape = avg_mase = None

    out = {
        "n_stores_trained": len(summary),
        "avg_smape": avg_smape,
        "avg_mase": avg_mase,
        "per_store": results
    }
    with open("results_summary.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\nDone. Summary saved to results_summary.json")
    if avg_smape is not None:
        print(f"Average SMAPE across trained stores: {avg_smape:.3f}")
        print(f"Average MASE across trained stores: {avg_mase:.3f}")

if __name__ == "__main__":
    main()
