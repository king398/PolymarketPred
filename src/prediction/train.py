import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# Ensure these imports work in your local environment structure
from dataset import TimeSeriesDataset
from model import BiLSTMPriceForecast
from feature_calc import add_all_talib_features

warnings.filterwarnings("ignore")


class CFG:
    crypto_name = "BTC"
    train_start = "2025-12-15 00:00:00"
    train_end = "2025-12-25 23:59:59"
    valid_start = "2025-12-26 00:00:00"
    valid_end = "2025-12-30 23:59:59"

    lookback_buffer_size = 300

    input_window = 60 * 10
    output_window = 1
    forecast_horizon = 60 * 5

    batch_size = 64
    epochs = 10
    learning_rate = 1e-4

    # Scheduler settings
    min_lr = 1e-6  # Minimum learning rate for Cosine Annealing

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = "best_model.pth"


# ==========================================
# 0. CUSTOM LOSS FUNCTION
# ==========================================
class DirectionalMSELoss(nn.Module):
    """
    Composite Loss: MSE + Directional Penalty.

    Financial time series care about Direction (sign) as much as Magnitude.
    This loss adds a penalty 'alpha' when the sign of prediction differs
    from the sign of the target.
    """

    def __init__(self, alpha=5.0):
        super(DirectionalMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        # 1. Standard Regression Loss (MSE)
        mse_loss = self.mse(y_pred, y_true)

        # 2. Directional Penalty
        # If signs match: y_pred * y_true > 0 => -product < 0 => ReLU(-product) = 0
        # If signs differ: y_pred * y_true < 0 => -product > 0 => ReLU(-product) = Positive Penalty
        direction_penalty = torch.mean(torch.relu(-y_pred * y_true))

        return mse_loss + (self.alpha * direction_penalty)


# ==========================================
# 1. ISOLATED FEATURE ENGINEERING
# ==========================================
def process_features(df):
    df = add_all_talib_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill().fillna(0.0)

    # Target: Log return over the next N steps
    future_close = df["close"].shift(-CFG.forecast_horizon)
    current_close = df["close"]
    df["target_return"] = np.log(future_close / current_close)

    df["target_return"] = df["target_return"].replace([np.inf, -np.inf], np.nan)
    df["target_return"] = df["target_return"].fillna(0.0)

    return df


# ==========================================
# 2. NORMALIZATION LOGIC
# ==========================================
def get_normalization_stats(train_df, feature_cols):
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std()
    return mean, std


def apply_normalization(df, feature_cols, mean, std):
    df_norm = df.copy()
    df_norm[feature_cols] = (df_norm[feature_cols] - mean) / (std + 1e-8)
    return df_norm


# ==========================================
# TRAINING HELPERS
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="Training", leave=False)

    for batch in pbar:
        x = batch["input"].to(device)
        y = batch["target"].to(device)

        optimizer.zero_grad()
        preds = model(x)

        if preds.shape != y.shape:
            preds = preds.view_as(y)

        loss = criterion(preds, y)
        loss.backward()

        # Optional: Gradient Clipping prevents exploding gradients in LSTMs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            x = batch["input"].to(device)
            y = batch["target"].to(device)

            preds = model(x)

            if preds.shape != y.shape:
                preds = preds.view_as(y)

            loss = criterion(preds, y)

            running_loss += loss.item()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    y_pred = np.concatenate(all_preds).flatten()
    y_true = np.concatenate(all_targets).flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100

    return {"Loss": running_loss / len(loader), "RMSE": rmse, "R2": r2, "Dir_Acc": dir_acc}


# ==========================================
# MAIN
# ==========================================
def main():
    # 1. Load Data
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'binance_data', CFG.crypto_name,
                             f'{CFG.crypto_name}.parquet')

    raw_df = pd.read_parquet(DATA_PATH)
    raw_df["datetime"] = pd.to_datetime(raw_df["open_time"], unit="us", utc=True)
    raw_df = raw_df.sort_values("datetime").reset_index(drop=True)

    print("--> Splitting Raw Data...")
    train_mask = (raw_df["datetime"] >= CFG.train_start) & (raw_df["datetime"] <= CFG.train_end)
    raw_train = raw_df[train_mask].copy()

    valid_mask = (raw_df["datetime"] >= CFG.valid_start) & (raw_df["datetime"] <= CFG.valid_end)
    raw_valid_main = raw_df[valid_mask].copy()

    valid_start_idx = raw_valid_main.index[0]
    buffer_start_idx = max(0, valid_start_idx - CFG.lookback_buffer_size)
    raw_valid_buffered = raw_df.iloc[
        buffer_start_idx: valid_start_idx + len(raw_valid_main) + CFG.forecast_horizon].copy()

    # 3. Process Features
    print("--> Processing Features...")
    train_processed = process_features(raw_train)
    valid_processed_buffered = process_features(raw_valid_buffered)

    valid_processed = valid_processed_buffered[
        (valid_processed_buffered["datetime"] >= CFG.valid_start) &
        (valid_processed_buffered["datetime"] <= CFG.valid_end)
        ].copy()

    train_processed = train_processed.iloc[:-CFG.forecast_horizon]
    valid_processed = valid_processed.iloc[:-CFG.forecast_horizon]

    # 4. Normalize
    print("--> Normalizing...")
    target_col = "target_return"
    exclude_cols = ["open_time", "datetime", target_col]
    feature_cols = [c for c in train_processed.columns if c not in exclude_cols]

    mean_stats, std_stats = get_normalization_stats(train_processed, feature_cols)
    train_df = apply_normalization(train_processed, feature_cols, mean_stats, std_stats)
    valid_df = apply_normalization(valid_processed, feature_cols, mean_stats, std_stats)

    # 5. Datasets
    train_dataset = TimeSeriesDataset(train_df, feature_cols, target_col, CFG.input_window, 1)
    valid_dataset = TimeSeriesDataset(valid_df, feature_cols, target_col, CFG.input_window, 1)

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=os.cpu_count())
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=os.cpu_count())

    # 6. Model Setup
    model = BiLSTMPriceForecast(n_features=len(feature_cols), hidden_size=128, num_layers=3, z=CFG.output_window)
    model = model.to(CFG.device)

    optimizer = optim.Adam(model.parameters(), lr=CFG.learning_rate)

    # --- CHANGE 1: SCHEDULER ---
    # Cosine Annealing: Smoothly lowers LR from learning_rate to min_lr over CFG.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CFG.epochs,
        eta_min=CFG.min_lr
    )

    # --- CHANGE 2: BETTER LOSS ---
    # Using Custom Directional MSE
    criterion = DirectionalMSELoss(alpha=5.0)

    best_loss = float('inf')

    for epoch in range(CFG.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, CFG.device)
        val_metrics = validate(model, valid_loader, criterion, CFG.device)

        # Step the scheduler at the end of epoch
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch + 1}/{CFG.epochs} | LR: {current_lr:.2e} | Train Loss: {train_loss:.6f} | Val Loss: {val_metrics['Loss']:.6f}")
        print(f"RMSE: {val_metrics['RMSE']:.6f} | R2: {val_metrics['R2']:.4f} | Dir Acc: {val_metrics['Dir_Acc']:.2f}%")

        if val_metrics['Loss'] < best_loss:
            best_loss = val_metrics['Loss']
            torch.save(model.state_dict(), CFG.save_path)
            print(">>> Saved Best Model")

        print("-" * 50)


if __name__ == "__main__":
    main()
