import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from feature_calc import add_all_talib_features

@profile
class TimeSeriesDataset(Dataset):
    def __init__(self, data: pd.DataFrame, input_window: int, output_window: int):
        """
        data: numpy array or pandas DataFrame of shape (num_samples, num_features)
        input_window: number of time steps in the input sequence (in seconds)
        output_window: number of time steps to predict (in seconds)
        """
        self.data = data
        self.input_window = input_window
        self.output_window = output_window
        self.feature_cols = ["open", "high", "low", "close", "volume"]
        self.target_col = "close"

    def __len__(self):
        return len(self.data) - self.input_window - self.output_window + 1

    def __getitem__(self, idx):
        input_df = self.data.iloc[idx:idx + self.input_window]
        target_df = self.data.iloc[
            idx + self.input_window: idx + self.input_window + self.output_window
        ]

        # ---- INPUT FEATURES ----
        input_features = add_all_talib_features(input_df)

        mean = input_features.mean()
        std = input_features.std()
        input_features = (input_features - mean) / (std + 1e-8)

        # ---- TARGET: LOG RETURNS ----
        prices = target_df[self.target_col].values

        # prepend last input price for correct first return
        prev_price = input_df[self.target_col].iloc[-1]
        prices_with_prev = np.concatenate([[prev_price], prices])

        target_values = np.log(prices_with_prev[1:] / prices_with_prev[:-1])
        input_features = torch.tensor(input_features.values)
        target_values = torch.tensor(target_values, dtype=torch.float32)

        return {
            "input": input_features,
            "target": target_values,
            "mean": mean,
            "std": std
        }
def make_df(n=50_000):
    close = np.cumsum(np.random.randn(n)) + 100
    open_ = close + np.random.randn(n) * 0.1
    high = np.maximum(open_, close) + np.random.rand(n) * 0.2
    low = np.minimum(open_, close) - np.random.rand(n) * 0.2
    volume = np.random.randint(100, 10_000, size=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def run():
    datset = TimeSeriesDataset(make_df(100_000), input_window=60, output_window=5)
    for i in range(100):
        x = datset[i]


if __name__ == "__main__":
    run()
