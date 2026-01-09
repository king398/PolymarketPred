import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data: pd.DataFrame, feature_cols: list, target_col: str,
                 input_window: int, output_window: int):
        """
        data: DataFrame with PRE-CALCULATED features and PRE-CALCULATED log return targets.
        """
        self.data = data
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.input_window = input_window
        self.output_window = output_window


    def __len__(self):
        return len(self.data) - self.input_window - self.output_window + 1

    def __getitem__(self, idx) -> dict:
        # 1. Slice the DataFrame (Rows)
        # We need rows [idx : idx + input + output] to cover both windows
        # Note: .iloc is slow. If performance lags, convert self.data to numpy in __init__
        start_idx = idx
        mid_idx = idx + self.input_window
        end_idx = mid_idx + self.output_window

        # Slice Input Features
        input_slice = self.data.iloc[start_idx : mid_idx][self.feature_cols]

        # Slice Target (Log Returns)
        target_slice = self.data.iloc[mid_idx : end_idx][self.target_col]

        # 2. Convert to Tensor
        # .values converts pandas to numpy, then we wrap in torch.tensor
        input_features = torch.tensor(input_slice.values, dtype=torch.float32)
        target_values = torch.tensor(target_slice.values, dtype=torch.float32)

        return {
            "input": input_features,
            "target": target_values
        }