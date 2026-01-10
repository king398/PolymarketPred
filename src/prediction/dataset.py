import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data: pd.DataFrame, feature_cols: list, target_col: str,
                 input_window: int, output_window: int, device: str = 'cpu'):
        """
        data: DataFrame with PRE-CALCULATED features and PRE-CALCULATED log return targets.
        """
        self.input_window = input_window
        self.output_window = output_window

        # --- OPTIMIZATION: Convert to Tensor immediately ---
        # 1. Extract numpy arrays (much faster to slice later)
        feature_data = data[feature_cols].values.astype(np.float32)
        target_data = data[target_col].values.astype(np.float32)

        # 2. Convert to Tensors
        # If your dataset fits in GPU memory, pass device='cuda' to speed up transfer later
        self.features = torch.tensor(feature_data, dtype=torch.float32)

        # Ensure target is 2D (seq_len, 1) or 1D (seq_len) depending on model needs
        # Here we keep it simple, but usually targets need a feature dimension
        self.targets = torch.tensor(target_data, dtype=torch.float32)

        # Pre-calculate length
        self.length = len(data) - self.input_window - self.output_window + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx) -> dict:
        # --- OPTIMIZATION: Pure Tensor Slicing ---
        # No pandas overhead. This is O(1) stride access.

        mid_idx = idx + self.input_window
        end_idx = mid_idx + self.output_window

        # Slice Input Features [Input Window, Num Features]
        input_features = self.features[idx : mid_idx]

        # Slice Target [Output Window]
        target_values = self.targets[mid_idx : end_idx]

        return {
            "input": input_features,
            "target": target_values
        }