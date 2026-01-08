import torch
import torch.nn as nn


class BiLSTMPriceForecast(nn.Module):
    """
    Bidirectional LSTM forecaster for 1-D target (price).

    Input:
      x: (B, T, n_features)

    Output:
      y_hat: (B, z)   # z future steps (e.g., 5 minutes ahead if your step=1 minute and z=5)

    Notes:
    - Uses a BiLSTM encoder over the past window.
    - Predicts all z steps "directly" from the encoded representation (no autoregressive decoding),
      which is usually stable and fast for short horizons like 5 minutes.
    """
    def __init__(
            self,
            n_features: int,
            hidden_size: int,
            num_layers: int,
            z: int,
            dropout: float = 0.1,
            use_layernorm: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.z = z
        self.num_directions = 2

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        rep_dim = hidden_size * self.num_directions
        self.norm = nn.LayerNorm(rep_dim) if use_layernorm else nn.Identity()

        # MLP head -> z steps of future price
        self.head = nn.Sequential(
            nn.Linear(rep_dim, rep_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rep_dim, z),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, n_features)
        returns: (B, z)
        """
        if x.dim() != 3:
            raise ValueError(f"x must be (B,T,F). Got {tuple(x.shape)}")
        B, T, F = x.shape
        if F != self.n_features:
            raise ValueError(f"Expected n_features={self.n_features}, got {F}")

        out, (h_n, c_n) = self.lstm(x)

        # h_n: (num_layers * 2, B, hidden_size)
        # Take final-layer hidden states for both directions and concat:
        # forward = h_n[-2], backward = h_n[-1]
        h_fwd = h_n[-2]  # (B, hidden)
        h_bwd = h_n[-1]  # (B, hidden)
        rep = torch.cat([h_fwd, h_bwd], dim=1)  # (B, hidden*2)

        rep = self.norm(rep)
        y_hat = self.head(rep)  # (B, z)
        return y_hat


# --- quick example ---
if __name__ == "__main__":
    B, T, n_features = 32, 60, 10  # e.g., last 60 minutes with 10 features
    z = 5                          # predict next 5 minutes
    model = BiLSTMPriceForecast(n_features=n_features, hidden_size=128, num_layers=2, z=z)
    x = torch.randn(B, T, n_features)
    y = model(x)
    print(y.shape)  # torch.Size([32, 5])
