import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    """
    Computes a weighted sum of LSTM outputs to focus on relevant past time steps.
    """
    def __init__(self, hidden_size):
        super(AttentionBlock, self).__init__()
        # Learnable weights to calculate relevance score for each time step
        self.attention_linear = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output shape: (Batch, Seq_Len, Hidden_Size * 2)

        # 1. Calculate energy scores
        # shape: (Batch, Seq_Len, 1)
        energy = torch.tanh(self.attention_linear(lstm_output))

        # 2. Calculate attention weights (probability distribution over time)
        # shape: (Batch, Seq_Len, 1)
        weights = F.softmax(energy, dim=1)

        # 3. Calculate Context Vector (Weighted Sum of hidden states)
        # shape: (Batch, Hidden_Size * 2)
        context_vector = torch.sum(weights * lstm_output, dim=1)

        return context_vector

class BiLSTMPriceForecast(nn.Module):
    """
    Bidirectional LSTM with Attention for 1-D target (price).
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

        # Dimension of the LSTM output (Forward + Backward)
        rep_dim = hidden_size * self.num_directions

        # --- ADDED: Attention Block ---
        self.attention = AttentionBlock(rep_dim)

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

        # out shape: (B, T, hidden_size * 2)
        # Contains hidden states for ALL time steps
        out, _ = self.lstm(x)

        # --- CHANGED: Use Attention instead of manual h_n extraction ---
        # The attention block looks at the whole sequence 'out' and
        # summarizes it into 'context' based on learned importance.
        context = self.attention(out)  # (B, hidden*2)

        rep = self.norm(context)
        y_hat = self.head(rep)  # (B, z)

        return y_hat