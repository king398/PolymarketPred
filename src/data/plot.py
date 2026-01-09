import pandas as pd
import matplotlib.pyplot as plt
import  numpy as np
df = pd.read_parquet("/home/mithil/PycharmProjects/PolymarketPred/data/binance_data/SOL/SOL.parquet")
print(df.head())
max_diff = (df['open'] - df['close']).abs().max()
print(f"Max difference between open and close: {max_diff}")
# log returns
"""df["log_return"] = np.log(df["close"] / df["close"].shift(120))
df["log_return"].fillna(0, inplace=True)
df['log_return'].hist(bins=100)
plt.show()"""
def find_optimal_horizon(df, max_lag=600):
    """
    Calculates the correlation between current return and future return
    at different horizons.
    """
    correlations = []
    lags = range(1, max_lag, 10) # Check every 10 seconds

    # Calculate 1-step returns first
    # (Log returns are additive, so we can sum them later)
    log_returns = np.log(df['close'] / df['close'].shift(1)).fillna(0)

    for lag in lags:
        # We want to see if the past 'lag' duration predicts the FUTURE 'lag' duration
        # Momentum: past_return vs future_return
        past_return = log_returns.rolling(window=lag).sum()
        future_return = log_returns.rolling(window=lag).sum().shift(-lag)

        corr = past_return.corr(future_return,method="spearman")
        print("Lag:", lag, "Correlation:", corr)
        correlations.append(corr)

    plt.figure(figsize=(10, 6))
    plt.plot(lags, correlations)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title("Predictability Horizon (Autocorrelation)")
    plt.xlabel("Forecast Horizon (Seconds)")
    plt.ylabel("Correlation (Signal Strength)")
    plt.grid(True, alpha=0.3)
    plt.show()

# Usage
find_optimal_horizon(df)