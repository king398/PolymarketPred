import pandas as pd
import matplotlib.pyplot as plt
import  numpy as np
df = pd.read_parquet("/home/mithil/PycharmProjects/PolymarketPred/data/binance_data/SOL/SOL.parquet")
max_diff = (df['open'] - df['close']).abs().max()
print(f"Max difference between open and close: {max_diff}")
# log returns
df["log_return"] = np.log(df["close"] / df["close"].shift(1))
df["log_return"].fillna(0, inplace=True)
df['log_return'].plot()
plt.show()
