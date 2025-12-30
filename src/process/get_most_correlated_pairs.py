import time

import pandas as pd
from datasets import Dataset
import numpy as np
import plotly.graph_objects as go
from scipy.stats import kendalltau, norm
from scipy.stats import rankdata

top_k = 25
eps = 1e-6

df = pd.read_csv("/home/mithil/PycharmProjects/PolymarketPred/data/market_windows.csv")
df["end_ts_utc"] = pd.to_datetime(df["end_ts_utc"], utc=True)
df["start_ts_utc"] = pd.to_datetime(df["start_ts_utc"], utc=True)
df = df.sort_values(by="end_ts_utc").reset_index(drop=True)
df = df[
    (df["status"] == "ok")
   & (df["question"].str.contains("Up", na=False))
    ].reset_index(drop=True)
print(f"Filtered markets count: {len(df)}")
df.to_csv("/home/mithil/PycharmProjects/PolymarketPred/data/market_windows_filtered.csv", index=False)
# filter for those in between a certain duration range
start_et = pd.Timestamp("2025-10-10 01:00", tz="US/Eastern")
end_et = start_et + pd.Timedelta(days=1)

# Convert ET → UTC for comparison
start_utc = start_et.tz_convert("UTC")
end_utc = end_et.tz_convert("UTC")
df_range = df[
    (df["start_ts_utc"] <= start_utc) &
    (df["end_ts_utc"] >= end_utc)
    ].copy()

parquets = [f"/home/mithil/PycharmProjects/PolymarketPred/data/polymarket_minute_parquet/{uuid}.parquet" for uuid in
            df_range['uuid'].tolist()]
dataset = Dataset.from_parquet(parquets).to_pandas()
dataset["timestamp_et"] = dataset["timestamp"].dt.tz_convert("US/Eastern")

dataset_window = dataset[
    (dataset["timestamp_et"] >= start_et) &
    (dataset["timestamp_et"] <  end_et)
    ].copy().reset_index(drop=True)

dataset_window["timestamp_et"] = dataset_window["timestamp_et"].dt.floor("min")

price_pivot = dataset_window.pivot_table(
    index="timestamp_et",
    columns="question",
    values="price",
    aggfunc="last"
).sort_index()
print(f"Price pivot shape: {price_pivot.shape}")
col_std = price_pivot.std(skipna=True)
col_mean = price_pivot.mean(skipna=True)
keep_cols = col_std[
    (col_std > 1e-2) &
    (col_mean > 0.03) &
    (col_mean < 0.96)
    ].index

#price_pivot = price_pivot[keep_cols]
print(f"Filtered price pivot shape: {price_pivot.shape}")

# Compute Kendall tau
kendall_corr = price_pivot.corr(method="kendall")

# Remove self-correlations
np.fill_diagonal(kendall_corr.values, np.nan)
mask = np.triu(np.ones(kendall_corr.shape), k=1).astype(bool)

top_pairs = (
    kendall_corr
    .where(mask)
    .stack()
    .rename("kendall_tau")
    .dropna()
    .sort_values(key=lambda s: s.abs(), ascending=False)
    .head(top_k)
)
N = min(top_k, len(top_pairs))
pairs = list(top_pairs.index[:N])
taus = top_pairs.values[:N]
for (A, B) in top_pairs.index:
    pair_tau = float(kendall_corr.loc[A, B])  # from the precomputed matrix
    print(f"Evaluating pair: {A} ↔ {B} | pair_tau={pair_tau:+.4f}")
