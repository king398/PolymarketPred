import pandas as pd
from datasets import Dataset
df = pd.read_csv("/home/mithil/PycharmProjects/PolymarketPred/data/market_windows.csv")
df["end_ts_utc"] = pd.to_datetime(df["end_ts_utc"], utc=True)
df["start_ts_utc"] = pd.to_datetime(df["start_ts_utc"], utc=True)
df = df.sort_values(by="end_ts_utc").reset_index(drop=True)
df = df[
    (df['duration_hours'] >= 24) &
    (df['status'] == "ok")
    ].reset_index(drop=True)
print(df['question'])
df.to_csv("/home/mithil/PycharmProjects/PolymarketPred/data/market_windows_filtered.csv", index=False)
# filter for those in between a certain duration range
start_et = pd.Timestamp("2025-11-01 00:00", tz="US/Eastern")
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

target_date = pd.Timestamp("2025-11-01").date()


dataset_day = dataset[
    dataset["timestamp_et"].dt.date == target_date
    ].copy().reset_index(drop=True)
dataset_day["timestamp_et"] = dataset_day["timestamp_et"].dt.floor("min")
price_pivot = dataset_day.pivot_table(
    index="timestamp_et",
    columns="question",
    values="price",
    aggfunc="last"
).sort_index()
print(f"Price pivot shape: {price_pivot.shape}")
col_std = price_pivot.std(skipna=True)
keep_cols = col_std[col_std > 1e-4].index
price_pivot = price_pivot[keep_cols]
price_pivot = price_pivot.dropna()
print(f"Filtered price pivot shape: {price_pivot.shape}")
price_pivot.to_csv("/home/mithil/PycharmProjects/PolymarketPred/data/fair_price_self.csv")