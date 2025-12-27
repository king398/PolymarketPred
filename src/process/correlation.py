from datasets import  Dataset
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import glob
DIR = "/home/mithil/PycharmProjects/PolymarketPred/data/polymarket_minute_parquet"
REPO_ID = "Mithilss/polymarket_minute_parquet"

PARQUETS = glob.glob(f"{DIR}/*.parquet")

# Fast-path parquet -> Arrow Dataset
ds = Dataset.from_parquet(PARQUETS)
pattern = re.compile(r"\b(bitcoin|ethereum|solana|xrp|btc|eth)\b", re.IGNORECASE)

def crypto_filter_batch(batch):
    qs = batch["question"]
    return [bool(pattern.search(q)) if q else False for q in qs]

ds_crypto = ds.filter(
    crypto_filter_batch,
    batched=True,
    batch_size=50_000,
    num_proc=16,
    load_from_cache_file=True,
)

start = pd.Timestamp("2025-11-01", tz="UTC")
end = start + pd.Timedelta(days=7)  # exclusivea

def in_range_batch(batch):
    ts = pd.to_datetime(batch["timestamp"], utc=True, errors="coerce")
    return ts >= start

ds_range = ds_crypto.filter(
    in_range_batch,
    batched=True,
    batch_size=50_000,
    num_proc=32,
    load_from_cache_file=True,
)

# ---------- belief features (pandas step for correctness) ----------

df = ds_range.to_pandas()
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

# Choose grouping key (prefer token_id if available)
group_key = "market_id"

# log-odds (belief state)
eps = 1e-6
p = df["price"].astype("float64").clip(eps, 1 - eps)
df["log_odds"] = np.log(p / (1 - p))

# sort then belief update (Δlog-odds) within each market
df = df.sort_values([group_key, "timestamp"])
df["belief_update"] = df.groupby(group_key)["log_odds"].diff()

# optional: fill first observation per group
df["belief_update"] = df["belief_update"].fillna(0.0)
# -----------------------------
# DROP markets with no belief change
# -----------------------------

# total absolute belief movement per market
belief_movement = (
    df.groupby(group_key)["belief_update"]
    .apply(lambda x: np.abs(x).sum())
)

# keep only markets with some movement
active_markets = belief_movement[belief_movement > 0].index

df = df[df[group_key].isin(active_markets)].copy()

print("Remaining active markets:", df[group_key].nunique())
print("Remaining rows:", len(df))

# convert back to HF dataset

print("Rows (crypto + date range):", len(ds_range))
print("Unique questions:", len(ds_range.unique("question")))
# plot

question = df["question"].iloc[-1]
df_q = df[df["question"] == question].copy()

df_q = df_q.sort_values("timestamp")
print(len(df_q))
print("Question:")
print(question)
print("Rows:", len(df_q))

fig = go.Figure()

# Price (probability)
fig.add_trace(
    go.Scatter(
        x=df_q["timestamp"],
        y=df_q["price"],
        mode="lines",
        name="Price (Probability)",
        yaxis="y1",
    )
)

# Log-odds
fig.add_trace(
    go.Scatter(
        x=df_q["timestamp"],
        y=df_q["log_odds"],
        mode="lines",
        name="Log-Odds",
        yaxis="y2",
    )
)

# Belief update (Δ log-odds)
fig.add_trace(
    go.Scatter(
        x=df_q["timestamp"],
        y=df_q["belief_update"],
        mode="lines",
        name="Belief Update (Δ log-odds)",
        yaxis="y3",
    )
)

fig.update_layout(
    title=f"Prediction Market Belief Dynamics<br>{question}",
    xaxis=dict(title="Time (UTC)"),

    yaxis=dict(
        title="Price",
        side="left",
        range=[0, 1],
    ),
    yaxis2=dict(
        title="Log-Odds",
        overlaying="y",
        side="right",
        showgrid=False,
    ),
    yaxis3=dict(
        title="Belief Update",
        anchor="free",
        overlaying="y",
        side="right",
        position=0.95,
        showgrid=False,
    ),

    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    hovermode="x unified",
    height=600,
)

fig.show()
