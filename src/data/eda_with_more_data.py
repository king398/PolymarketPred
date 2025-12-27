import glob
import random

import pandas as pd
import pyarrow.dataset as ds
import plotly.graph_objects as go

DIR = "/home/mithil/PycharmProjects/PolymarketPred/data/polymarket_minute_parquet"

# -----------------------------
# 1) Load all parquet files
# -----------------------------
parquet_files = glob.glob(f"{DIR}/*.parquet")  # excludes .json, .tmp, etc.
dataset = ds.dataset(parquet_files, format="parquet")

df = dataset.to_table().to_pandas()

# Ensure timestamp is tz-aware UTC (you likely already have this)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# -----------------------------
# 2) Filter to Oct 1–10 (inclusive of Oct 10, exclusive of Oct 11)
# -----------------------------
start = pd.Timestamp("2025-10-01", tz="UTC")
end = pd.Timestamp("2025-10-04", tz="UTC")  # exclusive

df_oct = df.loc[(df["timestamp"] >= start) & (df["timestamp"] < end)].copy()

print("Loaded files:", len(parquet_files), "Rows (Oct range):", len(df_oct))

# -----------------------------
# 3) Sample 25 random questions
# -----------------------------
questions = df_oct["question"].dropna().unique().tolist()
k = min(5
        , len(questions))

sampled_questions = random.sample(questions, k)

print(f"Unique questions in date range: {len(questions)}")
print(f"Sampled {k} questions:")
for i, q in enumerate(sampled_questions, 1):
    print(f"{i:02d}. {q}")

# Keep only the sampled questions
df_s = df_oct[df_oct["question"].isin(sampled_questions)].copy()

# Sort for nice lines
df_s = df_s.sort_values(["question", "timestamp"])

# Optional: downsample per question if plots get too heavy (e.g., keep every Nth point)
# N = 5
# df_s = df_s.groupby("question", group_keys=False).apply(lambda g: g.iloc[::N])

# -----------------------------
# 4) Plot with Plotly (one trace per question)
# -----------------------------
fig = go.Figure()

for q in sampled_questions:
    g = df_s[df_s["question"] == q]
    if len(g) == 0:
        continue

    fig.add_trace(
        go.Scatter(
            x=g["timestamp"],
            y=g["price"],
            mode="lines",
            name=q[:60] + ("…" if len(q) > 60 else ""),  # shorten legend label
            hovertemplate=(
                "Question: %{customdata}<br>"
                "Time (UTC): %{x}<br>"
                "Price: %{y:.4f}<extra></extra>"
            ),
            customdata=[q] * len(g),
        )
    )

fig.update_layout(
    title=f"Polymarket Minute Prices — {k} Random Questions (2025-10-01 to 2025-10-10 UTC)",
    xaxis_title="Timestamp (UTC)",
    yaxis_title="Price",
    hovermode="x unified",
    legend_title="Question (truncated)",
)

fig.show()
