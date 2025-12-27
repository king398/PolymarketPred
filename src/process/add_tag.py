from datasets import load_dataset
import re
import random
import pandas as pd
import plotly.graph_objects as go

# -----------------------------
# 0) Load HF dataset
# -----------------------------
ds = load_dataset("Mithilss/polymarket_minute_parquet")["train"]

# -----------------------------
# 1) Fast crypto filter (batched)
# -----------------------------
#pattern = re.compile(r"\b(bitcoin|ethereum|solana|xrp|btc|eth)\b", re.IGNORECASE)
pattern = re.compile(r"\b(bitcoin|ethereum)\b", re.IGNORECASE)

def crypto_filter_batch(batch):
    qs = batch["question"]
    return [bool(pattern.search(q)) if q else False for q in qs]

ds_crypto = ds.filter(
    crypto_filter_batch,
    batched=True,
    batch_size=50_000,
    num_proc=16,
)

print("Filtered rows:", len(ds_crypto))

# -----------------------------
# 2) Filter by date range (UTC)
#    IMPORTANT: adjust column name if yours differs ("timestamp")
# -----------------------------
# If your dataset timestamp is already ISO strings, this works fine.
# If it's int unix seconds, convert accordingly (see note below).
start = pd.Timestamp("2025-10-01", tz="UTC")
end   = pd.Timestamp("2025-10-03", tz="UTC")  # exclusive

def in_range_batch(batch):
    ts = pd.to_datetime(batch["timestamp"], utc=True, errors="coerce")
    return (ts >= start) & (ts < end)

ds_range = ds_crypto.filter(
    in_range_batch,
    batched=True,
    batch_size=50_000,
    num_proc=16,
)

print("Rows (crypto + date range):", len(ds_range))

# -----------------------------
# 3) Convert ONLY the slice to pandas
# -----------------------------
df = ds_range.to_pandas()

# Ensure types
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
df = df.dropna(subset=["timestamp", "question", "price"])

print("Pandas rows:", len(df))

# -----------------------------
# 4) Sample k random questions
# -----------------------------
questions = df["question"].dropna().unique().tolist()
k = min(5, len(questions))

sampled_questions = random.sample(questions, k) if k > 0 else []
print(f"Unique questions in date range: {len(questions)}")
print(f"Sampled {k} questions:")
for i, q in enumerate(sampled_questions, 1):
    print(f"{i:02d}. {q}")

df_s = df[df["question"].isin(sampled_questions)].copy()
df_s = df_s.sort_values(["question", "timestamp"])

# -----------------------------
# 5) Plot with Plotly
# -----------------------------
fig = go.Figure()

for q in sampled_questions:
    g = df_s[df_s["question"] == q]
    if g.empty:
        continue

    fig.add_trace(
        go.Scatter(
            x=g["timestamp"],
            y=g["price"],
            mode="lines",
            name=q[:60] + ("…" if len(q) > 60 else ""),
            hovertemplate=(
                "Question: %{customdata}<br>"
                "Time (UTC): %{x}<br>"
                "Price: %{y:.4f}<extra></extra>"
            ),
            customdata=[q] * len(g),
        )
    )

fig.update_layout(
    title=f"Polymarket Minute Prices — {k} Random Crypto Questions ({start.date()} to {end.date()} UTC)",
    xaxis_title="Timestamp (UTC)",
    yaxis_title="Price",
    hovermode="x unified",
    legend_title="Question (truncated)",
)

fig.show()
