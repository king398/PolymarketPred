from datasets import load_dataset
import re
import random
import pandas as pd
import numpy as np

# -----------------------------
# 0) Load HF dataset
# -----------------------------
ds = load_dataset("Mithilss/polymarket_minute_parquet")["train"]

# -----------------------------
# 1) Fast crypto filter (batched)
# -----------------------------
# pattern = re.compile(r"\b(bitcoin|ethereum|solana|xrp|btc|eth)\b", re.IGNORECASE)
pattern = re.compile(r"\b(bitcoin|ethereum)\b", re.IGNORECASE)


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

print("Filtered rows:", len(ds_crypto))

start = pd.Timestamp("2025-10-01", tz="UTC")
end = start + pd.Timedelta(hours=3)  # exclusive


def in_range_batch(batch):
    ts = pd.to_datetime(batch["timestamp"], utc=True, errors="coerce")
    return (ts >= start) & (ts < end)


ds_range = ds_crypto.filter(
    in_range_batch,
    batched=True,
    batch_size=50_000,
    num_proc=16,
    load_from_cache_file=True,
)


def log_odds(row, eps=1e-6):
    p = np.array(row["price"], dtype=np.float64)
    p = np.clip(p, eps, 1 - eps)
    log_odds = np.log(p / (1 - p))
    return {"log_odds": log_odds}


ds_range = ds_range.map(
    log_odds,
    batched=True,
    batch_size=50_000,
    num_proc=16,
)
