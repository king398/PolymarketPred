import os
import glob
from datasets import Dataset

# Faster hub upload transport (works if hf_transfer installed)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

DIR = "/home/mithil/PycharmProjects/PolymarketPred/data/polymarket_minute_parquet"
REPO_ID = "Mithilss/polymarket_minute_parquet"

PARQUETS = glob.glob(f"{DIR}/*.parquet")

# Fast-path parquet -> Arrow Dataset
ds = Dataset.from_parquet(PARQUETS)

print(ds)
print("rows:", ds.num_rows, "cols:", ds.column_names)

# Fewer/larger shards = less hub overhead (tune size to your taste)
# (datasets will shard & write in parallel)
ds.push_to_hub(
    REPO_ID,
    private=False,
    max_shard_size="2GB",   # try 2GB–5GB if you have large data
    num_proc=16,
)

print("✅ pushed to hub:", REPO_ID)
