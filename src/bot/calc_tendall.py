import zmq
import numpy as np
from  tqdm import tqdm
import  json
from market_websocket import ASSET_ID_FILE
from scipy.stats import kendalltau
from itertools import combinations

tick_dtype = np.dtype([
    ("ts_ms", np.int64),
    ("bid",   np.float32),
    ("ask",   np.float32),
])

ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect("tcp://127.0.0.1:5567")

valid_clobs = []
clob_question_map = {}
with open(ASSET_ID_FILE, "r") as f:
    for line in f.readlines():
        obj = json.loads(line)
        valid_clobs.append(obj["clob_token_id"])
        clob_question_map[obj["clob_token_id"]] = obj["question"]
print(valid_clobs)
# subscribe to everything
sub.setsockopt(zmq.SUBSCRIBE, b"")
assets_ticks = {}
for _ in tqdm(range(100)):
    aid, payload = sub.recv_multipart()
    if aid.decode() not in valid_clobs:
        continue
    arr = np.frombuffer(payload, dtype=tick_dtype)
    mid = (arr["bid"] + arr["ask"]) * 0.5
    if (mid.std() < 1e-3) or (len(arr) < 10):
        continue
    assets_ticks[aid.decode()] = arr



mid_prices = {
    aid: (arr["bid"] + arr["ask"]) * 0.5
    for aid, arr in assets_ticks.items()
}

results = {}
import time
start = time.time()
for (aid1, s1), (aid2, s2) in combinations(mid_prices.items(), 2):
    n = min(len(s1), len(s2))
    if n < 2:
        continue

    tau, pval = kendalltau(s1[:n], s2[:n])
    results[(aid1, aid2)] = {
        "tau": tau,
        "p_value": pval,
        "n": n,
    }
print(f"Kendall tau computation took {time.time() - start:.2f} seconds")
# pretty print
top20 = sorted(
    results.items(),
    key=lambda x: abs(x[1]["tau"]),
    reverse=True
)[:20]

for (a, b), r in top20:
    print(f"{clob_question_map[a]} ↔ {clob_question_map[b]}: τ={r['tau']:.4f}, n={r['n']}")




