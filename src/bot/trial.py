import asyncio
import json
import math
import time
from collections import deque
from numpy_ringbuffer import RingBuffer
import numpy as np
HOST = "127.0.0.1"
PORT = 9000

asset_id_que = {}
BUCKET_MS = 1
tick_dtype = np.dtype([
    ("ts_ms", np.int64),
    ("bid",   np.float32),
    ("ask",   np.float32),
])

def emit(bucket_start_ms, asset_id, bid, ask):
    rec = {
        "ts": int(bucket_start_ms),
        "asset_id": asset_id,
        "bid": bid,
        "ask": ask,
    }
    # Fast JSON dump
    line = json.dumps(rec, separators=(",", ":"))
    now_ms = time.time_ns() // 1_000_000
    asset_id_que[asset_id].append((now_ms, bid, ask))

async def process_market_data():
    print(f"Connecting to tcp://{HOST}:{PORT}...", flush=True)
    reader, writer = await asyncio.open_connection(HOST, PORT)
    print("Connected! Aggregating buckets...", flush=True)


    # State tracking per Asset ID
    # Structure: { asset_id: { "last_bid": float, "last_ask": float, "next_boundary": int } }
    state = {}

    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            # 2. Parse
            try:
                line_str = line.strip()
                if not line_str: continue
                msg = json.loads(line_str)

                # Handle raw types safely
                # Ensure ts_ms handles both int and string input
                ts_ms = int(msg["ts_ms"])
                aid = msg["asset_id"]

                raw_bid = msg.get("bid")
                raw_ask = msg.get("ask")
                if raw_bid is None or raw_ask is None:
                    continue

                bid = float(raw_bid)
                ask = float(raw_ask)

            except (ValueError, KeyError, json.JSONDecodeError):
                continue

            # 3. Initialize New Asset if unseen
            if aid not in state:
                # Align to the next grid boundary
                first_boundary = (math.floor(ts_ms / BUCKET_MS) + 1) * BUCKET_MS
                state[aid] = {
                    "last_bid": bid,
                    "last_ask": ask,
                    "next_boundary": first_boundary
                }
                asset_id_que[aid] = RingBuffer(capacity=4096, dtype=tick_dtype)
                asset_id_que[aid].append((ts_ms, bid, ask))
                continue

            # 4. Forward Fill Logic (Per Asset)
            asset_state = state[aid]

            # While the current message time is past the asset's next bucket boundary
            while ts_ms >= asset_state["next_boundary"]:
                # Emit the LKV (state BEFORE this new tick) for that boundary
                emit(
                    asset_state["next_boundary"],
                    aid,
                    asset_state["last_bid"],
                    asset_state["last_ask"],

                )
                asset_state["next_boundary"] += BUCKET_MS

            # 5. Update State (LKV)
            asset_state["last_bid"] = bid
            asset_state["last_ask"] = ask

    except ConnectionError:
        print("Connection lost")
    finally:
        writer.close()
        await writer.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(process_market_data())
    except KeyboardInterrupt:
        print("\nStopped.")