import asyncio
import json
import math
import time
from numpy_ringbuffer import RingBuffer
import numpy as np
import zmq

HOST = "127.0.0.1"
PORT = 9000
BUCKET_SIZE = 600
asset_id_que = {}
BUCKET_MS = 1000
tick_dtype = np.dtype([
    ("ts_ms", np.int64),
    ("bid",   np.float32),
    ("ask",   np.float32),
])

ZMQ_PUB = "tcp://127.0.0.1:5567"
ctx = zmq.Context.instance()
pub = ctx.socket(zmq.PUB)
pub.bind(ZMQ_PUB)

def emit(bucket_start_ms, asset_id, bid, ask):
    asset_id_que[asset_id].append((int(bucket_start_ms), bid, ask))
    if len(asset_id_que[asset_id]) == BUCKET_SIZE:
        print(f"Filled bucket for asset {asset_id}:")

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
                asset_id_que[aid] = RingBuffer(capacity=BUCKET_SIZE, dtype=tick_dtype)
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
async def publish_all_assets_every_1s():
    while True:
        await asyncio.sleep(BUCKET_MS / 1000.0)
        # Snapshot keys to avoid dict-size-change issues
        asset_ids = list(asset_id_que.keys())

        for aid in asset_ids:
            rb = asset_id_que.get(aid)
            if rb is None or len(rb) == 0:
                continue

            arr = np.asarray(rb)



            # Broadcast: topic = asset_id
            pub.send_multipart([
                aid.encode(),
                arr.tobytes(),
            ])
async def main():
    await asyncio.gather(
        process_market_data(),
        publish_all_assets_every_1s(),
    )



if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")