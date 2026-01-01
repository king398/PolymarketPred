import asyncio
import json
import math
import time
from numpy_ringbuffer import RingBuffer
import numpy as np
import zmq

HOST = "127.0.0.1"
PORT = 9000

BUCKET_MS = 100
BUCKET_SIZE = 8192
BROADCAST_INTERVAL_S = 1.0

# --- Filter Config ---
# Adjust these based on your asset's real price range
MIN_VALID_PRICE = 0.00
MAX_VALID_PRICE = 0.99

tick_dtype = np.dtype([
    ("ts_ms", np.int64),
    ("bid",   np.float32),
    ("ask",   np.float32),
])

# Shared buffers + state
asset_id_que: dict[str, RingBuffer] = {}
state: dict[str, dict] = {}
state_lock = asyncio.Lock()

ZMQ_PUB = "tcp://127.0.0.1:5567"
ctx = zmq.Context.instance()
pub = ctx.socket(zmq.PUB)
pub.bind(ZMQ_PUB)

def emit(bucket_start_ms: int, aid: str, bid: float, ask: float):
    # assumes asset_id_que[aid] exists
    asset_id_que[aid].append((int(bucket_start_ms), float(bid), float(ask)))

async def process_market_data():
    print(f"Connecting to tcp://{HOST}:{PORT}...", flush=True)
    reader, writer = await asyncio.open_connection(HOST, PORT)
    print("Connected! Tracking last bid/ask per asset...", flush=True)

    try:
        while True:
            line = await reader.readline()
            if not line:
                break

            try:
                s = line.strip()
                if not s:
                    continue
                msg = json.loads(s)

                ts_ms = int(msg["ts_ms"])
                aid = msg["asset_id"]

                raw_bid = msg.get("bid")
                raw_ask = msg.get("ask")

                # 1. Basic Null Check
                if raw_bid is None or raw_ask is None:
                    continue

                bid = float(raw_bid)
                ask = float(raw_ask)

                # 2. DATA QUALITY FILTER (The Fix)
                # Ignore placeholder/reset values so they don't poison the state
                if bid <= MIN_VALID_PRICE or ask >= MAX_VALID_PRICE:
                    # Optional: Print once to confirm we are catching them
                    # print(f"Ignored garbage tick: {bid}/{ask}")
                    continue

                # 3. Sanity Check: Crossed Market (optional but recommended)
                if bid > ask:
                    continue

            except (ValueError, KeyError, json.JSONDecodeError):
                continue

            async with state_lock:
                if aid not in state:
                    now_ms = int(time.time() * 1000)
                    # Align to next grid point
                    first_boundary = (math.floor(now_ms / BUCKET_MS) + 1) * BUCKET_MS

                    state[aid] = {
                        "last_bid": bid,
                        "last_ask": ask,
                        "next_boundary": first_boundary,
                    }
                    asset_id_que[aid] = RingBuffer(capacity=BUCKET_SIZE, dtype=tick_dtype)

                    # Seed the first point
                    emit(first_boundary, aid, bid, ask)
                else:
                    # Update LKV (Last Known Value)
                    st = state[aid]
                    st["last_bid"] = bid
                    st["last_ask"] = ask

    except ConnectionError:
        print("Connection lost")
    except Exception as e:
        print(f"Reader error: {e}")
    finally:
        print("Closing connection...")
        writer.close()
        await writer.wait_closed()

async def bucket_timer():
    """
    Every BUCKET_MS, forward-fill each asset up to now_ms using last_bid/ask.
    """
    sleep_s = BUCKET_MS / 1000.0

    while True:
        # Calculate now once per loop
        now_ms = int(time.time() * 1000)

        async with state_lock:
            # Snapshot keys to allow safe iteration
            aids = list(state.keys())

            for aid in aids:
                st = state[aid]

                # Forward fill all missed buckets up to current time
                # If data stopped coming, this repeats the last good price.
                while st["next_boundary"] <= now_ms:
                    emit(st["next_boundary"], aid, st["last_bid"], st["last_ask"])
                    st["next_boundary"] += BUCKET_MS

        # Sleep accounting for drift could be added, but simple sleep is usually fine
        await asyncio.sleep(sleep_s)

async def publish_all_assets_every_1s():
    while True:
        await asyncio.sleep(BROADCAST_INTERVAL_S)

        # Snapshot keys
        asset_ids = list(asset_id_que.keys())

        for aid in asset_ids:
            rb = asset_id_que.get(aid)
            if rb is None or len(rb) == 0:
                continue

            # Creating an array from RingBuffer is safe in asyncio
            # (no context switch inside np.array constructor)
            arr = np.array(rb, copy=True)
            pub.send_multipart([aid.encode(), arr.tobytes()])

async def main():
    await asyncio.gather(
        process_market_data(),
        bucket_timer(),
        publish_all_assets_every_1s(),
    )

if __name__ == "__main__":
    try:
        # Use uvloop if available for better performance, else standard
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")