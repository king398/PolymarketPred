import asyncio
import json
import math
import time
from numpy_ringbuffer import RingBuffer
import numpy as np
import zmq
from market_websocket import ASSET_ID_FILE

HOST = "127.0.0.1"
PORT = 9000

BUCKET_MS = 250
BUCKET_SIZE = 60
BROADCAST_INTERVAL_S = BUCKET_MS / 1000

# --- Filter Config ---
MIN_VALID_PRICE = 0.00
MAX_VALID_PRICE = 0.99

tick_dtype = np.dtype([
    ("ts_ms", np.int64),
    ("bid", np.float32),
    ("ask", np.float32),
])

# Shared buffers + state
asset_id_que: dict[str, RingBuffer] = {}
state: dict[str, dict] = {}
state_lock = asyncio.Lock()

# Global whitelist (updated dynamically by the publisher loop)
valid_clobs = set()
# add BTC and ETH for testing

ZMQ_PUB = "tcp://127.0.0.1:5567"
ctx = zmq.Context.instance()
pub = ctx.socket(zmq.PUB)
pub.bind(ZMQ_PUB)


def emit(bucket_start_ms: int, aid: str, bid: float, ask: float):
    if aid in asset_id_que:
        asset_id_que[aid].append((int(bucket_start_ms), float(bid), float(ask)))


async def process_market_data():
    """Ingests raw TCP data, filters it, and updates state."""
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

                aid = msg["asset_id"]

                # OPTIONAL: Check against the global whitelist.
                # Even though the publisher prunes, this prevents us from
                # creating NEW buffers for assets that were just removed.
                if valid_clobs and aid not in valid_clobs:
                    continue
                raw_bid = msg.get("bid")
                raw_ask = msg.get("ask")

                if raw_bid is None or raw_ask is None:
                    continue
                if aid != "BTC" and aid != "ETH":
                    if not (MIN_VALID_PRICE <= float(raw_bid) <= MAX_VALID_PRICE):
                        continue
                    if not (MIN_VALID_PRICE <= float(raw_ask) <= MAX_VALID_PRICE):
                        continue
                bid = float(raw_bid)
                ask = float(raw_ask)



                if bid > ask:
                    continue

            except (ValueError, KeyError, json.JSONDecodeError):
                continue

            async with state_lock:
                if aid not in state:
                    now_ms = int(time.time() * 1000)
                    first_boundary = (math.floor(now_ms / BUCKET_MS) + 1) * BUCKET_MS

                    state[aid] = {
                        "last_bid": bid,
                        "last_ask": ask,
                        "next_boundary": first_boundary,
                    }
                    asset_id_que[aid] = RingBuffer(capacity=BUCKET_SIZE, dtype=tick_dtype)
                    emit(first_boundary, aid, bid, ask)
                else:
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
    """Periodically forward-fills the price buckets."""
    sleep_s = BUCKET_MS / 1000.0
    while True:
        now_ms = int(time.time() * 1000)
        async with state_lock:
            aids = list(state.keys())
            for aid in aids:
                st = state[aid]
                while st["next_boundary"] <= now_ms:
                    emit(st["next_boundary"], aid, st["last_bid"], st["last_ask"])
                    st["next_boundary"] += BUCKET_MS
        await asyncio.sleep(sleep_s)


async def publish_all_assets_every_1s():
    """
    1. Loads the latest valid CLOBs from file.
    2. Prunes any memory state for IDs that are no longer in the file.
    3. Publishes the remaining valid assets via ZMQ.
    """
    global valid_clobs

    while True:
        await asyncio.sleep(BROADCAST_INTERVAL_S)

        # 1. READ FILE (Refresh Whitelist)
        current_valid_ids = set()
        current_valid_ids.update({"BTC", "ETH"})

        try:
            with open(ASSET_ID_FILE, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            obj = json.loads(line)
                            current_valid_ids.add(obj["clob_token_id"])
                        except json.JSONDecodeError:
                            pass

            # Update global set so process_market_data sees it too
            if len(current_valid_ids) > 0:
                valid_clobs = current_valid_ids
            else:
                # Safety: If file is empty or read failed, don't wipe everything immediately?
                # Or if you WANT to wipe everything if file is empty, remove this check.
                pass

        except Exception as e:
            print(f"Error reading asset file: {e}")
            # If read fails, skip pruning this tick to be safe
            continue

        # 2. PRUNE & PUBLISH
        async with state_lock:
            # Snapshot keys because we might delete
            tracked_assets = list(asset_id_que.keys())

            for aid in tracked_assets:
                # Check if the asset currently in memory is in our fresh list
                if aid not in valid_clobs:
                    # DELETE: It was removed from the file
                    del asset_id_que[aid]
                    if aid in state:
                        del state[aid]
                    # print(f"Pruned dropped asset: {aid}")
                    continue

                # PUBLISH
                rb = asset_id_que.get(aid)
                if rb is None or len(rb) == 0:
                    continue

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
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
