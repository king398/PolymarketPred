import asyncio
import json
import time
import numpy as np
import zmq
from market_websocket import ASSET_ID_FILE

# --- Config ---
HOST = "127.0.0.1"
PORT = 9000

BUCKET_MS = 50
BROADCAST_INTERVAL_S = BUCKET_MS / 1000

# --- Filter Config ---
MIN_VALID_PRICE = 0.00
MAX_VALID_PRICE = 0.99

tick_dtype = np.dtype([
    ("ts_ms", np.int64),
    ("bid", np.float32),
    ("ask", np.float32),
])

# Shared State
# Key: Asset ID
# Value: A single-element numpy array of tick_dtype containing the latest data
state: dict[str, np.ndarray] = {}
state_lock = asyncio.Lock()

# Global whitelist
valid_clobs = set()

# ZMQ Setup
ZMQ_PUB = "tcp://127.0.0.1:5569"
ctx = zmq.Context.instance()
pub = ctx.socket(zmq.PUB)
pub.bind(ZMQ_PUB)


async def process_market_data():
    """Ingests raw TCP data and updates the latest state in-place."""
    print(f"Connecting to tcp://{HOST}:{PORT}...", flush=True)
    reader, writer = await asyncio.open_connection(HOST, PORT)
    print("Connected! Tracking latest price per asset...", flush=True)

    try:
        while True:
            line = await reader.readline()
            if not line:
                break

            try:
                s = line.strip()
                if not s:
                    continue

                # Fast parse
                msg = json.loads(s)
                aid = msg.get("asset_id")

                # Whitelist Check
                if valid_clobs and aid not in valid_clobs:
                    continue

                raw_bid = msg.get("bid")
                raw_ask = msg.get("ask")

                if raw_bid is None or raw_ask is None:
                    continue

                # Filter Logic
                if aid != "BTC" and aid != "ETH":
                    if not (MIN_VALID_PRICE <= float(raw_bid) <= MAX_VALID_PRICE):
                        continue
                    if not (MIN_VALID_PRICE <= float(raw_ask) <= MAX_VALID_PRICE):
                        continue

                bid = float(raw_bid)
                ask = float(raw_ask)

                if bid > ask:
                    continue

                now_ms = int(time.time() * 1000)

                async with state_lock:
                    # If asset exists, update in-place (very fast)
                    if aid in state:
                        arr = state[aid]
                        # Accessing the first element [0] of the 1-item array
                        arr[0]['ts_ms'] = now_ms
                        arr[0]['bid'] = bid
                        arr[0]['ask'] = ask
                    else:
                        # Create new 1-element array
                        arr = np.zeros(1, dtype=tick_dtype)
                        arr[0]['ts_ms'] = now_ms
                        arr[0]['bid'] = bid
                        arr[0]['ask'] = ask
                        state[aid] = arr

            except (ValueError, KeyError, json.JSONDecodeError):
                continue

    except ConnectionError:
        print("Connection lost")
    except Exception as e:
        print(f"Reader error: {e}")
    finally:
        print("Closing connection...")
        writer.close()
        await writer.wait_closed()


async def publish_all_assets_loop():
    """
    Periodically publishes the CURRENT state of all valid assets.
    Does not forward-fill history, saving significant CPU.
    """
    global valid_clobs

    while True:
        await asyncio.sleep(BROADCAST_INTERVAL_S)

        # 1. READ FILE (Refresh Whitelist)
        # We assume the file is small. If this is still too heavy,
        # consider moving this block to run only every 5 seconds.
        current_valid_ids = {"BTC", "ETH"}

        try:
            with open(ASSET_ID_FILE, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            obj = json.loads(line)
                            current_valid_ids.add(obj["clob_token_id"])
                        except json.JSONDecodeError:
                            pass

            if len(current_valid_ids) > 0:
                valid_clobs = current_valid_ids

        except Exception as e:
            # Non-critical error, keep running with old whitelist
            pass

        # 2. PRUNE & PUBLISH
        async with state_lock:
            # Snapshot keys to allow deletion during iteration
            tracked_assets = list(state.keys())

            for aid in tracked_assets:
                # Prune dropped assets
                if aid not in valid_clobs:
                    del state[aid]
                    continue

                # Retrieve the pre-calculated numpy array
                arr = state[aid]

                # OPTIONAL: Update timestamp to 'now' so downstream sees it as fresh?
                # If you want strictly the "last trade time", comment this out.
                # If you want "current heartbeat", leave this in.
                arr[0]['ts_ms'] = int(time.time() * 1000)

                # Send raw bytes.
                # The downstream receives a valid numpy buffer of length 1.
                pub.send_multipart([aid.encode(), arr.tobytes()])


async def main():
    # We removed bucket_timer() as it is unnecessary for "latest value" tracking
    await asyncio.gather(
        process_market_data(),
        publish_all_assets_loop(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")