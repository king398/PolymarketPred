import asyncio
import websockets
import time
import re
from typing import Any, Dict, List, Set
import orjson
import json
import os
import zmq
import zmq.asyncio
import struct

# Try to use uvloop for performance
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

json_lib = orjson

# --- CONFIG ---
POLY_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
DATA_DIR = os.path.join(os.getcwd(), "data")
ASSET_ID_FILE = os.path.join(DATA_DIR, "clob_token_ids.jsonl")

# ZMQ CONFIG
ZMQ_PORT = 5567
ZMQ_ADDR = f"tcp://127.0.0.1:{ZMQ_PORT}"

FILE_CHECK_INTERVAL = 60

# --- UTILS ---
_LEADING_NUM_PREFIX = re.compile(r"^\s*\d+")

def _decode_events(raw: str) -> List[Dict[str, Any]]:
    s = raw.strip()
    if s and s[0].isdigit():
        s = _LEADING_NUM_PREFIX.sub("", s, count=1).lstrip()
    try:
        obj = json_lib.loads(s)
    except Exception:
        return []
    if isinstance(obj, dict): return [obj]
    if isinstance(obj, list): return [x for x in obj if isinstance(x, dict)]
    return []

def load_asset_ids():
    """Reads the JSONL file and returns a sorted list of unique IDs."""
    ids = []
    try:
        with open(ASSET_ID_FILE, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        ids.append(json.loads(line)["clob_token_id"])
                    except Exception:
                        pass
        return sorted(list(set(ids)))
    except FileNotFoundError:
        print(f"[System] Warning: {ASSET_ID_FILE} not found.")
        return []

# --- ZMQ PUBLISHER ---
class ZmqPublisher:
    def __init__(self):
        self.ctx = zmq.asyncio.Context()
        self.pub = self.ctx.socket(zmq.PUB)
        # We bind to the port so the Strategy can Connect to it
        self.pub.bind(ZMQ_ADDR)
        print(f"[ZMQ] Publishing Binary Data on {ZMQ_ADDR}")

    async def publish(self, asset_id: str, ts_ms: int, bid: float, ask: float):
        """
        Packs data into binary format for numpy compatibility.
        Format: <qff (Little-endian, int64, float32, float32)
        Matches: np.dtype([("ts_ms", np.int64), ("bid", np.float32), ("ask", np.float32)])
        """
        try:
            # Struct pack: timestamp (8 bytes), bid (4 bytes), ask (4 bytes)
            payload = struct.pack('<qff', ts_ms, bid, ask)
            # Topic is the asset_id
            await self.pub.send_multipart([asset_id.encode(), payload])
        except Exception as e:
            print(f"[ZMQ Error] {e}")

# --- POLYMARKET STREAMER ---
async def stream_polymarket(pub: ZmqPublisher, asset_ids: List[str]) -> None:
    """Streams data for specific IDs and publishes to ZMQ."""
    if not asset_ids:
        print("[Poly] No assets to stream.")
        return

    print(f"[Poly] Starting stream for {len(asset_ids)} assets...")
    market_state = {aid: (None, None) for aid in asset_ids}

    while True:
        try:
            async with websockets.connect(POLY_WS_URL, ping_interval=20, max_size=None) as ws:
                # Subscribe
                sub_msg = {"type": "market", "assets_ids": asset_ids}
                await ws.send(json_lib.dumps(sub_msg).decode())

                async for msg in ws:
                    events = _decode_events(msg)
                    for data in events:
                        ts_ms = data.get("timestamp") or int(time.time() * 1000)
                        et = data.get("event_type")
                        updates = []

                        if et == "price_change":
                            changes = data.get("price_changes") or []
                            if not changes and "asset_id" in data: changes = [data]
                            for ch in changes:
                                aid = ch.get("asset_id")
                                if aid in market_state:
                                    updates.append((aid, ch.get("best_bid"), ch.get("best_ask")))

                        elif et == "book":
                            aid = data.get("asset_id")
                            if aid in market_state:
                                bids = data.get("bids") or []
                                asks = data.get("asks") or []
                                bb = float(bids[0]["price"]) if bids else None
                                ba = float(asks[0]["price"]) if asks else None
                                updates.append((aid, bb, ba))

                        for aid, new_bid, new_ask in updates:
                            last_bid, last_ask = market_state[aid]
                            curr_bid = float(new_bid) if new_bid is not None else last_bid
                            curr_ask = float(new_ask) if new_ask is not None else last_ask

                            if curr_bid == last_bid and curr_ask == last_ask:
                                continue

                            market_state[aid] = (curr_bid, curr_ask)

                            # Publish to ZMQ if we have valid prices
                            if curr_bid is not None and curr_ask is not None:
                                await pub.publish(aid, int(ts_ms), curr_bid, curr_ask)

        except asyncio.CancelledError:
            print("[Poly] Stream cancelled (reloading IDs).")
            raise
        except Exception as e:
            print(f"[Poly] Error: {e}. Reconnecting in 2s...")
            await asyncio.sleep(2)

# --- MAIN MANAGER ---
async def main():
    # 1. Initialize ZMQ Publisher
    pub = ZmqPublisher()


    current_ids = []
    poly_task = None

    # 2. Manage Polymarket Stream (Dynamic Reloading)
    while True:
        # Check file for new IDs
        new_ids = await asyncio.to_thread(load_asset_ids)

        # Compare with current running IDs
        if new_ids and (new_ids != current_ids):
            if current_ids:
                print(f"[System] Poly ID Change! Old: {len(current_ids)}, New: {len(new_ids)}")
            else:
                print(f"[System] Poly Initial Load: {len(new_ids)} IDs found.")

            # Restart Polymarket Task
            if poly_task:
                poly_task.cancel()
                try:
                    await poly_task
                except asyncio.CancelledError:
                    pass

            current_ids = new_ids
            poly_task = asyncio.create_task(stream_polymarket(pub, current_ids))

        # Sleep before checking file again
        await asyncio.sleep(FILE_CHECK_INTERVAL)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down...")