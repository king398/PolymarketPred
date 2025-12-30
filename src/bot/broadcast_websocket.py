import asyncio
import websockets
import json
import time
import re
from typing import Any, Dict, List, Optional, Tuple, Set

# Try to use orjson for speed, fallback to standard json
try:
    import orjson
    json_lib = orjson
except ImportError:
    import json
    json_lib = json

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
ASSET_ID_FILE = "/home/mithil/PycharmProjects/PolymarketPred/data/clob_token_ids.jsonl"

# Load Asset IDs
ASSET_IDS = []
try:
    with open(ASSET_ID_FILE, "r") as f:
        for line in f:
            if line.strip():
                # Use standard json here as it's a one-time load
                ASSET_IDS.append(json.loads(line)["clob_token_id"])
except FileNotFoundError:
    print(f"Error: Could not find {ASSET_ID_FILE}")
    ASSET_IDS = []

# TCP broadcast server
BIND_HOST = "127.0.0.1"
BIND_PORT = 9000

_LEADING_NUM_PREFIX = re.compile(r"^\s*\d+")

def _decode_events(raw: str) -> List[Dict[str, Any]]:
    s = raw.strip()
    if s and s[0].isdigit():
        s = _LEADING_NUM_PREFIX.sub("", s, count=1).lstrip()

    try:
        obj = json_lib.loads(s)
    except Exception:
        return []

    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    return []

class Broadcaster:
    def __init__(self) -> None:
        self.clients: Set[asyncio.StreamWriter] = set()
        # Removed lock for pure speed (asyncio is single threaded)

    def add(self, writer: asyncio.StreamWriter):
        self.clients.add(writer)

    def remove(self, writer: asyncio.StreamWriter):
        self.clients.discard(writer)
        try:
            writer.close()
        except Exception:
            pass

    def broadcast_line(self, line: bytes):
        if not self.clients:
            return

        # Optimistic write (no await) to keep the feed moving fast
        dead = []
        for w in self.clients:
            try:
                w.write(line)
            except Exception:
                dead.append(w)

        for w in dead:
            self.remove(w)

async def tcp_server(b: Broadcaster) -> None:
    async def handle_client(reader, writer):
        addr = writer.get_extra_info("peername")
        print(f"[TCP] New client: {addr}")
        b.add(writer)
        try:
            # Keep connection open until client disconnects
            await reader.read()
        except Exception:
            pass
        finally:
            print(f"[TCP] Client disconnected: {addr}")
            b.remove(writer)

    server = await asyncio.start_server(handle_client, BIND_HOST, BIND_PORT)
    print(f"[TCP] Broadcasting on {BIND_HOST}:{BIND_PORT}")
    async with server:
        await server.serve_forever()

async def stream_polymarket_and_publish(b: Broadcaster) -> None:
    print(f"[WS] Streaming {len(ASSET_IDS)} assets...")

    # Dictionary to track last known state: { asset_id: (bid, ask) }
    # This prevents sending duplicate data if the price hasn't actually changed
    market_state = {aid: (None, None) for aid in ASSET_IDS}

    async with websockets.connect(
            WS_URL,
            ping_interval=None, # Disable ping for lower latency
            max_size=None,
    ) as ws:

        # Subscribe
        sub_msg = {"type": "market", "assets_ids": ASSET_IDS}
        await ws.send(json.dumps(sub_msg) if json_lib == json else json_lib.dumps(sub_msg).decode())

        async for msg in ws:
            # 1. Decode
            events = _decode_events(msg)

            for data in events:
                ts_ms = data.get("timestamp") or int(time.time() * 1000)
                et = data.get("event_type")

                updates = [] # List of (asset_id, bid, ask)

                # 2. Extract Data
                if et == "price_change":
                    changes = data.get("price_changes") or []
                    if not changes and "asset_id" in data:
                        changes = [data]

                    for ch in changes:
                        aid = ch.get("asset_id")
                        # Only process if this is an asset we care about
                        if aid in market_state:
                            updates.append((aid, ch.get("best_bid"), ch.get("best_ask")))

                elif et == "book":
                    aid = data.get("asset_id")
                    if aid in market_state:
                        bids = data.get("bids") or []
                        asks = data.get("asks") or []
                        # Safely get price or None
                        bb = float(bids[0]["price"]) if bids else None
                        ba = float(asks[0]["price"]) if asks else None
                        updates.append((aid, bb, ba))

                # 3. Update State & Broadcast (O(1) - only for changed assets)
                for aid, new_bid_raw, new_ask_raw in updates:
                    # Retrieve current state
                    last_bid, last_ask = market_state[aid]

                    # Parse new values (keep old if new is None)
                    current_bid = float(new_bid_raw) if new_bid_raw is not None else last_bid
                    current_ask = float(new_ask_raw) if new_ask_raw is not None else last_ask

                    # If nothing changed, skip
                    if (current_bid == last_bid) and (current_ask == last_ask):
                        continue

                    # Update State
                    market_state[aid] = (current_bid, current_ask)

                    # Prepare Output
                    out = {
                        "ts_ms": ts_ms,
                        "asset_id": aid,
                        "bid": current_bid,
                        "ask": current_ask
                    }

                    # Serialize and Broadcast
                    if json_lib == json:
                        line_str = json.dumps(out, separators=(",", ":")) + "\n"
                        line = line_str.encode("utf-8")
                    else:
                        line = json_lib.dumps(out) + b"\n"

                    b.broadcast_line(line)

async def main():
    b = Broadcaster()
    await asyncio.gather(
        tcp_server(b),
        stream_polymarket_and_publish(b),
    )

if __name__ == "__main__":
    try:
        # Optional: Use uvloop if available for linux performance
        try:
            import uvloop
            uvloop.install()
        except ImportError:
            pass

        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")