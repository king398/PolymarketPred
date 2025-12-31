import asyncio
import websockets
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

# --- CONFIG ---
POLY_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
BINANCE_WS_URL = "wss://stream.binance.com:9443/stream?streams=btcusdt@bookTicker/ethusdt@bookTicker/xrpusdt@bookTicker/solusdt@bookTicker"

ASSET_ID_FILE = "/home/mithil/PycharmProjects/PolymarketPred/data/clob_token_ids.jsonl"
BIND_HOST = "127.0.0.1"
BIND_PORT = 9000

# Mapping Binance symbols to your preferred "Asset IDs"
CRYPTO_MAP = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",
    "XRPUSDT": "XRP",
    "SOLUSDT": "SOL"
}

# Load Asset IDs
ASSET_IDS = []
try:
    with open(ASSET_ID_FILE, "r") as f:
        for line in f:
            if line.strip():
                ASSET_IDS.append(json.loads(line)["clob_token_id"])
except FileNotFoundError:
    print(f"Error: Could not find {ASSET_ID_FILE}")
    ASSET_IDS = []

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
        b.add(writer)
        try:
            await reader.read()
        except Exception:
            pass
        finally:
            b.remove(writer)

    server = await asyncio.start_server(handle_client, BIND_HOST, BIND_PORT)
    print(f"[TCP] Broadcasting on {BIND_HOST}:{BIND_PORT}")
    async with server:
        await server.serve_forever()

# --- MODIFIED: THROTTLED CRYPTO STREAM (100ms) ---
async def stream_binance_prices(b: Broadcaster) -> None:
    print("[WS] Connecting to Binance (Sampling every 100ms)...")

    # Shared state: stores the absolute latest price seen
    # Structure: { "BTCUSDT": {"bid": 90000.0, "ask": 90001.0}, ... }
    latest_state: Dict[str, Dict[str, float]] = {}

    # 1. Background Task: The "Ticker"
    # This wakes up every 100ms and broadcasts the state
    async def ticker_loop():
        last_broadcast_state = {}

        while True:
            await asyncio.sleep(0.1) # Wait 100ms

            # Use current server time for the "bucket" timestamp
            ts_ms = int(time.time() * 1000)

            # Snapshot the dictionary to iterate safely
            current_snapshot = latest_state.copy()

            for symbol, data in current_snapshot.items():
                # OPTIONAL: Check if price changed since last 100ms tick.
                # If you want to force output even if price is same, remove this 'if'.
                prev_data = last_broadcast_state.get(symbol)

                if prev_data != data:
                    out = {
                        "ts_ms": ts_ms,
                        "asset_id": CRYPTO_MAP[symbol],
                        "bid": data["bid"],
                        "ask": data["ask"]
                    }

                    if json_lib == json:
                        line = (json.dumps(out) + "\n").encode("utf-8")
                    else:
                        line = json_lib.dumps(out) + b"\n"

                    b.broadcast_line(line)

                    # Update local cache
                    last_broadcast_state[symbol] = data

    # Start the ticker in background
    asyncio.create_task(ticker_loop())

    # 2. Main Loop: The "Ingester"
    # This reads the firehose and updates 'latest_state' as fast as possible
    while True:
        try:
            async with websockets.connect(BINANCE_WS_URL) as ws:
                async for msg in ws:
                    try:
                        data = json_lib.loads(msg)
                        payload = data.get("data", {})
                        symbol = payload.get("s")

                        if symbol in CRYPTO_MAP:
                            # Just update the variable, don't broadcast yet
                            latest_state[symbol] = {
                                "bid": float(payload["b"]),
                                "ask": float(payload["a"])
                            }
                    except Exception:
                        continue
        except Exception as e:
            print(f"[Binance] Error: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)

async def stream_polymarket_and_publish(b: Broadcaster) -> None:
    print(f"[WS] Streaming {len(ASSET_IDS)} Polymarket assets...")
    market_state = {aid: (None, None) for aid in ASSET_IDS}

    while True:
        try:
            async with websockets.connect(POLY_WS_URL, ping_interval=None, max_size=None) as ws:
                sub_msg = {"type": "market", "assets_ids": ASSET_IDS}
                await ws.send(json.dumps(sub_msg) if json_lib == json else json_lib.dumps(sub_msg).decode())

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

                        for aid, new_bid_raw, new_ask_raw in updates:
                            last_bid, last_ask = market_state[aid]
                            current_bid = float(new_bid_raw) if new_bid_raw is not None else last_bid
                            current_ask = float(new_ask_raw) if new_ask_raw is not None else last_ask

                            if (current_bid == last_bid) and (current_ask == last_ask):
                                continue

                            market_state[aid] = (current_bid, current_ask)
                            out = {
                                "ts_ms": ts_ms,
                                "asset_id": aid,
                                "bid": current_bid,
                                "ask": current_ask
                            }

                            if json_lib == json:
                                line = (json.dumps(out) + "\n").encode("utf-8")
                            else:
                                line = json_lib.dumps(out) + b"\n"
                            b.broadcast_line(line)

        except Exception as e:
            print(f"[Polymarket] Disconnected: {e}. Reconnecting in 2s...")
            await asyncio.sleep(2)

async def main():
    b = Broadcaster()
    await asyncio.gather(
        tcp_server(b),
        stream_polymarket_and_publish(b),
        ## stream_binance_prices(b),
    )

if __name__ == "__main__":
    try:
        try:
            import uvloop
            uvloop.install()
        except ImportError:
            pass
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")