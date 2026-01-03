import asyncio
import websockets
import time
import re
from typing import Any, Dict, List, Set
import orjson
import json

# Try to use uvloop for performance
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

json_lib = orjson

# --- CONFIG ---
POLY_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
BINANCE_WS_URL = "wss://fstream.binance.com/ws"  # USDS-M Futures
ASSET_ID_FILE = "/home/mithil/PycharmProjects/PolymarketPred/data/clob_token_ids.jsonl"
BIND_HOST = "127.0.0.1"
BIND_PORT = 9000
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

# --- BROADCASTER & SERVER ---
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
        if not self.clients: return
        dead = []
        for w in self.clients:
            try:
                w.write(line)
            except Exception:
                dead.append(w)
        for w in dead: self.remove(w)

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

# --- BINANCE STREAMER (BTC/ETH) ---
async def stream_binance(b: Broadcaster) -> None:
    """Streams BTC and ETH Book Tickers from Binance Spot."""
    # Spot combined stream URL format: /stream?streams=btcusdt@bookTicker/ethusdt@bookTicker
    spot_url = "wss://stream.binance.com:9443/stream?streams=btcusdt@bookTicker/ethusdt@bookTicker"

    while True:
        try:
            async with websockets.connect(spot_url, ping_interval=20) as ws:
                print("[Binance Spot] Subscribed to BTC/ETH.")

                async for msg in ws:
                    try:
                        raw_data = json_lib.loads(msg)
                        # Combined streams wrap data in a "data" key
                        data = raw_data.get("data")

                        symbol = data["s"] # e.g., "BTCUSDT"
                        asset_id = symbol.replace("USDT", "")

                        out = {
                            "ts_ms": int(time.time() * 1000), # Spot ticker doesn't always have 'T'
                            "asset_id": asset_id,
                            "bid": float(data["b"]),
                            "ask": float(data["a"])
                        }

                        line = json_lib.dumps(out) + b"\n"
                        b.broadcast_line(line)

                    except Exception as e:
                        continue

        except Exception as e:
            print(f"[Binance Spot] Error: {e}. Reconnecting...")
            await asyncio.sleep(2)
# --- POLYMARKET STREAMER ---
async def stream_polymarket(b: Broadcaster, asset_ids: List[str]) -> None:
    """Streams data for specific IDs. Raises CancelledError if task is cancelled."""
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
                            out = {
                                "ts_ms": ts_ms,
                                "asset_id": aid,
                                "bid": curr_bid,
                                "ask": curr_ask
                            }

                            line = (json_lib.dumps(out)) + b"\n"
                            b.broadcast_line(line)

        except asyncio.CancelledError:
            print("[Poly] Stream cancelled (reloading IDs).")
            raise
        except Exception as e:
            print(f"[Poly] Error: {e}. Reconnecting in 2s...")
            await asyncio.sleep(2)

# --- MAIN MANAGER ---
async def main():
    b = Broadcaster()

    # 1. Start TCP Server
    asyncio.create_task(tcp_server(b))

    # 2. Start Binance Stream (Permanent Background Task)
    asyncio.create_task(stream_binance(b))

    current_ids = []
    poly_task = None

    # 3. Manage Polymarket Stream (Dynamic Reloading)
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
            poly_task = asyncio.create_task(stream_polymarket(b, current_ids))

        # Sleep before checking file again
        await asyncio.sleep(FILE_CHECK_INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())