import asyncio
import websockets
import json
import time
import aiofiles
import re
from typing import Any, Dict, List, Optional, Tuple, Set

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
ASSET_ID = "115193106500252441347057856539225168339065498062654799670373647667592722696153"

OUTPUT_FILE = "market_quotes.jsonl"

# TCP broadcast server
BIND_HOST = "127.0.0.1"
BIND_PORT = 9000

_LEADING_NUM_PREFIX = re.compile(r"^\s*\d+")


def _decode_events(raw: str) -> List[Dict[str, Any]]:
    s = raw.strip()
    if s and s[0].isdigit():
        s = _LEADING_NUM_PREFIX.sub("", s, count=1).lstrip()
    obj = json.loads(s)
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    return []


def _best_from_book(book_msg: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    bids = book_msg.get("bids") or []
    asks = book_msg.get("asks") or []
    best_bid = max((float(x["price"]) for x in bids), default=None)
    best_ask = min((float(x["price"]) for x in asks), default=None)
    return best_bid, best_ask


class Broadcaster:
    """
    Simple TCP "fanout":
      - multiple clients connect
      - every update is sent as one JSON line (NDJSON)
      - on connect, we optionally send the last known quote immediately
    """
    def __init__(self) -> None:
        self.clients: Set[asyncio.StreamWriter] = set()
        self.lock = asyncio.Lock()
        self.last_line: Optional[bytes] = None

    async def add(self, writer: asyncio.StreamWriter) -> None:
        async with self.lock:
            self.clients.add(writer)
            # Send last quote immediately (nice for new clients)
            if self.last_line is not None:
                try:
                    writer.write(self.last_line)
                    await writer.drain()
                except Exception:
                    self.clients.discard(writer)

    async def remove(self, writer: asyncio.StreamWriter) -> None:
        async with self.lock:
            self.clients.discard(writer)
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass

    async def broadcast_line(self, line: bytes) -> None:
        # remember last
        self.last_line = line
        async with self.lock:
            writers = list(self.clients)

        if not writers:
            return

        dead: List[asyncio.StreamWriter] = []
        for w in writers:
            try:
                w.write(line)
            except Exception:
                dead.append(w)

        # Drain in a second pass (avoids one slow client blocking the others too hard)
        for w in writers:
            if w in dead:
                continue
            try:
                await w.drain()
            except Exception:
                dead.append(w)

        if dead:
            async with self.lock:
                for w in dead:
                    self.clients.discard(w)
                    try:
                        w.close()
                    except Exception:
                        pass


async def tcp_server(b: Broadcaster) -> None:
    async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info("peername")
        print(f"[TCP] client connected: {addr}")
        await b.add(writer)
        try:
            # keep connection open; if client sends anything, we ignore
            while True:
                data = await reader.read(1024)
                if not data:
                    break
        except Exception:
            pass
        finally:
            print(f"[TCP] client disconnected: {addr}")
            await b.remove(writer)

    server = await asyncio.start_server(handle_client, BIND_HOST, BIND_PORT)
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    print(f"[TCP] broadcasting on {addrs}")
    async with server:
        await server.serve_forever()


async def stream_polymarket_and_publish(b: Broadcaster) -> None:
    print(f"[WS] Streaming ONLY asset_id={ASSET_ID}")

    async with websockets.connect(
            WS_URL,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
            max_size=None,
    ) as ws:
        await ws.send(json.dumps({
            "type": "market",
            "assets_ids": [ASSET_ID],
        }))

        best_bid: Optional[float] = None
        best_ask: Optional[float] = None
        last_written: Tuple[Optional[float], Optional[float]] = (None, None)

        async with aiofiles.open(OUTPUT_FILE, "a") as f:
            async for msg in ws:
                try:
                    events = _decode_events(msg)
                except json.JSONDecodeError:
                    continue

                for data in events:
                    ts_ms = int(data.get("timestamp", time.time() * 1000))
                    et = data.get("event_type")

                    if et == "book" and data.get("asset_id") == ASSET_ID:
                        b0, a0 = _best_from_book(data)
                        if b0 is not None:
                            best_bid = b0
                        if a0 is not None:
                            best_ask = a0

                    elif et == "price_change":
                        for ch in data.get("price_changes", []):
                            if ch.get("asset_id") != ASSET_ID:
                                continue
                            if ch.get("best_bid") is not None:
                                best_bid = float(ch["best_bid"])
                            if ch.get("best_ask") is not None:
                                best_ask = float(ch["best_ask"])

                    current = (best_bid, best_ask)
                    if current != last_written and any(v is not None for v in current):
                        out = {"ts_ms": ts_ms, "bid": best_bid, "ask": best_ask}
                        line_str = json.dumps(out, separators=(",", ":")) + "\n"
                        line = line_str.encode("utf-8")

                        # write to file (optional but you wanted it)
                        await f.write(line_str)
                        await f.flush()

                        # broadcast to all TCP clients
                        await b.broadcast_line(line)

                        last_written = current


async def main():
    b = Broadcaster()
    await asyncio.gather(
        tcp_server(b),
        stream_polymarket_and_publish(b),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
