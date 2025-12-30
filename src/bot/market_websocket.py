import asyncio
import websockets
import json
import time
import aiofiles
import re
from typing import Any, Dict, List, Optional, Tuple

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
ASSET_ID = "94123550151947355465084155553085412002538984957339940876169723664064204017612"
OUTPUT_FILE = "market_quotes.jsonl"

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

async def stream():
    print(f"Streaming ONLY asset_id={ASSET_ID}")

    async with websockets.connect(
            WS_URL,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
            max_size=None,
    ) as ws:
        # subscribe ONLY to this token
        await ws.send(json.dumps({
            "type": "market",
            "assets_ids": [ASSET_ID],
        }))

        best_bid: Optional[float] = None
        best_ask: Optional[float] = None

        last_written = (None, None)

        async with aiofiles.open(OUTPUT_FILE, "a") as f:
            async for msg in ws:
                try:
                    events = _decode_events(msg)
                except json.JSONDecodeError:
                    continue

                for data in events:
                    ts_ms = int(data.get("timestamp", time.time() * 1000))
                    et = data.get("event_type")

                    # ---------- BOOK SNAPSHOT ----------
                    if et == "book" and data.get("asset_id") == ASSET_ID:
                        b, a = _best_from_book(data)
                        if b is not None: best_bid = b
                        if a is not None: best_ask = a

                    # ---------- LAST TRADE ----------

                    # ---------- INCREMENTAL UPDATES ----------
                    elif et == "price_change":
                        for ch in data.get("price_changes", []):
                            if ch.get("asset_id") != ASSET_ID:
                                continue
                            if ch.get("best_bid") is not None:
                                best_bid = float(ch["best_bid"])
                            if ch.get("best_ask") is not None:
                                best_ask = float(ch["best_ask"])

                    # ---------- WRITE ONLY IF CHANGED ----------
                    current = (best_bid, best_ask)
                    if current != last_written and any(v is not None for v in current):
                        out = {
                            "ts_ms": ts_ms,
                            "bid": best_bid,
                            "ask": best_ask,
                        }
                        await f.write(json.dumps(out) + "\n")
                        await f.flush()
                        last_written = current

if __name__ == "__main__":
    try:
        asyncio.run(stream())
    except KeyboardInterrupt:
        print("\nStopped.")
