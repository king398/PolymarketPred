import json
import time
import os
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Iterable, Set, List

# ===================== CONFIG =====================
DATA_DIR = "/home/mithil/PycharmProjects/PolymarketPred/data"
ASSET_ID_FILE = os.path.join(DATA_DIR, "clob_token_ids.jsonl")
OUTPUT_FILE = os.path.join(DATA_DIR, "market_1m_candle_opens.jsonl")

BINANCE_REST = "https://api.binance.com"
API_BUFFER_SEC = 2  # Seconds to wait AFTER market start before querying Binance (ensure data availability)

SYMBOL_MAP = {
    "btc": "BTCUSDT",
    "eth": "ETHUSDT",
    "xrp": "XRPUSDT",
    "sol": "SOLUSDT",
}

# ===================== TIME HELPERS =====================
def parse_iso_with_offset(ts: str) -> datetime:
    return datetime.fromisoformat(ts)

def to_utc_ms(dt: datetime) -> int:
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)

def now_utc_ms() -> int:
    return int(time.time() * 1000)

def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))

# ===================== MARKET → SYMBOL =====================
def infer_binance_symbol(mkt: Dict[str, Any]) -> Optional[str]:
    slug = (mkt.get("slug") or "").lower()
    question = (mkt.get("question") or "").lower()

    if "-" in slug:
        prefix = slug.split("-", 1)[0]
        if prefix in SYMBOL_MAP:
            return SYMBOL_MAP[prefix]

    if "bitcoin" in question: return SYMBOL_MAP["btc"]
    if "ethereum" in question: return SYMBOL_MAP["eth"]
    if "xrp" in question: return SYMBOL_MAP["xrp"]
    if "solana" in question: return SYMBOL_MAP["sol"]

    return None

# ===================== BINANCE QUERY =====================
def binance_1m_candle_at(symbol: str, open_ms: int, session: requests.Session) -> Optional[Dict[str, Any]]:
    """
    Fetch the 1-minute candle that STARTS exactly at open_ms.
    """
    params = {
        "symbol": symbol,
        "interval": "1m",
        "startTime": open_ms,
        "limit": 1,
    }
    try:
        r = session.get(f"{BINANCE_REST}/api/v3/klines", params=params, timeout=10)
        r.raise_for_status()
        kl = r.json()
        if not kl:
            return None

        k = kl[0]
        k_open_time = int(k[0])

        # Verify we got the exact candle (Binance snaps to nearest, we want exact match)
        # However, for 1m candles, snapping is usually accurate if timestamp is minute-aligned.
        # We allow a small tolerance or strict check. Strict is better for "Open Price".
        if abs(k_open_time - open_ms) > 1000:
            return None

        return {
            "openTime": k_open_time,
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "numTrades": int(k[8]),
        }
    except Exception as e:
        print(f"Error fetching Binance data for {symbol}: {e}")
        return None

# ===================== CORE LOGIC =====================
def get_strike_data(mkt: Dict[str, Any], session: requests.Session) -> Optional[Dict[str, Any]]:
    ts = mkt.get("primary_market_timestamp")
    if not ts: return None

    symbol = infer_binance_symbol(mkt)
    if not symbol: return None

    start_dt = parse_iso_with_offset(ts)
    start_ms = to_utc_ms(start_dt)

    # 1. Check if future
    if start_ms > now_utc_ms():
        return None # Too early

    # 2. Fetch
    k = binance_1m_candle_at(symbol, start_ms, session)
    if not k:
        return None

    return {
        "slug": mkt.get("slug"),
        "clob_token_id": mkt.get("clob_token_id"),
        "symbol": symbol,
        "market_start_iso": ts,
        "market_start_utc_ms": start_ms,
        "candle_open_utc_ms": k["openTime"],
        "open_price": k["open"],
        "mean_price": (k["open"] + k["close"]) / 2.0,
    }

def load_processed_ids() -> Set[str]:
    """Load IDs we have already fetched to avoid duplicates."""
    processed = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        processed.add(data.get("clob_token_id"))
                    except: pass
    return processed

def load_pending_markets(processed_ids: Set[str]) -> List[Dict[str, Any]]:
    """Reads the input file and returns markets we haven't processed yet."""
    pending = []
    if not os.path.exists(ASSET_ID_FILE):
        print(f"Warning: {ASSET_ID_FILE} not found.")
        return []

    with open(ASSET_ID_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                m = json.loads(line)
                if m.get("clob_token_id") not in processed_ids:
                    # Parse time for sorting
                    ts = m.get("primary_market_timestamp")
                    if ts:
                        m["_start_ms"] = to_utc_ms(parse_iso_with_offset(ts))
                        pending.append(m)
            except: pass

    # Sort by start time (earliest first)
    pending.sort(key=lambda x: x["_start_ms"])
    return pending

# ===================== MAIN LOOP =====================
def main():
    session = requests.Session()
    print(f"Starting indefinite monitor.")
    print(f"Input: {ASSET_ID_FILE}")
    print(f"Output: {OUTPUT_FILE}")

    while True:
        # 1. Load State
        processed_ids = load_processed_ids()
        pending_markets = load_pending_markets(processed_ids)

        if not pending_markets:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] No pending markets. Sleeping 60s...")
            time.sleep(60)
            continue

        # 2. Get the next immediate market
        next_mkt = pending_markets[0]
        start_ms = next_mkt["_start_ms"]
        current_ms = now_utc_ms()

        # Calculate wait time
        wait_ms = start_ms - current_ms

        # 3. If market is in the future, SLEEP
        if wait_ms > 0:
            wait_sec = wait_ms / 1000.0
            wake_time = datetime.now() + timedelta(seconds=wait_sec)
            slug = next_mkt.get('slug', 'unknown')
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Next market: {slug}")
            print(f"   Starts in: {format_duration(wait_sec)}")
            print(f"   Sleeping until: {wake_time.strftime('%H:%M:%S')} (plus buffer)")

            # Sleep exactly until start + buffer
            time.sleep(wait_sec + API_BUFFER_SEC)

            # Wake up and refresh current_ms
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Waking up to fetch data...")

        # 4. If market is in the past (or we just woke up)
        # We process ALL markets that have started by now
        # (This handles the case where multiple markets start at the same time)

        markets_to_fetch = [m for m in pending_markets if m["_start_ms"] <= now_utc_ms()]

        if not markets_to_fetch:
            # Should generally not happen if logic above is correct,
            # unless wait_ms was < 0 but barely.
            time.sleep(1)
            continue

        print(f"Processing {len(markets_to_fetch)} markets ready for data collection...")

        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for mkt in markets_to_fetch:
                res = get_strike_data(mkt, session)

                if res:
                    f.write(json.dumps(res) + "\n")
                    f.flush() # Ensure it's written immediately
                    processed_ids.add(mkt["clob_token_id"])
                    print(f"   [SAVED] {res['slug']} | OPEN: {res['open_price']}")
                else:
                    # If None is returned, it might be too early (data not on Binance yet)
                    # or the symbol mapping failed.
                    # If it's simply too early (e.g. seconds ago), we might want to retry later.
                    # If it's old (days ago) and fails, it's likely a mapping error.
                    age_sec = (now_utc_ms() - mkt["_start_ms"]) / 1000
                    if age_sec < 60:
                        print(f"   [RETRY LATER] {mkt.get('slug')} (Data likely not ready)")
                        # Don't add to processed_ids so it gets picked up next loop
                    else:
                        print(f"   [FAILED] {mkt.get('slug')} (Could not fetch/map)")
                        # Optional: Add to processed to stop infinite retrying on broken assets
                        processed_ids.add(mkt["clob_token_id"])

        # Small sleep between batches to prevent tight loops if something is weird
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")