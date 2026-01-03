import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Iterable
import requests

# ===================== CONFIG =====================
ASSET_ID_FILE = "/home/mithil/PycharmProjects/PolymarketPred/data/clob_token_ids.jsonl"  # JSONL file
BINANCE_REST = "https://api.binance.com"

SYMBOL_MAP = {
    "btc": "BTCUSDT",
    "eth": "ETHUSDT",
    "xrp": "XRPUSDT",
    "sol": "SOLUSDT",
}

# ===================== TIME HELPERS =====================
def parse_iso_with_offset(ts: str) -> datetime:
    """
    Example: "2026-01-03 07:00:00-05:00"
    Returns an aware datetime (includes offset).
    """
    return datetime.fromisoformat(ts)

def to_utc_ms(dt: datetime) -> int:
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)

def now_utc_ms() -> int:
    return int(time.time() * 1000)

# ===================== MARKET → SYMBOL =====================
def infer_binance_symbol(mkt: Dict[str, Any]) -> Optional[str]:
    slug = (mkt.get("slug") or "").lower()
    question = (mkt.get("question") or "").lower()

    # 1) slug prefix: btc-updown-..., eth-updown-..., sol-updown-..., xrp-updown-...
    if "-" in slug:
        prefix = slug.split("-", 1)[0]
        if prefix in SYMBOL_MAP:
            return SYMBOL_MAP[prefix]

    # 2) fallback: keyword in question
    if "bitcoin" in question:
        return SYMBOL_MAP["btc"]
    if "ethereum" in question:
        return SYMBOL_MAP["eth"]
    if "xrp" in question:
        return SYMBOL_MAP["xrp"]
    if "solana" in question:
        return SYMBOL_MAP["sol"]

    return None

# ===================== BINANCE KLINE QUERY (1m EXACT) =====================
def binance_1m_candle_at(
        symbol: str,
        open_ms: int,
        session: requests.Session
) -> Optional[Dict[str, Any]]:
    """
    Fetch the 1-minute candle that STARTS exactly at open_ms (UTC ms).
    Uses klines interval=1m, startTime=open_ms, limit=1, then verifies openTime.
    """
    params = {
        "symbol": symbol,
        "interval": "1m",
        "startTime": open_ms,
        "limit": 1,
    }
    r = session.get(f"{BINANCE_REST}/api/v3/klines", params=params, timeout=10)
    r.raise_for_status()
    kl = r.json()
    if not kl:
        return None

    k = kl[0]
    k_open_time = int(k[0])
    if k_open_time != open_ms:
        # Not the exact candle boundary you asked for
        return None

    return {
        "openTime": k_open_time,
        "open": float(k[1]),
        "high": float(k[2]),
        "low": float(k[3]),
        "close": float(k[4]),
        "volume": float(k[5]),
        "closeTime": int(k[6]),
        "numTrades": int(k[8]),
    }

# ===================== STRIKE (1m OPEN) LOGIC =====================
def strike_open_for_market_1m(
        mkt: Dict[str, Any],
        session: requests.Session
) -> Optional[Dict[str, Any]]:
    """
    Returns the 1-minute candle OPEN that begins exactly at primary_market_timestamp.
    """
    ts = mkt.get("primary_market_timestamp")
    if not ts:
        return None

    symbol = infer_binance_symbol(mkt)
    if not symbol:
        return None

    start_dt = parse_iso_with_offset(ts)
    start_ms = to_utc_ms(start_dt)

    # Skip future markets
    if start_ms > now_utc_ms():
        return None

    k = binance_1m_candle_at(symbol, start_ms, session)
    if not k:
        return None

    return {
        "slug": mkt.get("slug"),
        "clob_token_id": mkt.get("clob_token_id"),
        "symbol": symbol,
        "interval": "1m",
        "market_start_iso": ts,
        "market_start_utc_ms": start_ms,
        "candle_open_utc_ms": k["openTime"],
        "open_price": k["open"],     # STRIKE proxy (start price)
        "high": k["high"],
        "low": k["low"],
        "close_price": k["close"],   # useful for resolution checks later
        "closeTime": k["closeTime"],
        "volume": k["volume"],
        "numTrades": k["numTrades"],
        "method": "binance /api/v3/klines interval=1m startTime=primary_market_timestamp",
        "mean_price": (k["open"] + k["close"]) / 2.0,
    }

# ===================== FILE LOADER =====================
def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # skip bad lines
                continue

# ===================== MAIN =====================
def main():
    session = requests.Session()
    results = []

    for market in iter_jsonl(ASSET_ID_FILE):
        res = strike_open_for_market_1m(market, session)
        if res:
            results.append(res)
            print(
                f"{res['symbol']:7s} {res['interval']:2s} | "
                f"{res['market_start_iso']} | "
                f"OPEN={res['open_price']:.8f} "
                f"H={res['high']:.8f} L={res['low']:.8f} C={res['close_price']:.8f} "
                f"V={res['volume']:.6f} T={res['numTrades']}"
            )

    print(f"\nComputed 1m candle opens: {len(results)}")

    out_path = "market_1m_candle_opens.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
