import asyncio
import aiohttp
import json
import ast
import pytz
import re
import os
import logging
import warnings
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple

# --- CONFIG ---
DATA_DIR = os.path.join(os.getcwd(), "data")
MARKET_ID_FILE = os.path.join(DATA_DIR, "clob_options_token_ids.jsonl")
CANDLE_OUTPUT_FILE = os.path.join(DATA_DIR, "market_1m_candle_opens.jsonl")

POLY_BASE_URL = "https://gamma-api.polymarket.com/events/slug/{slug}"
BINANCE_KLINE_API = "https://api.binance.com/api/v3/klines"
BINANCE_PRICE_API = "https://api.binance.com/api/v3/ticker/price"

# Silence NumPy Deprecation Warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("PolyBot")

# Timezone
ET = pytz.timezone("US/Eastern")

# Symbols for Discovery
SYMBOLS_LONG = ["bitcoin", "ethereum",  ]

# Binance Symbol Mapping
BINANCE_MAP = {
    "btc": "BTCUSDT", "bitcoin": "BTCUSDT",
    "eth": "ETHUSDT", "ethereum": "ETHUSDT", "ether": "ETHUSDT",
    "xrp": "XRPUSDT", "ripple": "XRPUSDT",
    "sol": "SOLUSDT", "solana": "SOLUSDT",
}


# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------

def get_current_et():
    return datetime.now(ET)


def to_utc_ms(dt_str: str) -> int:
    """Converts ISO string to UTC timestamp in ms."""
    dt = datetime.fromisoformat(dt_str)
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def extract_strike_from_slug(slug: str) -> Optional[float]:
    """
    Parses strike price from slug (For Weekly Markets).
    Ex: 'bitcoin-above-92k-on...' -> 92000.0
    Ex: 'xrp-above-2pt1-on...' -> 2.1
    """
    try:
        match = re.search(r'above-(.*?)-on', slug)
        if not match:
            return None

        raw_val = match.group(1)

        if 'k' in raw_val:
            raw_val = raw_val.replace('k', '')
            multiplier = 1000
        else:
            multiplier = 1

        if 'pt' in raw_val:
            raw_val = raw_val.replace('pt', '.')

        val = float(raw_val) * multiplier
        return val
    except Exception:
        return None


def format_option_ticker(symbol: str, market_end_dt: datetime, strike: float, option_type: str) -> str:
    """
    Formats the option ticker: BTC-26DEC25-95000-C
    Logic: Symbol - (MarketEnd + 1 Day) - Strike - Type
    """
    try:
        # 1. Clean Symbol
        sym = symbol.upper()
        if sym in ["BITCOIN"]: sym = "BTC"
        if sym in ["ETHEREUM", "ETHER"]: sym = "ETH"
        if sym in ["RIPPLE"]: sym = "XRP"

        # 2. Calculate Option Date = Market End + 1 Day
        option_date = market_end_dt + timedelta(days=1)
        date_str = option_date.strftime("%d%b%y").upper() # Ex: 26DEC25

        # 3. Format Strike (Integer if possible)
        if strike % 1 == 0:
            strike_str = str(int(strike))
        else:
            strike_str = str(strike)

        # 4. Type Char
        type_char = "C" if option_type == "call" else "P"
        if option_type == "unknown": type_char = "?"

        return f"{sym}-{date_str}-{strike_str}-{type_char}"

    except Exception as e:
        log.error(f"Ticker format error: {e}")
        return "UNKNOWN-TICKER"


def infer_binance_symbol(slug: str, question: str) -> Optional[str]:
    """Guesses Binance ticker from slug/question."""
    slug_lower = slug.lower()
    q_lower = question.lower()

    for key, val in BINANCE_MAP.items():
        if f"{key}-" in slug_lower or key in q_lower:
            return val
    return None


def generate_buckets(interval, count) -> List[Tuple[str, datetime]]:
    """Generates time strings for URL slugs."""
    now = get_current_et()
    buckets = []
    if interval == "1d":
        if now.hour >= 12:
            start = now.replace(hour=12, minute=0, second=0, microsecond=0) + timedelta(days=1)
        else:
            start = now.replace(hour=12, minute=0, second=0, microsecond=0)
        delta = timedelta(days=1)
        fmt_func = lambda dt: f"{dt.strftime('%B').lower()}-{dt.day}"
    else:
        return []

    for i in range(count):
        dt = start + (delta * i)
        buckets.append((fmt_func(dt), dt))
    return buckets


def parse_tokens(market):
    raw = market.get("clobTokenIds")
    if isinstance(raw, str): raw = ast.literal_eval(raw)
    if raw and len(raw) >= 2: return raw[0]
    return None


def parse_prices(market):
    raw = market.get("outcomePrices")
    if isinstance(raw, str): return ast.literal_eval(raw)
    return [float(p) for p in raw]


# -----------------------------------------------------------------------------
# MARKET DISCOVERY (PRODUCER)
# -----------------------------------------------------------------------------

async def fetch_slug(session, slug):
    url = POLY_BASE_URL.format(slug=slug)
    try:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
    except Exception:
        pass
    return None


async def fetch_current_price(session, symbol: str) -> Optional[float]:
    params = {"symbol": symbol}
    try:
        async with session.get(BINANCE_PRICE_API, params=params, timeout=5) as response:
            if response.status == 200:
                data = await response.json()
                return float(data["price"])
    except Exception as e:
        log.warning(f"⚠️ Failed to get current price for {symbol}: {e}")
    return None


async def process_weekly_markets(session, count):
    results = []
    buckets = generate_buckets("1d", count)
    pattern = "{symbol}-above-on-{param}"

    for symbol in SYMBOLS_LONG:

        # 1. Get Binance Symbol & Spot Price
        binance_ticker = BINANCE_MAP.get(symbol)
        current_spot = None
        if binance_ticker:
            current_spot = await fetch_current_price(session, binance_ticker)
            if current_spot:
                log.info(f"💰 Current {binance_ticker} Price: {current_spot}")

        # 2. Generate tasks
        tasks = []
        for i, (url_param, ts) in enumerate(buckets):
            market_end = ts
            market_start = ts - timedelta(days=6)
            slug = pattern.format(symbol=symbol, param=url_param)
            tasks.append((slug, i, market_start, market_end))

        # 3. Fetch
        fetch_tasks = [fetch_slug(session, t[0]) for t in tasks]
        responses = await asyncio.gather(*fetch_tasks)

        for (slug, pos, start_ts, end_ts), data in zip(tasks, responses):
            if not data or "markets" not in data: continue

            for market in data["markets"]:
                try:
                    prices = parse_prices(market)
                    prices = [float(p) for p in prices if p is not None]
                    if not prices or max(prices) > 0.95 or min(prices) < 0.05:
                        continue

                    strike = extract_strike_from_slug(market['slug'])

                    # --- Logic: Call/Put ---
                    market_type = "unknown"
                    if strike is not None and current_spot is not None:
                        if strike > current_spot:
                            market_type = "call"
                        elif strike < current_spot:
                            market_type = "put"

                    # --- Logic: Option Ticker ---
                    # We pass end_ts (Market End)
                    option_ticker = "N/A"
                    if strike is not None:
                        option_ticker = format_option_ticker(symbol, end_ts, strike, market_type)

                    results.append({
                        "slug": market['slug'],
                        "clob_token_id": parse_tokens(market),
                        "market_position": pos,
                        "category": "7d",
                        "primary_market_timestamp": str(start_ts),
                        "market_end": str(end_ts),
                        "question": market["question"],
                        "strike_price": strike,
                        "current_spot_at_discovery": current_spot,
                        "type": market_type,
                        "related_option": option_ticker  # <--- NEW FIELD
                    })
                except Exception:
                    continue
    return results


# -----------------------------------------------------------------------------
# BINANCE FETCHER (CONSUMER)
# -----------------------------------------------------------------------------

async def fetch_binance_candle(session, symbol: str, open_ms: int) -> Optional[Dict]:
    params = {"symbol": symbol, "interval": "1m", "startTime": open_ms, "limit": 1}
    for i in range(5):
        try:
            async with session.get(BINANCE_KLINE_API, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data:
                        k = data[0]
                        k_open_time = int(k[0])
                        if abs(k_open_time - open_ms) < 1000:
                            return {
                                "open": float(k[1]),
                                "close": float(k[4]),
                                "openTime": k_open_time
                            }
        except Exception:
            pass
        await asyncio.sleep(2)
    return None


async def fulfill_market(market_data: Dict, processed_set: set):
    cid = market_data.get('clob_token_id')
    slug = market_data.get('slug')
    m_type = market_data.get('type', 'unknown')
    op_ticker = market_data.get('related_option', 'N/A')

    try:
        start_ts_str = market_data['primary_market_timestamp']
        start_ms = to_utc_ms(start_ts_str)
    except Exception as e:
        log.error(f"Date parse error {slug}: {e}")
        return

    wait_ms = start_ms - now_ms()
    wait_sec = (wait_ms / 1000.0) + 10.0

    if wait_sec > 0:
        if wait_sec < 3600:
            log.info(f"⏳ Waiting {wait_sec:.1f}s for open: {slug}")
        await asyncio.sleep(wait_sec)

    symbol = infer_binance_symbol(slug, market_data.get('question', ''))
    if not symbol:
        return

    async with aiohttp.ClientSession() as session:
        candle = await fetch_binance_candle(session, symbol, start_ms)

    if candle:
        slug_strike = market_data.get("strike_price")
        final_strike = candle["open"] if slug_strike is None else slug_strike

        output_row = {
            "slug": slug,
            "clob_token_id": cid,
            "symbol": symbol,
            "market_start_iso": start_ts_str,
            "candle_open_utc_ms": candle["openTime"],
            "open_price": candle["open"],
            "strike_price": final_strike,
            "type": m_type,
            "related_option": op_ticker # <--- SAVING OPTION TICKER
        }

        with open(CANDLE_OUTPUT_FILE, "a") as f:
            f.write(json.dumps(output_row) + "\n")

        log.info(f"✅ [CAPTURED] {op_ticker} | Strike: {final_strike}")
        processed_set.add(cid)
    else:
        log.warning(f"⚠️ Candle miss: {slug}")


# -----------------------------------------------------------------------------
# MAIN LOOPS
# -----------------------------------------------------------------------------

async def discovery_loop(queue: asyncio.Queue, seen_ids: set):
    log.info("[System] Discovery Loop Started (10s interval)")
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                new_batch = await process_weekly_markets(session, 5)
                added_count = 0
                for m in new_batch:
                    cid = m.get('clob_token_id')
                    if cid and cid not in seen_ids:
                        seen_ids.add(cid)
                        with open(MARKET_ID_FILE, "a") as f:
                            f.write(json.dumps(m) + "\n")
                        await queue.put(m)
                        added_count += 1
                if added_count > 0:
                    log.info(f"🔎 Discovered {added_count} new markets.")
        except Exception as e:
            log.error(f"Discovery Error: {e}")
        await asyncio.sleep(10)


async def processing_loop(queue: asyncio.Queue, processed_strikes: set):
    log.info("[System] Processing Loop Started")
    while True:
        market = await queue.get()
        asyncio.create_task(fulfill_market(market, processed_strikes))
        queue.task_done()


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

async def main():
    seen_ids = set()
    processed_strikes = set()

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if os.path.exists(MARKET_ID_FILE):
        with open(MARKET_ID_FILE, 'r') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if 'clob_token_id' in d: seen_ids.add(d['clob_token_id'])
                except: pass

    if os.path.exists(CANDLE_OUTPUT_FILE):
        with open(CANDLE_OUTPUT_FILE, 'r') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if 'clob_token_id' in d: processed_strikes.add(d['clob_token_id'])
                except: pass

    log.info(f"Loaded {len(seen_ids)} known markets and {len(processed_strikes)} captured strikes.")

    queue = asyncio.Queue()

    if os.path.exists(MARKET_ID_FILE):
        with open(MARKET_ID_FILE, 'r') as f:
            for line in f:
                try:
                    m = json.loads(line)
                    cid = m.get('clob_token_id')
                    if cid and cid not in processed_strikes:
                        queue.put_nowait(m)
                except: pass

    log.info(f"Queued {queue.qsize()} pending markets from file history.")

    await asyncio.gather(
        discovery_loop(queue, seen_ids),
        processing_loop(queue, processed_strikes)
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[System] Stopped by user.")