import asyncio
import aiohttp
import json
import ast
import pytz
import re
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple

# --- CONFIG ---
DATA_DIR = "/home/mithil/PycharmProjects/PolymarketPred/data"
MARKET_ID_FILE = os.path.join(DATA_DIR, "clob_token_ids.jsonl")
CANDLE_OUTPUT_FILE = os.path.join(DATA_DIR, "market_1m_candle_opens.jsonl")

POLY_BASE_URL = "https://gamma-api.polymarket.com/events/slug/{slug}"
BINANCE_API = "https://api.binance.com/api/v3/klines"

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
SYMBOLS_SHORT = ["btc", "eth", "xrp", "sol"]
SYMBOLS_LONG = ["bitcoin", "ethereum", "xrp", "solana"]

# Binance Symbol Mapping
BINANCE_MAP = {
    "btc": "BTCUSDT", "bitcoin": "BTCUSDT",
    "eth": "ETHUSDT", "ethereum": "ETHUSDT", "ether": "ETHUSDT",
    "xrp": "XRPUSDT", "ripple": "XRPUSDT",
    "sol": "SOLUSDT", "solana": "SOLUSDT",
    "doge": "DOGEUSDT", "dogecoin": "DOGEUSDT",
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
        # Regex: find text between "above-" and "-on"
        match = re.search(r'above-(.*?)-on', slug)
        if not match:
            return None

        raw_val = match.group(1)

        # Normalize 'k' (thousands)
        if 'k' in raw_val:
            raw_val = raw_val.replace('k', '')
            multiplier = 1000
        else:
            multiplier = 1

        # Normalize 'pt' (decimal points)
        if 'pt' in raw_val:
            raw_val = raw_val.replace('pt', '.')

        val = float(raw_val) * multiplier
        return val
    except Exception:
        return None

def infer_binance_symbol(slug: str, question: str) -> Optional[str]:
    """Guesses Binance ticker from slug/question."""
    slug_lower = slug.lower()
    q_lower = question.lower()

    # Check explicit keys
    for key, val in BINANCE_MAP.items():
        if f"{key}-" in slug_lower or key in q_lower:
            return val
    return None

def generate_buckets(interval, count) -> List[Tuple[str, datetime]]:
    """Generates time strings for URL slugs."""
    now = get_current_et()
    buckets = []

    if interval == "1h":
        start = now.replace(minute=0, second=0, microsecond=0)
        delta = timedelta(hours=1)
        fmt_func = lambda dt: f"{dt.strftime('%B').lower()}-{dt.day}-{dt.strftime('%I%p').lstrip('0').lower()}"
    elif interval == "4h":
        start = now.replace(hour=(now.hour // 4) * 4, minute=0, second=0, microsecond=0)
        delta = timedelta(hours=4)
        fmt_func = lambda dt: int(dt.timestamp())
    elif interval == "1d":
        # Daily Logic: If past 12pm ET, start tomorrow. Else, today.
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

async def process_weekly_markets(session, count):
    """
    Weekly logic:
    1. Slug date = End Date (12 PM ET)
    2. Start Date = End Date - 5 Days
    3. Extract Strike from slug
    """
    tasks = []
    buckets = generate_buckets("1d", count)
    pattern = "{symbol}-above-on-{param}"

    for symbol in SYMBOLS_LONG:
        for i, (url_param, ts) in enumerate(buckets):
            market_end = ts
            market_start = ts - timedelta(days=5)
            slug = pattern.format(symbol=symbol, param=url_param)
            tasks.append((slug, i, market_start, market_end))

    results = []
    fetch_tasks = [fetch_slug(session, t[0]) for t in tasks]
    responses = await asyncio.gather(*fetch_tasks)

    for (slug, pos, start_ts, end_ts), data in zip(tasks, responses):
        if not data or "markets" not in data: continue

        for market in data["markets"]:
            try:
                # Filter decided markets
                prices = parse_prices(market)
                prices = [float(p) for p in prices if p is not None]
                if not prices or max(prices) > 0.95 or min(prices) < 0.05:
                    continue

                # STRICTLY extract strike for weeklies
                strike = extract_strike_from_slug(market['slug'])

                results.append({
                    "slug": market['slug'],
                    "clob_token_id": parse_tokens(market),
                    "market_position": pos,
                    "category": "7d",
                    "primary_market_timestamp": str(start_ts),
                    "market_end": str(end_ts),
                    "question": market["question"],
                    "strike_price": strike
                })
            except Exception:
                continue
    return results

async def process_standard_markets(session, category, symbols, interval, count, pattern):
    tasks = []
    buckets = generate_buckets(interval, count)

    duration_map = {
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1)
    }

    for symbol in symbols:
        for i, (url_param, ts) in enumerate(buckets):
            slug = pattern.format(symbol=symbol, param=url_param)

            if category == "1d":
                market_end = ts
                market_start = ts - timedelta(days=1)
            else:
                market_start = ts
                market_end = ts + duration_map.get(category, timedelta(0))

            tasks.append((slug, i, market_start, market_end))

    results = []
    fetch_tasks = [fetch_slug(session, t[0]) for t in tasks]
    responses = await asyncio.gather(*fetch_tasks)

    for (slug, pos, start_ts, end_ts), data in zip(tasks, responses):
        if not data or "markets" not in data: continue
        try:
            market = data["markets"][0]

            # Standard markets do not have strike in slug (Up/Down)
            # We set this to None explicitly

            results.append({
                "slug": slug,
                "clob_token_id": parse_tokens(market),
                "market_position": pos,
                "category": category,
                "primary_market_timestamp": str(start_ts),
                "market_end": str(end_ts),
                "question": market["question"],
                "strike_price": None
            })
        except Exception:
            continue
    return results

# -----------------------------------------------------------------------------
# BINANCE FETCHER (CONSUMER)
# -----------------------------------------------------------------------------

async def fetch_binance_candle(session, symbol: str, open_ms: int) -> Optional[Dict]:
    params = {"symbol": symbol, "interval": "1m", "startTime": open_ms, "limit": 1}
    # Retry logic
    for i in range(5):
        try:
            async with session.get(BINANCE_API, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data:
                        k = data[0]
                        k_open_time = int(k[0])
                        # Tolerance check (1s)
                        if abs(k_open_time - open_ms) < 1000:
                            return {
                                "open": float(k[1]),
                                "close": float(k[4]),
                                "openTime": k_open_time
                            }
        except Exception:
            pass
        await asyncio.sleep(1.5 * (i + 1))
    return None

async def fulfill_market(market_data: Dict, processed_set: set):
    """Waits for start time, fetches binance price, writes to output."""
    cid = market_data.get('clob_token_id')
    slug = market_data.get('slug')

    # 1. Parse timestamps
    try:
        start_ts_str = market_data['primary_market_timestamp']
        start_ms = to_utc_ms(start_ts_str)
    except Exception as e:
        log.error(f"Date parse error {slug}: {e}")
        return

    # 2. Wait logic
    wait_ms = start_ms - now_ms()
    if wait_ms > 0:
        wait_sec = wait_ms / 1000.0
        # Only log if it's a short wait, otherwise it spams
        if wait_sec < 3600:
            log.info(f"⏳ Waiting {wait_sec:.1f}s for open: {slug}")
        await asyncio.sleep(wait_sec + 2.0) # +2s buffer

    # 3. Fetch Binance
    symbol = infer_binance_symbol(slug, market_data.get('question', ''))
    if not symbol:
        log.warning(f"❌ No Binance symbol for {slug}")
        return

    async with aiohttp.ClientSession() as session:
        candle = await fetch_binance_candle(session, symbol, start_ms)

    # 4. Save
    if candle:
        # LOGIC:
        # If strike_price (from slug) exists -> Use it (Weekly)
        # If strike_price is None -> Use Candle Open (Standard)
        slug_strike = market_data.get("strike_price")

        # --- FIXED LOGIC ---
        # If slug_strike is None, use the Candle Open price as the strike
        if slug_strike is None:
            final_strike = candle["open"]
        else:
            final_strike = slug_strike

        output_row = {
            "slug": slug,
            "clob_token_id": cid,
            "symbol": symbol,
            "market_start_iso": start_ts_str,
            "candle_open_utc_ms": candle["openTime"],
            "open_price": candle["open"],
            "strike_price": final_strike
        }

        with open(CANDLE_OUTPUT_FILE, "a") as f:
            f.write(json.dumps(output_row) + "\n")

        log.info(f"✅ [STRIKE CAPTURED] {slug} | {final_strike}")
        processed_set.add(cid)
    else:
        log.warning(f"⚠️ Candle miss: {slug}")

# -----------------------------------------------------------------------------
# MAIN LOOPS
# -----------------------------------------------------------------------------

async def discovery_loop(queue: asyncio.Queue, seen_ids: set):
    """Polls Polymarket every 10s."""
    log.info("[System] Discovery Loop Started (10s interval)")

    while True:
        try:
            async with aiohttp.ClientSession() as session:
                new_batch = []

                # 1. Standard Markets (Strike will be None)
                new_batch.extend(await process_standard_markets(
                    session, "1h", SYMBOLS_LONG, "1h", 2, "{symbol}-up-or-down-{param}-et"
                ))
                new_batch.extend(await process_standard_markets(
                    session, "4h", SYMBOLS_SHORT, "4h", 1, "{symbol}-updown-4h-{param}"
                ))
                new_batch.extend(await process_standard_markets(
                    session, "1d", SYMBOLS_LONG, "1d", 2, "{symbol}-up-or-down-on-{param}"
                ))

                # 2. Weekly Markets (Strike extracted from slug)
                new_batch.extend(await process_weekly_markets(session, 9))

                # 3. Process Batch
                added_count = 0
                for m in new_batch:
                    cid = m.get('clob_token_id')
                    if cid and cid not in seen_ids:
                        seen_ids.add(cid)
                        # Write to ID File
                        with open(MARKET_ID_FILE, "a") as f:
                            f.write(json.dumps(m) + "\n")
                        # Send to Processor
                        await queue.put(m)
                        added_count += 1

                if added_count > 0:
                    log.info(f"🔎 Discovered {added_count} new markets.")

        except Exception as e:
            log.error(f"Discovery Error: {e}")

        await asyncio.sleep(10)

async def processing_loop(queue: asyncio.Queue, processed_strikes: set):
    """Reads from queue and spawns background tasks to wait/fetch strikes."""
    log.info("[System] Processing Loop Started")
    while True:
        market = await queue.get()
        # Spawn a background task for this market so we don't block the queue
        asyncio.create_task(fulfill_market(market, processed_strikes))
        queue.task_done()

# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

async def main():
    # 1. Load History to avoid duplicates
    seen_ids = set()
    processed_strikes = set()

    # Load IDs
    if os.path.exists(MARKET_ID_FILE):
        with open(MARKET_ID_FILE, 'r') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if 'clob_token_id' in d: seen_ids.add(d['clob_token_id'])
                except: pass

    # Load Completed Strikes
    if os.path.exists(CANDLE_OUTPUT_FILE):
        with open(CANDLE_OUTPUT_FILE, 'r') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if 'clob_token_id' in d: processed_strikes.add(d['clob_token_id'])
                except: pass

    log.info(f"Loaded {len(seen_ids)} known markets and {len(processed_strikes)} captured strikes.")

    # 2. Setup Queue
    queue = asyncio.Queue()

    # 3. Populate Queue with existing IDs that haven't been struck yet (Resume capability)
    if os.path.exists(MARKET_ID_FILE):
        with open(MARKET_ID_FILE, 'r') as f:
            for line in f:
                try:
                    m = json.loads(line)
                    cid = m.get('clob_token_id')
                    if cid and cid not in processed_strikes:
                        # Reprocessing history: Check if it's too old to bother?
                        # For now, just add it. The fetcher handles expired data gracefully.
                        queue.put_nowait(m)
                except: pass

    log.info(f"Queued {queue.qsize()} pending markets from file history.")

    # 4. Start Loops
    await asyncio.gather(
        discovery_loop(queue, seen_ids),
        processing_loop(queue, processed_strikes)
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[System] Stopped by user.")