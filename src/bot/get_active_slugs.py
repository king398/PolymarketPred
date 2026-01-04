import asyncio
import aiohttp
import json
import ast
import pytz
from datetime import datetime, timedelta

# --- CONFIG ---
OUTPUT_FILE = "clob_token_ids.jsonl" # Adjusted path for portability
BASE_URL = "https://gamma-api.polymarket.com/events/slug/{slug}"

# Symbols
SYMBOLS_SHORT = ["btc", "eth", "xrp", "sol"]
SYMBOLS_LONG = ["bitcoin", "ethereum", "xrp", "solana"]

ET = pytz.timezone("US/Eastern")

# Mapping categories to timedeltas
DURATION_MAP = {
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
    "weekly": timedelta(days=7)
}

def get_current_et():
    return datetime.now(ET)

def generate_buckets(interval, count):
    """Generates time strings for URL slugs."""
    now = get_current_et()
    buckets = []

    if interval == "15m":
        start = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
        delta = timedelta(minutes=15)
        fmt_func = lambda dt: int(dt.timestamp())
    elif interval == "4h":
        start = now.replace(hour=(now.hour // 4) * 4, minute=0, second=0, microsecond=0)
        delta = timedelta(hours=4)
        fmt_func = lambda dt: int(dt.timestamp())
    elif interval == "1h":
        start = now.replace(minute=0, second=0, microsecond=0)
        delta = timedelta(hours=1)
        fmt_func = lambda dt: f"{dt.strftime('%B').lower()}-{dt.day}-{dt.strftime('%I%p').lstrip('0').lower()}"
    elif interval == "1d":
        # --- LOGIC FIX: DAILY MARKETS ---
        # If it is past 12:00 PM ET, today's daily is CLOSED. We start looking at Tomorrow.
        # If it is before 12:00 PM ET, today's daily is OPEN. We start looking at Today.
        if now.hour >= 12:
            start = now.replace(hour=12, minute=0, second=0, microsecond=0) + timedelta(days=1)
        else:
            start = now.replace(hour=12, minute=0, second=0, microsecond=0)

        # We want to look FORWARD (Active markets), not backward.
        delta = timedelta(days=1)
        fmt_func = lambda dt: f"{dt.strftime('%B').lower()}-{dt.day}"
    else:
        return []

    for i in range(count):
        dt = start + (delta * i)
        buckets.append((fmt_func(dt), dt))
    return buckets


async def fetch_slug(session, slug):
    """Async fetch of a single slug."""
    url = BASE_URL.format(slug=slug)
    try:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
    except Exception:
        pass
    return None


def parse_tokens(market):
    """Returns a tuple of (Yes_ID, No_ID)."""
    raw = market.get("clobTokenIds")
    if isinstance(raw, str):
        raw = ast.literal_eval(raw)

    # Safety check for list length
    if raw and len(raw) >= 2:
        return raw[0], raw[1]
    return None, None


async def process_standard_markets(session, category, symbols, interval, count, pattern):
    tasks = []
    buckets = generate_buckets(interval, count)

    for symbol in symbols:
        for i, (url_param, ts) in enumerate(buckets):
            slug = pattern.format(symbol=symbol, param=url_param)

            # Timestamp Logic:
            # For 1d markets: The slug date is the END date. Start date is 1 day prior.
            if category == "1d":
                market_end = ts
                market_start = ts - timedelta(days=1)
            else:
                market_start = ts
                market_end = ts + DURATION_MAP.get(category, timedelta(0))

            tasks.append((slug, i, market_start, market_end))

    results = []
    fetch_tasks = [fetch_slug(session, t[0]) for t in tasks]
    responses = await asyncio.gather(*fetch_tasks)

    for (slug, pos, start_ts, end_ts), data in zip(tasks, responses):
        if not data or "markets" not in data:
            continue
        try:
            market = data["markets"][0]

            # --- EXTRACT YES AND NO TOKENS ---
            yes_id, no_id = parse_tokens(market)

            results.append({
                "slug": slug,
                "yes_token_id": yes_id,
                "no_token_id": no_id,
                "market_position": pos,
                "category": category,
                "primary_market_timestamp": str(start_ts),
                "market_end": str(end_ts),
                "question": market["question"]
            })
        except Exception:
            continue
    return results


async def main():
    print(f"[System] Starting ID Fetcher Loop (Interval: 60s)")

    while True:
        start_time = datetime.now(tz=ET)
        print(f"\n[System] Fetch cycle started at {start_time.strftime('%H:%M:%S')} ET...")

        try:
            async with aiohttp.ClientSession() as session:
                all_data = []

                # 1. 15 Minute
                # print("Fetching 15m...")
                all_data.extend(await process_standard_markets(
                    session, "15m", SYMBOLS_SHORT, "15m", 2, "{symbol}-updown-15m-{param}"))

                # 2. 1 Hour
                # print("Fetching 1h...")
                all_data.extend(await process_standard_markets(
                    session, "1h", SYMBOLS_LONG, "1h", 2, "{symbol}-up-or-down-{param}-et"
                ))

                # 3. 4 Hour
                # print("Fetching 4h...")
                all_data.extend(await process_standard_markets(
                    session, "4h", SYMBOLS_SHORT, "4h", 1, "{symbol}-updown-4h-{param}"
                ))

                # 4. 1 Day (Daily)
                print("Fetching 1d (Daily)...")
                all_data.extend(await process_standard_markets(
                    session, "1d", SYMBOLS_LONG, "1d", 2, "{symbol}-up-or-down-on-{param}"
                ))

                if all_data:
                    with open(OUTPUT_FILE, "w") as f:
                        for row in all_data:
                            f.write(json.dumps(row) + "\n")
                    print(f"[System] Success: Updated {OUTPUT_FILE} with {len(all_data)} markets.")

                    # Debug print to verify correct Daily market capture
                    daily_markets = [m for m in all_data if m['category'] == '1d']
                    if daily_markets:
                        print(f"   > Captured Daily: {daily_markets[0]['question']} (End: {daily_markets[0]['market_end']})")
                else:
                    print("[System] Warning: No data fetched this cycle.")

        except Exception as e:
            print(f"[Error] Cycle failed: {e}")

        print("[System] Sleeping for 60s...")
        await asyncio.sleep(60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[System] Stopped by user.")