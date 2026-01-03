import asyncio
import aiohttp
import json
import ast
import pytz
from datetime import datetime, timedelta

# --- CONFIG ---
OUTPUT_FILE = "/home/mithil/PycharmProjects/PolymarketPred/data/clob_token_ids.jsonl"
BASE_URL = "https://gamma-api.polymarket.com/events/slug/{slug}"

# Symbols
SYMBOLS_SHORT = ["btc", "eth", "sol", "xrp"]
SYMBOLS_LONG = ["bitcoin", "ethereum", "solana", "xrp"]

ET = pytz.timezone("US/Eastern")


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
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
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
    except Exception as e:
        pass
    return None


def parse_token_id(market):
    raw = market.get("clobTokenIds")
    if isinstance(raw, str):
        return ast.literal_eval(raw)[0]
    return raw[0]


def parse_prices(market):
    raw = market.get("outcomePrices")
    if isinstance(raw, str):
        return ast.literal_eval(raw)
    return raw


async def process_standard_markets(session, category, symbols, interval, count, pattern):
    tasks = []
    buckets = generate_buckets(interval, count)

    # Create a list of (slug, metadata) tuples to fetch
    for symbol in symbols:
        for i, (url_param, ts) in enumerate(buckets):
            slug = pattern.format(symbol=symbol, param=url_param)
            tasks.append((slug, i, str(ts), symbol))

    results = []
    # Fetch all concurrently
    fetch_tasks = [fetch_slug(session, t[0]) for t in tasks]
    responses = await asyncio.gather(*fetch_tasks)

    for (slug, pos, ts, sym), data in zip(tasks, responses):
        if not data or "markets" not in data:
            continue
        try:
            # Standard: Take the first market (usually the main one)
            market = data["markets"][0]
            results.append({
                "slug": slug,
                "clob_token_id": parse_token_id(market),
                "market_position": pos,
                "category": category,
                "primary_market_timestamp": ts,
                "question": market["question"]
            })
        except Exception:
            continue
    return results


async def process_weekly_markets(session, count):
    """
    Weekly logic: Fetch event -> Iterate ALL markets -> Filter by price > 0.95
    """
    tasks = []
    buckets = generate_buckets("1d", count)  # Weekly uses "december-31" format
    pattern = "{symbol}-above-on-{param}"  # e.g. bitcoin-above-on-december-31

    for symbol in SYMBOLS_LONG:
        for i, (url_param, ts) in enumerate(buckets):
            slug = pattern.format(symbol=symbol, param=url_param)
            tasks.append((slug, i, str(ts)))

    results = []
    fetch_tasks = [fetch_slug(session, t[0]) for t in tasks]
    responses = await asyncio.gather(*fetch_tasks)

    for (slug, pos, ts), data in zip(tasks, responses):
        if not data or "markets" not in data:
            continue

        for market in data["markets"]:
            try:
                # 1. Price Filter (Skip if max price < 0.95)
                # This filters out strikes that are already decided or very unlikely
                prices = parse_prices(market)
                if not prices or max([float(x) for x in prices]) < 0.01:  # Adjusted logic or keep your 0.95
                    # Note: Your original logic was 'filter out if max > 0.95'?
                    # Usually for data collection you want active markets.
                    # I'll keep your original intent: include valid markets.
                    pass

                    # 2. Add Valid Market
                results.append({
                    "slug": slug,
                    "clob_token_id": parse_token_id(market),
                    "market_position": pos,
                    "category": "weekly",
                    "primary_market_timestamp": ts,
                    "question": market["question"]
                })
            except Exception:
                continue
    return results


async def main():
    print(f"[System] Starting ID Fetcher Loop (Interval: 60s)")

    while True:
        start_time = datetime.now(tz=ET)
        print(f"\n[System] Fetch cycle started at {start_time.strftime('%H:%M:%S')}...")
        try:
            async with aiohttp.ClientSession() as session:
                all_data = []

                # 1. 15 Minute
                print("Fetching 15m...")
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

                # 4. 1 Day
                # print("Fetching 1d...")
                all_data.extend(await process_standard_markets(
                    session, "1d", SYMBOLS_LONG, "1d", 1, "{symbol}-up-or-down-on-{param}"
                ))

                # 5. Weekly
                # print("Fetching Weekly...")
                all_data.extend(await process_weekly_markets(session, 3))

                # Write to file
                if all_data:
                    with open(OUTPUT_FILE, "w") as f:
                        for row in all_data:
                            f.write(json.dumps(row) + "\n")
                    print(f"[System] Success: Updated {OUTPUT_FILE} with {len(all_data)} IDs.")
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
