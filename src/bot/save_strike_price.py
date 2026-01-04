import asyncio
import json
import os
import aiohttp
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Set, List
from collections import deque

# --- RICH IMPORTS (Optional, falls back to print if missing) ---
try:
    from rich.console import Console
    from rich.logging import RichHandler
    import logging

    console = Console()
    logging.basicConfig(
        level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(console=console)]
    )
    log = logging.getLogger("rich")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    log = logging.getLogger("simple")
    console = None

# ===================== CONFIG =====================
DATA_DIR = "/home/mithil/PycharmProjects/PolymarketPred/data"
ASSET_ID_FILE = os.path.join(DATA_DIR, "clob_token_ids.jsonl")
OUTPUT_FILE = os.path.join(DATA_DIR, "market_1m_candle_opens.jsonl")

BINANCE_API = "https://api.binance.com/api/v3/klines"
MAX_RETRIES = 5  # Try fetching 5 times if data is missing
RETRY_DELAY = 1.5 # Seconds between retries

# Normalized mapping
SYMBOL_MAP = {
    "btc": "BTCUSDT", "bitcoin": "BTCUSDT",
    "eth": "ETHUSDT", "ethereum": "ETHUSDT", "ether": "ETHUSDT",
    "xrp": "XRPUSDT", "ripple": "XRPUSDT",
    "sol": "SOLUSDT", "solana": "SOLUSDT",
    "doge": "DOGEUSDT", "dogecoin": "DOGEUSDT",
}

# ===================== UTILS =====================
def to_utc_ms(dt_str: str) -> int:
    dt = datetime.fromisoformat(dt_str)
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)

def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)

def infer_symbol(mkt: Dict[str, Any]) -> Optional[str]:
    """Robustly guesses the Binance ticker from Polymarket metadata."""
    # 1. Try explicit symbol field if exists
    if "symbol" in mkt: return mkt["symbol"]

    # 2. Try parsing the slug (e.g. "eth-updown-4h...")
    slug = (mkt.get("slug") or "").lower()
    parts = slug.split("-")
    if parts and parts[0] in SYMBOL_MAP:
        return SYMBOL_MAP[parts[0]]

    # 3. Search question text
    q = (mkt.get("question") or "").lower()
    for key, val in SYMBOL_MAP.items():
        if key in q or key in slug:
            return val
    return None

# ===================== ASYNC WORKERS =====================

class StrikeFetcher:
    def __init__(self):
        self.processed_ids = self._load_processed()
        self.queue = asyncio.PriorityQueue()
        self.known_ids = set() # To prevent re-queueing same ID in memory

    def _load_processed(self) -> Set[str]:
        """Load IDs already saved to disk."""
        processed = set()
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "clob_token_id" in data:
                            processed.add(data["clob_token_id"])
                    except: pass
        log.info(f"Loaded {len(processed)} existing strikes.")
        return processed

    async def fetch_candle(self, session: aiohttp.ClientSession, symbol: str, open_ms: int) -> Optional[Dict]:
        """Async fetch from Binance with retries."""
        params = {
            "symbol": symbol,
            "interval": "1m",
            "startTime": open_ms,
            "limit": 1
        }

        for i in range(MAX_RETRIES):
            try:
                async with session.get(BINANCE_API, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data:
                            k = data[0]
                            k_open_time = int(k[0])
                            # Strict timestamp check (tolerance 1000ms)
                            if abs(k_open_time - open_ms) < 1000:
                                return {
                                    "open": float(k[1]),
                                    "close": float(k[4]),
                                    "high": float(k[2]),
                                    "low": float(k[3]),
                                    "openTime": k_open_time
                                }
            except Exception as e:
                log.warning(f"Error fetching {symbol}: {e}")

            # Wait before retry (data might not be finalized)
            await asyncio.sleep(RETRY_DELAY * (i + 1))

        return None

    async def process_market(self, mkt: Dict):
        """Worker function for a single market."""
        cid = mkt['clob_token_id']
        slug = mkt.get('slug', 'unknown')
        start_ms = mkt['_start_ms']

        # 1. Wait until start time
        wait_ms = start_ms - now_ms()
        if wait_ms > 0:
            wait_sec = wait_ms / 1000.0
            log.info(f"⏳ Waiting {wait_sec:.1f}s for market start: [bold cyan]{slug}[/]", extra={"markup": True})
            await asyncio.sleep(wait_sec + 2.0) # +2s buffer for Binance

        # 2. Fetch Data
        symbol = infer_symbol(mkt)
        if not symbol:
            log.error(f"❌ Could not map symbol for: {slug}")
            return

        log.info(f"🔍 Fetching strike for [cyan]{slug}[/] ({symbol})...", extra={"markup": True})

        async with aiohttp.ClientSession() as session:
            candle = await self.fetch_candle(session, symbol, start_ms)

        # 3. Save Result
        if candle:
            output = {
                "slug": slug,
                "clob_token_id": cid,
                "symbol": symbol,
                "market_start_iso": mkt.get("primary_market_timestamp"),
                "market_start_utc_ms": start_ms,
                "candle_open_utc_ms": candle["openTime"],
                "open_price": candle["open"],
                "mean_price": (candle["open"] + candle["close"]) / 2.0
            }

            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(output) + "\n")

            self.processed_ids.add(cid)
            log.info(f"✅ [bold green]SAVED[/] {slug} | Strike: {candle['open']}", extra={"markup": True})
        else:
            log.warning(f"⚠️ Data unavailable for {slug} (Too old or API issue).")

    async def file_watcher(self):
        """Continuously tails the input file for new lines."""
        log.info(f"👀 Watching {ASSET_ID_FILE} for new markets...")

        # Initial read
        try:
            with open(ASSET_ID_FILE, 'r') as f:
                # Move to end if we only wanted new, but here we want to process all
                # that aren't done yet. So we read from start.
                pass
        except FileNotFoundError:
            log.error("Input file not found.")
            return

        # Simple polling file reader
        last_pos = 0
        while True:
            if os.path.exists(ASSET_ID_FILE):
                with open(ASSET_ID_FILE, 'r') as f:
                    f.seek(last_pos)
                    lines = f.readlines()
                    last_pos = f.tell()

                    for line in lines:
                        if not line.strip(): continue
                        try:
                            m = json.loads(line)
                            cid = m.get('clob_token_id')

                            # Filter: Must be valid, not processed, not already queued
                            if cid and cid not in self.processed_ids and cid not in self.known_ids:
                                ts = m.get("primary_market_timestamp")
                                if ts:
                                    m['_start_ms'] = to_utc_ms(ts)
                                    self.known_ids.add(cid)
                                    # Create background task for this market
                                    asyncio.create_task(self.process_market(m))
                        except Exception as e:
                            pass

            await asyncio.sleep(2) # Check file every 2 seconds

    async def run(self):
        await self.file_watcher()

# ===================== ENTRY POINT =====================
if __name__ == "__main__":
    fetcher = StrikeFetcher()
    try:
        asyncio.run(fetcher.run())
    except KeyboardInterrupt:
        print("Stopped.")