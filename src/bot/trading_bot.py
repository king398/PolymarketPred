import asyncio
import websockets
import json
import time
import aiofiles

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
ASSET_ID = "37319737055026754221246259333626958086127811705981382118444192106091220298406"
OUTPUT_FILE = "market_data.jsonl"

async def stream():
    print(f"Starting instant stream. Saving to {OUTPUT_FILE}...")

    async with websockets.connect(
            WS_URL,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
            max_size=None,
    ) as ws:
        # Subscribe
        await ws.send(json.dumps({
            "type": "market",
            "assets_ids": [ASSET_ID],
        }))

        total_messages = 0
        interval_messages = 0
        start_time = time.perf_counter()
        last_log_time = start_time

        print("Listening...")

        # Open the file ONCE and keep it open
        async with aiofiles.open(OUTPUT_FILE, mode='a') as f:
            async for msg in ws:
                # 1. Write immediately (add newline manually)
                await f.write(msg + '\n')

                # 2. Force flush to disk immediately (Optional but ensures safety)
                # Without this, the OS might still buffer the write internally.
                await f.flush()

                # 3. Logging / Benchmarking
                total_messages += 1
                interval_messages += 1

                current_time = time.perf_counter()
                if current_time - last_log_time > 1.0:
                    elapsed = current_time - last_log_time
                    rate = interval_messages / elapsed
                    print(f"\rRate: {rate:.0f} msg/s | Total: {total_messages}", end="")
                    interval_messages = 0
                    last_log_time = current_time

if __name__ == "__main__":
    try:
        try:
            import uvloop
            uvloop.install()
        except ImportError:
            pass
        asyncio.run(stream())
    except KeyboardInterrupt:
        print("\nStopped.")