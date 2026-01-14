import asyncio
import websockets
import json

# CONFIGURATION
# NOTE: This date must exist in Deribit. Update this if testing on a different day.
EXPIRY_DATE = "23JAN26"
CURRENCY = "ETH"


async def get_atm_instrument():
    """
    Connects to Deribit (Public), finds the current price,
    and returns the instrument name of the closest Call option.
    """
    # We use a temporary connection just to find the right instrument
    async with websockets.connect("wss://www.deribit.com/ws/api/v2") as ws:

        # --- Step A: Get Current Index Price ---
        await ws.send(json.dumps({
            "jsonrpc": "2.0", "method": "public/get_index_price",
            "params": {"index_name": "btc_usd"}, "id": 1
        }))

        current_price = 0

        # Loop until we get the specific response for ID 1
        while True:
            response = json.loads(await ws.recv())
            if response.get("id") == 1:
                current_price = response['result']['index_price']
                print(f"Current BTC Price: ${current_price}")
                break

        # --- Step B: Get All Instruments ---
        await ws.send(json.dumps({
            "jsonrpc": "2.0", "method": "public/get_instruments",
            "params": {"currency": CURRENCY, "kind": "option", "expired": False}, "id": 2
        }))

        instruments = []

        # Loop until we get the specific response for ID 2
        while True:
            response = json.loads(await ws.recv())
            if response.get("id") == 2:
                instruments = response['result']
                break
        print(instruments)
        # --- Step C: Find the Closest Strike ---
        closest_instrument = None
        min_diff = float('inf')

        for instr in instruments:
            # Name format: BTC-13JAN26-90000-C
            name = instr['instrument_name']
            print(name)
            parts = name.split('-')

            # Logic: Check if it matches our Date and is a Call (C)
            if len(parts) >= 4 and parts[1] == EXPIRY_DATE and parts[3] == 'C':
                strike = float(parts[2])
                diff = abs(current_price - strike)

                if diff < min_diff:
                    min_diff = diff
                    closest_instrument = name

        return closest_instrument


async def stream_iv():
    """
    Subscribes to the live ticker of the found instrument
    """
    print("Finding ATM Option...")
    target = await get_atm_instrument()

    if not target:
        print(f"Error: No options found for date {EXPIRY_DATE}. Check the date configuration.")
        return

    print(f"Target Acquired: {target}")
    print("-" * 40)

    # Open a new connection for the persistent stream
    async with websockets.connect("wss://www.deribit.com/ws/api/v2") as ws:

        # Subscribe to the specific ticker (Public channel)
        msg = {
            "jsonrpc": "2.0",
            "method": "public/subscribe",
            "params": {"channels": [f"ticker.{target}.100ms",f"ticker.{target}.100ms"]},
            "id": 100
        }
        await ws.send(json.dumps(msg))

        while True:
            response = json.loads(await ws.recv())

            # Look for notification events
            if 'params' in response and 'data' in response['params']:
                data = response['params']['data']
                print(data)
                # Extract the IV data
                mark_iv = data.get('mark_iv', 0)
                bid_iv = data.get('bid_iv', 0)
                ask_iv = data.get('ask_iv', 0)
                underlying = data.get('index_price', 0)

                print(f"BTC: ${underlying} | Mark IV: {mark_iv:.2f}% | Spread: {bid_iv:.2f}% - {ask_iv:.2f}%")




if __name__ == "__main__":
    try:
        asyncio.run(stream_iv())
    except KeyboardInterrupt:
        print("\nStream stopped by user.")
