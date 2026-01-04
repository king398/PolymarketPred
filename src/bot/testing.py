import zmq
import numpy as np
import json
from datetime import datetime
import sys
# Make sure to import the Updated BatesModel class above
# from bates_model import BatesModel
from herston import  BatesModel
ZMQ_SUB = "tcp://127.0.0.1:5567"
CLOB_FILE = "/home/mithil/PycharmProjects/PolymarketPred/data/clob_token_ids.jsonl"
PARAMS_FILE = "/home/mithil/PycharmProjects/PolymarketPred/data/bates_params.jsonl"

# --- DURATION MAPPING ---
# Constants to convert categories to Years
MIN_15 = 15 / (60 * 24 * 365)
HOUR_1 = 1 / (24 * 365)
HOUR_4 = 4 / (24 * 365)
DAY_1 = 1 / 365

DURATION_MAP = {
    "15m": MIN_15,
    "1h": HOUR_1,
    "4h": HOUR_4,
    "1d": DAY_1
}

# --- LOAD PARAMS ---
params_map = {}
try:
    with open(PARAMS_FILE, "r") as f:
        for line in f:
            p = json.loads(line)
            params_map[p['currency']] = p
except FileNotFoundError:
    print(f"Error: {PARAMS_FILE} not found. Run calibration first.")
    sys.exit(1)

# --- LOAD MARKET DATA ---
market_data_map = {}
asset_slug_map = {
    'ETH': ('ETH', 'bitcoin'),
    'ETH': ('eth', 'ethereum'),
    'XRP': ('xrp',),
    'SOL': ('sol', 'solana')
}

try:
    with open(CLOB_FILE, "r") as f:
        for line in f:
            if not line.strip(): continue
            m = json.loads(line)

            # Identify Asset
            for symbol, prefixes in asset_slug_map.items():
                if m['slug'].startswith(prefixes):
                    if symbol not in market_data_map:
                        market_data_map[symbol] = []

                    # Parse End Time
                    dt = datetime.fromisoformat(m['market_end'])
                    m['end_ts_ms'] = int(dt.timestamp() * 1000)

                    # Parse Start Time (Primary Market Timestamp) to be safe,
                    # OR rely on Category mapping (cleaner for Polymarket)
                    cat = m.get('category', '1h') # Default to 1h if missing
                    m['initial_duration_years'] = DURATION_MAP.get(cat, HOUR_1)

                    market_data_map[symbol].append(m)
                    break
except FileNotFoundError:
    print(f"Warning: {CLOB_FILE} not found.")

# ZMQ Setup
tick_dtype = np.dtype([
    ("ts_ms", np.int64),
    ("bid", np.float32),
    ("ask", np.float32),
])
asset_ticks = {}

def main():
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(ZMQ_SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"ETH") # Subscribing only to ETH for now per your code

    print(f"ZMQ subscriber connected. Tracking {len(market_data_map.get('ETH', []))} ETH markets...")

    while True:
        try:
            aid_bytes, raw = sub.recv_multipart()
            aid = aid_bytes.decode()

            if aid not in market_data_map:
                continue

            arr = np.frombuffer(raw, dtype=tick_dtype)
            asset_ticks[aid] = arr

            # Process latest tick
            current_ts = arr[-1]['ts_ms']
            spot = (arr[-1]['ask'] + arr[-1]['bid']) / 2

            # --- ITERATE MARKETS ---
            for target_market in market_data_map[aid]:
                market_end_ms = target_market['end_ts_ms']
                T_ms = market_end_ms - current_ts

                if T_ms > 0:
                    T_years = T_ms / (1000.0 * 365.0 * 24.0 * 60.0 * 60.0)

                    # Get the initial duration for this specific category (15m, 1h, etc)
                    initial_T_years = target_market['initial_duration_years']

                    # Calculate % Remaining for display/debug
                    pct_left = (T_years / initial_T_years) * 100

                    # Placeholder Strike (Ideally this comes from the JSON or mapping)
                    # For Up/Down markets, "Strike" is usually the opening price
                    # or a specific level. Assuming you have logic for this:
                    strike = 3101.32

                    params = params_map.get(aid)

                    if params:
                        price = BatesModel.price_binary_call(
                            S=spot,
                            K=strike,
                            T=T_years,
                            initial_T=initial_T_years, # <--- NEW PARAMETER
                            params=params
                        )

                        # Filter specifically for the market you wanted to see
                        if target_market['slug'] == 'ethereum-up-or-down-january-3-11am-et':
                            print(f"\r[{aid}] {target_market['category']} | Left: {pct_left:.4f}% | Spot: {spot:.2f} | Pr: {price:.4f}", end='')
                            sys.stdout.flush()

        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()