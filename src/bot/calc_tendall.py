import time
import zmq
import numpy as np
import zmq.asyncio
import asyncio
import json
from itertools import combinations
from scipy.stats import kendalltau
from market_websocket import ASSET_ID_FILE
# from pricing import fair_price_np # Assumed to be available locally

# --- Mocking fair_price_np for standalone functionality (Replace with your import) ---
def fair_price_np(target_series, ref_series, ref_latest_price):
    """
    Simple mock of a cointegration/correlation model.
    Replace this with your actual import from pricing.
    """
    # Dummy logic: Assumes simple linear relation for demonstration
    ratio = np.mean(target_series) / np.mean(ref_series)
    fair = ref_latest_price * ratio
    return {'fair_mean': fair}
# -----------------------------------------------------------------------------------

tick_dtype = np.dtype([
    ("ts_ms", np.int64),
    ("bid", np.float32),
    ("ask", np.float32),
])

ctx = zmq.asyncio.Context.instance()
sub = ctx.socket(zmq.SUB)
sub.connect("tcp://127.0.0.1:5567")
sub.subscribe(b"")

valid_clobs = []
clob_question_map = {}

# Load Asset Map
try:
    with open(ASSET_ID_FILE, "r") as f:
        for line in f.readlines():
            obj = json.loads(line)
            valid_clobs.append(obj["clob_token_id"])
            clob_question_map[obj["clob_token_id"]] = obj["question"]
except FileNotFoundError:
    print(f"Warning: {ASSET_ID_FILE} not found. Running without name mapping.")

assets_ticks = {}

class Position:
    def __init__(self, asset_id, pair_id, entry_price, quantity, side="YES"):
        self.asset_id = asset_id
        self.pair_id = pair_id  # The asset correlated with this one
        self.entry_price = entry_price
        self.quantity = quantity
        self.side = side # "YES" or "NO"
        self.timestamp = time.time()

    def __repr__(self):
        name = clob_question_map.get(self.asset_id, self.asset_id)
        return f"<{self.side} {name} @ {self.entry_price:.4f}>"

class SimulatedTrader:
    def __init__(self):
        self.positions = []
        self.balance = 1000.0  # Starting cash
        self.trade_size = 10.0 # Dollars per trade

    def buy(self, asset_id, pair_id, ask_price, side="YES"):
        # Prevent duplicate trades on the same pair to avoid spamming
        for p in self.positions:
            if p.asset_id == asset_id and p.pair_id == pair_id:
                return

        if self.balance >= self.trade_size:
            qty = self.trade_size / ask_price
            pos = Position(asset_id, pair_id, ask_price, qty, side)
            self.positions.append(pos)
            self.balance -= self.trade_size
            print(f"OPEN TRADE: Bought {side} [{clob_question_map.get(asset_id, asset_id)}] @ {ask_price:.4f} (Pair: {clob_question_map.get(pair_id, pair_id)})")

    def update_positions(self, current_data):
        """
        Check exit conditions:
        1. We are making a profit (Bid > Entry)
        2. Price is above Fair Price (Reversion achieved)
        """
        for pos in self.positions[:]: # Copy list to modify safeley
            # Ensure we have data for both assets in the pair
            if pos.asset_id not in current_data or pos.pair_id not in current_data:
                continue

            # Get current market data
            ticks_a = current_data[pos.asset_id]
            ticks_b = current_data[pos.pair_id]

            curr_bid = ticks_a['bid'][-1]
            curr_ask_b = ticks_b['ask'][-1] # Assuming we track B's ask for conservative correlation

            # Re-calculate fair price
            # Note: Using 'ask' series for correlation history as per original script logic
            fp_result = fair_price_np(
                ticks_a['ask'],
                ticks_b['ask'],
                curr_ask_b
            )
            fair_val = fp_result['fair_mean']

            # EXIT LOGIC
            # 1. Profit check
            is_profitable = curr_bid > pos.entry_price

            # 2. Fair Value check (Are we "over" the fair price?)
            # If we bought low, we sell when it goes HIGH (above fair)
            is_over_fair = curr_bid > fair_val

            if is_profitable and is_over_fair:
                revenue = pos.quantity * curr_bid
                profit = revenue - (pos.quantity * pos.entry_price)
                self.balance += revenue
                print(f"CLOSE TRADE: Sold {pos.side} [{clob_question_map.get(pos.asset_id, pos.asset_id)}] @ {curr_bid:.4f}. Profit: ${profit:.2f}")
                self.positions.remove(pos)

trader = SimulatedTrader()

async def collect_ticks():
    global assets_ticks
    while True:
        aid_bytes, payload = await sub.recv_multipart()
        aid = aid_bytes.decode()

        arr = np.frombuffer(payload, dtype=tick_dtype)

        # Basic filtering
        if (arr["ask"].std() < 1e-3) or (len(arr) < 100):
            continue

        assets_ticks[aid] = arr

def run_strategy(assets_snapshot: dict[str, np.ndarray]):
    # 1. Update existing positions (Check for Exits)
    trader.update_positions(assets_snapshot)

    # 2. Scan for New Entries
    mid_prices = {
        aid: arr['ask'] # Using Ask as 'mid' proxy per original logic
        for aid, arr in assets_snapshot.items()
    }

    results = {}
    for (aid1, s1), (aid2, s2) in combinations(mid_prices.items(), 2):
        n = min(len(s1), len(s2))
        if n < 20: # Minimum data points
            continue

        tau, pval = kendalltau(s1[:n], s2[:n])

        # Only look at strong correlations
        if abs(tau) > 0.5:
            results[(aid1, aid2)] = {
                "tau": tau,
                "n": n,
            }

    # Sort by strength of correlation
    top_opps = sorted(results.items(), key=lambda x: abs(x[1]["tau"]), reverse=True)[:20]

    for (a, b), r in top_opps:
        price_a = mid_prices[a][-1]
        price_b = mid_prices[b][-1]

        fair_res_a = fair_price_np(mid_prices[a], mid_prices[b], price_b)
        fair_res_b = fair_price_np(mid_prices[b], mid_prices[a], price_a)

        fair_a = fair_res_a['fair_mean']
        fair_b = fair_res_b['fair_mean']

        mispricing_a = fair_a - price_a # Positive = Undervalued (Buy)
        mispricing_b = fair_b - price_b

        # Threshold
        THRESHOLD = 0.05

        # Check Asset A
        if abs(mispricing_a) > THRESHOLD:
            # If Mispricing > 0, Fair > Actual. We want to BUY.
            if mispricing_a > 0:
                actual_ask_a = assets_snapshot[a]['ask'][-1]
                # Check if we still have room (Fair Price > Current Ask)
                if fair_a > actual_ask_a:
                    trader.buy(a, b, actual_ask_a, side="YES")

            # If Mispricing < 0, Actual > Fair. We want to SELL (or Buy NO).
            # Assuming we can buy "NO" or just ignore if we don't hold it.
            # Implementation for "NO" would go here if token ID known.

        # Check Asset B
        if abs(mispricing_b) > THRESHOLD:
            if mispricing_b > 0:
                actual_ask_b = assets_snapshot[b]['ask'][-1]
                if fair_b > actual_ask_b:
                    trader.buy(b, a, actual_ask_b, side="YES")

async def periodic_logic():
    while True:
        await asyncio.sleep(1)
        if len(assets_ticks) < 2:
            continue

        snapshot = {k: v.copy() for k, v in assets_ticks.items()}
        # Run CPU-bound math in thread to avoid blocking asyncio loop
        await asyncio.to_thread(run_strategy, snapshot)

async def main():
    print("Starting strategy engine...")
    await asyncio.gather(collect_ticks(), periodic_logic())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down...")