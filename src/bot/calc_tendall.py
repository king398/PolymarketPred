import time
import zmq
import numpy as np
import zmq.asyncio
import asyncio
import json
import sys
from itertools import combinations
from scipy.stats import kendalltau
from market_websocket import ASSET_ID_FILE

# --- Mocking fair_price_np (Replace with your actual import) ---
def fair_price_np(target_series, ref_series, ref_latest_price):
    ratio = np.mean(target_series) / np.mean(ref_series)
    fair = ref_latest_price * ratio
    return {'fair_mean': fair}
# ---------------------------------------------------------------

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
    pass

assets_ticks = {}

class Position:
    def __init__(self, asset_id, pair_id, entry_price, quantity, side="YES"):
        self.asset_id = asset_id
        self.pair_id = pair_id
        self.entry_price = entry_price
        self.quantity = quantity
        self.side = side
        self.timestamp = time.time()

class SimulatedTrader:
    def __init__(self):
        self.positions = []
        self.starting_balance = 1000.0
        self.balance = self.starting_balance
        self.trade_size = 10.0
        self.stop_loss_amt = 0.20
        self.take_profit_pct = 0.10 # 10% Profit Target

    def log_trade(self, message):
        # Clear the status line first, print message, then force new line
        sys.stdout.write(f"\r{message}\n")
        sys.stdout.flush()

    def buy(self, asset_id, pair_id, ask_price, side="YES"):
        # 1. Price Range Filter (0.05 to 0.95)
        if ask_price <= 0.05 or ask_price >= 0.95:
            return

        # Prevent duplicate trades
        for p in self.positions:
            if p.asset_id == asset_id and p.pair_id == pair_id:
                return

        if self.balance >= self.trade_size:
            qty = self.trade_size / ask_price
            pos = Position(asset_id, pair_id, ask_price, qty, side)
            self.positions.append(pos)
            self.balance -= self.trade_size
            name = clob_question_map.get(asset_id, asset_id)
            self.log_trade(f"🟢 BUY: {name} @ {ask_price:.3f}")

    def update_positions(self, current_data):
        for pos in self.positions[:]:
            if pos.asset_id not in current_data or pos.pair_id not in current_data:
                continue

            ticks_a = current_data[pos.asset_id]
            ticks_b = current_data[pos.pair_id]

            curr_bid = ticks_a['bid'][-1]
            curr_ask_b = ticks_b['ask'][-1]

            fp_result = fair_price_np(ticks_a['ask'], ticks_b['ask'], curr_ask_b)
            fair_val = fp_result['fair_mean']

            # --- EXIT LOGIC ---

            # 1. STOP LOSS (Immediate Exit: Drop > 0.20)
            if curr_bid <= (pos.entry_price - self.stop_loss_amt):
                self._close_position(pos, curr_bid, reason="STOP LOSS 🛑")
                continue

            # 2. TAKE PROFIT (Immediate Exit: Gain >= 10%)
            roi = (curr_bid - pos.entry_price) / pos.entry_price
            if roi >= self.take_profit_pct:
                self._close_position(pos, curr_bid, reason=f"TAKE PROFIT (+{roi*100:.1f}%) 🚀")
                continue

            # 3. STRATEGIC EXIT (Profit > 0 AND Reverted to Fair Value)
            is_profitable = curr_bid > pos.entry_price
            is_over_fair = curr_bid > fair_val

            if is_profitable and is_over_fair:
                self._close_position(pos, curr_bid, reason="FAIR VAL EXIT 💰")

    def _close_position(self, pos, sell_price, reason):
        revenue = pos.quantity * sell_price
        profit = revenue - (pos.quantity * pos.entry_price)
        self.balance += revenue
        self.positions.remove(pos)
        name = clob_question_map.get(pos.asset_id, pos.asset_id)
        self.log_trade(f"🔴 SELL: {name} @ {sell_price:.3f} | {reason} | PnL: ${profit:.2f}")

    def print_status(self, current_data):
        equity = 0.0
        for p in self.positions:
            if p.asset_id in current_data:
                equity += p.quantity * current_data[p.asset_id]['bid'][-1]
            else:
                equity += p.quantity * p.entry_price

        total_value = self.balance + equity
        total_pnl = total_value - self.starting_balance

        # Single line status bar using carriage return \r
        status_str = (
            f"\r[ PnL: ${total_pnl:+.2f} ] "
            f"[ Cash: ${self.balance:.0f} ] "
            f"[ Eq: ${equity:.0f} ] "
            f"[ Pos: {len(self.positions)} ] "
            f"[ Ticks: {len(assets_ticks)} ]"
        )
        sys.stdout.write(status_str)
        sys.stdout.flush()

trader = SimulatedTrader()

async def collect_ticks():
    global assets_ticks
    while True:
        aid_bytes, payload = await sub.recv_multipart()
        aid = aid_bytes.decode()
        arr = np.frombuffer(payload, dtype=tick_dtype)
        if (arr["ask"].std() < 1e-3) or (len(arr) < 100):
            continue
        assets_ticks[aid] = arr

def run_strategy(assets_snapshot: dict[str, np.ndarray]):
    # 1. Manage Positions (Exits)
    trader.update_positions(assets_snapshot)

    # 2. Print Status Bar (Flushed)
    trader.print_status(assets_snapshot)

    # 3. Scan for New Entries
    mid_prices = {aid: arr['ask'] for aid, arr in assets_snapshot.items()}
    results = {}

    for (aid1, s1), (aid2, s2) in combinations(mid_prices.items(), 2):
        n = min(len(s1), len(s2))
        if n < 20: continue

        # Quick pre-filter to save CPU
        if np.abs(s1[-1] - s2[-1]) > 0.9: continue

        tau, pval = kendalltau(s1[:n], s2[:n])
        if abs(tau) > 0.6:
            results[(aid1, aid2)] = {"tau": tau}

    top_opps = sorted(results.items(), key=lambda x: abs(x[1]["tau"]), reverse=True)[:15]

    for (a, b), r in top_opps:
        price_a = mid_prices[a][-1]
        price_b = mid_prices[b][-1]

        fair_a = fair_price_np(mid_prices[a], mid_prices[b], price_b)['fair_mean']
        fair_b = fair_price_np(mid_prices[b], mid_prices[a], price_a)['fair_mean']

        mispricing_a = fair_a - price_a
        mispricing_b = fair_b - price_b

        THRESHOLD = 0.05

        if mispricing_a > THRESHOLD:
            actual_ask_a = assets_snapshot[a]['ask'][-1]
            if fair_a > actual_ask_a:
                trader.buy(a, b, actual_ask_a, side="YES")

        if mispricing_b > THRESHOLD:
            actual_ask_b = assets_snapshot[b]['ask'][-1]
            if fair_b > actual_ask_b:
                trader.buy(b, a, actual_ask_b, side="YES")

async def periodic_logic():
    print("Strategy Engine Started... Waiting for data.")
    while True:
        await asyncio.sleep(1)
        if len(assets_ticks) < 2: continue
        snapshot = {k: v.copy() for k, v in assets_ticks.items()}
        await asyncio.to_thread(run_strategy, snapshot)

async def main():
    await asyncio.gather(collect_ticks(), periodic_logic())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...")