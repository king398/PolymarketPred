import time
import zmq
import numpy as np
import zmq.asyncio
import asyncio
import json
import sys
import os
from itertools import combinations
from scipy.stats import kendalltau
from market_websocket import ASSET_ID_FILE

# --- ANSI COLORS ---
C_GREEN = "\033[92m"
C_RED = "\033[91m"
C_YELLOW = "\033[93m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_CYAN = "\033[96m"
C_WHITE = "\033[97m"

# --- Mocking fair_price_np ---
def fair_price_np(target_series, ref_series, ref_latest_price):
    # Avoid division by zero
    mean_ref = np.mean(ref_series)
    if mean_ref == 0: return {'fair_mean': ref_latest_price}

    ratio = np.mean(target_series) / mean_ref
    fair = ref_latest_price * ratio
    return {'fair_mean': fair}

tick_dtype = np.dtype([
    ("ts_ms", np.int64),
    ("bid", np.float32),
    ("ask", np.float32),
])

ctx = zmq.asyncio.Context.instance()
sub = ctx.socket(zmq.SUB)
sub.connect("tcp://127.0.0.1:5567")
sub.subscribe(b"")

# --- ASSET MAPPING LOGIC ---
valid_clobs = []
clob_question_map = {}
last_map_reload_ts = 0
MAP_RELOAD_COOLDOWN = 5

def load_asset_map():
    global valid_clobs, clob_question_map, last_map_reload_ts
    try:
        with open(ASSET_ID_FILE, "r") as f:
            for line in f.readlines():
                if not line.strip(): continue
                try:
                    obj = json.loads(line)
                    cid = obj["clob_token_id"]
                    clob_question_map[cid] = obj["question"]
                    if cid not in valid_clobs:
                        valid_clobs.append(cid)
                except json.JSONDecodeError:
                    continue
        last_map_reload_ts = time.time()
    except FileNotFoundError:
        pass

def get_asset_name(asset_id):
    if asset_id in clob_question_map:
        return clob_question_map[asset_id]

    if time.time() - last_map_reload_ts > MAP_RELOAD_COOLDOWN:
        load_asset_map()
        if asset_id in clob_question_map:
            return clob_question_map[asset_id]

    return asset_id

load_asset_map()

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
        self.realized_pnl = 0.0
        self.trade_size = 20.0
        self.stop_loss_pct = 0.2
        self.take_profit_pct = 0.15
        self.logs = []

    def log_trade(self, message):
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.logs.append(f"[{timestamp}] {message}")
        if len(self.logs) > 8:
            self.logs.pop(0)

    def buy(self, asset_id, pair_id, price, side="YES"):
        if price <= 0.10 or price >= 0.9: return
        for p in self.positions:
            if p.asset_id == asset_id and p.pair_id == pair_id: return

        if self.balance >= self.trade_size:
            qty = self.trade_size / price
            pos = Position(asset_id, pair_id, price, qty, side)
            self.positions.append(pos)
            self.balance -= self.trade_size

            name = get_asset_name(asset_id)
            side_color = C_GREEN if side == "YES" else C_RED
            self.log_trade(f"{C_GREEN}BUY{C_RESET} {side_color}{side}{C_RESET} {name} @ {price:.3f}")

    def update_positions(self, current_data):
        for pos in self.positions[:]:
            if pos.asset_id not in current_data or pos.pair_id not in current_data:
                continue

            ticks_a = current_data[pos.asset_id]
            ticks_b = current_data[pos.pair_id]

            curr_bid_yes = ticks_a['bid'][-1]
            curr_ask_yes = ticks_a['ask'][-1]

            # --- CALCULATE CURRENT VALUE ---
            curr_val = 0.0
            if pos.side == "YES":
                curr_val = curr_bid_yes
            else:
                # Value of NO = 1.0 - Cost to Buy YES immediately (Ask)
                curr_val = 1.0 - curr_ask_yes

            roi = (curr_val - pos.entry_price) / pos.entry_price

            # --- EXIT LOGIC ---
            if roi <= -self.stop_loss_pct:
                self._close_position(pos, curr_val, reason="STOP LOSS")
                continue

            if roi >= self.take_profit_pct:
                self._close_position(pos, curr_val, reason=f"TAKE PROFIT")
                continue

            # Fair Value Check
            fp_result = fair_price_np(ticks_a['ask'], ticks_b['ask'], ticks_b['ask'][-1])
            fair_val_yes = fp_result['fair_mean']

            should_exit = False

            if pos.side == "YES":
                if curr_val > pos.entry_price and curr_val > fair_val_yes:
                    should_exit = True
            else:
                if curr_val > pos.entry_price and curr_ask_yes < fair_val_yes:
                    should_exit = True

            if should_exit:
                self._close_position(pos, curr_val, reason="FAIR VAL EXIT")

    def _close_position(self, pos, sell_price, reason):
        revenue = pos.quantity * sell_price
        profit = revenue - (pos.quantity * pos.entry_price)

        self.balance += revenue
        self.realized_pnl += profit  # <--- NEW: Update Realized P&L
        self.positions.remove(pos)

        name = get_asset_name(pos.asset_id)
        color = C_GREEN if profit > 0 else C_RED
        side_color = C_GREEN if pos.side == "YES" else C_RED
        self.log_trade(f"{C_RED}SELL{C_RESET} {side_color}{pos.side}{C_RESET} {name} @ {sell_price:.3f} ({color}${profit:+.2f}{C_RESET}) [{reason}]")

trader = SimulatedTrader()

def draw_dashboard(current_data):
    # Calculate Equity and Prepare Data for Sorting
    equity = 0.0
    active_rows = []

    for p in trader.positions:
        curr_bid = 0.0
        curr_ask = 0.0

        if p.asset_id in current_data:
            ticks = current_data[p.asset_id]

            if p.side == "YES":
                curr_bid = ticks['bid'][-1]
                curr_ask = ticks['ask'][-1]
            else:
                # If we hold NO:
                # We sell NO at (1 - YES_Ask) -> This is the Bid for NO
                # We buy NO at (1 - YES_Bid) -> This is the Ask for NO
                curr_bid = 1.0 - ticks['ask'][-1]
                curr_ask = 1.0 - ticks['bid'][-1]
        else:
            curr_bid = p.entry_price
            curr_ask = p.entry_price

        # Update Total Equity (Liquidation Value)
        equity += p.quantity * curr_bid

        unrealized_pnl = (curr_bid - p.entry_price) * p.quantity

        active_rows.append({
            'obj': p,
            'curr_bid': curr_bid,
            'curr_ask': curr_ask,
            'pnl': unrealized_pnl
        })

    # Sort: Highest PnL at the top
    active_rows.sort(key=lambda x: x['pnl'], reverse=True)

    total_val = trader.balance + equity
    total_pnl = total_val - trader.starting_balance

    # Colors
    pnl_color = C_GREEN if total_pnl >= 0 else C_RED
    realized_color = C_GREEN if trader.realized_pnl >= 0 else C_RED

    # --- RENDER ---
    # Move cursor to top-left and clear screen
    print("\033[H\033[J", end="")

    print(f"{C_BOLD}{C_CYAN}=== 🤖 ALGOBET TRADING DASHBOARD (SORTED BY PnL) ==={C_RESET}")
    print("-" * 120)
    print(f"💰 Cash:      ${trader.balance:.2f}")
    # --- NEW: Display Realized P&L ---
    print(f"✅ Realized:  {realized_color}${trader.realized_pnl:+.2f}{C_RESET}")
    print(f"📈 Equity:    ${equity:.2f}")
    print(f"💎 Net Worth: ${total_val:.2f}  ({pnl_color}Total PnL: {total_pnl:+.2f}{C_RESET})")
    print(f"📊 Positions: {len(trader.positions)}")
    print("-" * 120)

    # Added BID / ASK columns
    print(f"{'SIDE':<5} {'ASSET (Full Name)':<55} {'ENTRY':<7} {'BID':<7} {'ASK':<7} {'QTY':<6} {'PnL ($)':<10}")
    print("-" * 120)

    if not active_rows:
        print(f"{C_YELLOW}   No active positions. Scanning market...{C_RESET}")
    else:
        for row in active_rows:
            p = row['obj']
            name = get_asset_name(p.asset_id)

            # Formatting
            u_color = C_GREEN if row['pnl'] >= 0 else C_RED
            side_color = C_GREEN if p.side == "YES" else C_RED

            clean_name = name[:53] + ".." if len(name) > 55 else name

            print(f"{side_color}{p.side:<5}{C_RESET} "
                  f"{clean_name:<55} "
                  f"{p.entry_price:<7.3f} "
                  f"{C_WHITE}{row['curr_bid']:<7.3f}{C_RESET} "  # Bid in White
                  f"{C_WHITE}{row['curr_ask']:<7.3f}{C_RESET} "  # Ask in White
                  f"{int(p.quantity):<6} "
                  f"{u_color}{row['pnl']:+.2f}{C_RESET}")

    print("\n" + "=" * 120)
    print(f"{C_BOLD}📝 RECENT ACTIVITY LOG{C_RESET}")
    print("-" * 120)
    for log in trader.logs:
        print(log)
    print("-" * 120)

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
    # 1. Update positions (check exits)
    trader.update_positions(assets_snapshot)

    # 2. Draw Dashboard (with latest updates)
    draw_dashboard(assets_snapshot)

    # 3. Look for new entries
    mid_prices = {aid: arr['ask'] for aid, arr in assets_snapshot.items()}
    results = {}

    for (aid1, s1), (aid2, s2) in combinations(mid_prices.items(), 2):
        n = min(len(s1), len(s2))
        if n < 20: continue
        if np.abs(s1[-1] - s2[-1]) > 0.9: continue

        tau, pval = kendalltau(s1[:n], s2[:n])
        if abs(tau) > 0.6:
            results[(aid1, aid2)] = {"tau": tau}

    top_opps = sorted(results.items(), key=lambda x: abs(x[1]["tau"]), reverse=True)
    THRESHOLD = 0.02

    for (a, b), r in top_opps:
        # Check existence in snapshot again to be safe
        if a not in assets_snapshot or b not in assets_snapshot: continue

        yes_ask_a = assets_snapshot[a]['ask'][-1]
        yes_bid_a = assets_snapshot[a]['bid'][-1]
        yes_ask_b = assets_snapshot[b]['ask'][-1]
        yes_bid_b = assets_snapshot[b]['bid'][-1]
        spread_a = yes_ask_a - yes_bid_a
        spread_b = yes_ask_b - yes_bid_b

        # If spread is greater than $0.03 (3 cents), skip this pair entirely
        # You can tighten this to 0.02 for safer trades.
        if spread_a > 0.03 or spread_b > 0.03:
            continue
        fair_yes_a = fair_price_np(mid_prices[a], mid_prices[b], yes_ask_b)['fair_mean']
        fair_yes_b = fair_price_np(mid_prices[b], mid_prices[a], yes_ask_a)['fair_mean']

        # Scoring Logic
        # A YES score
        score_a_yes = fair_yes_a - yes_ask_a
        # A NO score: Value(NO) - Cost(NO). Value(NO)=1-FairYES. Cost(NO)=1-BidYES.
        # Score = (1 - FairYES) - (1 - BidYES) = BidYES - FairYES
        score_a_no = yes_bid_a - fair_yes_a

        # B YES score
        score_b_yes = fair_yes_b - yes_ask_b
        # B NO score
        score_b_no = yes_bid_b - fair_yes_b

        candidates = [
            {'score': score_a_yes, 'asset': a, 'pair': b, 'price': yes_ask_a,       'side': 'YES'},
            {'score': score_a_no,  'asset': a, 'pair': b, 'price': 1.0 - yes_bid_a, 'side': 'NO'},
            {'score': score_b_yes, 'asset': b, 'pair': a, 'price': yes_ask_b,       'side': 'YES'},
            {'score': score_b_no,  'asset': b, 'pair': a, 'price': 1.0 - yes_bid_b, 'side': 'NO'}
        ]

        best_opp = max(candidates, key=lambda x: x['score'])

        if best_opp['score'] > THRESHOLD:
            trader.buy(
                best_opp['asset'],
                best_opp['pair'],
                best_opp['price'],
                side=best_opp['side']
            )

async def periodic_logic():
    print("Initializing...")
    while True:
        await asyncio.sleep(0.5)
        if len(assets_ticks) < 2: continue
        snapshot = {k: v.copy() for k, v in assets_ticks.items()}
        await asyncio.to_thread(run_strategy, snapshot)

async def main():
    await asyncio.gather(collect_ticks(), periodic_logic())

if __name__ == "__main__":
    try:
        # Initial Clear
        os.system('cls' if os.name == 'nt' else 'clear')
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{C_RED}Shutting down...{C_RESET}")