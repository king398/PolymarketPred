import time
import zmq
import numpy as np
import zmq.asyncio
import asyncio
import json
import os
import sys
from collections import deque
from itertools import combinations
from scipy.stats import kendalltau

# --- RICH IMPORTS ---
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.style import Style
from rich import box

# --- CONFIGURATION ---
# Assuming market_websocket is a local file, otherwise we mock the constant
try:
    from market_websocket import ASSET_ID_FILE
except ImportError:
    ASSET_ID_FILE = "asset_ids.json"

ZMQ_ADDR = "tcp://127.0.0.1:5567"
MAP_RELOAD_COOLDOWN = 5

# --- DATA STRUCTURES ---
tick_dtype = np.dtype([
    ("ts_ms", np.int64),
    ("bid", np.float32),
    ("ask", np.float32),
])

# --- STATE MANAGEMENT ---
class AppState:
    """Shared state between Async Logic and UI Renderer"""
    def __init__(self):
        self.clob_question_map = {}
        self.valid_clobs = []
        self.assets_ticks = {}
        self.last_map_reload_ts = 0
        self.logs = deque(maxlen=10)  # Efficient fixed-size queue
        self.scan_count = 0
        self.active_pairs_count = 0

state = AppState()

# --- UTILS ---
def load_asset_map():
    try:
        if not os.path.exists(ASSET_ID_FILE): return
        with open(ASSET_ID_FILE, "r") as f:
            for line in f.readlines():
                if not line.strip(): continue
                try:
                    obj = json.loads(line)
                    cid = obj["clob_token_id"]
                    state.clob_question_map[cid] = obj["question"]
                    if cid not in state.valid_clobs:
                        state.valid_clobs.append(cid)
                except json.JSONDecodeError:
                    continue
        state.last_map_reload_ts = time.time()
    except Exception:
        pass

def get_asset_name(asset_id):
    if asset_id in state.clob_question_map:
        return state.clob_question_map[asset_id]

    if time.time() - state.last_map_reload_ts > MAP_RELOAD_COOLDOWN:
        load_asset_map()
        if asset_id in state.clob_question_map:
            return state.clob_question_map[asset_id]

    return asset_id

def fair_price_np(target_series, ref_series, ref_latest_price):
    mean_ref = np.mean(ref_series)
    if mean_ref == 0: return {'fair_mean': ref_latest_price}
    ratio = np.mean(target_series) / mean_ref
    return {'fair_mean': ref_latest_price * ratio}

# --- TRADING ENGINE ---
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

    def log(self, message, style="white"):
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        # Store as tuple (timestamp, message, style) for Rich rendering
        state.logs.append((timestamp, message, style))

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
            color = "green" if side == "YES" else "red"
            self.log(f"BUY {side} {name} @ {price:.3f}", style=f"bold {color}")

    def update_positions(self, current_data):
        for pos in self.positions[:]:
            if pos.asset_id not in current_data or pos.pair_id not in current_data:
                continue

            ticks_a = current_data[pos.asset_id]
            ticks_b = current_data[pos.pair_id]

            # Helper to get safe last element
            if len(ticks_a['bid']) == 0: continue

            curr_bid_yes = ticks_a['bid'][-1]
            curr_ask_yes = ticks_a['ask'][-1]

            curr_val = curr_bid_yes if pos.side == "YES" else (1.0 - curr_ask_yes)
            roi = (curr_val - pos.entry_price) / pos.entry_price

            if roi >= self.take_profit_pct:
                self._close_position(pos, curr_val, reason="TAKE PROFIT")
                continue

            # Fair Value Logic
            fp_result = fair_price_np(ticks_a['ask'], ticks_b['ask'], ticks_b['ask'][-1])
            fair_val_yes = fp_result['fair_mean']

            should_exit = False
            if pos.side == "YES" and curr_val > fair_val_yes:
                should_exit = True
            elif pos.side == "NO" and curr_ask_yes < fair_val_yes:
                should_exit = True

            if should_exit:
                self._close_position(pos, curr_val, reason="FAIR VAL EXIT")

    def _close_position(self, pos, sell_price, reason):
        revenue = pos.quantity * sell_price
        profit = revenue - (pos.quantity * pos.entry_price)
        self.balance += revenue
        self.realized_pnl += profit
        self.positions.remove(pos)

        name = get_asset_name(pos.asset_id)
        color = "green" if profit > 0 else "red"
        self.log(f"SELL {pos.side} {name} @ {sell_price:.3f} (PnL: ${profit:+.2f}) [{reason}]", style=f"bold {color}")

trader = SimulatedTrader()

# --- RICH UI GENERATORS ---

def make_header() -> Table:
    """Creates the top statistics grid."""
    equity = 0.0
    for p in trader.positions:
        # Simple equity calc for display
        if p.asset_id in state.assets_ticks:
            ticks = state.assets_ticks[p.asset_id]
            if len(ticks) > 0:
                price = ticks['bid'][-1] if p.side == "YES" else (1.0 - ticks['ask'][-1])
                equity += p.quantity * price
        else:
            equity += p.quantity * p.entry_price

    total_val = trader.balance + equity
    total_pnl = total_val - trader.starting_balance

    grid = Table.grid(expand=True)
    grid.add_column(justify="center", ratio=1)
    grid.add_column(justify="center", ratio=1)
    grid.add_column(justify="center", ratio=1)
    grid.add_column(justify="center", ratio=1)

    pnl_color = "green" if total_pnl >= 0 else "red"
    realized_color = "green" if trader.realized_pnl >= 0 else "red"

    grid.add_row(
        f"💰 Cash: [bold gold1]${trader.balance:.2f}[/]",
        f"✅ Realized: [bold {realized_color}]${trader.realized_pnl:+.2f}[/]",
        f"📈 Equity: [bold blue]${equity:.2f}[/]",
        f"💎 Net Worth: [bold {pnl_color}]${total_val:.2f}[/] (PnL: {total_pnl:+.2f})"
    )
    return Panel(grid, style="white on black", title="🤖 ALGOBET TRADING DASHBOARD", border_style="blue")

def make_position_table() -> Table:
    """Creates the main table of positions."""
    table = Table(expand=True, border_style="dim", box=box.SIMPLE)

    table.add_column("Side", width=4, justify="center")
    table.add_column("Asset", ratio=1)
    table.add_column("Entry", justify="right", width=8)
    table.add_column("Bid", justify="right", width=8)
    table.add_column("Ask", justify="right", width=8)
    table.add_column("Qty", justify="right", width=6)
    table.add_column("PnL ($)", justify="right", width=10)

    # Collect and sort data
    rows = []
    for p in trader.positions:
        curr_bid, curr_ask = p.entry_price, p.entry_price

        if p.asset_id in state.assets_ticks and len(state.assets_ticks[p.asset_id]) > 0:
            ticks = state.assets_ticks[p.asset_id]
            if p.side == "YES":
                curr_bid = ticks['bid'][-1]
                curr_ask = ticks['ask'][-1]
            else:
                curr_bid = 1.0 - ticks['ask'][-1]
                curr_ask = 1.0 - ticks['bid'][-1]

        pnl = (curr_bid - p.entry_price) * p.quantity
        rows.append((p, curr_bid, curr_ask, pnl))

    # Sort by PnL Descending
    rows.sort(key=lambda x: x[3], reverse=True)

    if not rows:
        table.add_row("-", f"[italic yellow]Scanning {state.active_pairs_count} pairs...[/]", "-", "-", "-", "-", "-")
    else:
        for p, bid, ask, pnl in rows:
            side_style = "bold green" if p.side == "YES" else "bold red"
            pnl_style = "green" if pnl >= 0 else "red"
            name = get_asset_name(p.asset_id)
            if len(name) > 60: name = name[:57] + "..."

            table.add_row(
                f"[{side_style}]{p.side}[/]",
                name,
                f"{p.entry_price:.3f}",
                f"{bid:.3f}",
                f"{ask:.3f}",
                f"{int(p.quantity)}",
                f"[{pnl_style}]{pnl:+.2f}[/]"
            )

    return Panel(table, title=f"📊 Active Positions ({len(trader.positions)})", border_style="blue")

def make_log_panel() -> Panel:
    """Creates the scrolling log panel."""
    log_content = Table.grid(padding=(0, 1))
    log_content.add_column(justify="left", style="dim cyan", width=10)
    log_content.add_column(justify="left")

    for ts, msg, style in state.logs:
        log_content.add_row(ts, f"[{style}]{msg}[/]")

    return Panel(log_content, title="📝 Activity Log", height=12, border_style="blue")

def generate_layout() -> Layout:
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=14)
    )
    layout["header"].update(make_header())
    layout["main"].update(make_position_table())
    layout["footer"].update(make_log_panel())
    return layout

# --- ASYNC LOGIC ---

async def collect_ticks(sub):
    while True:
        try:
            aid_bytes, payload = await sub.recv_multipart()
            aid = aid_bytes.decode()
            arr = np.frombuffer(payload, dtype=tick_dtype)
            if len(arr) < 50: continue # Wait for some history
            state.assets_ticks[aid] = arr
        except Exception:
            await asyncio.sleep(0.1)

def run_strategy_logic():
    """Runs in a thread to avoid blocking UI."""
    snapshot = {k: v.copy() for k, v in state.assets_ticks.items()}
    state.active_pairs_count = len(snapshot)

    # 1. Update existing positions
    trader.update_positions(snapshot)

    # 2. Find Opportunities
    mid_prices = {aid: arr['ask'] for aid, arr in snapshot.items()}
    results = []

    # Optimization: Only check a subset or pairs with enough data
    for (aid1, s1), (aid2, s2) in combinations(mid_prices.items(), 2):
        n = min(len(s1), len(s2))
        if n < 20: continue

        # Quick price divergence check before running expensive KendallTau
        if abs(s1[-1] - s2[-1]) > 0.8: continue

        tau, _ = kendalltau(s1[:n], s2[:n])
        if abs(tau) > 0.6:
            results.append(((aid1, aid2), tau))

    # Sort best correlations
    results.sort(key=lambda x: abs(x[1]), reverse=True)

    THRESHOLD = 0.02

    for (a, b), tau in results[:5]: # Only trade top 5 signals per cycle to avoid spam
        if a not in snapshot or b not in snapshot: continue

        # Simple Fair Value / Spread Logic (Same as original)
        yes_ask_a = snapshot[a]['ask'][-1]
        yes_bid_a = snapshot[a]['bid'][-1]
        yes_ask_b = snapshot[b]['ask'][-1]

        if (yes_bid_a < 0.10 or yes_ask_a > 0.90): continue
        if (yes_ask_a - yes_bid_a) > 0.03: continue # Spread filter

        fair_yes_a = fair_price_np(mid_prices[a], mid_prices[b], yes_ask_b)['fair_mean']

        # Calc Scores
        score_a_yes = fair_yes_a - yes_ask_a
        score_a_no = yes_bid_a - fair_yes_a

        if score_a_yes > THRESHOLD:
            trader.buy(a, b, yes_ask_a, side="YES")
        elif score_a_no > THRESHOLD:
            trader.buy(a, b, 1.0 - yes_bid_a, side="NO")

    state.scan_count += 1

async def strategy_loop():
    print("Warming up strategy...")
    while True:
        await asyncio.sleep(0.5) # Run strategy every 500ms
        if len(state.assets_ticks) < 2: continue
        await asyncio.to_thread(run_strategy_logic)

async def main():
    ctx = zmq.asyncio.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(ZMQ_ADDR)
    sub.subscribe(b"")

    # Start ZMQ and Strategy in background
    collector_task = asyncio.create_task(collect_ticks(sub))
    strategy_task = asyncio.create_task(strategy_loop())

    load_asset_map()

    # Start Rich Live UI
    console = Console()
    with Live(generate_layout(), refresh_per_second=4, screen=True) as live:
        while True:
            live.update(generate_layout())
            await asyncio.sleep(0.25)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutdown complete.")