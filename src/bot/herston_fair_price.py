"""
BATES-MODEL VALUE ARBITRAGE SIM (Polymarket Up/Down)
- DETAILED LOGS: Full trade lifecycle (Entry/Exit px, Qty, Duration).
- FULL TEXT: Questions wrap instead of truncate.
- AUTO-CLEAN: Expired markets are hidden from the scanner.
- UNIFIED UI: Single table for scanner & portfolio.
"""

import time
import zmq
import zmq.asyncio
import asyncio
import json
import os
import requests
import warnings
import numpy as np
from collections import deque
from datetime import datetime

# --- RICH IMPORTS ---
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich import box
from rich.text import Text

# --- MODEL IMPORT ---
from herston import BatesModel

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
ZMQ_ADDR = "tcp://127.0.0.1:5567"
BINANCE_API = "https://api.binance.com/api/v3/ticker/price"

# File Paths
DATA_DIR = "/home/mithil/PycharmProjects/PolymarketPred/data"
ASSET_ID_FILE = os.path.join(DATA_DIR, "clob_token_ids.jsonl")
PARAMS_FILE = os.path.join(DATA_DIR, "bates_params_digital.jsonl")
STRIKES_FILE = os.path.join(DATA_DIR,"market_1m_candle_opens.jsonl")

# Strategy Parameters
MIN_EDGE = 0.03
MAX_POS_SIZE = 50.0
SLIPPAGE = 0.0002
LIQUIDATION_THRESH = 0.05  # 10% time remaining

# Execution Parameters
CHUNK_PCT = 0.5
CHUNK_DELAY = 2.0

# Time Constants
MIN_15 = 15 / (60 * 24 * 365)
HOUR_1 = 1 / (24 * 365)
HOUR_4 = 4 / (24 * 365)
DAY_1 = 1 / 365
DURATION_MAP = {"15m": MIN_15, "1h": HOUR_1, "4h": HOUR_4, "1d": DAY_1}
YEAR_MS = 365 * 24 * 3600 * 1000

# --- DATA STRUCTURES ---
tick_dtype = np.dtype([
    ("ts_ms", np.int64),
    ("bid", np.float32),
    ("ask", np.float32),
])

state_ticks = {} # Global ticker store

# ==============================================================================
# 1. DATA MANAGEMENT
# ==============================================================================
class DataManager:
    def __init__(self):
        self.clob_map = {}
        self.strikes = {}
        self.bates_params = {}
        self.spot_prices = {}
        self.symbol_map = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT"}

    def _load_sync(self):
        if os.path.exists(PARAMS_FILE):
            with open(PARAMS_FILE, "r") as f:
                for line in f:
                    try:
                        p = json.loads(line)
                        self.bates_params[p['currency']] = p
                    except: pass

        if os.path.exists(STRIKES_FILE):
            with open(STRIKES_FILE, "r") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if "clob_token_id" in obj and "open_price" in obj:
                            self.strikes[obj["clob_token_id"]] = float(obj["open_price"])
                    except: pass

        if os.path.exists(ASSET_ID_FILE):
            with open(ASSET_ID_FILE, "r") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        m = json.loads(line)
                        slug = m.get('slug', '').lower()
                        underlying = None
                        if 'bitcoin' in slug or 'btc' in slug: underlying = "BTC"
                        elif 'ethereum' in slug or 'eth' in slug: underlying = "ETH"
                        elif 'solana' in slug or 'sol' in slug: underlying = "SOL"
                        elif 'xrp' in slug: underlying = "XRP"

                        if underlying and m.get('clob_token_id'):
                            cat = m.get('category', '1h')
                            dt = datetime.fromisoformat(m['market_end'].replace('Z', '+00:00'))

                            t_years = DURATION_MAP.get(cat, HOUR_1)

                            self.clob_map[m['clob_token_id']] = {
                                "question": m.get('question', m.get('slug')),
                                "underlying": underlying,
                                "end_ts_ms": int(dt.timestamp() * 1000),
                                "initial_T": t_years,
                                "initial_duration_ms": t_years * YEAR_MS
                            }
                    except: pass
        return len(self.strikes)

    async def watch_metadata(self):
        loop = asyncio.get_event_loop()
        while True:
            await loop.run_in_executor(None, self._load_sync)
            await asyncio.sleep(5)

    async def update_spot_prices(self):
        while True:
            try:
                for sym, ticker in self.symbol_map.items():
                    r = requests.get(BINANCE_API, params={"symbol": ticker}, timeout=2)
                    if r.status_code == 200:
                        self.spot_prices[sym] = float(r.json()['price'])
            except Exception: pass
            await asyncio.sleep(1.0)

# ==============================================================================
# 2. TRADING ENGINE
# ==============================================================================
class Position:
    def __init__(self, asset_id, side, strike, model_prob):
        self.asset_id = asset_id
        self.side = side
        self.strike = strike
        self.initial_model_prob = model_prob

        self.avg_entry_px = 0.0
        self.size_qty = 0.0
        self.cost_basis = 0.0

        # Tracking for logs
        self.start_ts = time.time()
        self.target_cost = MAX_POS_SIZE
        self.last_fill_ts = 0
        self.is_accumulating = True

class SimulatedTrader:
    def __init__(self, data_manager: DataManager):
        self.dm = data_manager
        self.balance = 1000.0
        self.positions = []
        self.realized_pnl = 0.0
        self.logs = deque(maxlen=10)

    def log(self, msg, style="white"):
        ts = time.strftime("%H:%M:%S")
        self.logs.append((ts, msg, style))

    def get_position(self, aid):
        for p in self.positions:
            if p.asset_id == aid: return p
        return None

    def _format_duration(self, start_ts):
        diff = time.time() - start_ts
        if diff < 60: return f"{int(diff)}s"
        return f"{int(diff/60)}m"

    def _settle_position(self, pos, final_price_yes, q_text):
        """Resolves a position at expiration."""
        outcome = 1.0 if final_price_yes >= 0.50 else 0.0
        payout_val = outcome if pos.side == "YES" else (1.0 - outcome)
        total_payout = pos.size_qty * payout_val
        pnl = total_payout - pos.cost_basis

        self.balance += total_payout
        self.realized_pnl += pnl

        color = "green" if pnl >= 0 else "red"
        dur = self._format_duration(pos.start_ts)

        log_msg = (
            f"SETTLED {pos.side} | {q_text}\n"
            f"   >> Exit: {outcome:.0f} | Entry: {pos.avg_entry_px:.3f} | Qty: {pos.size_qty:.1f}\n"
            f"   >> Held: {dur} | PnL: ${pnl:.2f}"
        )
        self.log(log_msg, style=f"bold {color}")
        self.positions.remove(pos)

    def evaluate(self, aid, market_bid, market_ask):
        meta = self.dm.clob_map.get(aid)
        strike = self.dm.strikes.get(aid)
        if not (meta and strike): return

        now_ms = int(time.time() * 1000)
        rem_ms = meta['end_ts_ms'] - now_ms
        pos = self.get_position(aid)

        # 1. Check Expiration
        if rem_ms <= 0:
            if pos:
                mid = (market_bid + market_ask) / 2.0
                self._settle_position(pos, mid, meta['question'])
            return

        # 2. Check Auto-Liquidation (Time < 10%)
        time_ratio = rem_ms / meta['initial_duration_ms']
        if pos and time_ratio < LIQUIDATION_THRESH:
            self._check_exit(pos, -1, market_bid, market_ask, meta['question'], force=True)
            return

        underlying = meta['underlying']
        spot = self.dm.spot_prices.get(underlying)
        params = self.dm.bates_params.get(underlying)
        if not (spot and params): return

        T_years = rem_ms / (1000 * 365 * 24 * 3600.0)
        model_p = BatesModel.price_binary_call(spot, strike, T_years, meta['initial_T'], params)

        yes_edge = model_p - market_ask
        # no_edge calculation is removed/ignored for entry logic below

        # Position Management
        if pos:
            if pos.is_accumulating:
                if (time.time() - pos.last_fill_ts) >= CHUNK_DELAY:
                    # Only check YES edge for validity since we only trade YES
                    valid = (pos.side == "YES" and yes_edge > MIN_EDGE)
                    price = market_ask

                    if valid: self._execute_chunk(pos, price, meta['question'])
                    else: pos.is_accumulating = False
            else:
                self._check_exit(pos, model_p, market_bid, market_ask, meta['question'])
        else:
            # Entry logic (Don't enter if < 10% time left)
            if time_ratio > LIQUIDATION_THRESH:
                q_text = meta['question']

                # --- MODIFIED: ONLY ENTER YES POSITIONS ---
                if yes_edge > MIN_EDGE:
                    self._start_position(aid, "YES", market_ask, model_p, strike, q_text)

                # The logic for NO positions has been removed here.

    def _start_position(self, aid, side, price, model_p, strike, q_text):
        if price <= 0.05 or price >= 0.95: return
        pos = Position(aid, side, strike, model_p)
        self.positions.append(pos)
        self._execute_chunk(pos, price, q_text)

    def _execute_chunk(self, pos, price, q_text):
        remaining = pos.target_cost - pos.cost_basis
        chunk = min(remaining, MAX_POS_SIZE * CHUNK_PCT, self.balance)
        if chunk < 1.0:
            pos.is_accumulating = False; return

        price = float(price + SLIPPAGE)
        qty = chunk / price
        self.balance -= chunk
        pos.cost_basis += chunk
        pos.size_qty += qty
        pos.avg_entry_px = pos.cost_basis / pos.size_qty
        pos.last_fill_ts = time.time()

        color = "green" # Since only YES is traded, color is always green
        msg = f"BUY {pos.side} | {q_text} @ {price:.3f} (Qty: {qty:.1f})"
        self.log(msg, style=f"dim {color}")

        if pos.cost_basis >= pos.target_cost * 0.99:
            pos.is_accumulating = False

    def _check_exit(self, pos, model_p, bid, ask, q_text, force=False):
        # Exit logic simplified for YES only context (though NO logic is harmless if kept)
        exit_px = bid - SLIPPAGE

        pnl = (pos.size_qty * exit_px) - pos.cost_basis
        roi = pnl / max(pos.cost_basis, 1e-6)

        should_close = force
        if not force:
            if pos.side == "YES" and model_p < bid: should_close = True
            elif roi >= 0.25: should_close = True

        if should_close:
            self.balance += (pos.size_qty * exit_px)
            self.realized_pnl += pnl
            self.positions.remove(pos)

            color = "green" if pnl > 0 else "red"
            msg_type = "LIQ" if force else "CLOSE"
            dur = self._format_duration(pos.start_ts)

            log_msg = (
                f"{msg_type} {pos.side} | {q_text}\n"
                f"   >> Exit: {exit_px:.3f} | Entry: {pos.avg_entry_px:.3f} | Qty: {pos.size_qty:.1f}\n"
                f"   >> Held: {dur} | PnL: ${pnl:.2f}"
            )
            self.log(log_msg, style=f"bold {color}")

# ==============================================================================
# 3. UNIFIED UI GENERATION
# ==============================================================================
def format_time_left(ms, initial_ms):
    if ms < 0: return "EXP"
    ratio = ms / initial_ms if initial_ms > 0 else 0
    style_tag = "[red]" if ratio < LIQUIDATION_THRESH else ""
    end_tag = "[/]" if ratio < LIQUIDATION_THRESH else ""
    seconds = ms // 1000
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    txt = f"{h}h {m}m"
    if h == 0 and m < 10: txt = f"{m}m {s}s"
    return f"{style_tag}{txt}{end_tag}"

def get_unified_rows(trader):
    rows = []
    now_ms = int(time.time() * 1000)
    pos_map = {p.asset_id: p for p in trader.positions}

    for aid, meta in trader.dm.clob_map.items():
        strike = trader.dm.strikes.get(aid)
        ticks = state_ticks.get(aid)
        spot = trader.dm.spot_prices.get(meta['underlying'])
        params = trader.dm.bates_params.get(meta['underlying'])

        if not (strike and ticks is not None and spot): continue

        mkt_bid = float(ticks['bid'][-1])
        mkt_ask = float(ticks['ask'][-1])
        mkt_mid = (mkt_bid + mkt_ask) / 2.0

        rem_ms = meta['end_ts_ms'] - now_ms
        pos = pos_map.get(aid)

        # --- CLEANING LOGIC ---
        # If expired and no position, skip this row completely
        if rem_ms < 0 and pos is None:
            continue

        model_p = 0.0
        edge_val = 0.0

        if params and rem_ms > 0:
            T_yrs = rem_ms / (1000 * 365 * 24 * 3600.0)
            try:
                model_p = BatesModel.price_binary_call(spot, strike, T_yrs, meta['initial_T'], params)
                edge_yes = model_p - mkt_ask
                edge_no = mkt_bid - model_p
                edge_val = edge_yes if edge_yes > edge_no else edge_no
            except: pass

        pos_str = "-"
        entry_str = "-"
        pnl_str = "-"
        row_style = "white"

        if pos:
            row_style = "bold yellow" if pos.is_accumulating else "bold white"
            mark = mkt_bid if pos.side == "YES" else (1.0 - mkt_ask)
            val = pos.size_qty * mark
            unreal_pnl = val - pos.cost_basis
            p_color = "green" if unreal_pnl >= 0 else "red"
            pnl_str = f"[{p_color}]{unreal_pnl:+.2f}[/]"
            side_c = "green" if pos.side=="YES" else "red"
            pos_str = f"[{side_c}]{pos.side}[/] ({int(pos.size_qty)})"
            entry_str = f"{pos.avg_entry_px:.3f}"

        if pos is None and abs(edge_val) < 0.005 and rem_ms > 3600000:
            continue

        e_color = "green" if edge_val > 0 else "red"
        edge_str = f"[{e_color}]{edge_val:+.3f}[/]" if model_p > 0 else "-"

        rows.append({
            "question": meta['question'],
            "time": format_time_left(rem_ms, meta['initial_duration_ms']),
            "mid": mkt_mid,
            "fair": model_p,
            "edge": edge_val,
            "edge_str": edge_str,
            "pos": pos_str,
            "entry": entry_str,
            "pnl": pnl_str,
            "style": row_style,
            "has_pos": pos is not None
        })

    # Sort strictly by Question to prevent shifting
    rows.sort(key=lambda x: x['question'])
    return rows

def make_layout(trader):
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=20) # Increased size for detailed logs
    )

    stats = Table.grid(expand=True)
    stats.add_column(justify="center", ratio=1)
    stats.add_column(justify="center", ratio=1)
    stats.add_column(justify="center", ratio=1)

    spot_txt = " | ".join([f"{k} ${v:3f}" for k,v in trader.dm.spot_prices.items()])
    stats.add_row(
        f"[bold blue]Cash: ${trader.balance:.2f}[/]",
        f"[bold yellow]Realized PnL: ${trader.realized_pnl:.2f}[/]",
        f"[dim]{spot_txt}[/]"
    )
    layout["header"].update(Panel(stats, style="white on black"))

    table = Table(title="LIVE MARKET MONITOR & PORTFOLIO", expand=True, border_style="blue", box=box.SIMPLE)
    # Use overflow="fold" to wrap text instead of truncating
    table.add_column("Question", ratio=3, overflow="fold")
    table.add_column("Time", justify="center", style="dim")
    table.add_column("Mkt Mid", justify="right", style="magenta")
    table.add_column("Fair", justify="right", style="cyan")
    table.add_column("Edge", justify="right")
    table.add_column("Pos (Shares)", justify="center")
    table.add_column("Avg Entry", justify="right")
    table.add_column("Unreal PnL", justify="right")

    rows = get_unified_rows(trader)
    for r in rows:
        table.add_row(
            r['question'],
            r['time'],
            f"{r['mid']:.3f}",
            f"{r['fair']:.3f}" if r['fair'] > 0 else "-",
            r['edge_str'],
            r['pos'],
            r['entry'],
            r['pnl'],
            style=r['style']
        )

    layout["main"].update(table)

    log_text = Text()
    for ts, msg, style in list(trader.logs):
        # Adding separator for readability
        log_text.append(f"[{ts}] ", style="dim")
        log_text.append(f"{msg}\n", style=style)
        log_text.append("-" * 40 + "\n", style="dim")

    layout["footer"].update(Panel(log_text, title="Detailed Activity Log", border_style="dim"))

    return layout

# ==============================================================================
# 4. ASYNC LOOP
# ==============================================================================
async def zmq_loop(trader):
    ctx = zmq.asyncio.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(ZMQ_ADDR)
    sub.subscribe(b"")

    while True:
        try:
            aid_bytes, payload = await sub.recv_multipart()
            aid = aid_bytes.decode()
            arr = np.frombuffer(payload, dtype=tick_dtype)
            state_ticks[aid] = arr

            if len(arr) > 0:
                bid = float(arr['bid'][-1])
                ask = float(arr['ask'][-1])
                trader.evaluate(aid, bid, ask)

        except Exception:
            await asyncio.sleep(0.1)

async def main():
    dm = DataManager()
    dm._load_sync()
    print("Initializing Unified Bates Model Sim...")
    trader = SimulatedTrader(dm)

    asyncio.create_task(dm.update_spot_prices())
    asyncio.create_task(zmq_loop(trader))
    asyncio.create_task(dm.watch_metadata())

    with Live(make_layout(trader), refresh_per_second=4, screen=True) as live:
        while True:
            live.update(make_layout(trader))
            await asyncio.sleep(0.25)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass