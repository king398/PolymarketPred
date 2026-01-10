"""
BATES-MODEL Z-SCORE ARBITRAGE (YES-ONLY)
- STRATEGY: Mean Reversion on Model vs Market Residuals.
- SIGNAL: Z-Score of (Mid_Price - Model_Price).
- ENTRY: Buy YES when Z < -1.5 (Market is statistically cheap).
- EXIT:  Sell YES when Z >= 0.25 (Mispricing corrected).
- FILTER: Only trades YES. No short selling/NO shares.
- FIXES: Async loading, ZMQ polling, Non-blocking UI.
- ADDED: Live Z-Score Dashboard Column.
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
import csv
from collections import deque
from datetime import datetime
from rich.live import Live
from rich.layout import Layout
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

# --- MODEL IMPORT ---
# Ensure heston_model.py is in the same directory
try:
    from heston_model import FastHestonModel
except ImportError:
    # Dummy mock for demonstration if file is missing
    class FastHestonModel:
        @staticmethod
        def price_binary_call(S, K, T, T_total, params):
            return 0.50

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
ZMQ_ADDR = "tcp://127.0.0.1:5567"
BINANCE_API = "https://api.binance.com/api/v3/ticker/price"

# File Paths
DATA_DIR = os.path.join(os.getcwd(), "data")
ASSET_ID_FILE = os.path.join(DATA_DIR, "clob_token_ids.jsonl")
PARAMS_FILE = os.path.join(DATA_DIR, "bates_params_digital.jsonl")
STRIKES_FILE = os.path.join(DATA_DIR, "market_1m_candle_opens.jsonl")
TRADES_LOG_FILE = os.path.join(DATA_DIR, "sim_trade_history.csv")

# --- Z-SCORE STRATEGY PARAMS ---
ENTRY_Z_SCORE = -1.5   # Buy YES when Market is 1.5 sigma BELOW Model
EXIT_Z_SCORE = 0.25    # Exit when Market returns near Model price
ROLLING_WINDOW = 300   # Ticks to calculate volatility of the residual
MIN_HISTORY = 30       # Warmup ticks before trading
MAX_SPREAD = 0.08      # Max Bid/Ask spread allowed
MAX_POS_SIZE = 100.0   # Max capital per trade
SLIPPAGE = 0.0002
CHUNK_PCT = 0.2
CHUNK_DELAY = 1.0

# Time Constants
YEAR_MS = 365 * 24 * 3600 * 1000
DURATION_MAP = {"15m": 15 / (60 * 24 * 365), "1h": 1 / (24 * 365), "4h": 4 / (24 * 365), "1d": 1 / 365}

# --- DATA STRUCTURES ---
tick_dtype = np.dtype([
    ("ts_ms", np.int64),
    ("bid", np.float32),
    ("ask", np.float32),
])

state_ticks = {}


# ==============================================================================
# 1. STATISTICS ENGINE
# ==============================================================================
class RollingStats:
    """
    Tracks the rolling Mean and StdDev of the 'Residual' (Market Mid - Model Prob).
    """
    def __init__(self, window_size=300):
        self.window = window_size
        self.values = deque(maxlen=window_size)

    def update(self, val):
        self.values.append(val)

    def get_stats(self):
        """Returns (mean, std_dev) of the rolling window."""
        if len(self.values) < MIN_HISTORY:
            return 0.0, 0.0
        arr = np.array(self.values)
        return np.mean(arr), np.std(arr)

    def get_z_score(self, current_val):
        mean, std = self.get_stats()
        if std < 1e-6: return 0.0
        return (current_val - mean) / std

    def ready(self):
        return len(self.values) >= MIN_HISTORY


# ==============================================================================
# 2. DATA MANAGEMENT
# ==============================================================================
class DataManager:
    def __init__(self):
        self.clob_map = {}
        self.strikes = {}
        self.bates_params = {}
        self.spot_prices = {}
        self.symbol_map = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT"}

    def _load_sync(self):
        # Load Model Params
        if os.path.exists(PARAMS_FILE):
            with open(PARAMS_FILE, "r") as f:
                for line in f:
                    try:
                        p = json.loads(line)
                        self.bates_params[p['currency']] = p
                    except: pass

        # Load Strikes
        if os.path.exists(STRIKES_FILE):
            with open(STRIKES_FILE, "r") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if "clob_token_id" in obj:
                            self.strikes[obj["clob_token_id"]] = float(obj["strike_price"])
                    except: pass

        # Load Metadata
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
                            dt = datetime.fromisoformat(m['market_end'].replace('Z', '+00:00'))
                            t_years = DURATION_MAP.get(m.get('category', '1h'), 1 / (24 * 365))
                            self.clob_map[m['clob_token_id']] = {
                                "question": m.get('question', m.get('slug')),
                                "underlying": underlying,
                                "end_ts_ms": int(dt.timestamp() * 1000),
                                "initial_T": t_years,
                                "initial_duration_ms": t_years * YEAR_MS
                            }
                    except: pass

    async def watch_metadata(self):
        loop = asyncio.get_event_loop()
        while True:
            # Run heavy load in thread executor
            await loop.run_in_executor(None, self._load_sync)
            await asyncio.sleep(5)

    async def update_spot_prices(self):
        while True:
            try:
                for sym, ticker in self.symbol_map.items():
                    r = requests.get(BINANCE_API, params={"symbol": ticker}, timeout=2)
                    if r.status_code == 200: self.spot_prices[sym] = float(r.json()['price'])
            except: pass
            await asyncio.sleep(1.0)


# ==============================================================================
# 3. TRADING ENGINE
# ==============================================================================
class Position:
    def __init__(self, asset_id, side, strike, model_prob, entry_z, entry_spot):
        self.asset_id = asset_id
        self.side = side
        self.strike = strike
        self.initial_model_prob = model_prob
        self.initial_spot = entry_spot
        self.entry_z = entry_z
        self.avg_entry_px = 0.0
        self.size_qty = 0.0
        self.cost_basis = 0.0
        self.start_ts = time.time()
        self.target_cost = MAX_POS_SIZE
        self.last_fill_ts = 0
        self.is_accumulating = True

class SimulatedTrader:
    def __init__(self, data_manager: DataManager):
        self.dm = data_manager
        self.balance = 2000.0
        self.positions = []
        self.realized_pnl = 0.0
        self.logs = deque(maxlen=20)
        self.residuals = {}

        # --- NEW: Store Live Analytics for UI ---
        self.live_analytics = {}

        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(TRADES_LOG_FILE):
            try:
                with open(TRADES_LOG_FILE, mode='w', newline='') as f:
                    csv.writer(f).writerow([
                        "timestamp", "action", "question", "side", "price",
                        "fair_value", "z_score", "spot", "sigma", "pnl", "reason"
                    ])
            except: pass

    def _save_trade(self, action, pos, price, model_p, z, spot, sigma, pnl, reason, q_text):
        try:
            with open(TRADES_LOG_FILE, mode='a', newline='') as f:
                csv.writer(f).writerow([
                    datetime.now().isoformat(), action, q_text, pos.side,
                    f"{price:.4f}", f"{model_p:.4f}", f"{z:.2f}",
                    f"{spot:.2f}", f"{sigma:.4f}",
                    f"{pnl:.4f}" if pnl else "", reason
                ])
        except: pass

    def log(self, msg, style="white"):
        self.logs.append((time.strftime("%H:%M:%S"), msg, style))

    def get_position(self, aid):
        for p in self.positions:
            if p.asset_id == aid: return p
        return None

    def evaluate(self, aid, market_bid, market_ask):
        meta = self.dm.clob_map.get(aid)
        strike = self.dm.strikes.get(aid)
        if not (meta and strike): return

        # 1. Market Data
        now_ms = int(time.time() * 1000)
        rem_ms = meta['end_ts_ms'] - now_ms
        if rem_ms <= 0: return

        mid_price = (market_bid + market_ask) / 2
        spread = market_ask - market_bid

        # 2. Model Price
        spot = self.dm.spot_prices.get(meta['underlying'])
        params = self.dm.bates_params.get(meta['underlying'])
        if not (spot and params): return

        T_years = rem_ms / (1000 * 365 * 24 * 3600.0)
        model_p = FastHestonModel.price_binary_call(spot, strike, T_years, meta['initial_T'], params)

        # 3. Z-Score Calculation
        residual = mid_price - model_p

        if aid not in self.residuals: self.residuals[aid] = RollingStats(ROLLING_WINDOW)
        stats = self.residuals[aid]
        stats.update(residual)

        if not stats.ready():
            # Store partial data even if not ready to trade
            self.live_analytics[aid] = {
                "fair": model_p,
                "z": 0.0,
                "residual": residual,
                "ready": False
            }
            return

        mean_resid, sigma = stats.get_stats()
        if sigma < 1e-6: sigma = 0.001

        z_score = (residual - mean_resid) / sigma

        # --- NEW: Update Live Analytics for UI ---
        self.live_analytics[aid] = {
            "fair": model_p,
            "z": z_score,
            "residual": residual,
            "ready": True
        }

        pos = self.get_position(aid)

        # --- TRADING LOGIC ---
        if pos:
            # Accumulate
            if pos.is_accumulating:
                if (time.time() - pos.last_fill_ts) > CHUNK_DELAY:
                    if z_score < (ENTRY_Z_SCORE + 0.5) and spread < MAX_SPREAD:
                        self._execute_chunk(pos, market_ask, model_p, spot, sigma, meta['question'], z_score)
                    else:
                        pos.is_accumulating = False

            # Exit Logic
            if z_score >= EXIT_Z_SCORE:
                self._close_position(pos, market_bid, model_p, spot, sigma, meta['question'], z_score, "MEAN_REV")

        elif spread < MAX_SPREAD:
            # Entry Logic
            if z_score < ENTRY_Z_SCORE:
                self._start_position(aid, "YES", market_ask, strike, model_p, spot, sigma, meta['question'], z_score)

    def _start_position(self, aid, side, price, strike, model_p, spot, sigma, q_text, z):
        if price > 0.98 or price < 0.02: return
        pos = Position(aid, side, strike, model_p, z, spot)
        self.positions.append(pos)
        self._execute_chunk(pos, price, model_p, spot, sigma, q_text, z)

    def _execute_chunk(self, pos, price, model_p, spot, sigma, q_text, z):
        remaining = pos.target_cost - pos.cost_basis
        chunk = min(remaining, MAX_POS_SIZE * CHUNK_PCT, self.balance)
        if chunk < 5.0: pos.is_accumulating = False; return

        price = float(price + SLIPPAGE)
        qty = chunk / price
        self.balance -= chunk
        pos.cost_basis += chunk
        pos.size_qty += qty
        pos.avg_entry_px = pos.cost_basis / pos.size_qty
        pos.last_fill_ts = time.time()

        resid = price - model_p
        msg = (f"[BUY YES] {q_text[:30]}...\n"
               f"   >> Px:{price:.3f} | Fair:{model_p:.3f} | Resid:{resid:+.3f}\n"
               f"   >> Z:{z:.2f}    | Vol:{sigma:.3f}  | Spot:${spot:,.2f}")

        self.log(msg, style="green")
        self._save_trade("BUY", pos, price, model_p, z, spot, sigma, None, "ENTRY", q_text)

    def _close_position(self, pos, price, model_p, spot, sigma, q_text, z, reason):
        price = float(price - SLIPPAGE)
        proceeds = pos.size_qty * price
        pnl = proceeds - pos.cost_basis

        self.balance += proceeds
        self.realized_pnl += pnl
        self.positions.remove(pos)

        color = "bold green" if pnl > 0 else "bold red"
        spot_delta = spot - pos.initial_spot
        spot_pct = (spot_delta / pos.initial_spot) * 100

        msg = (f"[SELL YES] {q_text[:30]}...\n"
               f"   >> Exit:{price:.3f} | Fair:{model_p:.3f} | Spot:${spot:,.2f} ({spot_pct:+.1f}%)\n"
               f"   >> Z:{z:.2f}     | PnL:${pnl:.2f} | Reason:{reason}")

        self.log(msg, style=color)
        self._save_trade("SELL", pos, price, model_p, z, spot, sigma, pnl, reason, q_text)


# ==============================================================================
# 4. LAYOUT / UI GENERATION (INLINED)
# ==============================================================================
def format_time_remaining(end_ts_ms):
    now = time.time() * 1000
    diff = (end_ts_ms - now) / 1000
    if diff < 0: return "ENDED"
    hours = int(diff // 3600)
    mins = int((diff % 3600) // 60)
    return f"{hours}h {mins}m"

def make_layout(trader, state_ticks):
    """
    Generates the Rich layout with the Live Z-Score Column.
    """
    layout = Layout()

    # --- Header Table (Portfolio) ---
    header_table = Table(box=box.SIMPLE_HEAD, expand=True, show_header=True)
    header_table.add_column("Metric", style="dim")
    header_table.add_column("Value", style="bold white")
    header_table.add_column("Metric", style="dim")
    header_table.add_column("Value", style="bold white")

    total_positions = len(trader.positions)
    total_invested = sum(p.cost_basis for p in trader.positions)

    header_table.add_row(
        "Cash Balance", f"${trader.balance:.2f}",
        "Realized PnL", f"${trader.realized_pnl:.2f}"
    )
    header_table.add_row(
        "Open Positions", str(total_positions),
        "Invested", f"${total_invested:.2f}"
    )

    # --- Market Monitor Table ---
    market_table = Table(
        title="LIVE MARKET MONITOR & Z-SCORE ARBITRAGE",
        box=box.SIMPLE,
        expand=True,
        header_style="bold cyan"
    )

    market_table.add_column("Question", ratio=4)
    market_table.add_column("Time", justify="right", width=8)
    market_table.add_column("Bid", justify="right", width=6, style="green")
    market_table.add_column("Ask", justify="right", width=6, style="red")
    market_table.add_column("Fair", justify="right", width=6, style="yellow")
    market_table.add_column("Edge", justify="right", width=8)
    market_table.add_column("Z-Score", justify="right", width=8) # <--- NEW COLUMN
    market_table.add_column("Pos (Shares)", justify="center", width=12)
    market_table.add_column("Unreal PnL", justify="right", width=10)

    # Sort active markets by Liquidity (roughly) or Activity
    active_aids = list(state_ticks.keys())

    for aid in active_aids:
        ticks = state_ticks[aid]
        if len(ticks) == 0: continue

        bid = ticks['bid'][-1]
        ask = ticks['ask'][-1]

        meta = trader.dm.clob_map.get(aid)
        if not meta: continue

        q_text = meta['question']
        time_rem = format_time_remaining(meta['end_ts_ms'])

        # Analytics
        analytics = trader.live_analytics.get(aid, {})
        fair_val = analytics.get('fair', 0.0)
        edge = analytics.get('residual', 0.0)
        z_score = analytics.get('z', 0.0)
        is_ready = analytics.get('ready', False)

        # Formatting Z-Score
        z_str = f"{z_score:.2f}"
        z_style = "dim white"
        if is_ready:
            if z_score <= ENTRY_Z_SCORE: z_style = "bold green" # Buy Signal
            elif z_score >= EXIT_Z_SCORE: z_style = "bold red"  # Sell Signal
        else:
            z_str = "Wait" # Warming up

        # Formatting Edge
        edge_str = f"{edge:+.3f}"
        edge_style = "green" if edge > 0 else "red"

        # Position Info
        pos = trader.get_position(aid)
        pos_str = "-"
        pnl_str = "-"

        if pos:
            pos_str = f"{int(pos.size_qty)}"
            # Est PnL based on mid
            mid = (bid + ask) / 2
            unreal = (pos.size_qty * mid) - pos.cost_basis
            pnl_color = "green" if unreal > 0 else "red"
            pnl_str = f"[{pnl_color}]{unreal:+.2f}[/{pnl_color}]"

        market_table.add_row(
            q_text,
            time_rem,
            f"{bid:.3f}",
            f"{ask:.3f}",
            f"{fair_val:.3f}",
            f"[{edge_style}]{edge_str}[/{edge_style}]",
            f"[{z_style}]{z_str}[/{z_style}]",
            pos_str,
            pnl_str
        )

    # --- Logs Panel ---
    log_text = Text()
    for t, msg, style in trader.logs:
        log_text.append(f"{t} ", style="dim")
        log_text.append(msg + "\n", style=style)

    # Compose Layout
    layout.split(
        Layout(Panel(header_table, title="Portfolio"), size=8),
        Layout(market_table, name="body"),
        Layout(Panel(log_text, title="Trade Logs", border_style="blue"), size=10)
    )

    return layout


# ==============================================================================
# 5. ASYNC LOOP
# ==============================================================================
async def zmq_loop(trader):
    """
    Polls ZMQ with a timeout to prevent blocking the Event Loop.
    """
    ctx = zmq.asyncio.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(ZMQ_ADDR)
    sub.subscribe(b"")

    poller = zmq.asyncio.Poller()
    poller.register(sub, zmq.POLLIN)

    while True:
        try:
            events = await poller.poll(timeout=100)
            if sub in dict(events):
                aid, payload = await sub.recv_multipart()
                arr = np.frombuffer(payload, dtype=tick_dtype)
                state_ticks[aid.decode()] = arr
                if len(arr) > 0:
                    trader.evaluate(aid.decode(), float(arr['bid'][-1]), float(arr['ask'][-1]))

            await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            break
        except Exception as e:
            await asyncio.sleep(0.1)

async def main():
    print("--- STARTING TRADER ---")
    dm = DataManager()

    # 1. Load Data in Background (Non-Blocking)
    print("Loading Metadata (this may take 10-20 seconds)...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, dm._load_sync)
    print(f"Loaded {len(dm.clob_map)} assets.")

    trader = SimulatedTrader(dm)

    # 2. Launch Tasks
    t1 = asyncio.create_task(dm.update_spot_prices())
    t2 = asyncio.create_task(zmq_loop(trader))
    t3 = asyncio.create_task(dm.watch_metadata())

    print("Entering Live Dashboard...")
    try:
        # Pass both trader and state_ticks to the layout
        with Live(make_layout(trader, state_ticks), refresh_per_second=4, screen=True) as live:
            while True:
                live.update(make_layout(trader, state_ticks))
                await asyncio.sleep(0.25)
    except asyncio.CancelledError:
        print("Stopping...")
    finally:
        t1.cancel()
        t2.cancel()
        t3.cancel()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass