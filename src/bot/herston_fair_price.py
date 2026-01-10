"""
BATES-MODEL Z-SCORE ARBITRAGE (YES-ONLY)
- STRATEGY: Mean Reversion on Model vs Market Residuals.
- SIGNAL: Z-Score of (Mid_Price - Model_Price).
- ENTRY: Buy YES when Z < -1.5 (Market is statistically cheap).
- EXIT:  Sell YES when Z >= 0.0 (Mispricing corrected).
- FILTER: Only trades YES. No short selling/NO shares.
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
from layout import make_layout
from rich.live import Live

# --- MODEL IMPORT ---
from heston_model import FastHestonModel

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
ENTRY_Z_SCORE = -1.5  # Buy YES when Market is 1.5 sigma BELOW Model
EXIT_Z_SCORE = 0.25  # Exit when Market returns to Model price
ROLLING_WINDOW = 600  # Ticks to calculate volatility of the residual
MIN_HISTORY = 30  # Warmup ticks before trading
MAX_SPREAD = 0.08  # Max Bid/Ask spread allowed
MAX_POS_SIZE = 100.0  # Max capital per trade
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
    This tells us if the current deviation is 'normal noise' or a 'signal'.
    """

    def __init__(self, window_size=300):
        self.window = window_size
        self.values = deque(maxlen=window_size)

    def update(self, val):
        self.values.append(val)

    def get_z_score(self, current_val):
        if len(self.values) < MIN_HISTORY: return 0.0

        # Calculate stats of history
        arr = np.array(self.values)
        mean = np.mean(arr)
        std = np.std(arr)

        if std < 1e-6: return 0.0  # Avoid division by zero

        # Z = (Current - Mean) / Std
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
                        self.bates_params[json.loads(line)['currency']] = json.loads(line)
                    except:
                        pass

        # Load Strikes
        if os.path.exists(STRIKES_FILE):
            with open(STRIKES_FILE, "r") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if "clob_token_id" in obj: self.strikes[obj["clob_token_id"]] = float(obj["strike_price"])
                    except:
                        pass

        # Load Metadata
        if os.path.exists(ASSET_ID_FILE):
            with open(ASSET_ID_FILE, "r") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        m = json.loads(line)
                        slug = m.get('slug', '').lower()
                        underlying = None
                        if 'bitcoin' in slug or 'btc' in slug:
                            underlying = "BTC"
                        elif 'ethereum' in slug or 'eth' in slug:
                            underlying = "ETH"
                        elif 'solana' in slug or 'sol' in slug:
                            underlying = "SOL"
                        elif 'xrp' in slug:
                            underlying = "XRP"

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
                    except:
                        pass

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
                    if r.status_code == 200: self.spot_prices[sym] = float(r.json()['price'])
            except:
                pass
            await asyncio.sleep(1.0)


# ==============================================================================
# 3. TRADING ENGINE
# ==============================================================================
class Position:
    def __init__(self, asset_id, side, strike, model_prob, entry_z):
        self.asset_id = asset_id
        self.side = side
        self.strike = strike
        self.initial_model_prob = model_prob
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
        self.logs = deque(maxlen=10)
        self.residuals = {}  # Map[aid] -> RollingStats
        self._init_csv()
        self._restore_history()

    def _init_csv(self):
        if not os.path.exists(TRADES_LOG_FILE):
            try:
                with open(TRADES_LOG_FILE, mode='w', newline='') as f:
                    csv.writer(f).writerow(["timestamp", "action", "question", "side", "price", "pnl", "z_score"])
            except:
                pass

    def _restore_history(self):
        # Omitted for brevity: Use previous restore logic if needed
        pass

    def _save_trade(self, action, pos, price, pnl, z_score, q_text):
        try:
            with open(TRADES_LOG_FILE, mode='a', newline='') as f:
                csv.writer(f).writerow([
                    datetime.now().isoformat(), action, q_text, pos.side,
                    f"{price:.4f}", f"{pnl:.4f}" if pnl else "", f"{z_score:.2f}"
                ])
        except:
            pass

    def log(self, msg, style="white"):
        self.logs.append((time.strftime("%H:%M:%S"), msg, style))

    def get_position(self, aid):
        for p in self.positions:
            if p.asset_id == aid: return p
        return None

    def evaluate(self, aid, market_bid, market_ask):
        """
        Core Z-Score Logic:
        1. Calculate Fair Value (Model Price).
        2. Calculate Residual = Market_Mid - Model_Price.
        3. Update Rolling Stats of Residual.
        4. Calculate Z-Score.
        5. Trade: Buy YES if Z < -1.5 (Market Cheap).
        """
        meta = self.dm.clob_map.get(aid)
        strike = self.dm.strikes.get(aid)
        if not (meta and strike): return

        # 1. Market Data
        now_ms = int(time.time() * 1000)
        rem_ms = meta['end_ts_ms'] - now_ms
        if rem_ms <= 0: return  # Expired

        mid_price = (market_bid + market_ask) / 2
        spread = market_ask - market_bid

        # 2. Model Price
        spot = self.dm.spot_prices.get(meta['underlying'])
        params = self.dm.bates_params.get(meta['underlying'])
        if not (spot and params): return

        T_years = rem_ms / (1000 * 365 * 24 * 3600.0)
        model_p = FastHestonModel.price_binary_call(spot, strike, T_years, meta['initial_T'], params)

        # 3. Z-Score Calculation
        # Residual: Positive = Market Expensive | Negative = Market Cheap
        residual = mid_price - model_p

        if aid not in self.residuals: self.residuals[aid] = RollingStats(ROLLING_WINDOW)
        stats = self.residuals[aid]
        stats.update(residual)

        if not stats.ready(): return  # Wait for history

        z_score = stats.get_z_score(residual)
        pos = self.get_position(aid)

        # --- TRADING LOGIC ---

        # Manage Existing Position
        if pos:
            # Accumulate
            if pos.is_accumulating:
                if (time.time() - pos.last_fill_ts) > CHUNK_DELAY:
                    # Keep buying if Z is still low (market still cheap)
                    if z_score < (ENTRY_Z_SCORE + 0.5) and spread < MAX_SPREAD:
                        self._execute_chunk(pos, market_ask, meta['question'], z_score)
                    else:
                        pos.is_accumulating = False

            # Check Exit (Mean Reversion)
            # Exit if Z returns to 0 (Market Price rose to match Model)
            if z_score >= EXIT_Z_SCORE:
                self._close_position(pos, market_bid, meta['question'], z_score, "MEAN_REV")

        # Check New Entry
        elif spread < MAX_SPREAD:
            # ONLY BUY YES
            # Entry: Z < -1.5 (Market Mid is 1.5 deviations below Model)
            if z_score < ENTRY_Z_SCORE:
                self._start_position(aid, "YES", market_ask, strike, model_p, meta['question'], z_score)

    def _start_position(self, aid, side, price, strike, model_p, q_text, z):
        if price > 0.98 or price < 0.02: return
        pos = Position(aid, side, strike, model_p, z)
        self.positions.append(pos)
        self._execute_chunk(pos, price, q_text, z)

    def _execute_chunk(self, pos, price, q_text, z):
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

        self.log(f"[BUY YES] Z:{z:.2f} | {q_text[:40]}... | Px:{price:.3f}", style="green")
        self._save_trade("BUY", pos, price, None, z, q_text)

    def _close_position(self, pos, price, q_text, z, reason):
        price = float(price - SLIPPAGE)
        proceeds = pos.size_qty * price
        pnl = proceeds - pos.cost_basis

        self.balance += proceeds
        self.realized_pnl += pnl
        self.positions.remove(pos)

        color = "bold green" if pnl > 0 else "bold red"
        self.log(f"[SELL YES] Z:{z:.2f} | PnL:${pnl:.2f} ({reason}) | {q_text[:40]}...", style=color)
        self._save_trade("SELL", pos, price, pnl, z, q_text)


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
            aid, payload = await sub.recv_multipart()
            arr = np.frombuffer(payload, dtype=tick_dtype)
            state_ticks[aid.decode()] = arr
            if len(arr) > 0: trader.evaluate(aid.decode(), float(arr['bid'][-1]), float(arr['ask'][-1]))
        except:
            await asyncio.sleep(0.01)


async def main():
    dm = DataManager()
    dm._load_sync()
    print("Initializing Z-Score YES-Only Trader...")
    trader = SimulatedTrader(dm)
    asyncio.create_task(dm.update_spot_prices())
    asyncio.create_task(zmq_loop(trader))
    asyncio.create_task(dm.watch_metadata())

    with Live(make_layout(trader, state_ticks), refresh_per_second=4, screen=True) as live:
        while True:
            live.update(make_layout(trader, state_ticks))
            await asyncio.sleep(0.25)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
