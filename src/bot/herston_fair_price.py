"""
BATES-MODEL VALUE ARBITRAGE SIM (Polymarket Up/Down)
- DETAILED LOGS: Multi-line logs with Spot, Strike, Model Price, Edge, ROI, Duration.
- DYNAMIC EDGE: Requires larger margin of safety for cheap options (Anti-Theta).
- UNIFIED UI: Single table for scanner & portfolio.
- FAILSAFE EXIT: Hysteresis added to stop whipsaw exits.
- STATE RESTORE: Reloads positions from CSV.
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

# --- CONFIGURATION ---
ZMQ_ADDR = "tcp://127.0.0.1:5567"
BINANCE_API = "https://api.binance.com/api/v3/ticker/price"

# File Paths
DATA_DIR = os.path.join(os.getcwd(), "data")
ASSET_ID_FILE = os.path.join(DATA_DIR, "clob_token_ids.jsonl")
PARAMS_FILE = os.path.join(DATA_DIR, "bates_params_digital.jsonl")
STRIKES_FILE = os.path.join(DATA_DIR, "market_1m_candle_opens.jsonl")
TRADES_LOG_FILE = os.path.join(DATA_DIR, "sim_trade_history.csv")

# Strategy Parameters
BASE_MIN_EDGE = 0.02       # Standard edge for ITM/ATM
HIGH_CONVICTION_EDGE = 0.05 # Higher edge for OTM (Cheap) options
MAX_POS_SIZE = 50.0
SLIPPAGE = 0.0002
LIQUIDATION_THRESH = 0.10
COOLDOWN_DURATION = 60.0
MAX_SPREAD = 0.10

# Risk Management
THESIS_TOLERANCE = 0.05
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

state_ticks = {}


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
                    except:
                        pass

        if os.path.exists(STRIKES_FILE):
            with open(STRIKES_FILE, "r") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if "clob_token_id" in obj and "strike_price" in obj:
                            self.strikes[obj["clob_token_id"]] = float(obj["strike_price"])
                    except:
                        pass

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
                    except:
                        pass
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
            except Exception:
                pass
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
        self.cooldowns = {}
        self._init_csv()
        self._restore_history()

    def _init_csv(self):
        if not os.path.exists(TRADES_LOG_FILE):
            try:
                with open(TRADES_LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "action", "question", "asset_id", "side", "price", "quantity", "cost", "pnl", "reason"])
            except Exception as e:
                self.log(f"CSV Init Error: {e}", style="red")

    def _restore_history(self):
        if not os.path.exists(TRADES_LOG_FILE): return
        self.log("Restoring history from CSV...", style="bold yellow")
        try:
            with open(TRADES_LOG_FILE, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                active_positions = {}
                for row in reader:
                    action = row['action']
                    aid = row['asset_id']
                    side = row['side']
                    try:
                        price = float(row['price']) if row['price'] else 0.0
                        qty = float(row['quantity']) if row['quantity'] else 0.0
                        cost = float(row['cost']) if row['cost'] else 0.0
                        pnl = float(row['pnl']) if row['pnl'] else 0.0
                    except: continue

                    if action == 'BUY':
                        self.balance -= cost
                        if aid not in active_positions:
                            strike = self.dm.strikes.get(aid, 0.0)
                            active_positions[aid] = Position(aid, side, strike, 0.0)
                        pos = active_positions[aid]
                        pos.cost_basis += cost
                        pos.size_qty += qty
                        if pos.size_qty > 0: pos.avg_entry_px = pos.cost_basis / pos.size_qty
                        pos.is_accumulating = False
                    elif action in ['SELL', 'LIQ', 'TAKE-PROFIT', 'STOP-LOSS', 'THESIS-BROKEN', 'MODEL-ARB-EXIT']:
                        self.balance += (qty * price)
                        self.realized_pnl += pnl
                        if aid in active_positions: del active_positions[aid]
                    elif action == 'SETTLE':
                        self.balance += (qty * price)
                        self.realized_pnl += pnl
                        if aid in active_positions: del active_positions[aid]
                self.positions = list(active_positions.values())
        except Exception as e:
            self.log(f"Error restoring: {e}", style="red")

    def _save_trade(self, action, pos, price, qty, cost, pnl, reason, question):
        try:
            with open(TRADES_LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([datetime.now().isoformat(), action, question, pos.asset_id, pos.side, f"{price:.4f}", f"{qty:.4f}", f"{cost:.4f}", f"{pnl:.4f}" if pnl is not None else "", reason])
        except: pass

    def log(self, msg, style="white"):
        self.logs.append((time.strftime("%H:%M:%S"), msg, style))

    def get_position(self, aid):
        for p in self.positions:
            if p.asset_id == aid: return p
        return None

    def _format_duration(self, start_ts):
        diff = time.time() - start_ts
        return f"{int(diff)}s" if diff < 60 else f"{int(diff/60)}m"

    def _settle_position(self, pos, final_price_yes, q_text):
        outcome = 1.0 if final_price_yes >= 0.50 else 0.0
        payout = pos.size_qty * outcome
        pnl = payout - pos.cost_basis
        self.balance += payout
        self.realized_pnl += pnl

        color = "green" if pnl >= 0 else "red"
        dur = self._format_duration(pos.start_ts)

        # DETAILED SETTLE LOG
        msg = (f"[SETTLE] {pos.side} | {q_text}\n"
               f"   >> Outcome: {outcome:.0f} | Entry: {pos.avg_entry_px:.3f} | ROI: {(pnl/pos.cost_basis)*100:.1f}%\n"
               f"   >> PnL: ${pnl:.2f} | Held: {dur}")

        self.log(msg, style=f"bold {color}")
        self._save_trade("SETTLE", pos, outcome, pos.size_qty, pos.cost_basis, pnl, "EXPIRATION", q_text)
        self.positions.remove(pos)

    def evaluate(self, aid, market_bid, market_ask):
        meta = self.dm.clob_map.get(aid)
        strike = self.dm.strikes.get(aid)
        if not (meta and strike): return

        now_ms = int(time.time() * 1000)
        rem_ms = meta['end_ts_ms'] - now_ms
        pos = self.get_position(aid)
        spread = market_ask - market_bid

        if rem_ms <= 0:
            if pos: self._settle_position(pos, (market_bid+market_ask)/2, meta['question'])
            return

        if pos and market_bid <= 0.02: return # Dead market protection

        spot = self.dm.spot_prices.get(meta['underlying'])
        params = self.dm.bates_params.get(meta['underlying'])
        if not (spot and params): return

        T_years = rem_ms / (1000 * 365 * 24 * 3600.0)
        t_days = rem_ms / (1000 * 3600 * 24.0)
        model_p = FastHestonModel.price_binary_call(spot, strike, T_years, meta['initial_T'], params)
        yes_edge = model_p - market_ask

        # --- DYNAMIC EDGE CALCULATION ---
        required_edge = HIGH_CONVICTION_EDGE if market_ask < 0.40 else BASE_MIN_EDGE

        if pos:
            if pos.is_accumulating:
                if (time.time() - pos.last_fill_ts) >= CHUNK_DELAY:
                    if spread > MAX_SPREAD: return
                    if pos.side == "YES" and yes_edge > required_edge:
                        # PASS DETAILED INFO TO EXECUTE
                        self._execute_chunk(pos, market_ask, meta['question'], model_p, spot, yes_edge, t_days)
                    else:
                        pos.is_accumulating = False
            else:
                self._check_exit(pos, model_p, market_bid, market_ask, meta['question'], t_days)
        else:
            if aid in self.cooldowns:
                if (time.time() - self.cooldowns[aid]) < COOLDOWN_DURATION: return
                else: del self.cooldowns[aid]

            if spread > MAX_SPREAD: return

            if (rem_ms/meta['initial_duration_ms']) > LIQUIDATION_THRESH:
                if yes_edge > required_edge:
                    # PASS DETAILED INFO TO START
                    self._start_position(aid, "YES", market_ask, model_p, strike, meta['question'], spot, yes_edge, t_days)

    def _start_position(self, aid, side, price, model_p, strike, q_text, spot, edge, t_days):
        if price <= 0.15 or price >= 0.95: return
        pos = Position(aid, side, strike, model_p)
        self.positions.append(pos)
        self._execute_chunk(pos, price, q_text, model_p, spot, edge, t_days)

    def _execute_chunk(self, pos, price, q_text, model_p, spot, edge, t_days):
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

        # DETAILED BUY LOG
        msg = (f"[BUY] {pos.side} | {q_text}\n"
               f"   >> Px: {price:.3f} | Fair: {model_p:.3f} | Edge: {edge:.3f}\n"
               f"   >> Spot: ${spot:,.2f} | Strike: {pos.strike:,.0f} | Exp: {t_days:.1f}d")

        self.log(msg, style="dim green")
        self._save_trade("BUY", pos, price, qty, chunk, None, "ENTRY_CHUNK", q_text)
        if pos.cost_basis >= pos.target_cost * 0.99: pos.is_accumulating = False

    def _check_exit(self, pos, model_p, bid, ask, q_text, t_days, force=False):
        exit_px = bid - SLIPPAGE
        pnl = (pos.size_qty * exit_px) - pos.cost_basis
        roi = pnl / max(pos.cost_basis, 1e-6)
        should_close = force
        msg_type = "LIQ" if force else "CLOSE"

        if not force:
            # Take Profit
            if roi >= 0.10:
                should_close = True
                msg_type = "TAKE-PROFIT"

            # Model Exit with Hysteresis
            elif pos.side == "YES" and model_p < (bid - 0.05):
                should_close = True
                msg_type = "MODEL-ARB-EXIT"

        if should_close:
            self.balance += (pos.size_qty * exit_px)
            self.realized_pnl += pnl
            self.positions.remove(pos)
            color = "green" if pnl > 0 else "red"
            dur = self._format_duration(pos.start_ts)

            # DETAILED EXIT LOG
            msg = (f"[{msg_type}] {pos.side} | {q_text}\n"
                   f"   >> Exit: {exit_px:.3f} | Entry: {pos.avg_entry_px:.3f} | ROI: {roi*100:.1f}%\n"
                   f"   >> Fair: {model_p:.3f} | PnL: ${pnl:.2f} | Held: {dur}")

            self.log(msg, style=f"bold {color}")
            self._save_trade("SELL", pos, exit_px, pos.size_qty, pos.cost_basis, pnl, msg_type, q_text)

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
        except: await asyncio.sleep(0.1)

async def main():
    dm = DataManager()
    dm._load_sync()
    print("Initializing Unified Bates Model Sim...")
    trader = SimulatedTrader(dm)
    asyncio.create_task(dm.update_spot_prices())
    asyncio.create_task(zmq_loop(trader))
    asyncio.create_task(dm.watch_metadata())
    with Live(make_layout(trader, state_ticks), refresh_per_second=4, screen=True) as live:
        while True:
            live.update(make_layout(trader, state_ticks))
            await asyncio.sleep(0.25)

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass