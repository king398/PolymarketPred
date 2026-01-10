"""
STRATEGY: SPOT-DRIVEN LATENCY SCALPER (HFT)
- CORE LOGIC: Arbitrage the time delay between Binance Spot moves and Prediction Market updates.
- SIGNAL: Instant Gap = (Heston_Fair_Value - Market_Price).
- SPEED: Reacts immediately to Spot volatility. No history/windows.
- STYLE: Aggressive Taker.
- ENTRY: Buy YES if Fair_Value > Ask + Min_Edge.
- EXIT: Sell YES immediately if Fair_Value < Bid (Edge gone).
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
try:
    from heston_model import FastHestonModel
except ImportError:
    class FastHestonModel:
        @staticmethod
        def price_binary_call(S, K, T, T_total, params):
            # Dummy pricing if file missing
            d = (S - K)
            return 0.5 + (d * 0.001)

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. HFT CONFIGURATION
# ==============================================================================
ZMQ_ADDR = "tcp://127.0.0.1:5567"
BINANCE_API = "https://api.binance.com/api/v3/ticker/price"

# --- SCALPING PARAMS ---
MIN_EDGE_TO_ENTER = 0.025   # We need 2.5% edge (covers fees + slippage + profit)
MIN_EDGE_TO_HOLD = 0.005    # If edge drops below 0.5%, we dump the position
MAX_SPREAD = 0.10           # Don't trade illiquid garbage
MAX_POS_SIZE = 200.0        # Max capital per trade (Aggressive sizing)
MAX_TOTAL_RISK = 5000.0     # Max total inventory
REFRESH_RATE = 0.1          # 100ms eval loop

# File Paths
DATA_DIR = os.path.join(os.getcwd(), "data")
ASSET_ID_FILE = os.path.join(DATA_DIR, "clob_token_ids.jsonl")
PARAMS_FILE = os.path.join(DATA_DIR, "bates_params_digital.jsonl")
STRIKES_FILE = os.path.join(DATA_DIR, "market_1m_candle_opens.jsonl")
TRADES_LOG_FILE = os.path.join(DATA_DIR, "hft_scalp_log.csv")

# Constants
YEAR_MS = 365 * 24 * 3600 * 1000
DURATION_MAP = {"15m": 15/(60*24*365), "1h": 1/(24*365), "4h": 4/(24*365), "1d": 1/365}

tick_dtype = np.dtype([("ts_ms", np.int64), ("bid", np.float32), ("ask", np.float32)])
state_ticks = {}


# ==============================================================================
# 1. DATA MANAGER
# ==============================================================================
class DataManager:
    def __init__(self):
        self.clob_map = {}
        self.strikes = {}
        self.bates_params = {}
        self.spot_prices = {}
        self.symbol_map = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT"}

    def _load_sync(self):
        # 1. Load Params
        if os.path.exists(PARAMS_FILE):
            with open(PARAMS_FILE, "r") as f:
                for line in f:
                    try:
                        p = json.loads(line)
                        self.bates_params[p['currency']] = p
                    except: pass

        # 2. Load Strikes
        if os.path.exists(STRIKES_FILE):
            with open(STRIKES_FILE, "r") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if "clob_token_id" in obj:
                            self.strikes[obj["clob_token_id"]] = float(obj["strike_price"])
                    except: pass

        # 3. Load Metadata
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

                        if underlying and m.get('clob_token_id'):
                            dt = datetime.fromisoformat(m['market_end'].replace('Z', '+00:00'))
                            t_years = DURATION_MAP.get(m.get('category', '1h'), 1/(24*365))
                            self.clob_map[m['clob_token_id']] = {
                                "question": m.get('question', m.get('slug')),
                                "underlying": underlying,
                                "end_ts_ms": int(dt.timestamp() * 1000),
                                "initial_T": t_years
                            }
                    except: pass

    async def watch_metadata(self):
        loop = asyncio.get_event_loop()
        while True:
            await loop.run_in_executor(None, self._load_sync)
            await asyncio.sleep(10)

    async def stream_spot_prices(self):
        """High frequency polling for Spot Prices"""
        while True:
            try:
                for sym, ticker in self.symbol_map.items():
                    r = requests.get(BINANCE_API, params={"symbol": ticker}, timeout=1)
                    if r.status_code == 200:
                        self.spot_prices[sym] = float(r.json()['price'])
            except: pass
            await asyncio.sleep(0.5) # Poll every 500ms

# ==============================================================================
# 2. SCALPING ENGINE
# ==============================================================================
class Position:
    def __init__(self, asset_id, entry_px, fair_at_entry):
        self.asset_id = asset_id
        self.entry_px = entry_px
        self.qty = MAX_POS_SIZE / entry_px
        self.cost_basis = MAX_POS_SIZE
        self.fair_at_entry = fair_at_entry
        self.timestamp = time.time()

class ScalperBot:
    def __init__(self, data_manager: DataManager):
        self.dm = data_manager
        self.cash = 5000.0
        self.positions = {} # dict {asset_id: Position}
        self.realized_pnl = 0.0
        self.logs = deque(maxlen=15)
        self.live_analytics = {}
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(TRADES_LOG_FILE):
            with open(TRADES_LOG_FILE, 'w', newline='') as f:
                csv.writer(f).writerow(["timestamp", "action", "symbol", "price", "fair", "spot", "pnl"])

    def log(self, msg, color="white"):
        ts = datetime.now().strftime("%H:%M:%S")
        self.logs.append((ts, msg, color))

    def evaluate(self, aid, bid, ask):
        meta = self.dm.clob_map.get(aid)
        strike = self.dm.strikes.get(aid)
        if not (meta and strike): return

        spot = self.dm.spot_prices.get(meta['underlying'])
        params = self.dm.bates_params.get(meta['underlying'])
        if not (spot and params): return

        # 1. Calculate Instant Fair Value
        now_ms = time.time() * 1000
        rem_ms = meta['end_ts_ms'] - now_ms
        if rem_ms <= 0: return

        T_years = rem_ms / (1000 * 365 * 24 * 3600.0)
        fair_price = FastHestonModel.price_binary_call(spot, strike, T_years, meta['initial_T'], params)

        # 2. Store Analytics for UI
        spread = ask - bid
        buy_edge = fair_price - ask
        sell_edge = bid - fair_price # For exiting long

        self.live_analytics[aid] = {
            "fair": fair_price,
            "buy_edge": buy_edge,
            "sell_edge": sell_edge,
            "spot": spot
        }

        # 3. TRADING LOGIC
        pos = self.positions.get(aid)

        if pos:
            # --- EXIT LOGIC ---
            # If our edge has evaporated (or reversed), DUMP IT.
            # We don't wait for profit target, we wait for edge decay.
            current_hold_value = fair_price - bid

            # Logic: If fair price is now LOWER than bid (market is overpaying), SELL.
            # Or if the edge is just too small to justify risk (< 0.5%)
            if fair_price < bid or (fair_price - bid) < MIN_EDGE_TO_HOLD:
                self._execute_sell(pos, bid, fair_price, spot, meta['question'])

        else:
            # --- ENTRY LOGIC ---
            # Buy if Fair Value is significantly higher than Ask (Market Lagging Upward Move)
            if buy_edge > MIN_EDGE_TO_ENTER and spread < MAX_SPREAD:
                if self.cash > MAX_POS_SIZE:
                    self._execute_buy(aid, ask, fair_price, spot, meta['question'])

    def _execute_buy(self, aid, price, fair, spot, question):
        if price >= 0.99: return
        pos = Position(aid, price, fair)
        self.positions[aid] = pos
        self.cash -= pos.cost_basis

        # Log
        edge_pct = ((fair - price)/price)*100
        self.log(f"BUY  | {question[:20]} | Gap: {edge_pct:.1f}% | Spot: {spot:.1f}", "bold green")
        self._write_csv("BUY", question, price, fair, spot, 0)

    def _execute_sell(self, pos, price, fair, spot, question):
        proceeds = pos.qty * price
        pnl = proceeds - pos.cost_basis
        self.cash += proceeds
        self.realized_pnl += pnl
        del self.positions[pos.asset_id]

        color = "bold cyan" if pnl > 0 else "bold red"
        self.log(f"SELL | {question[:20]} | PnL: ${pnl:.2f} | Spot: {spot:.1f}", color)
        self._write_csv("SELL", question, price, fair, spot, pnl)

    def _write_csv(self, side, sym, px, fair, spot, pnl):
        with open(TRADES_LOG_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([datetime.now().isoformat(), side, sym, px, fair, spot, pnl])

# ==============================================================================
# 3. HIGH-SPEED DASHBOARD
# ==============================================================================
def make_layout(bot):
    layout = Layout()

    # Metrics
    table_header = Table(box=box.SIMPLE, show_header=False, expand=True)
    table_header.add_row(
        f"CASH: [bold green]${bot.cash:.2f}[/]",
        f"PNL: [bold yellow]${bot.realized_pnl:.2f}[/]",
        f"ACTIVE SCALPS: [bold white]{len(bot.positions)}[/]"
    )

    # Market Grid
    table_mkt = Table(title="LATENCY SCALPER (Spot vs Model Gap)", box=box.SIMPLE_HEAD, expand=True)
    table_mkt.add_column("Market", ratio=3)
    table_mkt.add_column("Spot", justify="right")
    table_mkt.add_column("Mkt Ask", justify="right", style="red")
    table_mkt.add_column("Fair Val", justify="right", style="cyan")
    table_mkt.add_column("Gap (Edge)", justify="right", style="bold")
    table_mkt.add_column("Status", justify="center")

    sorted_aids = sorted(bot.live_analytics.keys(), key=lambda x: bot.live_analytics[x]['buy_edge'], reverse=True)

    for aid in sorted_aids[:12]: # Show top 12 opps
        data = bot.live_analytics[aid]
        meta = bot.dm.clob_map.get(aid)
        if not meta: continue

        # Edge Coloring
        edge = data['buy_edge']
        edge_style = "dim white"
        if edge > MIN_EDGE_TO_ENTER: edge_style = "bold green blink"
        elif edge < 0: edge_style = "red"

        # Status
        status = "-"
        if aid in bot.positions:
            pnl_unreal = (data['fair'] - bot.positions[aid].entry_px) * bot.positions[aid].qty
            c = "green" if pnl_unreal > 0 else "red"
            status = f"[{c}]OPEN ${pnl_unreal:.1f}[/{c}]"

        table_mkt.add_row(
            meta['question'][:35],
            f"{data['spot']:.1f}",
            f"{state_ticks.get(aid, np.zeros(1, dtype=tick_dtype))['ask'][-1]:.3f}",
            f"{data['fair']:.3f}",
            f"[{edge_style}]{edge:+.3f}[/{edge_style}]",
            status
        )

    # Logs
    log_txt = Text()
    for t, m, c in bot.logs:
        log_txt.append(f"[{t}] {m}\n", style=c)

    layout.split(
        Layout(Panel(table_header, title="Wallet"), size=5),
        Layout(table_mkt),
        Layout(Panel(log_txt, title="Scalp Log"), size=8)
    )
    return layout

# ==============================================================================
# 4. MAIN LOOP
# ==============================================================================
async def zmq_listener(bot):
    ctx = zmq.asyncio.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(ZMQ_ADDR)
    sub.subscribe(b"")

    while True:
        try:
            aid, payload = await sub.recv_multipart()
            arr = np.frombuffer(payload, dtype=tick_dtype)
            if len(arr) > 0:
                aid_str = aid.decode()
                state_ticks[aid_str] = arr
                bot.evaluate(aid_str, float(arr['bid'][-1]), float(arr['ask'][-1]))
        except Exception:
            await asyncio.sleep(0.01)

async def main():
    dm = DataManager()
    bot = ScalperBot(dm)

    # Background Tasks
    asyncio.create_task(dm.watch_metadata())
    asyncio.create_task(dm.stream_spot_prices())
    asyncio.create_task(zmq_listener(bot))

    # Give it a second to warm up
    print("Warming up models...")
    await asyncio.sleep(2)

    with Live(make_layout(bot), refresh_per_second=10, screen=True) as live:
        while True:
            live.update(make_layout(bot))
            await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(main())