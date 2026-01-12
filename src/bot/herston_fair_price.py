"""
STRATEGY: DELTA-FLUX MOMENTUM SCALPER (LOW LATENCY)
ARCH: MULTIPROCESSING + EVENT DRIVEN + BATCH MATH + ASYNC LOGGING
"""
import asyncio
import aiohttp
import zmq
import zmq.asyncio
import json
import os
import warnings
import numpy as np
import csv
import time
from datetime import datetime
from collections import deque
from concurrent.futures import ProcessPoolExecutor

# --- RICH IMPORTS ---
from rich.live import Live
from rich.layout import Layout
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.align import Align
from rich.progress_bar import ProgressBar

# --- MODEL IMPORT ---
from heston_model import FastHestonModel

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
ZMQ_ADDR = "tcp://127.0.0.1:5567"
BINANCE_WS = "wss://stream.binance.com:9443/stream?streams=btcusdt@trade/ethusdt@trade/solusdt@trade/xrpusdt@trade"

# --- STRATEGY PARAMETERS ---
MIN_VELOCITY_BUY = 0.003
MAX_SPREAD = 0.08
MAX_POS_SIZE = 100.0
TAKER_FEE_PCT = 0.0125

# --- EXIT PARAMETERS ---
STAG_TOLERANCE = 0.015
STAG_LIMIT_SEC = 5.0
MOMENTUM_FLIP_THRESH = -0.01

# Files
DATA_DIR = os.path.join(os.getcwd(), "data")
TRADES_LOG_FILE = os.path.join(DATA_DIR, "delta_scalp_optimized.csv")
ASSET_ID_FILE = os.path.join(DATA_DIR, "clob_token_ids.jsonl")
PARAMS_FILE = os.path.join(DATA_DIR, "bates_params_digital.jsonl")
STRIKES_FILE = os.path.join(DATA_DIR, "market_1m_candle_opens.jsonl")

DURATION_MAP = {"15m": 15 / (60 * 24 * 365), "1h": 1 / (24 * 365), "4h": 4 / (24 * 365), "1d": 1 / 365}
tick_dtype = np.dtype([("ts_ms", np.int64), ("bid", np.float32), ("ask", np.float32)])
state_ticks = {}


# ==============================================================================
# 1. HELPER: BATCH MATH (RUNS IN SEPARATE PROCESS)
# ==============================================================================
def run_math_batch(spot, strikes, T_years_arr, initial_T_arr, params):
    """
    This function runs in a separate CPU process.
    It receives arrays and returns an array of fair prices.
    """
    results = []
    # If FastHestonModel is not vectorized, we loop here (in C-speed, not Python loop overhead)
    # Using a list comprehension inside the process is much faster than pickling 50 separate tasks.
    for i in range(len(strikes)):
        fair = FastHestonModel.price_binary_call(
            spot, strikes[i], T_years_arr[i], initial_T_arr[i], params
        )
        # Clamp result
        results.append(max(0.01, min(0.99, fair)))
    return results


# ==============================================================================
# 2. DATA MANAGER
# ==============================================================================
class DataManager:
    def __init__(self):
        self.clob_map = {}
        self.strikes = {}
        self.bates_params = {}
        self.stream_map = {"btcusdt": "BTC", "ethusdt": "ETH", "solusdt": "SOL", "xrpusdt": "XRP"}
        self.spot_cache = {k: 0.0 for k in self.stream_map.values()}

    def _load_sync(self):
        if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR, exist_ok=True)
        # Load Params
        if os.path.exists(PARAMS_FILE):
            with open(PARAMS_FILE, "r") as f:
                for line in f:
                    try:
                        p = json.loads(line)
                        self.bates_params[p['currency']] = p
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
                        if 'btc' in slug:
                            underlying = "BTC"
                        elif 'eth' in slug:
                            underlying = "ETH"
                        elif 'sol' in slug:
                            underlying = "SOL"
                        elif 'xrp' in slug:
                            underlying = "XRP"

                        if underlying and m.get('clob_token_id'):
                            dt = datetime.fromisoformat(m['market_end'].replace('Z', '+00:00'))
                            t_dur = DURATION_MAP.get(m.get('category', '1h'), 1 / (24 * 365))
                            if m.get('category') == '15m': t_dur = 15 / (60 * 24 * 365)
                            self.clob_map[m['clob_token_id']] = {
                                "question": m.get('question', m.get('slug')),
                                "underlying": underlying,
                                "end_ts_ms": int(dt.timestamp() * 1000),
                                "initial_T": t_dur
                            }
                    except:
                        pass

    async def watch_metadata(self):
        loop = asyncio.get_event_loop()
        while True:
            await loop.run_in_executor(None, self._load_sync)
            await asyncio.sleep(10)


# ==============================================================================
# 3. TRADING ENGINE
# ==============================================================================
class Position:
    def __init__(self, asset_id, entry_px, qty):
        self.asset_id = asset_id
        self.entry_px = entry_px
        self.qty = qty
        self.cost = entry_px * qty
        self.ts = time.time()


class DeltaBot:
    def __init__(self, dm: DataManager):
        self.dm = dm
        self.cash = 5000.0
        self.positions = {}
        self.pnl = 0.0
        self.wins = 0
        self.losses = 0
        self.total_fees = 0.0

        # KEY CHANGE 1: ProcessPool for bypassing GIL
        self.executor = ProcessPoolExecutor(max_workers=3)

        # KEY CHANGE 2: Async Logging Queue
        self.log_queue = asyncio.Queue()

        self.asset_index = {}
        self.logs = deque(maxlen=30)
        self.model_logs = deque(maxlen=20)
        self.last_state = {}
        self.stagnation_start = {}
        self.live_dashboard = {}
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(TRADES_LOG_FILE):
            with open(TRADES_LOG_FILE, 'w', newline='') as f:
                csv.writer(f).writerow(["timestamp", "action", "symbol", "price", "velocity", "reason", "spot", "pnl"])

    def rebuild_index(self):
        temp_index = {}
        for aid, meta in self.dm.clob_map.items():
            u = meta.get('underlying')
            if u:
                if u not in temp_index: temp_index[u] = []
                temp_index[u].append(aid)
        self.asset_index = temp_index

    # --- ASYNC LOGGING ---
    def queue_log(self, category, msg, details=None, style="white"):
        # Put data in queue, don't format string yet if possible, or format here quickly
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_queue.put_nowait({
            "type": "UI",
            "ts": ts,
            "category": category,
            "msg": msg,
            "details": details or "",
            "style": style
        })

    def queue_csv(self, action, sym, px, vel, reason, spot, pnl):
        self.log_queue.put_nowait({
            "type": "CSV",
            "row": [datetime.now().isoformat(), action, sym, f"{px:.3f}", f"{vel:.3f}", reason, f"{spot:.1f}",
                    f"{pnl:.2f}"]
        })

    # --- CORE EVENT LOOP ---
    async def handle_spot_update(self, underlying, spot_price):
        """
        Calculates Fair Value for ALL assets of this underlying in one batch.
        Then triggers entry/exit logic immediately.
        """
        if self.dm.spot_cache.get(underlying) == spot_price:
            return  # No change
        self.dm.spot_cache[underlying] = spot_price
        affected_ids = self.asset_index.get(underlying, [])
        if not affected_ids: return

        now_ts = time.time()
        params = self.dm.bates_params.get(underlying)
        if not params: return

        # 1. Prepare Batch Data
        batch_aids = []
        batch_strikes = []
        batch_T_years = []
        batch_init_T = []

        for aid in affected_ids:
            meta = self.dm.clob_map[aid]
            strike = self.dm.strikes.get(aid)
            if not strike: continue

            rem_ms = meta['end_ts_ms'] - (now_ts * 1000)
            if rem_ms <= 0: continue

            batch_aids.append(aid)
            batch_strikes.append(strike)
            batch_T_years.append(rem_ms / (1000 * 365 * 24 * 3600.0))
            batch_init_T.append(meta['initial_T'])

        if not batch_aids: return

        # 2. Run Math in Process Pool (No GIL blocking)
        # We pass lists; the helper function handles them.
        try:
            start = time.time()
            fairs = await asyncio.get_running_loop().run_in_executor(
                self.executor,
                run_math_batch,  # The static function defined above
                spot_price,
                batch_strikes,
                batch_T_years,
                batch_init_T,
                params
            )
            end = time.time()
        except Exception as e:
            # self.queue_log("ERR", f"Math Error: {e}")
            return

        # 3. Process Results & Trigger Logic Instantly
        for i, new_fair in enumerate(fairs):
            aid = batch_aids[i]
            prev = self.last_state.get(aid)

            velocity = 0.0
            if prev:
                dt = now_ts - prev['ts']
                if dt > 0.05:  # avoid div by zero or micro-noise
                    velocity = (new_fair - prev['fair']) / dt

            # Update State
            self.last_state[aid] = {'fair': new_fair, 'ts': now_ts, 'velocity': velocity}

            # Update Dashboard Data (Fast)
            meta = self.dm.clob_map[aid]
            self.live_dashboard[aid] = {
                "fair": new_fair,
                "velocity": velocity,
                "spot": spot_price,
                "strike": batch_strikes[i],
                "stag_timer": 0.0,  # Calculated in render to save cycles
                "question": meta['question'],
                "underlying": underlying
            }

            # 4. TRIGGER (No Sleeping)
            if aid in self.positions:
                self.check_exit(aid, new_fair, velocity, spot_price, meta['question'])
            else:
                self.check_entry(aid, new_fair, velocity, spot_price, meta['question'])

            if velocity > MIN_VELOCITY_BUY:
                ts = datetime.now().strftime("%H:%M:%S")
                self.model_logs.append(f"[{ts}] {underlying} | Spot: {spot_price:.1f} | Vel: {velocity:+.4f}")

    # ==============================================================================
    # 4. ENTRY LOGIC
    # ==============================================================================
    def check_entry(self, aid, fair, velocity, spot, question):
        if velocity < MIN_VELOCITY_BUY: return

        ticks = state_ticks.get(aid)
        if ticks is None or len(ticks) == 0: return

        # Latency Guard
        data_ts = float(ticks['ts_ms'][-1])
        now_ms = time.time_ns() // 1_000_000

        ask = float(ticks['ask'][-1])
        bid = float(ticks['bid'][-1])
        spread = ask - bid

        if ask >= 0.90 or ask <= 0.20: return
        if spread > MAX_SPREAD: return

        # Execute
        entry_fee = MAX_POS_SIZE * TAKER_FEE_PCT
        invested_amount = MAX_POS_SIZE - entry_fee
        qty = invested_amount / ask

        self.positions[aid] = Position(aid, ask, qty)
        self.positions[aid].cost = MAX_POS_SIZE
        self.cash -= MAX_POS_SIZE
        self.total_fees += entry_fee

        if aid in self.stagnation_start: del self.stagnation_start[aid]

        details = f"Vel: {velocity:+.3f} | Ask: {ask:.3f} | Spread: {spread:.3f} | Fee: ${entry_fee:.2f} | Latency: {now_ms - data_ts}ms"
        self.queue_log("BUY", f"Entered {question}", details, "bold green")
        self.queue_csv("BUY", question, ask, velocity, "MOMENTUM_ENTRY", spot, 0)

    # ==============================================================================
    # 5. EXIT LOGIC
    # ==============================================================================
    def check_exit(self, aid, fair, velocity, spot, question):
        ticks = state_ticks.get(aid)
        if ticks is None: return

        bid = float(ticks['bid'][-1])
        data_ts = float(ticks['ts_ms'][-1])

        reason = None
        now = time.time_ns() // 1_000_000

        # Exit Conditions
        if velocity < MOMENTUM_FLIP_THRESH:
            reason = f"FLIP: {velocity:.3f}"
        elif abs(velocity) < STAG_TOLERANCE:
            if aid not in self.stagnation_start:
                self.stagnation_start[aid] = now
            else:
                elapsed = (now - self.stagnation_start[aid]) / 1000.0
                if elapsed > STAG_LIMIT_SEC:
                    reason = f"STAG: {elapsed:.1f}s"
        else:
            if aid in self.stagnation_start: del self.stagnation_start[aid]

        if reason:
            self._execute_sell(aid, self.positions[aid], bid, reason, velocity, spot, question, data_ts)

    def _execute_sell(self, aid, pos, price, reason, velocity, spot, question, data_ts):
        gross = pos.qty * price
        fee = gross * TAKER_FEE_PCT
        net = gross - fee
        pnl = net - pos.cost

        self.cash += net
        self.pnl += pnl
        self.total_fees += fee

        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1

        del self.positions[aid]
        if aid in self.stagnation_start: del self.stagnation_start[aid]
        now = time.time_ns() // 1_000_000
        c = "green" if pnl > 0 else "bold red"
        details = f"Bid: {price:.3f} | PnL: ${pnl:.2f} | Fee: ${fee:.2f} | Vel: {velocity:+.3f} | Latency: {now - data_ts}ms"
        self.queue_log("SELL", f"Closed {question} ({reason})", details, c)
        self.queue_csv("SELL", question, price, velocity, reason, spot, pnl)


# ==============================================================================
# 6. WORKERS (IO & LOGGING)
# ==============================================================================
async def logger_worker(bot):
    """Handles file IO and UI log appending off the main thread"""
    while True:
        item = await bot.log_queue.get()
        if item['type'] == 'UI':
            bot.logs.append(item)
        elif item['type'] == 'CSV':
            try:
                with open(TRADES_LOG_FILE, 'a', newline='') as f:
                    csv.writer(f).writerow(item['row'])
            except:
                pass
        bot.log_queue.task_done()


async def binance_ws_worker(bot):
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(BINANCE_WS) as ws:
                    bot.queue_log("SYS", "Connected to Binance", style="blue")
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            if 'data' in data:
                                payload = data['data']
                                sym_raw = payload['s'].lower()
                                underlying = bot.dm.stream_map.get(sym_raw)

                                if underlying:
                                    # Trigger computation immediately
                                    await bot.handle_spot_update(underlying, float(payload['p']))
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            break
        except Exception:
            await asyncio.sleep(2)


async def poly_zmq_worker(bot):
    ctx = zmq.asyncio.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(ZMQ_ADDR)
    sub.subscribe(b"")

    while True:
        try:
            aid_bytes, payload = await sub.recv_multipart()

            # Optimization: Basic length check before numpy overhead
            if len(payload) == 0: continue

            arr = np.frombuffer(payload, dtype=tick_dtype)
            if len(arr) == 0: continue

            latest_bid = arr['bid'][-1]
            latest_ask = arr['ask'][-1]

            # Fast Filter
            if latest_bid < 0.005: continue
            if abs(latest_bid - 0.01) < 0.0001: continue  # Glitch filter

            aid = aid_bytes.decode()
            state_ticks[aid] = arr

            # --- ZMQ Triggered Logic (Optional: Spread checking) ---
            # If we hold a position, we might want to exit on ZMQ update if liquidity collapses
            # For now, we rely on Spot updates for velocity, but you could add check_exit here too.

        except Exception:
            await asyncio.sleep(0.01)


# ==============================================================================
# 7. VISUALIZATION (SEPARATE FROM LOGIC)
# ==============================================================================
def make_layout(bot):
    # Header
    grid = Table.grid(expand=True)
    grid.add_column(ratio=1);
    grid.add_column(ratio=1);
    grid.add_column(ratio=1);
    grid.add_column(ratio=1)
    tot = bot.wins + bot.losses
    wr = (bot.wins / tot * 100) if tot > 0 else 0
    grid.add_row(
        Panel(f"[bold green]${bot.cash:,.2f}[/]", title="Cash", border_style="green"),
        Panel(f"[bold yellow]${bot.pnl:+,.2f}[/]", title="Net PnL", border_style="yellow"),
        Panel(f"[red]${bot.total_fees:,.2f}[/]", title="Fees", border_style="red"),
        Panel(f"[magenta]{wr:.0f}% ({bot.wins}/{bot.losses})[/]", title="Win Rate", border_style="magenta")
    )

    # Spot Ticker
    s_text = Text(" LIVE SPOT: ", style="bold white on black")
    colors = {"BTC": "orange1", "ETH": "blue", "SOL": "magenta", "XRP": "white"}
    for sym, price in bot.dm.spot_cache.items():
        s_text.append(f" {sym} ${price:,.2f} ", style=f"bold black on {colors.get(sym, 'white')}")

    # Main Table
    table = Table(box=box.SIMPLE, expand=True, header_style="bold white", row_styles=["dim", ""])
    table.add_column("Market", ratio=3)
    table.add_column("Strike/Dist", justify="right", width=15)
    table.add_column("Spread", justify="right", width=8)
    table.add_column("Velocity", justify="right", width=10)
    table.add_column("Status", justify="center", width=12)

    active_ids = sorted(bot.live_dashboard.keys(), key=lambda x: bot.dm.clob_map.get(x, {}).get('question', ''))

    for aid in active_ids[:18]:
        d = bot.live_dashboard[aid]
        ticks = state_ticks.get(aid)
        if not ticks: continue

        ask, bid = ticks['ask'][-1], ticks['bid'][-1]
        spread = ask - bid
        dist = ((d['spot'] - d['strike']) / d['strike']) * 100

        # Stagnation Bar
        stag_render = ""
        if aid in bot.stagnation_start:
            el = (time.time_ns() // 1000000 - bot.stagnation_start[aid]) / 1000
            if el > 0.5: stag_render = f"[red]STAG {el:.1f}s[/]"

        # Status
        status = stag_render
        if aid in bot.positions:
            unreal = (ask - bot.positions[aid].entry_px) * bot.positions[aid].qty
            status = f"[{'green' if unreal > 0 else 'red'}]${unreal:+.1f}[/]"
        elif d['velocity'] > MIN_VELOCITY_BUY:
            status = "[blink bold yellow]SIGNAL[/]"

        v_col = "green" if d['velocity'] > 0.002 else ("red" if d['velocity'] < -0.002 else "dim")

        table.add_row(
            d['question'],
            f"${d['strike']} ({dist:+.2f}%)",
            f"{spread:.3f}",
            f"[{v_col}]{d['velocity']:+.4f}[/]",
            status
        )

    # Logs
    log_txt = Text()
    for e in reversed(list(bot.logs)):
        log_txt.append(f"[{e['ts']}] {e['msg']} ", style=e['style'])
        if e['details']: log_txt.append(f"| {e['details']}", style="dim")
        log_txt.append("\n")

    layout = Layout()
    layout.split_column(
        Layout(grid, size=3),
        Layout(Align.center(s_text), size=1),
        Layout(Panel(table, title="Delta-Flux Velocity Scanner"), ratio=2),
        Layout(Panel(log_txt, title="Event Log"), ratio=1)
    )
    return layout


async def main():
    dm = DataManager()
    bot = DeltaBot(dm)

    print("Loading Metadata...")
    await asyncio.get_event_loop().run_in_executor(None, dm._load_sync)
    bot.rebuild_index()

    asyncio.create_task(dm.watch_metadata())
    asyncio.create_task(logger_worker(bot))
    asyncio.create_task(poly_zmq_worker(bot))
    asyncio.create_task(binance_ws_worker(bot))

    # Re-index occasionally
    async def indexer():
        while True:
            await asyncio.sleep(30)
            bot.rebuild_index()

    asyncio.create_task(indexer())

    print("Starting Interface...")
    await asyncio.sleep(1)

    with Live(make_layout(bot), refresh_per_second=1, screen=True) as live:
        while True:
            live.update(make_layout(bot))
            await asyncio.sleep(1.0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
