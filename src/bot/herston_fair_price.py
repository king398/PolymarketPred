"""
STRATEGY: DELTA-FLUX MOMENTUM SCALPER (PURE VELOCITY)
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

# --- RICH IMPORTS ---
from rich.live import Live
from rich.layout import Layout
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.align import Align
from rich.console import Group
from rich.progress_bar import ProgressBar
from rich.style import Style

# --- MODEL IMPORT ---
from heston_model import FastHestonModel

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
ZMQ_ADDR = "tcp://127.0.0.1:5567"
BINANCE_WS = "wss://stream.binance.com:9443/stream?streams=btcusdt@trade/ethusdt@trade/solusdt@trade/xrpusdt@trade"

# --- STRATEGY PARAMETERS ---
MIN_VELOCITY_BUY = 0.10      # How fast price must be moving up to buy
MAX_SPREAD = 0.08            # Max difference between Bid/Ask to allow entry
MAX_POS_SIZE = 100.0         # Dollars per trade

# --- EXIT PARAMETERS ---
STAG_TOLERANCE = 0.015       # Velocity close to 0 is stagnation
STAG_LIMIT_SEC = 5.0         # Seconds to hold while stagnant
MOMENTUM_FLIP_THRESH = -0.05 # Sell immediately if velocity drops below this

# Files
DATA_DIR = os.path.join(os.getcwd(), "data")
TRADES_LOG_FILE = os.path.join(DATA_DIR, "delta_scalp_momentum_log.csv")
ASSET_ID_FILE = os.path.join(DATA_DIR, "clob_token_ids.jsonl")
PARAMS_FILE = os.path.join(DATA_DIR, "bates_params_digital.jsonl")
STRIKES_FILE = os.path.join(DATA_DIR, "market_1m_candle_opens.jsonl")

DURATION_MAP = {"15m": 15 / (60 * 24 * 365), "1h": 1 / (24 * 365), "4h": 4 / (24 * 365), "1d": 1 / 365}
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
        self.stream_map = {"btcusdt": "BTC", "ethusdt": "ETH", "solusdt": "SOL", "xrpusdt": "XRP"}
        self.spot_cache = {k: 0.0 for k in self.stream_map.values()}

    def _load_sync(self):
        if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR, exist_ok=True)
        if os.path.exists(PARAMS_FILE):
            with open(PARAMS_FILE, "r") as f:
                for line in f:
                    try:
                        p = json.loads(line);
                        self.bates_params[p['currency']] = p
                    except:
                        pass
        if os.path.exists(STRIKES_FILE):
            with open(STRIKES_FILE, "r") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if "clob_token_id" in obj: self.strikes[obj["clob_token_id"]] = float(obj["strike_price"])
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
# 2. TRADING ENGINE
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
        self.start_cash = 5000.0
        self.positions = {}
        self.pnl = 0.0
        self.wins = 0
        self.losses = 0

        # Logs
        self.logs = deque(maxlen=30)
        self.model_logs = deque(maxlen=20)

        # State
        self.last_state = {}
        self.stagnation_start = {}
        self.live_dashboard = {}
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(TRADES_LOG_FILE):
            with open(TRADES_LOG_FILE, 'w', newline='') as f:
                csv.writer(f).writerow(["timestamp", "action", "symbol", "price", "velocity", "reason", "spot", "pnl"])

    def log(self, category, msg, details=None, style="white"):
        ts = datetime.now().strftime("%H:%M:%S")
        self.logs.append({
            "ts": ts,
            "category": category,
            "msg": msg,
            "details": details or "",
            "style": style
        })

    def log_model(self, symbol, spot, fair, vel):
        ts = datetime.now().strftime("%H:%M:%S")
        self.model_logs.append(f"[{ts}] {symbol} | Spot: {spot:.1f} | Vel: {vel:+.4f}")

    def handle_spot_update(self, underlying, spot_price):
        self.dm.spot_cache[underlying] = spot_price
        now_ts = time.time()
        affected_ids = [aid for aid, m in self.dm.clob_map.items() if m['underlying'] == underlying]

        for aid in affected_ids:
            meta = self.dm.clob_map.get(aid)
            strike = self.dm.strikes.get(aid)
            params = self.dm.bates_params.get(underlying)

            if meta['end_ts_ms'] <= (now_ts * 1000): continue
            if not (strike and params): continue

            rem_ms = meta['end_ts_ms'] - (now_ts * 1000)
            T_years = rem_ms / (1000 * 365 * 24 * 3600.0)
            if T_years <= 0: continue

            # --- MODEL CALCULATION (Used ONLY for Velocity now) ---
            new_fair = FastHestonModel.price_binary_call(spot_price, strike, T_years, meta['initial_T'], params)
            new_fair = max(0.01, min(0.99, new_fair))

            # --- VELOCITY CALCULATION ---
            prev = self.last_state.get(aid)
            velocity = 0.0
            if prev:
                dt = now_ts - prev['ts']
                if dt > 0.0001:
                    velocity = (new_fair - prev['fair']) / dt

            self.last_state[aid] = {'fair': new_fair, 'ts': now_ts}

            # Update Dashboard Data
            stag_timer = 0.0
            if aid in self.stagnation_start:
                stag_timer = now_ts - self.stagnation_start[aid]

            self.live_dashboard[aid] = {
                "fair": new_fair,
                "velocity": velocity,
                "spot": spot_price,
                "strike": strike,
                "stag_timer": stag_timer,
                "question": meta['question'],
                "underlying": underlying
            }

            if velocity > MIN_VELOCITY_BUY:
                self.log_model(underlying, spot_price, new_fair, velocity)

            if aid in self.positions:
                self.check_exit(aid, new_fair, velocity, spot_price, meta['question'])
            else:
                self.check_entry(aid, new_fair, velocity, spot_price, meta['question'])

    # ==========================================================================
    # MODIFIED: PURE MOMENTUM ENTRY
    # ==========================================================================
    def check_entry(self, aid, fair, velocity, spot, question):
        # 1. VELOCITY CHECK
        if velocity < MIN_VELOCITY_BUY: return

        # 2. SPREAD / MAX PRICE CHECK
        ticks = state_ticks.get(aid)
        if ticks is None or len(ticks) == 0: return
        ask = float(ticks['ask'][-1])
        bid = float(ticks['bid'][-1])

        # Don't buy if spread is massive or price is near max
        if ask >= 0.98 or (ask - bid) > MAX_SPREAD: return

        # --- REMOVED: VALUE GAP CHECK ---
        # We now enter purely because velocity is high, regardless of "Fair Value"

        qty = MAX_POS_SIZE / ask
        self.positions[aid] = Position(aid, ask, qty)
        self.cash -= MAX_POS_SIZE
        if aid in self.stagnation_start: del self.stagnation_start[aid]

        details = (
            f"Momentum Entry @ {ask:.3f}\n"
            f"Vel: {velocity:+.3f} (Req: >{MIN_VELOCITY_BUY})"
        )
        self.log("BUY", f"Entered {question}", details, "bold green")
        self._write_csv("BUY", question, ask, velocity, "MOMENTUM_ENTRY", spot, 0)

    # ==========================================================================
    # MODIFIED: EXIT LOGIC (NO VALUE CHECK, ADDED CRASH GUARD)
    # ==========================================================================
    def check_exit(self, aid, fair, velocity, spot, question):
        ticks = state_ticks.get(aid)
        if ticks is None: return
        bid = float(ticks['bid'][-1])
        pos = self.positions[aid]
        reason = None
        now = time.time()

        # 1. CRASH GUARD (Negative Velocity)
        # If momentum flips negative, sell immediately to protect gains/limit loss
        if velocity < MOMENTUM_FLIP_THRESH:
            reason = f"MOMENTUM FLIP: Vel {velocity:.3f} < {MOMENTUM_FLIP_THRESH}"

        # 2. STAGNATION EXIT
        elif abs(velocity) < STAG_TOLERANCE:
            if aid not in self.stagnation_start:
                self.stagnation_start[aid] = now
            else:
                elapsed = now - self.stagnation_start[aid]
                if elapsed > STAG_LIMIT_SEC:
                    reason = f"STAGNATION: {elapsed:.1f}s > {STAG_LIMIT_SEC}s"
        else:
            # Momentum is still positive, keep holding
            if aid in self.stagnation_start:
                del self.stagnation_start[aid]

        if reason:
            self._execute_sell(aid, pos, bid, reason, velocity, fair, spot, question)

    def _execute_sell(self, aid, pos, price, reason, velocity, fair, spot, question):
        proceeds = pos.qty * price
        pnl = proceeds - pos.cost
        roi = (pnl / pos.cost) * 100
        self.cash += proceeds
        self.pnl += pnl

        if pnl > 0: self.wins += 1
        else: self.losses += 1

        del self.positions[aid]
        if aid in self.stagnation_start: del self.stagnation_start[aid]

        c = "green" if pnl > 0 else "bold red"
        details = (
            f"Exit Price: {price:.3f} | Entry: {pos.entry_px:.3f}\n"
            f"Reason: {reason}\n"
            f"ROI: {roi:+.2f}%"
        )
        self.log("SELL", f"Closed {question}", details, c)
        self._write_csv("SELL", question, price, velocity, reason, spot, pnl)

    def _write_csv(self, action, sym, px, vel, reason, spot, pnl):
        with open(TRADES_LOG_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(), action, sym,
                f"{px:.3f}", f"{vel:.3f}", reason,
                f"{spot:.1f}", f"{pnl:.2f}"
            ])


# ==============================================================================
# 3. WORKERS
# ==============================================================================
async def binance_ws_worker(bot):
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(BINANCE_WS) as ws:
                    bot.log("SYS", "Connected to Binance Stream", "", "blue")
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            if 'data' in data:
                                payload = data['data']
                                sym_raw = payload['s'].lower()
                                underlying = bot.dm.stream_map.get(sym_raw)
                                if underlying:
                                    bot.handle_spot_update(underlying, float(payload['p']))
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
            aid, payload = await sub.recv_multipart()
            arr = np.frombuffer(payload, dtype=tick_dtype)

            # --- ARTIFACT FILTER APPLIED HERE ---
            if len(arr) > 0:
                # User reported 0.010 glitch ticks.
                # We filter anything <= 0.02 to be safe and ensure global state
                # only holds valid prices.
                latest_bid = arr['bid'][-1]
                latest_ask = arr['ask'][-1]

                if latest_bid > 0.02 and latest_ask > 0.02:
                    state_ticks[aid.decode()] = arr
        except:
            await asyncio.sleep(0.01)


# ==============================================================================
# 4. ENHANCED DASHBOARD
# ==============================================================================
def make_header(bot):
    grid = Table.grid(expand=True)
    grid.add_column(justify="center", ratio=1)
    grid.add_column(justify="center", ratio=1)
    grid.add_column(justify="center", ratio=1)
    grid.add_column(justify="center", ratio=1)

    total_trades = bot.wins + bot.losses
    win_rate = (bot.wins / total_trades * 100) if total_trades > 0 else 0
    exposure = sum([p.qty * p.entry_px for p in bot.positions.values()])

    grid.add_row(
        Panel(f"[bold green]${bot.cash:,.2f}[/]", title="Available Cash", border_style="green"),
        Panel(f"[bold yellow]${bot.pnl:+,.2f}[/]", title="Realized PnL", border_style="yellow"),
        Panel(f"[cyan]${exposure:,.2f}[/]", title="Active Exposure", border_style="cyan"),
        Panel(f"[magenta]{win_rate:.0f}% ({bot.wins}W/{bot.losses}L)[/]", title="Win Rate", border_style="magenta")
    )
    return grid


def make_spot_ticker(bot):
    s = bot.dm.spot_cache
    text = Text()
    text.append(" LIVE SPOT:  ", style="bold white on black")

    colors = {"BTC": "orange1", "ETH": "blue", "SOL": "magenta", "XRP": "white"}
    for sym, price in s.items():
        text.append(f" {sym} ", style=f"bold black on {colors.get(sym, 'white')}")
        text.append(f" ${price:,.2f} ", style="bold white on black")
        text.append(" │ ", style="dim grey")

    return Align.center(text)


def make_market_table(bot):
    table = Table(
        box=box.SIMPLE,
        expand=True,
        header_style="bold white",
        row_styles=["dim", ""]
    )
    table.add_column("Market / Strike", ratio=3)
    table.add_column("Moneyness", justify="right", width=12)
    # Replaced "Edge" with "Spread" for context
    table.add_column("Spread", justify="right", width=10)
    table.add_column("Market Ask", justify="right", width=10)
    table.add_column("Velocity", justify="right", width=10)
    table.add_column("Stagnation (Exit)", justify="left", width=20)
    table.add_column("Status", justify="center", width=12)

    active_ids = sorted(
        bot.live_dashboard.keys(),
        key=lambda x: (
            0 if x in bot.positions else 1,
            -abs(bot.live_dashboard[x]['velocity']),
            bot.dm.clob_map.get(x, {}).get('question', '')
        )
    )

    if not active_ids:
        table.add_row("[dim]Waiting for data stream...[/]", "", "", "", "", "", "")

    for aid in active_ids[:18]:
        data = bot.live_dashboard[aid]
        meta = bot.dm.clob_map.get(aid)
        if not meta: continue

        vel = data['velocity']
        spot = data['spot']
        strike = data['strike']
        stag_t = data['stag_timer']

        ticks = state_ticks.get(aid)
        if ticks is not None and len(ticks) > 0:
            mkt_ask = ticks['ask'][-1]
            mkt_bid = ticks['bid'][-1]
            spread = mkt_ask - mkt_bid
        else:
            mkt_ask = 0.0
            spread = 0.0

        dist_pct = ((spot - strike) / strike) * 100
        mon_style = "green" if dist_pct > 0 else "red"
        moneyness_str = f"[{mon_style}]{dist_pct:+.2f}%[/]"

        v_style = "dim"
        if vel > MIN_VELOCITY_BUY: v_style = "bold green"
        elif vel < MOMENTUM_FLIP_THRESH: v_style = "bold red"

        if aid in bot.stagnation_start:
            completed = min(stag_t, STAG_LIMIT_SEC)
            bar = ProgressBar(total=STAG_LIMIT_SEC, completed=completed, width=15, style="red",
                              complete_style="bold red", finished_style="blink bold red")
            stag_render = bar
        else:
            stag_render = Text("-", style="dim")

        status = "-"
        if aid in bot.positions:
            pos = bot.positions[aid]
            unreal = (mkt_ask - pos.entry_px) * pos.qty
            c = "green" if unreal > 0 else "red"
            status = f"[{c}]${unreal:+.1f}[/{c}]"
        elif vel > MIN_VELOCITY_BUY:
            status = "[blink bold yellow]SIGNAL[/]"

        table.add_row(
            f"{meta['question']}\n[dim]Strike: ${strike}[/]",
            moneyness_str,
            f"{spread:.3f}",
            f"{mkt_ask:.3f}",
            f"[{v_style}]{vel:+.4f}[/{v_style}]",
            stag_render,
            status
        )
    return table


def make_logs_panel(bot):
    # Left: Trade Execution Log
    trade_text = Text()
    if not bot.logs:
        trade_text.append("No trades yet...", style="dim")
    else:
        for entry in reversed(list(bot.logs)):
            trade_text.append(f"[{entry['ts']}] {entry['category']}: {entry['msg']}\n", style=entry['style'])
            if entry['details']:
                trade_text.append(f"{entry['details']}\n", style="dim white")
            trade_text.append("─"*30 + "\n", style="dim grey")

    # Right: Model Internal Stream
    model_text = Text()
    if not bot.model_logs:
        model_text.append("Model waiting for high velocity...", style="dim")
    else:
        for line in reversed(list(bot.model_logs)):
            model_text.append(line + "\n", style="cyan")

    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)

    grid.add_row(
        Panel(trade_text, title="Execution Log (Newest First)", border_style="white"),
        Panel(model_text, title="Momentum Stream (Newest First)", border_style="cyan")
    )

    return grid

def make_layout(bot):
    layout = Layout()
    layout.split_column(
        Layout(make_header(bot), size=3),
        Layout(make_spot_ticker(bot), size=1),
        Layout(Panel(make_market_table(bot), title="Delta-Flux Scanner (Momentum Only)"), ratio=2),
        Layout(make_logs_panel(bot), ratio=1)
    )
    return layout


# ==============================================================================
# MAIN
# ==============================================================================
async def main():
    dm = DataManager()
    bot = DeltaBot(dm)

    print("Loading Metadata...")
    await asyncio.get_event_loop().run_in_executor(None, dm._load_sync)

    # Background Tasks
    asyncio.create_task(dm.watch_metadata())
    asyncio.create_task(poly_zmq_worker(bot))
    asyncio.create_task(binance_ws_worker(bot))

    print("Starting Interface...")
    await asyncio.sleep(2)

    with Live(make_layout(bot), refresh_per_second=4, screen=True) as live:
        while True:
            live.update(make_layout(bot))
            await asyncio.sleep(0.1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass