"""
PROFIT-IMPROVED PAIR MEAN-REVERSION SIM (Polymarket-style YES/NO)
- Hedged pair trades (long one leg, short the other)
- Logit-space OLS spread + Z-score
- Mean-reversion filter via half-life estimate
- Kendall tau filter
- Cooldowns + max exposure
- Time stop, dollar stop-loss, take-profit
- Conservative marking (bid for YES exits, 1-ask for NO exits)
- Light slippage model (you can set to 0.0)

NOTE: This is still a SIM. “Profit” depends on data quality + market structure.
This version fixes several logic issues in the original and adds real risk controls.
"""

import time
import zmq
import numpy as np
import zmq.asyncio
import asyncio
import json
import os
from collections import deque
from itertools import combinations
from scipy.stats import kendalltau

# --- RICH IMPORTS ---
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich import box

# --- CONFIGURATION ---
try:
    from market_websocket import ASSET_ID_FILE
except ImportError:
    ASSET_ID_FILE = "asset_ids.json"

ZMQ_ADDR = "tcp://127.0.0.1:5567"
MAP_RELOAD_COOLDOWN = 5

# Strategy controls
MIN_HISTORY = 120
WINDOW = 200                 # rolling window for pair stats (clamped by available history)
MAX_HISTORY_STORE = 1200     # keep last N ticks per asset
TAU_MIN = 0.60               # monotonic correlation filter
MIN_SPREAD_STD = 0.12        # in logit space (too small => noisy z)
ENTRY_Z_BASE = 2.6
EXIT_Z = 2.0                # exit when z comes back near 0
MAX_HALF_LIFE = 180.0        # seconds-equivalent in "ticks"; approximate filter
MIN_HALF_LIFE = 4.0

# Execution / sim realism
SLIPPAGE_BPS = 10            # 10 bps = 0.10% (set 0 to disable)
SLIPPAGE = SLIPPAGE_BPS / 10000.0

# --- DATA STRUCTURES ---
tick_dtype = np.dtype([
    ("ts_ms", np.int64),
    ("bid", np.float32),
    ("ask", np.float32),
])

# --- STATE MANAGEMENT ---
class AppState:
    def __init__(self):
        self.clob_question_map = {}
        self.valid_clobs = []
        self.assets_ticks = {}             # aid -> np array of tick_dtype
        self.last_map_reload_ts = 0
        self.logs = deque(maxlen=12)
        self.scan_count = 0
        self.active_pairs_count = 0
        self.last_pair_debug = ""          # optional UI debug line

state = AppState()

# --- UTILS ---
def load_asset_map():
    try:
        if not os.path.exists(ASSET_ID_FILE):
            return
        with open(ASSET_ID_FILE, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    cid = obj.get("clob_token_id")
                    q = obj.get("question", "")
                    if cid:
                        state.clob_question_map[cid] = q
                        if cid not in state.valid_clobs:
                            state.valid_clobs.append(cid)
                except json.JSONDecodeError:
                    continue
        state.last_map_reload_ts = time.time()
    except Exception:
        pass

def get_asset_name(asset_id: str) -> str:
    if asset_id in state.clob_question_map:
        return state.clob_question_map[asset_id]
    if time.time() - state.last_map_reload_ts > MAP_RELOAD_COOLDOWN:
        load_asset_map()
        if asset_id in state.clob_question_map:
            return state.clob_question_map[asset_id]
    return asset_id

def clip_prob(p):
    return float(np.clip(p, 1e-4, 1 - 1e-4))

def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def mid_series(arr):
    # arr: structured ticks
    return (arr["bid"].astype(np.float64) + arr["ask"].astype(np.float64)) * 0.5

def safe_last_mid(arr):
    if len(arr) == 0:
        return np.nan
    return float((arr["bid"][-1] + arr["ask"][-1]) * 0.5)

def estimate_half_life(spread: np.ndarray) -> float:
    """
    Approximate mean reversion half-life from AR(1) on spread:
    ds = a + b*s_{t-1}; b < 0 indicates mean reversion
    half_life ~ -ln(2) / b
    Returned in "ticks" units (not seconds).
    """
    if spread.size < 30:
        return np.inf
    s = spread.astype(np.float64)
    ds = s[1:] - s[:-1]
    lag = s[:-1]
    var = np.var(lag)
    if var < 1e-12:
        return np.inf
    b = np.cov(lag, ds, ddof=0)[0, 1] / (var + 1e-12)
    # mean reversion requires b < 0
    if not np.isfinite(b) or b >= -1e-6:
        return np.inf
    hl = -np.log(2.0) / b
    if not np.isfinite(hl) or hl <= 0:
        return np.inf
    return float(hl)

def compute_pair_signal(a_mid: np.ndarray, b_mid: np.ndarray):
    """
    Build logit-space linear relationship:
      y = alpha + beta*x + noise
      spread = y - (alpha + beta*x)
      z = (spread[-1] - mean)/std
    Plus filters: Kendall tau, spread std, half-life.
    """
    n = min(a_mid.size, b_mid.size)
    if n < MIN_HISTORY:
        return None

    w = min(WINDOW, n)
    a = a_mid[-w:]
    b = b_mid[-w:]

    # Kendall tau on levels (you can switch to returns if you want)
    tau, _ = kendalltau(a, b)
    if not np.isfinite(tau) or abs(tau) < TAU_MIN:
        return None

    y = logit(a)
    x = logit(b)

    x_var = np.var(x)
    if x_var < 1e-10:
        return None

    beta = np.cov(x, y, ddof=0)[0, 1] / (x_var + 1e-12)
    alpha = float(np.mean(y) - beta * np.mean(x))

    spread = y - (alpha + beta * x)
    s_std = float(np.std(spread))
    if not np.isfinite(s_std) or s_std < MIN_SPREAD_STD:
        return None

    s_mean = float(np.mean(spread))
    z = float((spread[-1] - s_mean) / (s_std + 1e-12))

    hl = estimate_half_life(spread)
    if not (MIN_HALF_LIFE <= hl <= MAX_HALF_LIFE):
        return None

    # adaptive entry threshold: higher when spread is jumpier (noise)
    # (using std of delta spread as a noise proxy)
    noise = float(np.std(np.diff(spread))) if spread.size > 10 else 0.0
    z_entry = ENTRY_Z_BASE + min(0.8, max(0.0, (noise / (s_std + 1e-9)) - 0.6))

    return {
        "z": z,
        "alpha": alpha,
        "beta": float(beta),
        "spread_mean": s_mean,
        "spread_std": s_std,
        "tau": float(tau),
        "half_life": hl,
        "z_entry": float(z_entry),
    }

# --- TRADING ENGINE (HEDGED PAIRS) ---
def exec_buy_yes(ask_yes: float) -> float:
    # pay ask with slippage
    px = clip_prob(ask_yes * (1.0 + SLIPPAGE))
    return px

def exec_sell_yes(bid_yes: float) -> float:
    # receive bid minus slippage
    px = clip_prob(bid_yes * (1.0 - SLIPPAGE))
    return px

def exec_buy_no_from_yes_bid(bid_yes: float) -> float:
    # simplified: NO entry price approx (1 - bid_yes), with slippage
    px = clip_prob((1.0 - bid_yes) * (1.0 + SLIPPAGE))
    return px

def exec_sell_no_from_yes_ask(ask_yes: float) -> float:
    # simplified: NO exit value approx (1 - ask_yes), with slippage
    px = clip_prob((1.0 - ask_yes) * (1.0 - SLIPPAGE))
    return px

class PairPosition:
    """
    direction:
      +1 => Long A / Short B  (expect z to rise toward 0)
      -1 => Short A / Long B  (expect z to fall toward 0)
    """
    def __init__(self, a_id, b_id, direction, entry_z, alpha, beta, spread_std, dollars_per_leg):
        self.a_id = a_id
        self.b_id = b_id
        self.direction = int(direction)
        self.entry_z = float(entry_z)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.spread_std = float(spread_std)

        self.open_ts = time.time()
        self.dollars_per_leg = float(dollars_per_leg)

        # legs: (side, entry_price, qty)
        # side in {"YES","NO"}
        self.a_side = None
        self.b_side = None
        self.a_entry = 0.0
        self.b_entry = 0.0
        self.a_qty = 0.0
        self.b_qty = 0.0

    def key(self):
        # normalized pair key
        return tuple(sorted((self.a_id, self.b_id)))

class SimulatedTrader:
    def __init__(self):
        self.positions = []                 # list[PairPosition]
        self.starting_balance = 1000.0
        self.balance = self.starting_balance
        self.realized_pnl = 0.0

        # risk controls
        self.dollars_per_leg = 20.0
        self.max_positions = 10
        self.max_positions_per_asset = 4
        self.pair_cooldown_s = 45.0
        self.last_trade_time = {}           # pair_key -> ts

        # exits
        self.max_hold_s = 240.0             # time stop
        self.stop_loss_per_pair = -4.0      # dollars
        self.take_profit_per_pair = 3.5     # dollars

    def log(self, message, style="white"):
        ts = time.strftime("%H:%M:%S", time.localtime())
        state.logs.append((ts, message, style))

    def _asset_exposure_count(self, aid):
        c = 0
        for p in self.positions:
            if p.a_id == aid or p.b_id == aid:
                c += 1
        return c

    def _has_pair_open(self, a, b):
        k = tuple(sorted((a, b)))
        for p in self.positions:
            if p.key() == k:
                return True
        return False

    def _pair_on_cooldown(self, a, b):
        k = tuple(sorted((a, b)))
        last = self.last_trade_time.get(k, 0.0)
        return (time.time() - last) < self.pair_cooldown_s

    def open_pair(self, a_id, b_id, direction, entry_z, sig, snapshot):
        """
        direction +1 => Long A / Short B
        direction -1 => Short A / Long B
        We implement "short" via buying NO on that asset.
        """
        if len(self.positions) >= self.max_positions:
            return
        if self._has_pair_open(a_id, b_id):
            return
        if self._pair_on_cooldown(a_id, b_id):
            return
        if self._asset_exposure_count(a_id) >= self.max_positions_per_asset:
            return
        if self._asset_exposure_count(b_id) >= self.max_positions_per_asset:
            return

        # need current quotes
        if a_id not in snapshot or b_id not in snapshot:
            return
        a_ticks = snapshot[a_id]
        b_ticks = snapshot[b_id]
        if len(a_ticks) == 0 or len(b_ticks) == 0:
            return

        a_bid = float(a_ticks["bid"][-1]); a_ask = float(a_ticks["ask"][-1])
        b_bid = float(b_ticks["bid"][-1]); b_ask = float(b_ticks["ask"][-1])

        # skip illiquid / too wide / near extremes
        if not (0.02 < a_bid < 0.98 and 0.02 < a_ask < 0.98 and 0.02 < b_bid < 0.98 and 0.02 < b_ask < 0.98):
            return
        if (a_ask - a_bid) > 0.06 or (b_ask - b_bid) > 0.06:
            return

        dollars = self.dollars_per_leg
        total_need = 2.0 * dollars
        if self.balance < total_need:
            return

        pos = PairPosition(
            a_id=a_id,
            b_id=b_id,
            direction=direction,
            entry_z=entry_z,
            alpha=sig["alpha"],
            beta=sig["beta"],
            spread_std=sig["spread_std"],
            dollars_per_leg=dollars
        )

        # Decide sides
        if direction == +1:
            # Long A (buy YES A), Short B (buy NO B)
            pos.a_side = "YES"
            pos.b_side = "NO"
            pos.a_entry = exec_buy_yes(a_ask)
            pos.b_entry = exec_buy_no_from_yes_bid(b_bid)
        else:
            # Short A (buy NO A), Long B (buy YES B)
            pos.a_side = "NO"
            pos.b_side = "YES"
            pos.a_entry = exec_buy_no_from_yes_bid(a_bid)
            pos.b_entry = exec_buy_yes(b_ask)

        # sanity
        if not (0.01 < pos.a_entry < 0.99 and 0.01 < pos.b_entry < 0.99):
            return

        pos.a_qty = dollars / pos.a_entry
        pos.b_qty = dollars / pos.b_entry

        self.positions.append(pos)
        self.balance -= total_need

        k = pos.key()
        self.last_trade_time[k] = time.time()

        a_name = get_asset_name(a_id)
        b_name = get_asset_name(b_id)
        color = "green" if direction == +1 else "red"
        self.log(
            f"OPEN PAIR ({pos.a_side} {a_name}) vs ({pos.b_side} {b_name}) | z={entry_z:+.2f} tau={sig['tau']:+.2f} hl={sig['half_life']:.0f}",
            style=f"bold {color}"
        )

    def _mark_leg_value(self, aid, side, qty, snapshot):
        """
        Conservative marking:
          YES -> value at bid (as if we sell)
          NO  -> value at 1-ask (as if we sell NO)
        """
        if aid not in snapshot or len(snapshot[aid]) == 0:
            return 0.0
        t = snapshot[aid]
        bid = float(t["bid"][-1])
        ask = float(t["ask"][-1])
        if side == "YES":
            px = clip_prob(bid)
        else:
            px = clip_prob(1.0 - ask)
        return qty * px

    def _close_pair(self, pos: PairPosition, snapshot, reason: str):
        # compute exit prices with slippage
        a_ticks = snapshot.get(pos.a_id)
        b_ticks = snapshot.get(pos.b_id)
        if a_ticks is None or b_ticks is None or len(a_ticks) == 0 or len(b_ticks) == 0:
            return

        a_bid = float(a_ticks["bid"][-1]); a_ask = float(a_ticks["ask"][-1])
        b_bid = float(b_ticks["bid"][-1]); b_ask = float(b_ticks["ask"][-1])

        if pos.a_side == "YES":
            a_exit_px = exec_sell_yes(a_bid)
        else:
            a_exit_px = exec_sell_no_from_yes_ask(a_ask)

        if pos.b_side == "YES":
            b_exit_px = exec_sell_yes(b_bid)
        else:
            b_exit_px = exec_sell_no_from_yes_ask(b_ask)

        a_rev = pos.a_qty * a_exit_px
        b_rev = pos.b_qty * b_exit_px
        revenue = a_rev + b_rev

        cost = (pos.a_qty * pos.a_entry) + (pos.b_qty * pos.b_entry)
        pnl = revenue - cost

        self.balance += revenue
        self.realized_pnl += pnl

        try:
            self.positions.remove(pos)
        except ValueError:
            pass

        a_name = get_asset_name(pos.a_id)
        b_name = get_asset_name(pos.b_id)
        c = "green" if pnl >= 0 else "red"
        self.log(
            f"CLOSE PAIR {a_name} vs {b_name} | PnL ${pnl:+.2f} [{reason}]",
            style=f"bold {c}"
        )

    def update_positions(self, snapshot, pair_signals_cache):
        """
        Exit logic:
          - Z mean reversion back near 0
          - Time stop
          - Dollar stop-loss / take-profit
        """
        now = time.time()

        for pos in self.positions[:]:
            # mark-to-market pair value
            val_a = self._mark_leg_value(pos.a_id, pos.a_side, pos.a_qty, snapshot)
            val_b = self._mark_leg_value(pos.b_id, pos.b_side, pos.b_qty, snapshot)
            curr_value = val_a + val_b
            cost = (pos.a_qty * pos.a_entry) + (pos.b_qty * pos.b_entry)
            pnl = curr_value - cost

            # time stop
            if (now - pos.open_ts) > self.max_hold_s:
                self._close_pair(pos, snapshot, reason="TIME STOP")
                continue

            # pnl stops
            if pnl <= self.stop_loss_per_pair:
                self._close_pair(pos, snapshot, reason="STOP LOSS")
                continue
            if pnl >= self.take_profit_per_pair:
                self._close_pair(pos, snapshot, reason="TAKE PROFIT")
                continue

            # z reversion exit: recompute current z for this pair
            k = pos.key()
            sig = pair_signals_cache.get(k)
            if sig is None:
                # try compute from latest data if not cached
                if pos.a_id in snapshot and pos.b_id in snapshot:
                    a_mid = mid_series(snapshot[pos.a_id])
                    b_mid = mid_series(snapshot[pos.b_id])
                    sig = compute_pair_signal(a_mid, b_mid)
                if sig is None:
                    continue

            z = sig["z"]

            # If we entered direction +1 on negative z, we want z to rise toward 0
            # If direction -1 on positive z, we want z to fall toward 0
            if pos.direction == +1:
                if z >= +EXIT_Z:
                    self._close_pair(pos, snapshot, reason=f"Z EXIT (z={z:+.2f})")
            else:
                if z <= -EXIT_Z:
                    self._close_pair(pos, snapshot, reason=f"Z EXIT (z={z:+.2f})")

trader = SimulatedTrader()

# --- RICH UI ---
def make_header() -> Panel:
    equity = 0.0
    snapshot = state.assets_ticks

    for p in trader.positions:
        equity += trader._mark_leg_value(p.a_id, p.a_side, p.a_qty, snapshot)
        equity += trader._mark_leg_value(p.b_id, p.b_side, p.b_qty, snapshot)

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
        f"💎 Net Worth: [bold {pnl_color}]${total_val:.2f}[/] (PnL: {total_pnl:+.2f})",
    )
    subtitle = f"pairs scanned: {state.active_pairs_count} | ticks: {len(state.assets_ticks)} | {state.last_pair_debug}"
    return Panel(grid, style="white on black", title="🤖 ALGOBET DASHBOARD (HEDGED PAIRS)", subtitle=subtitle, border_style="blue")

def make_position_table() -> Panel:
    table = Table(expand=True, border_style="dim", box=box.SIMPLE)
    table.add_column("Dir", width=3, justify="center")
    table.add_column("Leg A", ratio=1)
    table.add_column("Leg B", ratio=1)
    table.add_column("Entry z", justify="right", width=7)
    table.add_column("Now z", justify="right", width=7)
    table.add_column("Age(s)", justify="right", width=7)
    table.add_column("PnL($)", justify="right", width=9)

    # precompute current pair z if possible
    snapshot = state.assets_ticks
    rows = []
    for p in trader.positions:
        a_name = get_asset_name(p.a_id)
        b_name = get_asset_name(p.b_id)
        if len(a_name) > 42: a_name = a_name[:39] + "..."
        if len(b_name) > 42: b_name = b_name[:39] + "..."

        # mark pnl
        val_a = trader._mark_leg_value(p.a_id, p.a_side, p.a_qty, snapshot)
        val_b = trader._mark_leg_value(p.b_id, p.b_side, p.b_qty, snapshot)
        curr_value = val_a + val_b
        cost = (p.a_qty * p.a_entry) + (p.b_qty * p.b_entry)
        pnl = curr_value - cost

        # now z
        now_z = 0.0
        if p.a_id in snapshot and p.b_id in snapshot:
            a_mid = mid_series(snapshot[p.a_id])
            b_mid = mid_series(snapshot[p.b_id])
            sig = compute_pair_signal(a_mid, b_mid)
            if sig is not None:
                now_z = sig["z"]

        age = time.time() - p.open_ts
        rows.append((p, a_name, b_name, now_z, age, pnl))

    rows.sort(key=lambda x: x[-1], reverse=True)

    if not rows:
        table.add_row("-", "[italic yellow]Waiting for signals...[/]", "-", "-", "-", "-", "-")
    else:
        for p, a_name, b_name, now_z, age, pnl in rows:
            dir_txt = "↑" if p.direction == +1 else "↓"
            dir_style = "bold green" if p.direction == +1 else "bold red"
            pnl_style = "green" if pnl >= 0 else "red"

            legA = f"{p.a_side} {a_name} @ {p.a_entry:.3f}"
            legB = f"{p.b_side} {b_name} @ {p.b_entry:.3f}"

            table.add_row(
                f"[{dir_style}]{dir_txt}[/]",
                legA,
                legB,
                f"{p.entry_z:+.2f}",
                f"{now_z:+.2f}",
                f"{age:,.0f}",
                f"[{pnl_style}]{pnl:+.2f}[/]"
            )

    return Panel(table, title=f"📊 Open Pair Positions ({len(trader.positions)})", border_style="blue")

def make_log_panel() -> Panel:
    log_content = Table.grid(padding=(0, 1))
    log_content.add_column(justify="left", style="dim cyan", width=10)
    log_content.add_column(justify="left")

    for ts, msg, style in state.logs:
        log_content.add_row(ts, f"[{style}]{msg}[/]")

    return Panel(log_content, title="📝 Activity Log", height=12, border_style="blue")

def generate_layout() -> Layout:
    layout = Layout()
    layout.split(
        Layout(name="header", size=4),
        Layout(name="main"),
        Layout(name="footer", size=14),
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
            if arr.size < MIN_HISTORY:
                continue

            # Keep last MAX_HISTORY_STORE ticks for stability + speed
            if arr.size > MAX_HISTORY_STORE:
                arr = arr[-MAX_HISTORY_STORE:]

            # Copy to detach from message buffer
            state.assets_ticks[aid] = arr.copy()
        except Exception:
            await asyncio.sleep(0.05)

def run_strategy_logic():
    # snapshot current ticks
    snapshot = {k: v.copy() for k, v in state.assets_ticks.items()}
    aids = list(snapshot.keys())

    # count pairs
    state.active_pairs_count = (len(aids) * (len(aids) - 1)) // 2

    # cache pair signals for exits + to avoid recompute
    pair_sig_cache = {}

    # Update exits first (needs signals too; we fill cache lazily)
    # But we compute signals below and pass cache in.
    # We'll do a quick first pass fill for pairs we already hold:
    held_pairs = set(p.key() for p in trader.positions)
    for (a, b) in held_pairs:
        if a in snapshot and b in snapshot:
            sig = compute_pair_signal(mid_series(snapshot[a]), mid_series(snapshot[b]))
            if sig is not None:
                pair_sig_cache[(a, b)] = sig

    trader.update_positions(snapshot, pair_sig_cache)

    if len(aids) < 2:
        return

    # Build mid series dict once
    mids = {}
    for aid in aids:
        arr = snapshot[aid]
        if arr.size >= MIN_HISTORY:
            mids[aid] = mid_series(arr)

    # Scan all pairs
    last_debug = ""
    for aid1, aid2 in combinations(mids.keys(), 2):
        # skip if already open / on cooldown
        if trader._has_pair_open(aid1, aid2) or trader._pair_on_cooldown(aid1, aid2):
            continue

        a_mid = mids[aid1]
        b_mid = mids[aid2]
        sig = compute_pair_signal(a_mid, b_mid)
        if sig is None:
            continue

        k = tuple(sorted((aid1, aid2)))
        pair_sig_cache[k] = sig

        z = sig["z"]
        z_entry = sig["z_entry"]

        # Determine which is A vs B in the signal? compute_pair_signal treats inputs as (a,b)
        # We'll use it as (aid1, aid2) for decision.
        # If z is very negative => a is cheap vs b => long a / short b (direction +1)
        # If z is very positive => a is expensive vs b => short a / long b (direction -1)
        if z <= -z_entry:
            trader.open_pair(aid1, aid2, direction=+1, entry_z=z, sig=sig, snapshot=snapshot)
            last_debug = f"last: z={z:+.2f} entry={z_entry:.2f} tau={sig['tau']:+.2f}"
        elif z >= +z_entry:
            trader.open_pair(aid1, aid2, direction=-1, entry_z=z, sig=sig, snapshot=snapshot)
            last_debug = f"last: z={z:+.2f} entry={z_entry:.2f} tau={sig['tau']:+.2f}"

    state.last_pair_debug = last_debug
    state.scan_count += 1

async def strategy_loop():
    # warm up
    while True:
        await asyncio.sleep(0.25)  # 4 Hz
        if len(state.assets_ticks) < 2:
            continue
        await asyncio.to_thread(run_strategy_logic)

async def main():
    ctx = zmq.asyncio.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(ZMQ_ADDR)
    sub.subscribe(b"")

    load_asset_map()

    collector_task = asyncio.create_task(collect_ticks(sub))
    strategy_task = asyncio.create_task(strategy_loop())

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
