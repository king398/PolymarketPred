from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich import box
from rich.text import Text
import time
from heston_model import FastHestonModel

LIQUIDATION_THRESH = 0.10

def make_layout(trader, state_ticks):
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=20)
    )

    # --- PORTFOLIO CALCULATIONS ---
    invested_cash = sum(p.cost_basis for p in trader.positions)
    positions_value = 0.0

    # Calculate Market Value of Open Positions
    for p in trader.positions:
        ticks = state_ticks.get(p.asset_id)
        if ticks is not None and len(ticks) > 0:
            current_bid = float(ticks['bid'][-1])
            current_ask = float(ticks['ask'][-1])
            # Conservative Liquidation Value
            # If YES: Sell into Bid. If NO: Buy back at Ask.
            price = current_bid if p.side == "YES" else (1.0 - current_ask)
            positions_value += p.size_qty * price
        else:
            # Fallback to cost if no live data
            positions_value += p.cost_basis

    total_account_value = trader.balance + positions_value
    total_pnl = total_account_value - 1000.0  # Assumes 1000 start

    # Colors
    pnl_color = "green" if total_pnl >= 0 else "red"
    realized_color = "green" if trader.realized_pnl >= 0 else "red"

    # --- HEADER DISPLAY ---
    stats = Table.grid(expand=True)
    stats.add_column(justify="center", ratio=1)
    stats.add_column(justify="center", ratio=1)
    stats.add_column(justify="center", ratio=1)

    spot_txt = " | ".join([f"{k} ${v:3f}" for k, v in trader.dm.spot_prices.items()])

    stats.add_row(
        f"[bold]Total Value: ${total_account_value:.2f}[/]  |  [bold {pnl_color}]Net PnL: ${total_pnl:+.2f}[/]",
        f"Cash: ${trader.balance:.2f}  |  [yellow]Inv: ${invested_cash:.2f}[/]  |  [bold {realized_color}]Realized: ${trader.realized_pnl:+.2f}[/]",
        f"[dim]{spot_txt}[/]"
    )
    layout["header"].update(Panel(stats, style="white on black"))

    # --- MAIN TABLE ---
    table = Table(title="LIVE MARKET MONITOR & PORTFOLIO", expand=True, border_style="blue", box=box.SIMPLE)
    table.add_column("Question", ratio=3, overflow="fold")
    table.add_column("Time", justify="center", style="dim")
    table.add_column("Bid", justify="right", style="green")
    table.add_column("Ask", justify="right", style="red")
    table.add_column("Fair", justify="right", style="cyan")
    table.add_column("Edge", justify="right")
    table.add_column("Pos (Shares)", justify="center")
    table.add_column("Avg Entry", justify="right")
    table.add_column("Unreal PnL", justify="right")

    rows = get_unified_rows(trader, state_ticks)
    for r in rows:
        table.add_row(
            r['question'],
            r['time'],
            f"{r['bid']:.3f}",
            f"{r['ask']:.3f}",
            f"{r['fair']:.3f}" if r['fair'] > 0 else "-",
            r['edge_str'],
            r['pos'],
            r['entry'],
            r['pnl'],
            style=r['style']
        )

    layout["main"].update(table)

    # --- FOOTER LOGS ---
    log_text = Text()
    for ts, msg, style in list(trader.logs):
        log_text.append(f"[{ts}] ", style="dim")
        log_text.append(f"{msg}\n", style=style)
        log_text.append("-" * 40 + "\n", style="dim")

    layout["footer"].update(Panel(log_text, title="Detailed Activity Log", border_style="dim"))

    return layout


def get_unified_rows(trader, state_ticks):
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
        rem_ms = meta['end_ts_ms'] - now_ms
        pos = pos_map.get(aid)

        if rem_ms < 0 or pos is None:
            continue

        model_p = 0.0
        edge_val = 0.0
        if params and rem_ms > 0:
            T_yrs = rem_ms / (1000 * 365 * 24 * 3600.0)
            try:
                model_p = FastHestonModel.price_binary_call(spot, strike, T_yrs, meta['initial_T'], params)
                edge_yes = model_p - mkt_ask
                edge_no = mkt_bid - model_p
                edge_val = edge_yes if edge_yes > edge_no else edge_no
            except:
                pass

        pos_str = "-"
        entry_str = "-"
        pnl_str = "-"
        row_style = "white"

        if pos:
            row_style = "bold yellow" if pos.is_accumulating else "bold white"
            # Calculate Unrealized PnL based on Mark Price
            mark = mkt_bid if pos.side == "YES" else (1.0 - mkt_ask)
            val = pos.size_qty * mark
            unreal_pnl = val - pos.cost_basis
            p_color = "green" if unreal_pnl >= 0 else "red"
            pnl_str = f"[{p_color}]{unreal_pnl:+.2f}[/]"
            side_c = "green" if pos.side == "YES" else "red"
            pos_str = f"[{side_c}]{pos.side}[/] ({int(pos.size_qty)})"
            entry_str = f"{pos.avg_entry_px:.3f}"

        # Check cooldown state
        is_cooling = aid in trader.cooldowns

        if pos is None and not is_cooling and abs(edge_val) < 0.005 and rem_ms > 3600000:
            continue

        e_color = "green" if edge_val > 0 else "red"
        edge_str = f"[{e_color}]{edge_val:+.3f}[/]" if model_p > 0 else "-"

        rows.append({
            "question": meta['question'],
            "time": format_time_left(rem_ms, meta['initial_duration_ms']),
            "time_ms": rem_ms,
            "bid": mkt_bid,
            "ask": mkt_ask,
            "fair": model_p,
            "edge": edge_val,
            "edge_str": edge_str,
            "pos": pos_str,
            "entry": entry_str,
            "pnl": pnl_str,
            "style": row_style,
            "has_pos": pos is not None
        })

    rows.sort(key=lambda x: x['time_ms'])
    return rows


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
