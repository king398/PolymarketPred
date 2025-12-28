import pandas as pd
from datasets import Dataset
import numpy as np
import plotly.graph_objects as go
from scipy.stats import kendalltau, norm

top_k = 50
eps = 1e-6

df = pd.read_csv("/home/mithil/PycharmProjects/PolymarketPred/data/market_windows.csv")
df["end_ts_utc"] = pd.to_datetime(df["end_ts_utc"], utc=True)
df["start_ts_utc"] = pd.to_datetime(df["start_ts_utc"], utc=True)
df = df.sort_values(by="end_ts_utc").reset_index(drop=True)
df = df[
    (df['duration_hours'] >= 24) &
    (df['status'] == "ok")
    ].reset_index(drop=True)
df.to_csv("/home/mithil/PycharmProjects/PolymarketPred/data/market_windows_filtered.csv", index=False)
# filter for those in between a certain duration range
start_et = pd.Timestamp("2025-11-01 00:00", tz="US/Eastern")
end_et = start_et + pd.Timedelta(days=1)

# Convert ET → UTC for comparison
start_utc = start_et.tz_convert("UTC")
end_utc = end_et.tz_convert("UTC")
df_range = df[
    (df["start_ts_utc"] <= start_utc) &
    (df["end_ts_utc"] >= end_utc)
    ].copy()

parquets = [f"/home/mithil/PycharmProjects/PolymarketPred/data/polymarket_minute_parquet/{uuid}.parquet" for uuid in
            df_range['uuid'].tolist()]
dataset = Dataset.from_parquet(parquets).to_pandas()
dataset["timestamp_et"] = dataset["timestamp"].dt.tz_convert("US/Eastern")

target_date = pd.Timestamp("2025-11-01").date()

dataset_day = dataset[
    dataset["timestamp_et"].dt.date == target_date
    ].copy().reset_index(drop=True)
dataset_day["timestamp_et"] = dataset_day["timestamp_et"].dt.floor("min")
price_pivot = dataset_day.pivot_table(
    index="timestamp_et",
    columns="question",
    values="price",
    aggfunc="last"
).sort_index()
print(f"Price pivot shape: {price_pivot.shape}")
col_std = price_pivot.std(skipna=True)
col_mean = price_pivot.mean(skipna=True)
keep_cols = col_std[
    (col_std > 1e-2) &
    (col_mean > 0.03) &
    (col_mean < 0.96)
    ].index

price_pivot = price_pivot[keep_cols]
print(f"Filtered price pivot shape: {price_pivot.shape}")

# Compute Kendall tau
kendall_corr = price_pivot.corr(method="kendall")

# Remove self-correlations
np.fill_diagonal(kendall_corr.values, np.nan)
mask = np.triu(np.ones(kendall_corr.shape), k=1).astype(bool)

top_pairs = (
    kendall_corr
    .where(mask)
    .stack()
    .rename("kendall_tau")
    .dropna()
    .sort_values(key=lambda s: s.abs(), ascending=False)
    .head(top_k)
)

price_pivot.to_csv("/home/mithil/PycharmProjects/PolymarketPred/data/fair_price_self.csv")
N = min(top_k, len(top_pairs))

pairs = list(top_pairs.index[:N])
taus = top_pairs.values[:N]

fig = go.Figure()
print(f"\nTop {top_k} |Kendall τ| correlated market pairs:\n")
for i, ((q1, q2), tau) in enumerate(top_pairs.items(), start=1):
    sign = "+" if tau > 0 else "-"
    print(f"{i:02d}. τ={tau:+.4f} | {q1}  ↔  {q2}")

# Pre-add all traces (2 per pair), then toggle visibility via dropdown
visibility = []
for i, ((q1, q2), tau) in enumerate(zip(pairs, taus)):
    pair_df = price_pivot[[q1, q2]].copy()

    # normalize for comparability
    pair_df = pair_df

    fig.add_trace(go.Scatter(
        x=pair_df.index, y=pair_df[q1],
        mode="lines", name=q1[:80],
        visible=(i == 0),
        hovertemplate="Time: %{x}<br>Norm: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=pair_df.index, y=pair_df[q2],
        mode="lines", name=q2[:80],
        visible=(i == 0),
        hovertemplate="Time: %{x}<br>Norm: %{y:.4f}<extra></extra>",
    ))

# Build dropdown buttons
buttons = []
for i, ((q1, q2), tau) in enumerate(zip(pairs, taus)):
    vis = [False] * (2 * N)
    vis[2 * i] = True
    vis[2 * i + 1] = True

    label = f"{i + 1:02d} | τ={tau:.3f} | {q1[:35]}  ↔  {q2[:35]}"
    buttons.append(dict(
        label=label,
        method="update",
        args=[
            {"visible": vis},
            {"title": f"Top Kendall τ Pair #{i + 1} (normalized)<br>τ = {tau:.4f}"}
        ],
    ))

fig.update_layout(
    title=f"Top Kendall τ Pair #1 (normalized)<br>τ = {taus[0]:.4f}",
    template="plotly_dark",
    height=650,
    hovermode="x unified",
    updatemenus=[dict(
        buttons=buttons,
        direction="down",
        x=0.0,
        y=1.15,
        xanchor="left",
        yanchor="top",
    )],
    margin=dict(t=120),
    legend=dict(orientation="h", y=1.02, x=0),
)

fig.update_xaxes(title="Time (US/Eastern)", rangeslider=dict(visible=True))
fig.update_yaxes(title="Normalized price (p / p0)")

fig.show()
def empirical_cdf_value(sample: np.ndarray, x0: float) -> float:
    """Empirical CDF value F(x0) from sample."""
    sample = np.asarray(sample, dtype=float)
    return float(np.clip(np.mean(sample <= x0), eps, 1 - eps))


def pit_rank(x: np.ndarray) -> np.ndarray:
    """Empirical PIT using ranks -> values in (0,1)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    ranks = pd.Series(x).rank(method="average").to_numpy()
    u = (ranks - 0.5) / n
    return np.clip(u, eps, 1 - eps)


def fit_copula_params(u: np.ndarray, v: np.ndarray, family: str = "gaussian") -> dict:
    tau, _ = kendalltau(u, v)
    tau = float(np.clip(tau, -0.999, 0.999))
    fam = family.lower()

    if fam == "gaussian":
        rho = float(np.clip(np.sin(np.pi * tau / 2.0), -0.999, 0.999))
        return {"tau": tau, "rho": rho}

    if fam == "clayton":
        if tau <= 0:
            raise ValueError("Clayton requires positive dependence (tau > 0).")
        theta = float(2 * tau / (1 - tau))
        return {"tau": tau, "theta": theta}

    if fam == "gumbel":
        if tau < 0:
            raise ValueError("Gumbel typically assumes tau >= 0.")
        theta = float(1.0 / (1 - tau))
        return {"tau": tau, "theta": theta}

    raise ValueError("family must be one of: gaussian, clayton, gumbel")


def sample_u_given_v_gaussian(v0: float, rho: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z2 = norm.ppf(v0)
    z1 = rng.normal(loc=rho * z2, scale=np.sqrt(1 - rho ** 2), size=n)
    return np.clip(norm.cdf(z1), eps, 1 - eps)


def empirical_ppf(sample: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Empirical inverse CDF (quantile)."""
    sample = np.asarray(sample, dtype=float)
    q = np.asarray(q, dtype=float)
    return np.quantile(sample, q, method="linear")


def fair_price(
        A_series: pd.Series,
        B_series: pd.Series,
        pB_new: float,
        family: str,
        n_mc: int = 5000,
        seed: int = 42,
):
    u_A = pit_rank(A_series.values)
    u_B = pit_rank(B_series.values)

    try:
        params = fit_copula_params(u_A, u_B, family)
    except ValueError:
        return None

    v0 = empirical_cdf_value(B_series.values, pB_new)
    u_samp = sample_u_given_v_gaussian(v0, params["rho"], n_mc, seed)

    A_samp = empirical_ppf(A_series.values, u_samp)

    return {
        "tau": params["tau"],
        "copula_param": params.get("rho", params.get("theta", np.nan)),
        "pB_new": float(pB_new),
        "fair_mean": float(np.mean(A_samp)),
        "fair_median": float(np.median(A_samp)),
        "fair_p10": float(np.quantile(A_samp, 0.10)),
        "fair_p90": float(np.quantile(A_samp, 0.90)),
        "n_obs": int(len(u_A)),
    }


H = 10  # holding period in minutes
threshold = 0.03
min_hist = 60
n_mc = 10_000

sample_pair = price_pivot[list(top_pairs.index[0])]
A, B = top_pairs.index[1]

price_A = price_pivot[A]
price_B = price_pivot[B]
df_ab = pd.concat([price_A.rename("A"), price_B.rename("B")], axis=1).dropna()

trades = []

for t_idx in range(min_hist, len(df_ab) - H):
    hist = df_ab.iloc[:t_idx + 1]

    pA_now = hist["A"].iloc[-1]
    pB_now = hist["B"].iloc[-1]  # conditioning value

    result = fair_price(
        A_series=hist["A"],
        B_series=hist["B"],
        pB_new=pB_now,
        family="gaussian",
        n_mc=n_mc,
        seed=42,
    )
    if result is None:
        continue

    fair = result["fair_mean"]
    mispricing = fair - pA_now

    # Decide trade direction
    if mispricing > threshold:
        side = "YES"  # YES undervalued
        s = +1
    elif mispricing < -threshold:
        side = "NO"  # YES overvalued => buy NO
        s = -1
    else:
        continue  # no trade

    # Exit after H minutes
    t_enter = hist.index[-1]
    pA_exit = df_ab["A"].iloc[t_idx + H]
    t_exit = df_ab.index[t_idx + H]

    # Profit per $1 notional (ignoring fees/slippage)
    profit = s * (pA_exit - pA_now)

    trades.append({
        "t_enter": t_enter,
        "t_exit": t_exit,
        "side": side,
        "pA_enter": float(pA_now),
        "pA_exit": float(pA_exit),
        "fair": float(fair),
        "mispricing": float(mispricing),
        "profit_per_$1": float(profit),
        "tau": float(result["tau"]),
        "param": float(result["copula_param"]),
        "n_obs": int(result["n_obs"]),
    })

trades_df = pd.DataFrame(trades)

if len(trades_df) == 0:
    print("No trades triggered. Lower threshold or min_hist, or try different pairs.")
else:
    total_pnl = trades_df["profit_per_$1"].sum()
    avg_pnl = trades_df["profit_per_$1"].mean()
    win_rate = (trades_df["profit_per_$1"] > 0).mean()
    med_pnl = trades_df["profit_per_$1"].median()

    print(f"Trades: {len(trades_df)} | Hold: {H} min | Threshold: {threshold}")
    print(f"Total PnL per $1 notional: {total_pnl:+.4f}")
    print(f"Avg PnL per trade:        {avg_pnl:+.4f} (median {med_pnl:+.4f})")
    print(f"Win rate:                 {win_rate * 100:.1f}%")

    # Optional: show worst/best trades
    print("\nTop 5 best trades:")
    print(trades_df.sort_values("profit_per_$1", ascending=False).head(5)[
              ["t_enter", "side", "pA_enter", "pA_exit", "fair", "mispricing", "profit_per_$1"]
          ])

    print("\nTop 5 worst trades:")
    print(trades_df.sort_values("profit_per_$1", ascending=True).head(5)[
              ["t_enter", "side", "pA_enter", "pA_exit", "fair", "mispricing", "profit_per_$1"]
          ])

    # Save for analysis
    out_path = "/home/mithil/PycharmProjects/PolymarketPred/data/fair_price_backtest_trades.csv"
    trades_df.to_csv(out_path, index=False)
    print(f"\nSaved trades to: {out_path}")
