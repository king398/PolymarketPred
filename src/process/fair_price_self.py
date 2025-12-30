import time

import pandas as pd
from datasets import Dataset
import numpy as np
import plotly.graph_objects as go
from scipy.stats import kendalltau, norm
from scipy.stats import rankdata

top_k = 25
eps = 1e-6

df = pd.read_csv("/home/mithil/PycharmProjects/PolymarketPred/data/market_windows.csv")
df["end_ts_utc"] = pd.to_datetime(df["end_ts_utc"], utc=True)
df["start_ts_utc"] = pd.to_datetime(df["start_ts_utc"], utc=True)
df = df.sort_values(by="end_ts_utc").reset_index(drop=True)
df = df[
    (df["status"] == "ok") &
    (df["question"].str.contains("Up", na=False))
    ].reset_index(drop=True)
print(f"Filtered markets count: {len(df)}")
df.to_csv("/home/mithil/PycharmProjects/PolymarketPred/data/market_windows_filtered.csv", index=False)
# filter for those in between a certain duration range
start_et = pd.Timestamp("2025-10-05 00:00", tz="US/Eastern")
end_et = start_et + pd.Timedelta(hours=4)

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

target_date = pd.Timestamp("2025-10-05").date()

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


def empirical_cdf_value(sample: np.ndarray, x0: float) -> float:
    """Empirical CDF value F(x0) from sample."""
    sample = np.asarray(sample, dtype=float)
    return float(np.clip(np.mean(sample <= x0), eps, 1 - eps))


def pit_rank(x: np.ndarray) -> np.ndarray:
    """Empirical PIT using ranks -> values in (0,1)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    ranks = rankdata(x, method="average")
    u = (ranks - 0.5) / n
    return np.clip(u, eps, 1 - eps)


def sample_u_given_v_gaussian(v0: float, rho: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z2 = norm.ppf(v0)
    z1 = rng.normal(loc=rho * z2, scale=np.sqrt(1 - rho ** 2), size=n)
    return np.clip(norm.cdf(z1), eps, 1 - eps)

def empirical_ppf(sample: np.ndarray, q) -> float:
    """Empirical inverse CDF (quantile). Returns float."""
    sample = np.asarray(sample, dtype=float)

    q = float(q)  # ensure scalar
    if not np.isfinite(q):
        return np.nan

    q = float(np.clip(q, eps, 1 - eps))
    return float(np.quantile(sample, q, method="linear"))


def fair_price(
        A_series: pd.Series,
        B_series: pd.Series,
        pB_new: float,
):
    """
    Gaussian copula ONLY
    Deterministic, no Monte Carlo, no leakage
    """

    # PIT
    u_A = pit_rank(A_series.values)
    u_B = pit_rank(B_series.values)

    # Fit Gaussian copula via Kendall tau
    tau, _ = kendalltau(u_A, u_B)
    tau = float(np.clip(tau, -0.999, 0.999))
    rho = float(np.clip(np.sin(np.pi * tau / 2.0), -0.999, 0.999))

    # Percentile of B at current price
    v0 = empirical_cdf_value(B_series.values, pB_new)

    # Conditional mean in copula space
    z = norm.ppf(v0)
    u_cond_mean = norm.cdf(rho * z)

    # Map back to price space
    fair = float(empirical_ppf(A_series.values, u_cond_mean))

    return {
        "tau": tau,
        "copula_param": rho,
        "pB_new": float(pB_new),
        "fair_mean": fair,
        "n_obs": int(len(u_A)),
    }


H = 20  # holding period (minutes)
threshold = 0.03
min_hist = 120
eps = 1e-6

trades = []

start_profile_time = time.time()

for (A, B) in top_pairs.index:
    pair_tau = float(kendall_corr.loc[A, B])  # from the precomputed matrix
    print(f"Evaluating pair: {A} ↔ {B} | pair_tau={pair_tau:+.4f}")


    # --- Build aligned dataframe ONCE ---
    price_A = price_pivot[A]
    price_B = price_pivot[B]
    df_ab = pd.concat(
        [price_A.rename("A"), price_B.rename("B")],
        axis=1
    ).dropna()

    if len(df_ab) < min_hist + H:
        continue

    # --- NumPy views (NO pandas in loop) ---
    A_vals = df_ab["A"].values
    B_vals = df_ab["B"].values
    times = df_ab.index.values

    # --- PRECOMPUTE PIT + COPULA PARAMS (ONCE) ---

    # ================= MAIN TIME LOOP =================
    for t_idx in range(min_hist, len(df_ab) - H):

        # --- expanding history ONLY (no leakage) ---
        A_hist = df_ab["A"].iloc[:t_idx + 1]
        B_hist = df_ab["B"].iloc[:t_idx + 1]

        pA_now = A_vals[t_idx]
        pB_now = B_vals[t_idx]

        # --- Fair(A | B) ---
        res_A = fair_price(
            A_series=A_hist,
            B_series=B_hist,
            pB_new=pB_now,
        )
        if res_A is None:
            continue
        fairA = res_A["fair_mean"]
        misA = fairA - pA_now  # + => A undervalued vs fair

        # --- Fair(B | A) ---
        res_B = fair_price(
            A_series=B_hist,  # swap!
            B_series=A_hist,
            pB_new=pA_now,  # swap conditioning price!
        )
        if res_B is None:
            continue
        fairB = res_B["fair_mean"]
        misB = fairB - pB_now  # + => B undervalued vs fair

        # --- Decide which market is more out-of-fair ---
        # (use absolute deviation; trade the bigger dislocation)
        if abs(misA) >= abs(misB):
            trade_market = "A"
            fair = fairA
            mispricing = misA
            tau = res_A["tau"]
            rho = res_A["copula_param"]
            p_enter = pA_now
            p_exit = A_vals[t_idx + H]
        else:
            trade_market = "B"
            fair = fairB
            mispricing = misB
            tau = res_B["tau"]
            rho = res_B["copula_param"]
            p_enter = pB_now
            p_exit = B_vals[t_idx + H]

        # --- Trading rule (same as before, but applied to chosen market) ---
        if mispricing > threshold:
            side = "YES"
            s = +1
        else:
            continue

        profit = p_exit - p_enter
        roi = profit / max(p_enter, 1e-6)

        trades.append({
            "t_enter": times[t_idx],
            "t_exit": times[t_idx + H],

            "pair_A": A,
            "pair_B": B,
            "trade_market": trade_market,  # "A" or "B"

            "side": side,

            # record BOTH currents + BOTH fairs
            "pA_enter": float(pA_now),
            "pB_enter": float(pB_now),
            "fairA": float(fairA),
            "fairB": float(fairB),
            "misA": float(misA),
            "misB": float(misB),

            # record the chosen trade’s numbers
            "p_enter": float(p_enter),
            "p_exit": float(p_exit),
            "fair": float(fair),
            "mispricing": float(mispricing),

            "profit_per_$1": float(profit),
            "profit_per_share": float(profit),
            "roi": float(roi),

            "tau": float(tau),
            "rho": float(rho),
            "n_obs": int(len(A_hist)),
        })

end_profile_time = time.time()
average_time = (end_profile_time - start_profile_time) / max(1, len(top_pairs))
print(f"Average time per pair: {average_time:.4f} seconds")

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

    avg_roi = trades_df["roi"].mean()
    med_roi = trades_df["roi"].median()
    roi_win_rate = (trades_df["roi"] > 0).mean()
    total_roi = trades_df["roi"].sum()
    print("\n=== ROI Metrics ===")
    print(f"Total ROI (sum):         {total_roi:+.4f}")
    print(f"Avg ROI per trade:       {avg_roi:+.4f}")
    print(f"Median ROI per trade:    {med_roi:+.4f}")
    print(f"Win rate (ROI):          {roi_win_rate * 100:.1f}%")

    print(trades_df.sort_values("roi", ascending=False).head(5))

    print("\nTop 5 worst trades:")
    print(trades_df.sort_values("roi", ascending=True).head(5))
    # Save for analysis
    out_path = "/home/mithil/PycharmProjects/PolymarketPred/data/fair_price_backtest_trades.csv"
    trades_df.to_csv(out_path, index=False)
    print(f"\nSaved trades to: {out_path}")
