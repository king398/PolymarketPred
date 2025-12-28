#!/usr/bin/env python3
"""
Copula Fair Price Engine for Polymarket Crypto Markets
-----------------------------------------------------

What this script does (end-to-end):
1) Loads your local Polymarket minute parquet files (HuggingFace Datasets).
2) Filters to crypto-related questions + a time range.
3) Builds a PRICE matrix (time x market(question)) with forward-fill alignment.
4) Computes a dependency score between markets (Kendall tau -> copula params).
5) For strong pairs, computes "fair price" of A given B is at its latest price:
      Fair(A | B=pB_now) via empirical PIT + copula conditional sampling + inverse marginal
6) Saves:
   - polymarket_price_pivot.csv
   - copula_fair_prices.csv
   - top_opportunities.csv   (ranked mispricings)
   - (optional) HTML scatter plot for a chosen pair

Notes:
- This is designed for probability series (YES prices in [0,1]), not belief updates.
- Gaussian copula is fastest; Clayton/Gumbel capture tail dependence (slower).
- Conditioning uses empirical CDF of B at pB_now, then samples U|V=v0 and maps back to A.

Usage:
  python copula_fair_price.py

Edit CONFIG below for dates, thresholds, family, etc.
"""

import glob
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset

from scipy.stats import kendalltau, norm
from scipy.optimize import brentq

# ----------------------------
# CONFIG
# ----------------------------
DIR = "/home/mithil/PycharmProjects/PolymarketPred/data/polymarket_minute_parquet"

CRYPTO_REGEX = r"\b(bitcoin|ethereum|solana|xrp|btc|eth)\b"

START_UTC = "2025-10-02"
DAYS = 2

FREQ = "1min"                 # alignment frequency
FAMILY = "gaussian"           # "gaussian" | "clayton" | "gumbel"
STRONG_TAU_ABS = 0.35         # threshold on |Kendall tau| (tune: 0.25–0.5)
MAX_PAIRS = 200               # cap to avoid huge runtime
N_MC = 30_000                 # conditional sampling size
SEED = 42

# Output files
OUT_PRICE_PIVOT = "polymarket_price_pivot.csv"
OUT_FAIR = "copula_fair_prices.csv"
OUT_TOP = "top_opportunities.csv"

EPS = 1e-6


# ----------------------------
# Utility: empirical PIT + PPF
# ----------------------------
def pit_rank(x: np.ndarray) -> np.ndarray:
    """Empirical PIT using ranks -> values in (0,1)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    ranks = pd.Series(x).rank(method="average").to_numpy()
    u = (ranks - 0.5) / n
    return np.clip(u, EPS, 1 - EPS)


def empirical_cdf_value(sample: np.ndarray, x0: float) -> float:
    """Empirical CDF value F(x0) from sample."""
    sample = np.asarray(sample, dtype=float)
    return float(np.clip(np.mean(sample <= x0), EPS, 1 - EPS))


def empirical_ppf(sample: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Empirical inverse CDF (quantile)."""
    sample = np.asarray(sample, dtype=float)
    q = np.asarray(q, dtype=float)
    return np.quantile(sample, q, method="linear")


# ----------------------------
# Copula fitting (via Kendall tau)
# ----------------------------
def fit_copula_params(u: np.ndarray, v: np.ndarray, family: str):
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


# ----------------------------
# Conditional sampling U|V=v0
# ----------------------------
def sample_u_given_v_gaussian(v0: float, rho: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z2 = norm.ppf(v0)
    z1 = rng.normal(loc=rho * z2, scale=np.sqrt(1 - rho**2), size=n)
    return np.clip(norm.cdf(z1), EPS, 1 - EPS)


def clayton_inv_conditional_cdf(q: float, v: float, theta: float) -> float:
    # Inversion of Clayton conditional CDF (derived from ∂C/∂v)
    A = (q * (v ** (theta + 1.0))) ** (-theta / (1.0 + theta))
    u_neg_theta = A - (v ** (-theta)) + 1.0
    if u_neg_theta <= 0:
        return EPS
    return float(u_neg_theta ** (-1.0 / theta))


def sample_u_given_v_clayton(v0: float, theta: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.random(n)
    u = np.array([clayton_inv_conditional_cdf(qi, v0, theta) for qi in q], dtype=float)
    return np.clip(u, EPS, 1 - EPS)


def gumbel_conditional_cdf(u: float, v: float, theta: float) -> float:
    lu = -np.log(u)
    lv = -np.log(v)
    a = (lu**theta + lv**theta)
    t = a ** (1.0 / theta)
    C = np.exp(-t)
    return float(C * (a**(1.0/theta - 1.0)) * (lv**(theta - 1.0)) / v)


def gumbel_inv_conditional_cdf(q: float, v: float, theta: float) -> float:
    f = lambda uu: gumbel_conditional_cdf(uu, v, theta) - q
    return float(brentq(f, EPS, 1 - EPS, maxiter=200))


def sample_u_given_v_gumbel(v0: float, theta: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.random(n)
    u = np.array([gumbel_inv_conditional_cdf(qi, v0, theta) for qi in q], dtype=float)
    return np.clip(u, EPS, 1 - EPS)


# ----------------------------
# Fair price core
# ----------------------------
def fair_price_A_given_B(
        A_series: pd.Series,
        B_series: pd.Series,
        pB_new: float,
        family: str,
        n_mc: int,
        seed: int,
):
    joined = pd.concat([A_series, B_series], axis=1).dropna()
    if len(joined) < 200:
        return None  # too little data

    joined.columns = ["A", "B"]
    A = joined["A"].to_numpy(dtype=float)
    B = joined["B"].to_numpy(dtype=float)

    # PIT for dependence fitting
    u = pit_rank(A)
    v = pit_rank(B)

    try:
        params = fit_copula_params(u, v, family)
    except ValueError:
        return None

    v0 = empirical_cdf_value(B, pB_new)

    fam = family.lower()
    if fam == "gaussian":
        u_samp = sample_u_given_v_gaussian(v0, params["rho"], n_mc, seed)
    elif fam == "clayton":
        u_samp = sample_u_given_v_clayton(v0, params["theta"], n_mc, seed)
    elif fam == "gumbel":
        u_samp = sample_u_given_v_gumbel(v0, params["theta"], n_mc, seed)
    else:
        raise ValueError("family must be gaussian/clayton/gumbel")

    A_samp = empirical_ppf(A, u_samp)

    return {
        "tau": params["tau"],
        "copula_param": params.get("rho", params.get("theta", np.nan)),
        "pB_new": float(pB_new),
        "fair_mean": float(np.mean(A_samp)),
        "fair_median": float(np.median(A_samp)),
        "fair_p10": float(np.quantile(A_samp, 0.10)),
        "fair_p90": float(np.quantile(A_samp, 0.90)),
        "n_obs": int(len(joined)),
    }


# ----------------------------
# Main pipeline
# ----------------------------
def main():
    parquets = glob.glob(f"{DIR}/*.parquet")
    if not parquets:
        raise FileNotFoundError(f"No parquet files found in: {DIR}")

    print(f"Found {len(parquets)} parquet files.")
    ds = Dataset.from_parquet(parquets)

    pattern = re.compile(CRYPTO_REGEX, re.IGNORECASE)

    def crypto_filter_batch(batch):
        qs = batch["question"]
        return [bool(pattern.search(q)) if q else False for q in qs]

    ds_crypto = ds.filter(
        crypto_filter_batch,
        batched=True,
        batch_size=50_000,
        num_proc=32,
    )

    start = pd.Timestamp(START_UTC, tz="UTC")
    end = start + pd.Timedelta(days=DAYS)

    def in_range_batch(batch):
        ts = pd.to_datetime(batch["timestamp"], utc=True, errors="coerce")
        return (ts >= start) & (ts <= end)

    ds_range = ds_crypto.filter(
        in_range_batch,
        batched=True,
        batch_size=50_000,
        num_proc=32,
        load_from_cache_file=True,
    )

    df = ds_range.to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "question", "price"])
    df["price"] = df["price"].astype("float64").clip(EPS, 1 - EPS)

    # Align time
    df["timestamp_min"] = df["timestamp"].dt.floor(FREQ)

    # Price pivot (levels)
    price_pivot = df.pivot_table(
        index="timestamp_min",
        columns="question",
        values="price",
        aggfunc="last"
    ).sort_index()
    print(price_pivot)

    # Forward-fill within each column for continuity (then drop all-empty rows)
    price_pivot = price_pivot.ffill().dropna(how="all")

    # Drop columns that are almost constant / dead
    col_std = price_pivot.std(skipna=True)
    keep_cols = col_std[col_std > 1e-5].index
    price_pivot = price_pivot[keep_cols]

    price_pivot.to_csv(OUT_PRICE_PIVOT)
    print(f"✅ Saved price pivot: {OUT_PRICE_PIVOT}")
    print(f"Price pivot shape: {price_pivot.shape} (Time x Markets)")

    # Latest price per market
    latest_price = df.sort_values("timestamp").groupby("question")["price"].last()

    markets = price_pivot.columns.tolist()
    n = len(markets)
    print(f"Active markets for copula: {n}")

    # Compute Kendall tau matrix (cheap) to pick candidate pairs
    # We do pairwise tau on aligned window (dropna pairwise)
    candidates = []
    print("Computing Kendall tau for candidate pairs...")
    for i in tqdm(range(n)):
        A = markets[i]
        a = price_pivot[A]
        for j in range(i + 1, n):
            B = markets[j]
            b = price_pivot[B]
            joined = pd.concat([a, b], axis=1).dropna()
            if len(joined) < 300:
                continue
            tau, _ = kendalltau(joined.iloc[:, 0].values, joined.iloc[:, 1].values)
            if not np.isfinite(tau):
                continue
            if abs(tau) >= STRONG_TAU_ABS:
                candidates.append((A, B, float(tau), int(len(joined))))

    if not candidates:
        print("No pairs passed tau threshold. Lower STRONG_TAU_ABS and rerun.")
        return

    # Sort candidates strongest first and cap
    candidates.sort(key=lambda x: abs(x[2]), reverse=True)
    candidates = candidates[:MAX_PAIRS]
    print(f"Selected {len(candidates)} strong pairs (|tau| >= {STRONG_TAU_ABS}).")

    # Compute fair prices
    rows = []
    print(f"Computing copula fair prices (family={FAMILY}, N_MC={N_MC})...")
    for idx, (A, B, tau, n_obs_pair) in enumerate(tqdm(candidates)):
        pA_now = float(latest_price.get(A, np.nan))
        pB_now = float(latest_price.get(B, np.nan))
        if not (np.isfinite(pA_now) and np.isfinite(pB_now)):
            continue

        stats_A_given_B = fair_price_A_given_B(
            price_pivot[A], price_pivot[B],
            pB_new=pB_now,
            family=FAMILY,
            n_mc=N_MC,
            seed=SEED + idx * 2,
        )
        stats_B_given_A = fair_price_A_given_B(
            price_pivot[B], price_pivot[A],
            pB_new=pA_now,
            family=FAMILY,
            n_mc=N_MC,
            seed=SEED + idx * 2 + 1,
        )
        if stats_A_given_B is None or stats_B_given_A is None:
            continue

        rows.append({
            "A": A, "B": B,
            "tau": tau,
            "n_obs_pair": n_obs_pair,
            "family": FAMILY,
            "copula_param_A_given_B": stats_A_given_B["copula_param"],
            "copula_param_B_given_A": stats_B_given_A["copula_param"],
            "pA_now": pA_now,
            "pB_now": pB_now,

            "fair_A_given_B_mean": stats_A_given_B["fair_mean"],
            "fair_A_given_B_p10": stats_A_given_B["fair_p10"],
            "fair_A_given_B_p90": stats_A_given_B["fair_p90"],

            "fair_B_given_A_mean": stats_B_given_A["fair_mean"],
            "fair_B_given_A_p10": stats_B_given_A["fair_p10"],
            "fair_B_given_A_p90": stats_B_given_A["fair_p90"],

            "n_obs_used_A_given_B": stats_A_given_B["n_obs"],
            "n_obs_used_B_given_A": stats_B_given_A["n_obs"],
        })

    fair_df = pd.DataFrame(rows)
    if fair_df.empty:
        print("No fair price results produced (try gaussian family or lower thresholds).")
        return

    fair_df.to_csv(OUT_FAIR, index=False)
    print(f"✅ Saved fair prices: {OUT_FAIR}")

    # Rank opportunities by absolute mispricing
    fair_df["mispricing_A"] = fair_df["pA_now"] - fair_df["fair_A_given_B_mean"]
    fair_df["mispricing_B"] = fair_df["pB_now"] - fair_df["fair_B_given_A_mean"]
    fair_df["abs_mispricing_max"] = np.maximum(fair_df["mispricing_A"].abs(), fair_df["mispricing_B"].abs())

    top = fair_df.sort_values("abs_mispricing_max", ascending=False).copy()
    top.to_csv(OUT_TOP, index=False)
    print(f"✅ Saved top opportunities: {OUT_TOP}")

    # Print a concise view
    show_cols = [
        "A", "B", "tau", "pA_now", "fair_A_given_B_mean", "mispricing_A",
        "pB_now", "fair_B_given_A_mean", "mispricing_B", "abs_mispricing_max"
    ]
    print("\nTop 15 opportunities (by max abs mispricing):")
    print(top[show_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
