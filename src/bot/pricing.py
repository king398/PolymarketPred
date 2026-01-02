import numpy as np
from scipy.stats import kendalltau, norm, rankdata

eps = 1e-6

def empirical_cdf_value(sample: np.ndarray, x0: float) -> float:
    sample = np.asarray(sample, dtype=float)
    return float(np.clip(np.mean(sample <= x0), eps, 1 - eps))

def pit_rank(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.size
    ranks = rankdata(x, method="average")
    u = (ranks - 0.5) / n
    return np.clip(u, eps, 1 - eps)

def empirical_ppf(sample: np.ndarray, q) -> float:
    sample = np.asarray(sample, dtype=float)
    q = float(q)
    if not np.isfinite(q):
        return np.nan
    q = float(np.clip(q, eps, 1 - eps))
    return float(np.quantile(sample, q, method="linear"))

def fair_price_np(A: np.ndarray, B: np.ndarray, pB_new: float) -> dict:
    """
    Gaussian copula ONLY
    Deterministic, no Monte Carlo, no leakage
    Inputs:
      A, B: 1D numpy arrays (or array-likes)
      pB_new: scalar (float)
    """
    A = np.asarray(A, dtype=float).ravel()
    B = np.asarray(B, dtype=float).ravel()

    if A.size < 2 or B.size < 2:
        raise ValueError("A and B must each have at least 2 observations.")
    if not np.isfinite(pB_new):
        raise ValueError("pB_new must be a finite scalar.")

    # If lengths differ, align to common length (same behavior you used earlier with min(n))
    n = min(A.size, B.size)
    A = A[:n]
    B = B[:n]

    # Drop non-finite pairs
    mask = np.isfinite(A) & np.isfinite(B)
    A = A[mask]
    B = B[mask]
    if A.size < 2:
        raise ValueError("Not enough finite paired observations after filtering.")

    # PIT
    u_A = pit_rank(A)
    u_B = pit_rank(B)

    # Fit Gaussian copula via Kendall tau
    tau, _ = kendalltau(u_A, u_B)
    tau = float(np.clip(tau, -0.999, 0.999))
    rho = float(np.clip(np.sin(np.pi * tau / 2.0), -0.999, 0.999))

    # Percentile of B at current price
    v0 = empirical_cdf_value(B, float(pB_new))

    # Conditional mean in copula space
    z = norm.ppf(v0)
    u_cond_mean = norm.cdf(rho * z)

    # Map back to price space
    fair = float(empirical_ppf(A, u_cond_mean))

    return {
        "tau": tau,
        "copula_param": rho,
        "pB_new": float(pB_new),
        "fair_mean": fair,
        "n_obs": int(A.size),
    }
