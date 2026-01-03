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
def fair_price_copula(A: np.ndarray, B: np.ndarray, pB_latest: float) -> dict:
    """
    Improved Gaussian Copula for Pair Trading.
    Estimates the 'fair' price of A based on the current price of B.
    """
    # 1. Alignment and Cleaning
    n = min(len(A), len(B))
    if n < 10: return {"fair_mean": A[-1] if len(A) > 0 else 0.5, "tau": 0}

    a_vec = A[-n:]
    b_vec = B[-n:]

    # 2. Kendall's Tau -> Rho (Gaussian Param)
    # Using a fast approximation for Kendall Tau if performance is an issue
    tau, _ = kendalltau(a_vec, b_vec)
    if np.isnan(tau): tau = 0
    rho = np.sin(np.pi * tau / 2.0)
    rho = np.clip(rho, -0.99, 0.99)

    # 3. Probability Integral Transform (PIT)
    # Where does the current price of B sit relative to its history?
    # We use a small epsilon to avoid +/- infinity in norm.ppf
    v_b = np.mean(b_vec <= pB_latest)
    v_b = np.clip(v_b, 0.001, 0.999)

    # 4. Conditional Expectation in Gaussian Space
    # E[U_a | U_b = v_b] = Phi(rho * Phi^-1(v_b))
    z_b = norm.ppf(v_b)
    u_a_cond = norm.cdf(rho * z_b)

    # 5. Inverse CDF (Quantile) to get price
    fair_a = np.quantile(a_vec, u_a_cond)

    return {
        "fair_mean": float(fair_a),
        "tau": float(tau),
        "rho": float(rho)
    }
