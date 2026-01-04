import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import pandas as pd
import requests
from datetime import datetime
from typing import Tuple, List, Optional
import warnings

warnings.filterwarnings("ignore")

# --- Optional Numba JIT for Speed ---
try:
    from numba import jit
    HAS_NUMBA = True
    jit_decorator = jit(nopython=True, cache=True, fastmath=True)
except ImportError:
    HAS_NUMBA = False

    def jit_decorator(func):
        return func


# -------------------------------------------------------------------------
# 1. Heston Model Math (JIT)
# -------------------------------------------------------------------------

@jit_decorator
def _heston_cf_core(phi: complex, T: float, params: np.ndarray, P_num: int) -> complex:
    """
    Heston characteristic function core (normalized S=1, r=0, q=0).
    params: [kappa, theta, sigma_v, rho, v0]
    """
    kappa, theta, sigma_v, rho, v0 = params
    i = 1j

    if P_num == 1:
        u = 0.5
        b = kappa - rho * sigma_v
    else:
        u = -0.5
        b = kappa

    a = kappa * theta
    sig_v2 = sigma_v * sigma_v + 1e-20

    d = np.sqrt((rho * sigma_v * phi * i - b) ** 2 - sig_v2 * (2.0 * u * phi * i - phi * phi))
    g = (b - rho * sigma_v * phi * i - d) / (b - rho * sigma_v * phi * i + d + 1e-20)
    exp_dt = np.exp(-d * T)

    val_log = (1.0 - g * exp_dt) / (1.0 - g + 1e-20)
    C = (a / sig_v2) * ((b - rho * sigma_v * phi * i - d) * T - 2.0 * np.log(val_log))
    D = ((b - rho * sigma_v * phi * i - d) / sig_v2) * ((1.0 - exp_dt) / (1.0 - g * exp_dt + 1e-20))

    return np.exp(C + D * v0)


@jit_decorator
def _integrand_real_heston(phi: float, K_norm: float, T: float, params: np.ndarray, P_num: int) -> float:
    if np.abs(phi) < 1e-9:
        return 0.0
    cf_val = _heston_cf_core(phi, T, params, P_num)
    val = np.exp(-1j * phi * np.log(K_norm)) * cf_val / (1j * phi)
    return float(np.real(val))


class HestonModel:
    """
    Pure Heston stochastic volatility model (normalized S space).
    We price the DIGITAL call via P2 using Gil-Pelaez.
    """

    @staticmethod
    def _adjust_params(T: float, params: List[float]) -> np.ndarray:
        p = np.array(params, dtype=np.float64)
        # Mild stabilization for very short T to prevent numerical blowups
        if T < 0.02:
            p[2] *= (T / 0.02) ** 0.5  # damp sigma_v
        return p

    @staticmethod
    def price_binary_call(S: float, K: float, T: float, params: List[float]) -> float:
        """
        Digital call paying 1 if ST > K, with r≈0.
        For digitals, price = P2 under this convention.
        """
        if T <= 1e-6:
            return 1.0 if S > K else 0.0

        p_arr = HestonModel._adjust_params(T, params)
        K_norm = K / S

        limit = 500.0
        try:
            val_int, _ = quad(
                _integrand_real_heston,
                1e-9,
                limit,
                args=(K_norm, T, p_arr, 2),
                limit=50,
            )
            p2 = 0.5 + (1.0 / np.pi) * val_int
            return float(np.clip(p2, 0.0, 1.0))
        except Exception:
            return 0.5


# -------------------------------------------------------------------------
# 2. Deribit Data & Calibration
# -------------------------------------------------------------------------

class DeribitDataManager:
    BASE_URL = "https://www.deribit.com/api/v2/public/"

    def __init__(self, asset: str):
        self.asset = asset.upper()
        # Deribit options are basically BTC/ETH inverse-style
        if self.asset in ["BTC", "ETH"]:
            self.api_currency = self.asset
            self.is_inverse = True
        else:
            # Deribit doesn't really list SOL/XRP options like BTC/ETH;
            # leaving your previous logic, but it may return empty.
            self.api_currency = "USDC"
            self.is_inverse = False

    def get_spot_price(self) -> float:
        potential_names = [f"{self.asset.lower()}_usdc", f"{self.asset.lower()}_usd"]
        for name in potential_names:
            try:
                resp = requests.get(
                    f"{self.BASE_URL}get_index_price",
                    params={"index_name": name},
                    timeout=5,
                ).json()
                if "result" in resp:
                    return float(resp["result"]["index_price"])
            except Exception:
                continue
        return 0.0

    def get_liquid_chain(self, target_days: int = 25) -> Tuple[float, float, pd.DataFrame]:
        """
        Returns: spot, T_years, chain_df with columns:
          strike, norm_call, expiry, days, norm_strike
        """
        print(f"Fetching {self.asset} chain (via {self.api_currency} scope)...")
        spot = self.get_spot_price()
        if spot == 0.0:
            print(f"  [Error] Could not fetch spot price for {self.asset}")
            return 0.0, 0.0, pd.DataFrame()

        params = {"currency": self.api_currency, "kind": "option"}
        try:
            resp = requests.get(
                f"{self.BASE_URL}get_book_summary_by_currency",
                params=params,
                timeout=10,
            ).json()
        except Exception:
            return 0.0, 0.0, pd.DataFrame()

        data = []
        prefix = f"{self.asset}-" if self.is_inverse else f"{self.asset}_"

        for item in resp.get("result", []):
            inst = item.get("instrument_name", "")
            if not inst.startswith(prefix):
                continue

            parts = inst.split("-")
            # Example BTC-26JAN24-50000-C
            if len(parts) < 4 or parts[-1] != "C":
                continue

            try:
                strike = float(parts[-2])
                expiry = datetime.strptime(parts[1], "%d%b%y")
                mark = float(item.get("mark_price", 0.0))

                # Normalize call price into "S=1" space:
                # - inverse: Deribit mark_price is already in BTC terms, but your prior code used raw mark
                # - non-inverse: mark / spot
                if self.is_inverse:
                    norm_price = mark
                else:
                    norm_price = mark / spot

                data.append({"strike": strike, "norm_call": norm_price, "expiry": expiry})
            except Exception:
                continue

        df = pd.DataFrame(data)
        if df.empty:
            print(f"  [Warning] No raw options data found for {self.asset}")
            return 0.0, 0.0, pd.DataFrame()

        now = datetime.now()
        df["days"] = (df["expiry"] - now).dt.total_seconds() / 86400.0
        df = df[df["days"] > 2.0]

        if df.empty:
            print("  [Warning] No future expiries found.")
            return 0.0, 0.0, pd.DataFrame()

        # Choose expiry closest to target_days
        unique_days = np.array(sorted(df["days"].unique()))
        chosen_days = float(unique_days[np.argmin(np.abs(unique_days - target_days))])

        chain = df[df["days"] == chosen_days].copy()

        # Dynamic strike filtering around spot
        chain = chain[(chain["strike"] >= spot * 0.6) & (chain["strike"] <= spot * 1.4)]
        if chain.empty:
            print("  [Warning] Chain empty after strike filtering.")
            return spot, chosen_days / 365.0, pd.DataFrame()

        chain["norm_strike"] = chain["strike"] / spot
        chain = chain.sort_values("norm_strike").reset_index(drop=True)

        return spot, chosen_days / 365.0, chain


class HestonCalibrator:
    def __init__(self, asset: str):
        self.asset = asset.upper()
        self.dm = DeribitDataManager(self.asset)

    def calibrate(self) -> Optional[dict]:
        # 1) Fetch option chain
        spot, T, chain = self.dm.get_liquid_chain()
        if chain.empty or len(chain) < 3:
            print(f"  [Skipping {self.asset}] Insufficient option strikes found.")
            return None

        K = chain["norm_strike"].values.astype(float)
        C = chain["norm_call"].values.astype(float)

        # Smooth call prices a bit to stabilize numerical gradient
        if len(C) > 5:
            kernel = np.ones(3, dtype=float) / 3.0
            C = np.convolve(C, kernel, mode="same")

        # Market "digital" estimate: D(K) ~ -dC/dK
        D_mkt = -np.gradient(C, K)
        D_mkt = np.clip(D_mkt, 0.0, 1.0)

        # 2) Objective: fit Heston params to digital curve
        def objective(p):
            # p = [kappa, theta, sigma_v, rho, v0]
            kappa, theta, sigma_v, rho, v0 = p

            # quick sanity penalties to keep solver stable
            if theta <= 0 or sigma_v <= 0 or v0 <= 0:
                return 1e9
            if abs(rho) >= 0.999:
                return 1e9

            err = 0.0
            for i, k_val in enumerate(K):
                model_val = HestonModel.price_binary_call(1.0, float(k_val), float(T), list(p))

                # Weight ATM more heavily
                w = np.exp(-((k_val - 1.0) / 0.15) ** 2)
                diff = model_val - D_mkt[i]
                err += w * diff * diff

            # Regularization: keep v0 near theta
            err += 0.1 * (v0 - theta) ** 2
            return float(err)

        # Initial guesses and bounds
        guess = np.array([2.0, 0.5, 0.5, -0.5, 0.5], dtype=float)

        bounds = [
            (0.1, 20.0),    # kappa
            (0.01, 5.0),    # theta
            (0.01, 5.0),    # sigma_v
            (-0.99, 0.99),  # rho
            (0.01, 5.0),    # v0
        ]

        print(f"  Optimizing {self.asset} (Heston-only) (T={T:.3f}y, {len(K)} strikes)...")

        try:
            res = minimize(
                objective,
                guess,
                method="L-BFGS-B",
                bounds=bounds,
                tol=1e-5,
            )

            return {
                "currency": self.asset,
                "spot": float(spot),
                "timestamp": datetime.now().isoformat(),
                "kappa": float(res.x[0]),
                "theta": float(res.x[1]),
                "sigma_v": float(res.x[2]),
                "rho": float(res.x[3]),
                "v0": float(res.x[4]),
                "success": bool(res.success),
                "fun": float(res.fun),
                "nfev": int(getattr(res, "nfev", -1)),
            }
        except Exception as e:
            print(f"  Optimization failed: {e}")
            return None


if __name__ == "__main__":
    assets = ["BTC", "ETH", "SOL", "XRP"]
    results = []

    print("--- HESTON CALIBRATION (DERIBIT OPTIONS -> DIGITAL FIT) ---")
    if HAS_NUMBA:
        print(">> JIT Acceleration: ACTIVE")

    for a in assets:
        print(f"\n[{a}] Starting...")
        cal = HestonCalibrator(a)
        res = cal.calibrate()
        if res and res.get("success", False):
            results.append(res)
            print(f"  >> SUCCESS: v0={res['v0']:.4f}, rho={res['rho']:.3f}")
        elif res:
            # still save failed runs if you want; here we just print diagnostics
            print(f"  >> FAILED: fun={res.get('fun')}, nfev={res.get('nfev')}")

    if results:
        df = pd.DataFrame(results)
        print("\n--- FINAL PARAMETERS ---")
        print(df[["currency", "kappa", "theta", "sigma_v", "rho", "v0"]].head())

        filename = "/home/mithil/PycharmProjects/PolymarketPred/data/bates_params_digital.jsonl"
        df.to_json(filename, orient="records", lines=True)
        print(f"\nSaved to {filename}")
    else:
        print("\nNo successful calibrations to save.")
