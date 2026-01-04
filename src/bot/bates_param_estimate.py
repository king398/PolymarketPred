import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Union
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
# 1. Historical Jump Analysis (Binance)
# -------------------------------------------------------------------------

class BinanceHistory:
    BASE_URL = "https://api.binance.com/api/v3/klines"

    @staticmethod
    def get_jump_params(symbol: str, lookback_days: int = 7) -> Tuple[float, float, float]:
        """
        Fetches historical data to estimate Jump Diffusion parameters.
        Returns: (lambda_j, mu_j, sigma_j)
        """
        # Map generic symbol to Binance pair
        pair_map = {
            'BTC': 'BTCUSDT',
            'ETH': 'ETHUSDT',
            'SOL': 'SOLUSDT',
            'XRP': 'XRPUSDT',
            'BNB': 'BNBUSDT'
        }
        pair = pair_map.get(symbol.upper(), f"{symbol.upper()}USDT")

        # 1h candles for 7 days = 24 * 7 = 168 data points
        limit = lookback_days * 24

        try:
            params = {
                "symbol": pair,
                "interval": "5m",
                "limit": limit
            }
            resp = requests.get(BinanceHistory.BASE_URL, params=params, timeout=5)
            data = resp.json()

            if not isinstance(data, list) or len(data) < 50:
                print(f"  [Binance] Insufficient data for {symbol}, using defaults.")
                return 1.0, -0.05, 0.10

            # Parse Closing Prices (index 4)
            closes = np.array([float(x[4]) for x in data])

            # Log Returns
            log_rets = np.diff(np.log(closes))

            # --- Iterative Jump Detection ---
            # We filter out moves > 3 stdevs to find "normal" volatility,
            # then classify everything else as a jump.

            std_diffusive = np.std(log_rets)
            # Refine std by excluding initial outliers
            clean_rets = log_rets[np.abs(log_rets) < 3 * std_diffusive]
            std_diffusive = np.std(clean_rets)

            # Threshold for Jumps (3 sigma event)
            threshold = 3.0 * std_diffusive

            jump_indices = np.where(np.abs(log_rets) > threshold)[0]
            jumps = log_rets[jump_indices]

            # 1. Lambda (Annualized frequency)
            # count / (days / 365)
            n_jumps = len(jumps)
            T_year = lookback_days / 365.0
            lambda_j = max(0.1, n_jumps / T_year) # Avoid 0 division

            if n_jumps > 0:
                mu_j = np.mean(jumps)
                sigma_j = np.std(jumps)
                # Floor sigma_j to avoid singularities
                if sigma_j < 0.01: sigma_j = 0.01
            else:
                # No jumps in last 7 days; assume quiet regime
                # Defaults: Rare small jumps
                lambda_j = 0.5
                mu_j = 0.0
                sigma_j = 0.05

            print(f"  [Binance] {symbol}: Found {n_jumps} jumps (Vol: {std_diffusive:.4f}). Lambda={lambda_j:.2f}, Mu={mu_j:.4f}")
            return lambda_j, mu_j, sigma_j

        except Exception as e:
            print(f"  [Binance] Error fetching {symbol}: {e}")
            return 2.0, -0.05, 0.10

# -------------------------------------------------------------------------
# 2. Bates Model Math (JIT)
# -------------------------------------------------------------------------

@jit_decorator
def _bates_cf_core(phi: complex, T: float, params: np.ndarray, P_num: int) -> complex:
    # params: [kappa, theta, sigma_v, rho, v0, lambda_j, mu_j, sigma_j]
    kappa, theta, sigma_v, rho, v0, lambda_j, mu_j, sigma_j = params
    i = 1j

    if P_num == 1:
        u = 0.5
        b = kappa - rho * sigma_v
    else:
        u = -0.5
        b = kappa

    a = kappa * theta
    sig_v2 = sigma_v * sigma_v + 1e-20
    d = np.sqrt((rho * sigma_v * phi * i - b)**2 - sig_v2 * (2 * u * phi * i - phi * phi))
    g = (b - rho * sigma_v * phi * i - d) / (b - rho * sigma_v * phi * i + d + 1e-20)
    exp_dt = np.exp(-d * T)

    val_log = (1.0 - g * exp_dt) / (1.0 - g + 1e-20)
    C = (a / sig_v2) * ((b - rho * sigma_v * phi * i - d) * T - 2.0 * np.log(val_log))
    D = ((b - rho * sigma_v * phi * i - d) / sig_v2) * ((1.0 - exp_dt) / (1.0 - g * exp_dt + 1e-20))

    heston_cf = np.exp(C + D * v0)

    # Jump Component (Merton)
    # k is the expected jump size in log-space correction
    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    jump_part = lambda_j * T * (np.exp(i * phi * mu_j - 0.5 * sigma_j**2 * phi**2) - 1.0 - i * phi * k)

    return heston_cf * np.exp(jump_part)

@jit_decorator
def _integrand_real(phi: float, K_norm: float, T: float, params: np.ndarray, P_num: int) -> float:
    if np.abs(phi) < 1e-9: return 0.0
    cf_val = _bates_cf_core(phi, T, params, P_num)
    val = np.exp(-1j * phi * np.log(K_norm)) * cf_val / (1j * phi)
    return np.real(val)

class BatesModel:
    @staticmethod
    def _adjust_params(T: float, params: List[float]) -> np.ndarray:
        p = np.array(params, dtype=np.float64)
        # Empirical adjustment for very short expiries to prevent explosion
        if T < 0.02:
            p[2] *= (T/0.02)**0.5
        return p

    @staticmethod
    def price_binary_call(S: float, K: float, T: float, params: List[float]) -> float:
        if T <= 1e-6: return 1.0 if S > K else 0.0
        p_arr = BatesModel._adjust_params(T, params)
        K_norm = K / S

        limit = 500.0
        try:
            val_int, _ = quad(_integrand_real, 1e-9, limit, args=(K_norm, T, p_arr, 2), limit=50)
            p2 = 0.5 + (1.0 / np.pi) * val_int
            return float(np.clip(p2, 0.0, 1.0))
        except:
            return 0.5

# -------------------------------------------------------------------------
# 3. Deribit Data & Calibration
# -------------------------------------------------------------------------

class DeribitDataManager:
    BASE_URL = "https://www.deribit.com/api/v2/public/"

    def __init__(self, asset: str):
        self.asset = asset.upper()
        if self.asset in ['BTC', 'ETH']:
            self.api_currency = self.asset
            self.is_inverse = True
        else:
            self.api_currency = 'USDC'
            self.is_inverse = False

    def get_spot_price(self) -> float:
        potential_names = [f"{self.asset.lower()}_usdc", f"{self.asset.lower()}_usd"]
        for name in potential_names:
            try:
                resp = requests.get(f"{self.BASE_URL}get_index_price", params={'index_name': name}, timeout=5).json()
                if 'result' in resp:
                    return float(resp['result']['index_price'])
            except Exception:
                continue
        return 0.0

    def get_liquid_chain(self, target_days: int = 25) -> Tuple[float, float, pd.DataFrame]:
        print(f"Fetching {self.asset} chain (via {self.api_currency} scope)...")
        spot = self.get_spot_price()
        if spot == 0:
            print(f"  [Error] Could not fetch spot price for {self.asset}")
            return 0.0, 0.0, pd.DataFrame()

        params = {'currency': self.api_currency, 'kind': 'option'}
        try:
            resp = requests.get(f"{self.BASE_URL}get_book_summary_by_currency", params=params, timeout=10).json()
        except Exception:
            return 0.0, 0.0, pd.DataFrame()

        data = []
        prefix = f"{self.asset}-" if self.is_inverse else f"{self.asset}_"

        for item in resp.get('result', []):
            inst = item['instrument_name']
            if not inst.startswith(prefix): continue

            parts = inst.split('-')
            if len(parts) < 4 or parts[-1] != 'C': continue

            try:
                strike = float(parts[-2])
                expiry = datetime.strptime(parts[1], "%d%b%y")
                mark = item['mark_price']

                if self.is_inverse:
                    norm_price = mark
                else:
                    norm_price = mark / spot

                data.append({'strike': strike, 'norm_call': norm_price, 'expiry': expiry})
            except:
                continue

        df = pd.DataFrame(data)
        if df.empty:
            print(f"  [Warning] No raw options data found for {self.asset}")
            return 0.0, 0.0, pd.DataFrame()

        now = datetime.now()
        df['days'] = (df['expiry'] - now).dt.total_seconds() / 86400.0
        df = df[df['days'] > 2]

        if df.empty:
            print("  [Warning] No future expiries found.")
            return 0.0, 0.0, pd.DataFrame()

        chosen_days = min(df['days'].unique(), key=lambda x: abs(x - target_days))
        chain = df[df['days'] == chosen_days].copy()

        # Dynamic strike filtering
        chain = chain[(chain['strike'] >= spot * 0.6) & (chain['strike'] <= spot * 1.4)]
        chain['norm_strike'] = chain['strike'] / spot
        chain = chain.sort_values('norm_strike')

        return spot, float(chosen_days/365.0), chain

class BatesCalibrator:
    def __init__(self, asset: str):
        self.asset = asset
        self.dm = DeribitDataManager(asset)

    def calibrate(self):
        # 1. Fetch Option Chain
        spot, T, chain = self.dm.get_liquid_chain()
        if chain.empty or len(chain) < 3:
            print(f"  [Skipping {self.asset}] Insufficient option strikes found.")
            return None

        # 2. Fetch Historical Jumps (Binance)
        # We perform this BEFORE calibration to fix these parameters
        lambda_j, mu_j, sigma_j = BinanceHistory.get_jump_params(self.asset, lookback_days=7)

        K = chain['norm_strike'].values
        C = chain['norm_call'].values

        # Smooth prices if dense to avoid noise in gradient
        if len(C) > 5:
            C = np.convolve(C, np.ones(3)/3, mode='same')

        # Calculate Market Digital Prices (Slope of Call)
        D_mkt = -np.gradient(C, K)
        D_mkt = np.clip(D_mkt, 0.0, 1.0)

        # 3. Objective Function (Optimize only Heston params)
        def objective(p):
            # p = [kappa, theta, sigma_v, rho, v0]
            # Fixed = [lambda_j, mu_j, sigma_j]
            full_params = [*p, lambda_j, mu_j, sigma_j]

            err = 0.0
            for i, k_val in enumerate(K):
                model_val = BatesModel.price_binary_call(1.0, k_val, T, full_params)

                # Weight ATM more heavily
                w = np.exp(-((k_val-1.0)/0.15)**2)
                err += w * (model_val - D_mkt[i])**2

            # Regularization: keep v0 near theta (long term mean)
            err += 0.1 * (p[4] - p[1])**2
            return err

        # Initial Guesses
        guess = [2.0, 0.5, 0.5, -0.5, 0.5]

        # Bounds: [kappa, theta, sigma_v, rho, v0]
        bounds = [
            (0.1, 20.0), # kappa
            (0.01, 5.0), # theta
            (0.01, 5.0), # sigma_v
            (-0.99, 0.99), # rho
            (0.01, 5.0)  # v0
        ]

        print(f"  Optimizing {self.asset} (T={T:.3f}y, {len(K)} strikes)...")

        try:
            res = minimize(
                objective,
                guess,
                method='L-BFGS-B',
                bounds=bounds,
                tol=1e-5
            )

            return {
                'currency': self.asset,
                'spot': spot,
                'timestamp': datetime.now().isoformat(),
                'kappa': res.x[0],
                'theta': res.x[1],
                'sigma_v': res.x[2],
                'rho': res.x[3],
                'v0': res.x[4],
                'lambda_j': lambda_j,
                'mu_j': mu_j,
                'sigma_j': sigma_j
            }
        except Exception as e:
            print(f"  Optimization failed: {e}")
            return None

if __name__ == "__main__":
    assets = ['BTC', 'ETH', 'SOL', 'XRP']
    results = []

    print("--- BATES CALIBRATION (BINANCE HISTORICAL + DERIBIT OPTIONS) ---")
    if HAS_NUMBA: print(">> JIT Acceleration: ACTIVE")

    for a in assets:
        print(f"\n[{a}] Starting...")
        cal = BatesCalibrator(a)
        res = cal.calibrate()
        if res:
            results.append(res)
            print(f"  >> SUCCESS: v0={res['v0']:.2%}, Jumps(λ)={res['lambda_j']:.2f}")

    if results:
        df = pd.DataFrame(results)
        print("\n--- FINAL PARAMETERS ---")
        print(df[['currency', 'kappa', 'v0', 'lambda_j', 'mu_j']].head())

        filename = "/home/mithil/PycharmProjects/PolymarketPred/data/bates_params_digital.jsonl"
        df.to_json(filename, orient='records', lines=True)
        print(f"\nSaved to {filename}")