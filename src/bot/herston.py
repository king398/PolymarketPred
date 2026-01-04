import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Union
import warnings

warnings.filterwarnings("ignore")

# --- Optional Numba JIT for 100x Speedup ---
# If numba is installed, the math kernels become machine code.
# If not, it falls back to standard Python seamlessly.
try:
    from numba import jit
    HAS_NUMBA = True
    jit_decorator = jit(nopython=True, cache=True, fastmath=True)
except ImportError:
    HAS_NUMBA = False
    def jit_decorator(func):
        return func

# -------------------------------------------------------------------------
# JIT-Compiled Math Kernels (Stateless)
# -------------------------------------------------------------------------

@jit_decorator
def _bates_cf_core(phi: complex, T: float, params: np.ndarray, P_num: int) -> complex:
    """
    Core Characteristic Function using the 'Albrecher/Gatheral' Stable Form.
    This form avoids the 'Heston Trap' (branch cut discontinuities).

    params array: [kappa, theta, sigma_v, rho, v0, lambda_j, mu_j, sigma_j]
    """
    kappa = params[0]
    theta = params[1]
    sigma_v = params[2]
    rho = params[3]
    v0 = params[4]
    lambda_j = params[5]
    mu_j = params[6]
    sigma_j = params[7]

    i = 1j

    # Gil-Pelaez coefficients
    if P_num == 1:
        u = 0.5
        b = kappa - rho * sigma_v
    else:
        u = -0.5
        b = kappa

    a = kappa * theta
    # Small epsilon to prevent division by zero
    sig_v2 = sigma_v * sigma_v + 1e-20

    # Characteristic equation root
    # d = sqrt( (rho*sig*phi*i - b)^2 - sig^2*(2*u*phi*i - phi^2) )
    d_sq = (rho * sigma_v * phi * i - b)**2 - sig_v2 * (2 * u * phi * i - phi * phi)
    d = np.sqrt(d_sq)

    # Stable g: g = (b - rho*sig*phi*i - d) / (b - rho*sig*phi*i + d)
    g_num = b - rho * sigma_v * phi * i - d
    g_den = b - rho * sigma_v * phi * i + d
    g = g_num / (g_den + 1e-20)

    exp_dt = np.exp(-d * T)

    # Heston Log-Price CF components
    # C(T)
    # The stable form logarithm: log( (1-g*exp(-dt))/(1-g) )
    val_log = (1.0 - g * exp_dt) / (1.0 - g + 1e-20)
    C = (a / sig_v2) * (g_num * T - 2.0 * np.log(val_log))

    # D(T)
    D = (g_num / sig_v2) * ((1.0 - exp_dt) / (1.0 - g * exp_dt + 1e-20))

    # Heston CF (S_internal normalized to 1.0, so log(S)=0 term vanishes)
    heston_cf = np.exp(C + D * v0)

    # Merton Jumps Component
    # Compensator k = E[e^J] - 1
    # J ~ N(mu_j, sigma_j^2) => E[e^J] = exp(mu_j + 0.5*sigma_j^2)
    k = np.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0

    # Jump CF part: lambda * T * ( E[e^{i*phi*J}] - 1 - i*phi*k )
    # E[e^{i*phi*J}] = exp(i*phi*mu_j - 0.5*sigma_j^2 * phi^2)
    jump_part = lambda_j * T * (
            np.exp(i * phi * mu_j - 0.5 * sigma_j * sigma_j * phi * phi) - 1.0 - i * phi * k
    )

    return heston_cf * np.exp(jump_part)

@jit_decorator
def _integrand_real(phi: float, K_norm: float, T: float, params: np.ndarray, P_num: int) -> float:
    """
    The integrand for Gil-Pelaez: Re( exp(-i*phi*k) * CF(phi) / (i*phi) )
    """
    # Removable singularity at phi=0 handled by integration limits (starting at 1e-9)
    if np.abs(phi) < 1e-9:
        return 0.0

    i = 1j
    cf_val = _bates_cf_core(phi, T, params, P_num)
    # term: exp(-i * phi * log(K)) * cf / (i * phi)
    val = np.exp(-i * phi * np.log(K_norm)) * cf_val / (i * phi)
    return np.real(val)

# -------------------------------------------------------------------------
# Bates Model Class
# -------------------------------------------------------------------------

class BatesModel:
    """
    Optimized Bates Model (Heston + Merton Jumps).
    Features:
      - Numba JIT support for high-performance calibration.
      - Dynamic integration limits for accurate short-expiry pricing.
      - Stable Albrecher characteristic function.
    """

    @staticmethod
    def _adjust_params_for_expiry(T: float, initial_T: float, params) -> List[float]:
        """
        Adjusts parameters for short expiries ("expiry crush").
        Returns a flat list of 8 floats.
        """
        # --- Normalize params to list of floats ---
        p_raw = []
        if isinstance(params, dict):
            p_raw = [
                params.get("kappa", 0), params.get("theta", 0), params.get("sigma_v", 0),
                params.get("rho", 0), params.get("v0", 0), params.get("lambda_j", 0),
                params.get("mu_j", 0), params.get("sigma_j", 0)
            ]
        elif isinstance(params, (list, tuple, np.ndarray)):
            # Handle nested array case if accidentally passed
            flat = np.array(params).flatten()
            p_raw = flat.tolist()

        # Ensure exactly 8 params (pad if necessary)
        if len(p_raw) < 8:
            p_raw += [0.0] * (8 - len(p_raw))

        p = [float(x) for x in p_raw[:8]]

        # --- Expiry Logic ---
        if initial_T <= 0:
            initial_T = T + 1e-9

        # Safety clamp for numerical stability
        safe_T = max(T, 1e-6)

        pct_remaining = max(0.0, min(1.0, safe_T / initial_T))
        minutes_remaining = safe_T * 525600.0 # 365*24*60

        kill_threshold = 0.2
        if pct_remaining < kill_threshold:
            zone_progress = pct_remaining / kill_threshold
            decay_factor = zone_progress ** 8

            # Decay volatility driving parameters
            p[2] *= decay_factor  # sigma_v
            p[5] *= decay_factor  # lambda_j

            # Crush spot vol, but maintain floor to prevent singularities
            p[4] = max(1e-6, p[4] * (zone_progress ** 2))

        if minutes_remaining < 5:
            # Locking factor for very last moments
            locking_factor = float(np.exp((15.0 - minutes_remaining) / 3.0))
            p[0] *= locking_factor  # kappa

        return p

    @staticmethod
    def _get_dynamic_limit(T: float, params: List[float]) -> float:
        """
        Calculates integration limit based on Time and Volatility.
        Short expiries require MUCH larger limits to capture the tail.
        """
        # params: [kappa, theta, sigma_v, rho, v0, ...]
        # Approximation of vol for scaling: sqrt(v0) or sqrt(theta)
        vol_est = max(np.sqrt(params[4]), np.sqrt(params[1]), 0.1)

        # Scale limit inversely with sqrt(T).
        # If T is small, CF is wide -> High limit.
        safe_T = max(T, 1e-5)

        # Base heuristic: High enough to capture tails for very short expiries
        limit = 150.0 / (vol_est * np.sqrt(safe_T))

        # Clamp to reasonable range [200, 5000]
        return float(np.clip(limit, 200.0, 1000.0))

    @staticmethod
    def _P(K_norm: float, T: float, params: List[float], P_num: int) -> float:
        """
        Gil-Pelaez Pricing Formula using JIT-compiled integrand.
        """
        # Convert params to numpy array for JIT compatibility
        p_arr = np.array(params, dtype=np.float64)

        # Determine integration limit dynamically
        limit = BatesModel._get_dynamic_limit(T, params)

        try:
            val_int, _ = quad(
                _integrand_real,
                1e-9,
                limit,
                args=(K_norm, T, p_arr, P_num),
                limit=100,      # sub-intervals for quad
                epsabs=1e-5,    # Moderate tolerance for speed/accuracy balance
                epsrel=1e-5
            )
        except Exception:
            # Fallback for rare integration failures
            return 0.5

        val = 0.5 + (1.0 / np.pi) * val_int
        return float(np.clip(val, 0.0, 1.0))

    @staticmethod
    def price_vanilla_call(S: float, K: float, T: float, params: List[float]) -> float:
        """
        Standard Bates Call Price (P1, P2)
        """
        if T <= 1e-6:
            return max(S - K, 0.0)

        K_norm = K / S
        # P1 uses +0.5 shift, P2 uses -0.5 shift
        p1 = BatesModel._P(K_norm, T, params, 1)
        p2 = BatesModel._P(K_norm, T, params, 2)

        return float(max(0.0, S * p1 - K * p2))

    @staticmethod
    def price_binary_call(S: float, K: float, T: float, initial_T: float, params: List[float]) -> float:
        """
        Price a Digital Call (Cash-or-Nothing) paying $1 if S_T > K.
        Interface matches request: (S, K, T, initial_T, params)
        """
        if T <= 1e-6:
            return 1.0 if S > K else 0.0

        # Apply the expiry crush logic
        adj_params = BatesModel._adjust_params_for_expiry(T, initial_T, params)

        K_norm = K / S

        # For digitals (r=0), Price = P(S_T > K) = P2
        # We compute P2 directly rather than differentiating calls for stability.
        p2 = BatesModel._P(K_norm, T, adj_params, 2)

        return float(p2)

# -------------------------------------------------------------------------
# Data & Calibration Classes
# -------------------------------------------------------------------------

class DeribitDataManager:
    BASE_URL = "https://www.deribit.com/api/v2/public/"

    def __init__(self, currency: str = 'BTC'):
        self.currency = currency.upper()

    def get_spot_price(self) -> float:
        params = {'index_name': f'{self.currency.lower()}_usd'}
        try:
            resp = requests.get(f"{self.BASE_URL}get_index_price", params=params).json()
            return float(resp['result']['index_price'])
        except Exception as e:
            print(f"Error fetching spot: {e}")
            return 0.0

    def get_historical_data(self, days: int = 90) -> pd.DataFrame:
        print(f"Fetching last {days} days of historical data for Jump Estimation...")
        end_ts = int(datetime.now().timestamp() * 1000)
        start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        params = {
            'instrument_name': f'{self.currency}-PERPETUAL',
            'start_timestamp': start_ts,
            'end_timestamp': end_ts,
            'resolution': '60'  # hourly
        }
        url = f"{self.BASE_URL}get_tradingview_chart_data"
        try:
            resp = requests.get(url, params=params).json()
            if 'result' not in resp or resp['result']['status'] != 'ok':
                raise ValueError("Failed to fetch history")

            df = pd.DataFrame({'timestamp': resp['result']['ticks'], 'close': resp['result']['close']})
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error fetching history: {e}")
            return pd.DataFrame()

    def get_liquid_chain(self, target_days: int = 30) -> Tuple[float, float, pd.DataFrame]:
        print(f"Fetching {self.currency} option chain...")
        params = {'currency': self.currency, 'kind': 'option'}
        try:
            resp = requests.get(f"{self.BASE_URL}get_book_summary_by_currency", params=params).json()
        except Exception as e:
            raise ValueError(f"API Error: {e}")

        data = []
        for item in resp.get('result', []):
            inst_name = item['instrument_name']
            parts = inst_name.split('-')
            # Format: BTC-29MAR24-60000-C
            if len(parts) != 4 or parts[-1] != 'C':
                continue
            mp = item.get('mark_price', 0.0)
            if mp == 0:
                continue

            try:
                expiry_date = datetime.strptime(parts[1], "%d%b%y")
                strike = float(parts[2])
                data.append({'strike': strike, 'mark_price': float(mp), 'expiry': expiry_date})
            except ValueError:
                continue

        df = pd.DataFrame(data)
        if df.empty:
            raise ValueError("No valid options found.")

        now = datetime.now()
        df['days_to_expiry'] = (df['expiry'] - now).dt.total_seconds() / 86400.0
        # Filter for future expiries
        df = df[df['days_to_expiry'] > 0.5]

        if df.empty:
            raise ValueError("No future expiries found.")

        # Pick expiry closest to target
        chosen_days = min(df['days_to_expiry'].unique(), key=lambda x: abs(x - target_days))
        chain = df[df['days_to_expiry'] == chosen_days].copy()

        spot = self.get_spot_price()
        if spot == 0:
            raise ValueError("Invalid spot price")

        # Filter strikes: +/- 20% around spot for liquidity
        chain = chain[(chain['strike'] > spot * 0.80) & (chain['strike'] < spot * 1.20)]
        chain = chain.sort_values('strike').reset_index(drop=True)

        # Normalize
        chain['norm_strike'] = chain['strike'] / spot
        chain['norm_call'] = chain['mark_price'] # Deribit prices are already in BTC (C/S)

        return spot, float(chosen_days / 365.0), chain


class JumpEstimator:
    @staticmethod
    def estimate(df: pd.DataFrame) -> Tuple[float, float, float]:
        if df.empty:
            return 0.1, 0.0, 0.01

        df = df.copy()
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df = df.dropna()

        mean_ret = df['log_ret'].mean()
        std_ret = df['log_ret'].std()

        # 3-sigma jump detection
        threshold = 3.0 * std_ret

        jumps = df[np.abs(df['log_ret'] - mean_ret) > threshold]['log_ret']

        hours_per_year = 24 * 365
        total_hours = len(df)
        n_jumps = len(jumps)

        if n_jumps == 0:
            print("No jumps detected. Using nominal defaults.")
            return 0.1, 0.0, 0.01

        lambda_j = n_jumps * (hours_per_year / total_hours)
        mu_j = float(jumps.mean())
        sigma_j = float(jumps.std(ddof=1)) if len(jumps) > 1 else float(std_ret * 2)

        print(f"  -> Found {n_jumps} jumps in {total_hours} hours.")
        return float(lambda_j), mu_j, sigma_j


def implied_digital_from_calls(K: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Digital(K) ≈ -dC/dK
    """
    K = np.asarray(K, float)
    C = np.asarray(C, float)
    # Gradient handles non-uniform spacing
    dC_dK = np.gradient(C, K)
    # Digital price is negative derivative of call wrt strike
    dig = -dC_dK
    return np.clip(dig, 0.0, 1.0)


class BatesCalibrator:
    def __init__(self, currency: str = 'BTC'):
        self.data_manager = DeribitDataManager(currency)

    def calibrate(self) -> Tuple[Dict[str, float], float]:
        spot, T, chain = self.data_manager.get_liquid_chain(target_days=30)
        hist_data = self.data_manager.get_historical_data(days=90)
        lambda_j, mu_j, sigma_j = JumpEstimator.estimate(hist_data)

        print(f"Calibrating Bates on IMPLIED DIGITALS: Spot=${spot:.2f} | T={T:.3f}y")
        print(f"Fixed Jump Params: Lambda={lambda_j:.2f}, Mu_J={mu_j:.4f}, Sigma_J={sigma_j:.4f}")

        # Build market implied digital curve
        K = chain['norm_strike'].to_numpy()
        C_mkt = chain['norm_call'].to_numpy()
        D_mkt = implied_digital_from_calls(K, C_mkt)

        # Initial guess: [kappa, theta, sigma_v, rho, v0]
        initial_guess = np.array([2.0, 0.25, 1.0, -0.5, 0.25], dtype=float)

        def objective(h_params: np.ndarray) -> float:
            kappa, theta, sigma_v, rho, v0 = h_params

            # Soft boundaries
            if not (0.1 <= kappa <= 25.0 and
                    1e-4 <= theta <= 2.0 and
                    0.01 <= sigma_v <= 10.0 and
                    -0.99 <= rho <= 0.99 and
                    1e-4 <= v0 <= 2.0):
                return 1e12

            full_params = [kappa, theta, sigma_v, rho, v0, lambda_j, mu_j, sigma_j]

            # Model digitals
            # Use T as both expiry and initial_T for calibration snapshot
            D_mod = np.array([
                BatesModel.price_binary_call(1.0, float(k), T, T, full_params)
                for k in K
            ], dtype=float)

            # Weights: Focus on ATM
            atm_w = np.exp(-((K - 1.0) / 0.08) ** 2)
            w = 0.25 + 0.75 * atm_w

            err = np.sum(w * (D_mod - D_mkt) ** 2)

            # Regularization
            err += 0.05 * (sigma_v ** 2) + 0.02 * (v0 ** 2)

            return float(err)

        print("Optimizing Heston diffusion parameters...")
        result = minimize(
            objective,
            initial_guess,
            method='Nelder-Mead',
            tol=1e-5,
            options={'maxiter': 300, 'disp': True}
        )

        p = result.x
        final_params = {
            'kappa': float(round(p[0], 6)),
            'theta': float(round(p[1], 6)),
            'sigma_v': float(round(p[2], 6)),
            'rho': float(round(p[3], 6)),
            'v0': float(round(p[4], 6)),
            'lambda_j': float(round(lambda_j, 6)),
            'mu_j': float(round(mu_j, 6)),
            'sigma_j': float(round(sigma_j, 6)),
        }
        return final_params, spot


if __name__ == "__main__":
    currencies = ['BTC', 'ETH']
    all_results = []

    print("--- BATES MODEL MULTI-ASSET CALIBRATION (OPTIMIZED) ---")
    if HAS_NUMBA:
        print(">> Numba JIT detected. Acceleration ENABLED.")
    else:
        print(">> Numba not found. Running in standard Python mode.")

    for currency in currencies:
        print(f"\n[{currency}] STARTING CALIBRATION...")
        try:
            calibrator = BatesCalibrator(currency)
            params, current_spot = calibrator.calibrate()

            print(f"[{currency}] Calibrated Parameters:")
            for k, v in params.items():
                print(f"  {k}: {v}")

            record = {'currency': currency, 'spot': current_spot, 'timestamp': datetime.now().isoformat()}
            record.update(params)
            all_results.append(record)

        except Exception as e:
            print(f"[{currency}] Error during calibration: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        df_results = pd.DataFrame(all_results)
        filename = "bates_params_digital.jsonl"
        df_results.to_json(filename, orient='records', lines=True)
        print(f"\n--- SUCCESS ---")
        print(f"Parameters for {len(all_results)} assets saved to '{filename}'.")
        print(df_results)
    else:
        print("\n--- NO RESULTS ---")