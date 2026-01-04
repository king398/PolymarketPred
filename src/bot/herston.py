import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings("ignore")


eps = 1e-12


class BatesModel:
    """
    Bates = Heston stochastic vol + Merton lognormal jumps.
    We implement:
      - vanilla call via P1/P2 (Gil-Pelaez)
      - digital call = P2 (risk-neutral Prob(ST > K))
      - optional "expiry crush" adjustment for short-duration binaries
    """

    @staticmethod
    def _adjust_params_for_expiry(T: float, initial_T: float, params) -> List[float]:
        """
        Always returns a flat list of 8 python floats:
        [kappa, theta, sigma_v, rho, v0, lambda_j, mu_j, sigma_j]
        """

        # --- normalize input into a flat list ---
        if isinstance(params, dict):
            p = [
                params["kappa"], params["theta"], params["sigma_v"], params["rho"],
                params["v0"], params["lambda_j"], params["mu_j"], params["sigma_j"]
            ]
        else:
            # if someone accidentally passes nested stuff, flatten 1 level
            if len(params) == 4 and isinstance(params[0], (list, tuple, np.ndarray)):
                # e.g. [[kappa,theta,sigma_v,rho,v0], lambda, mu, sigma]
                p = list(params[0]) + list(params[1:])
            else:
                p = list(params)

        # --- force numeric floats (this prevents your TypeError) ---
        try:
            p = [float(x) for x in p]
        except Exception as e:
            raise TypeError(f"Non-numeric Bates params detected: {p}") from e

        # Protect against division by zero
        if initial_T <= 0:
            initial_T = T + 1e-9

        pct_remaining = max(0.0, min(1.0, T / initial_T))
        minutes_remaining = T * 365.0 * 24.0 * 60.0

        kill_threshold = 0.2
        if pct_remaining < kill_threshold:
            zone_progress = pct_remaining / kill_threshold
            decay_factor = zone_progress ** 8
            p[2] *= decay_factor  # sigma_v
            p[5] *= decay_factor  # lambda_j
            p[4] = max(1e-6, p[4] * (zone_progress ** 2))  # v0

        if minutes_remaining < 5:
            locking_factor = float(np.exp((15.0 - minutes_remaining) / 3.0))
            p[0] *= locking_factor  # kappa

        return p


    @staticmethod
    def _cf(phi: float, T: float, params: List[float], P_num: int) -> complex:
        # params: [kappa, theta, sigma_v, rho, v0, lambda_j, mu_j, sigma_j]
        kappa, theta, sigma_v, rho, v0, lambda_j, mu_j, sigma_j = params

        i = 1j
        S_internal = 1.0  # we work in normalized S space

        if P_num == 1:
            u, b = 0.5, kappa - rho * sigma_v
        else:
            u, b = -0.5, kappa

        a = kappa * theta
        sig_v2 = sigma_v * sigma_v

        d_sq = (rho * sigma_v * phi * i - b) ** 2 - sig_v2 * (2 * u * phi * i - phi * phi)
        d = np.sqrt(d_sq)

        g_num = b - rho * sigma_v * phi * i - d
        g_den = b - rho * sigma_v * phi * i + d
        g = g_num / (g_den + eps)

        exp_dt = np.exp(-d * T)

        C = (a / (sig_v2 + eps)) * (g_num * T - 2.0 * np.log((1 - g * exp_dt) / (1 - g + eps)))
        D = (g_num / (sig_v2 + eps)) * ((1 - exp_dt) / (1 - g * exp_dt + eps))

        heston_cf = np.exp(C + D * v0 + i * phi * np.log(S_internal))

        # Merton jumps
        k = np.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0
        jump_cf = np.exp(
            lambda_j * T * (np.exp(i * phi * mu_j - 0.5 * sigma_j * sigma_j * phi * phi) - 1.0)
            - i * phi * lambda_j * k * T
        )

        return heston_cf * jump_cf

    @staticmethod
    def _p_integrand(phi: float, K_norm: float, T: float, params: List[float], P_num: int) -> float:
        i = 1j
        cf = BatesModel._cf(phi, T, params, P_num)
        return np.real(np.exp(-i * phi * np.log(K_norm)) * cf / (i * phi + eps))

    @staticmethod
    def _P(K_norm: float, T: float, params: List[float], P_num: int, limit: float = 400.0) -> float:
        # Gil-Pelaez: P = 1/2 + 1/pi ∫ Re( e^{-i phi ln K} CF / (i phi) ) dphi
        val = 0.5 + (1.0 / np.pi) * quad(
            BatesModel._p_integrand, 1e-8, limit, args=(K_norm, T, params, P_num), limit=200
        )[0]
        return float(np.clip(val, 0.0, 1.0))

    @staticmethod
    def price_vanilla_call(S: float, K: float, T: float, params: List[float]) -> float:
        # r=0, q=0, normalized CF uses S_internal=1; scale handled by K_norm = K/S
        if T <= 0:
            return max(S - K, 0.0)

        K_norm = K / S
        P1 = BatesModel._P(K_norm, T, params, P_num=1, limit=400.0)
        P2 = BatesModel._P(K_norm, T, params, P_num=2, limit=400.0)
        return float(np.clip(S * P1 - K * P2, 0.0, 1e9))

    @staticmethod
    def price_binary_call(S: float, K: float, T: float, initial_T: float, params: List[float]) -> float:
        """
        Digital call paying 1 if ST > K.
        Your short-expiry 'crush' is applied here (optional).
        """
        if T <= 0:
            return 1.0 if S > K else 0.0

        adj_params = BatesModel._adjust_params_for_expiry(T, initial_T, params)
        K_norm = K / S

        # For digitals, P2 is the price (r~0)
        try:
            P2 = BatesModel._P(K_norm, T, adj_params, P_num=2, limit=1000.0)
            return float(np.clip(P2, 0.0, 1.0))
        except Exception:
            return 1.0 if S > K else 0.0


class DeribitDataManager:
    BASE_URL = "https://www.deribit.com/api/v2/public/"

    def __init__(self, currency: str = 'BTC'):
        self.currency = currency.upper()

    def get_spot_price(self) -> float:
        params = {'index_name': f'{self.currency.lower()}_usd'}
        resp = requests.get(f"{self.BASE_URL}get_index_price", params=params).json()
        return float(resp['result']['index_price'])

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
        resp = requests.get(url, params=params).json()
        if 'result' not in resp or resp['result']['status'] != 'ok':
            raise ValueError("Failed to fetch history")

        df = pd.DataFrame({'timestamp': resp['result']['ticks'], 'close': resp['result']['close']})
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def get_liquid_chain(self, target_days: int = 30) -> Tuple[float, float, pd.DataFrame]:
        print(f"Fetching {self.currency} option chain...")
        params = {'currency': self.currency, 'kind': 'option'}
        resp = requests.get(f"{self.BASE_URL}get_book_summary_by_currency", params=params).json()

        data = []
        for item in resp['result']:
            inst_name = item['instrument_name']
            parts = inst_name.split('-')
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
        chosen_days = min(df['days_to_expiry'].unique(), key=lambda x: abs(x - target_days))

        chain = df[df['days_to_expiry'] == chosen_days].copy()
        spot = self.get_spot_price()

        chain = chain[(chain['strike'] > spot * 0.80) & (chain['strike'] < spot * 1.20)]
        chain = chain.sort_values('strike').reset_index(drop=True)

        chain['norm_strike'] = chain['strike'] / spot
        # Deribit mark_price is quoted in underlying units -> already dimensionless when dividing by spot later.
        # For our normalized S=1 world, treat mark_price as "C/Spot".
        chain['norm_call'] = chain['mark_price']

        return spot, float(chosen_days / 365.0), chain


class JumpEstimator:
    @staticmethod
    def estimate(df: pd.DataFrame) -> Tuple[float, float, float]:
        df = df.copy()
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df = df.dropna()

        mean_ret = df['log_ret'].mean()
        std_ret = df['log_ret'].std()
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
        print(f"  -> Threshold: +/- {threshold:.4f} log-ret")
        return float(lambda_j), mu_j, sigma_j


def implied_digital_from_calls(K: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Digital(K) ≈ -dC/dK
    K and C are in normalized units (S=1, C=Call/Spot).
    """
    K = np.asarray(K, float)
    C = np.asarray(C, float)

    # gradient handles non-uniform spacing too
    dC_dK = np.gradient(C, K)
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

        # Build market implied digital curve from vanilla calls
        K = chain['norm_strike'].to_numpy()
        C_mkt = chain['norm_call'].to_numpy()
        D_mkt = implied_digital_from_calls(K, C_mkt)

        # Heston diffusion initial guess: [kappa, theta, sigma_v, rho, v0]
        initial_guess = np.array([2.0, 0.25, 1.0, -0.5, 0.25], dtype=float)

        def objective(h_params: np.ndarray) -> float:
            kappa, theta, sigma_v, rho, v0 = h_params

            # soft bounds
            if not (0.1 <= kappa <= 25.0 and
                    1e-4 <= theta <= 2.0 and
                    0.01 <= sigma_v <= 10.0 and
                    -0.99 <= rho <= 0.99 and
                    1e-4 <= v0 <= 2.0):
                return 1e12

            full_params = [kappa, theta, sigma_v, rho, v0, lambda_j, mu_j, sigma_j]

            # Model digitals directly via P2 (your binary pricer)
            # initial_T should be the session's start-to-expiry time; for a single expiry, use initial_T=T
            D_mod = np.array([
                BatesModel.price_binary_call(1.0, float(k), T, T, full_params)
                for k in K
            ], dtype=float)

            # Weighting: emphasize around-the-money digitals (most informative about tail + vol)
            atm_w = np.exp(-((K - 1.0) / 0.08) ** 2)  # ~8% band
            w = 0.25 + 0.75 * atm_w

            err = np.sum(w * (D_mod - D_mkt) ** 2)

            # regularize too-crazy sigma_v + v0 (prevents fat tails that keep digitals mushy)
            err += 0.05 * (sigma_v ** 2) + 0.02 * (v0 ** 2)

            return float(err)

        print("Optimizing Heston diffusion parameters (fit digitals)...")
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

    print("--- BATES MODEL MULTI-ASSET CALIBRATION (DIGITAL FIT) ---")
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
        filename = "/home/mithil/PycharmProjects/PolymarketPred/data/bates_params_digital.jsonl"
        df_results.to_json(filename, orient='records', lines=True)
        print(f"\n--- SUCCESS ---")
        print(f"Parameters for {len(all_results)} assets saved to '{filename}'.")
        print(df_results)
    else:
        print("\n--- NO RESULTS ---")
        print("Calibration failed for all assets.")
