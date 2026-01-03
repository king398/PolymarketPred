import numpy as np
import requests
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional
import warnings

warnings.filterwarnings("ignore")

# --- 1. MODELING LOGIC (BATES: HESTON + JUMPS) ---

class BatesModel:
    """
    Implements the Bates Model (Heston + Merton Jumps).
    """

    @staticmethod
    def _integrand(phi: float, K_norm: float, T: float, params: List[float], P_num: int) -> float:
        """
        Bates Characteristic Function = Heston CF * Jump CF
        """
        # Unpack params:
        # Heston: [kappa, theta, sigma_v, rho, v0]
        # Jumps:  [lambda_j, mu_j, sigma_j]
        kappa, theta, sigma_v, rho, v0, lambda_j, mu_j, sigma_j = params

        S_internal = 1.0
        i = 1j

        # --- Heston Component ---
        if P_num == 1:
            u, b = 0.5, kappa - rho * sigma_v
        else:
            u, b = -0.5, kappa

        a = kappa * theta
        d_sq = (rho * sigma_v * phi * i - b)**2 - sigma_v**2 * (2 * u * phi * i - phi**2)
        d = np.sqrt(d_sq)
        g = (b - rho * sigma_v * phi * i - d) / (b - rho * sigma_v * phi * i + d)

        C = (a / sigma_v**2) * ((b - rho * sigma_v * phi * i - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
        D = (b - rho * sigma_v * phi * i - d) / sigma_v**2 * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))

        heston_cf = np.exp(C + D * v0 + i * phi * np.log(S_internal))

        # --- Jump Component (Merton) ---
        # We need to adjust the drift so the process remains risk-neutral (Martingale condition).
        # The compensator 'k' ensures E[S_T] = S_0 * e^rT

        # Expected percentage jump size
        k = np.exp(mu_j + 0.5 * sigma_j**2) - 1

        # Jump Characteristic Function Part
        # Standard formula: exp( lambda * T * ( E[e^{i*phi*J}] - 1 - i*phi*k ) )
        # where J is Normal(mu_j, sigma_j)

        jump_part_val = np.exp(i * phi * mu_j - 0.5 * sigma_j**2 * phi**2) # Characteristic func of Normal dist for log-jump

        # Note: If P_num==1 (calculating Delta), the measure changes, but typically
        # for standard pricing we use the risk-neutral form directly or transform u.
        # Below is the standard Bates CF adaptation for the P1/P2 formulation:

        if P_num == 1:
            # Under Measure Q1, the jump intensity and mean change
            # However, a common approximation in code libraries is to stick to the generalized CF
            # with the specific u parameters (0.5 vs -0.5) handling the transform.
            # We simply add the Jump Exponent to the Heston Exponent.

            # Re-calculating u for Jump CF context explicitly:
            # The input 'phi' here is actually the transformed variable from the Gil-Pelaez formula.
            pass

        # Explicit Bates Log-CF addition:
        # Term 1: Random Jump Diff
        # Term 2: Drift Correction (-lambda * k * i * phi * T)

        jump_drift_correction = -lambda_j * k * i * phi * T
        jump_randomness = lambda_j * T * (np.exp(mu_j * i * phi - 0.5 * sigma_j**2 * phi**2) - 1)

        # Adjust for P1 vs P2 shifts if necessary.
        # Heston standard approach handles P1/P2 via 'u' and 'b'.
        # For Jumps, we can use the standard characteristic exponent with the specific phi.

        # A robust way for P1/P2 integration is to treat the Jump component as an independent factor:
        jump_cf = np.exp(lambda_j * T * (np.exp(i * phi * mu_j - 0.5 * sigma_j**2 * phi**2 + (i*phi if P_num==1 else 0)*sigma_j**2 ) - 1)
                         - i * phi * lambda_j * k * T)

        # Simplified standard Bates CF used in most libraries (Gatheral):
        # We add the jump exponent to the total exponent.
        # The drift correction is vital.

        val_jump = lambda_j * T * (np.exp(mu_j * i * phi - 0.5 * sigma_j**2 * phi**2) - 1) - i * phi * lambda_j * k * T

        # Apply Gil-Pelaez normalization
        full_cf = heston_cf * np.exp(val_jump)

        return np.real(np.exp(-i * phi * np.log(K_norm)) * full_cf / (i * phi))

    @staticmethod
    def price_vanilla_call(S: float, K: float, T: float, params: List[float]) -> float:
        K_norm = K / S
        # params: [kappa, theta, sigma_v, rho, v0, lambda, mu_j, sigma_j]
        P1 = 0.5 + (1 / np.pi) * quad(BatesModel._integrand, 1e-8, 200, args=(K_norm, T, params, 1))[0]
        P2 = 0.5 + (1 / np.pi) * quad(BatesModel._integrand, 1e-8, 200, args=(K_norm, T, params, 2))[0]
        return S * P1 - K * P2

    @staticmethod
    def price_binary_call(S: float, K: float, T: float, params: List[float]) -> float:
        K_norm = K / S
        P2 = 0.5 + (1 / np.pi) * quad(BatesModel._integrand, 1e-8, 5000, args=(K_norm, T, params, 2))[0]
        return P2


# --- 2. DATA MANAGEMENT (DERIBIT) ---

class DeribitDataManager:
    BASE_URL = "https://www.deribit.com/api/v2/public/"

    def __init__(self, currency: str = 'BTC'):
        self.currency = currency.upper()

    def get_spot_price(self) -> float:
        params = {'index_name': f'{self.currency.lower()}_usd'}
        try:
            resp = requests.get(f"{self.BASE_URL}get_index_price", params=params).json()
            return float(resp['result']['index_price'])
        except Exception:
            return 0.0

    def get_historical_data(self, days: int = 30) -> pd.DataFrame:
        """Fetches hourly OHLC data for the last 'days'."""
        print(f"Fetching last {days} days of historical data for Jump Estimation...")
        end_ts = int(datetime.now().timestamp() * 1000)
        start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        # '60' resolution = 1 hour. Good balance between granularity and noise.
        params = {
            'instrument_name': f'{self.currency}-PERPETUAL',
            'start_timestamp': start_ts,
            'end_timestamp': end_ts,
            'resolution': '1'
        }

        url = f"{self.BASE_URL}get_tradingview_chart_data"
        resp = requests.get(url, params=params).json()

        if 'result' not in resp or resp['result']['status'] != 'ok':
            raise ValueError("Failed to fetch history")

        df = pd.DataFrame({
            'timestamp': resp['result']['ticks'],
            'close': resp['result']['close']
        })
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def get_liquid_chain(self, target_days: int = 3) -> Tuple[float, float, pd.DataFrame]:
        print(f"Fetching {self.currency} option chain...")
        params = {'currency': self.currency, 'kind': 'option'}
        resp = requests.get(f"{self.BASE_URL}get_book_summary_by_currency", params=params).json()

        data = []
        for item in resp['result']:
            inst_name = item['instrument_name']
            parts = inst_name.split('-')
            if len(parts) != 4 or parts[-1] != 'C': continue
            if item.get('mark_price', 0) == 0: continue

            try:
                expiry_date = datetime.strptime(parts[1], "%d%b%y")
                strike = float(parts[2])
                data.append({'strike': strike, 'price_btc': item['mark_price'], 'expiry': expiry_date})
            except ValueError:
                continue

        df = pd.DataFrame(data)
        if df.empty: raise ValueError("No valid options found.")

        now = datetime.now()
        df['days_to_expiry'] = (df['expiry'] - now).dt.total_seconds() / 86400

        unique_expiries = df['days_to_expiry'].unique()
        chosen_days = min(unique_expiries, key=lambda x: abs(x - target_days))

        chain = df[df['days_to_expiry'] == chosen_days].copy()
        spot = self.get_spot_price()

        # Filter for liquidity near the money
        chain = chain[(chain['strike'] > spot * 0.80) & (chain['strike'] < spot * 1.20)]
        chain['norm_strike'] = chain['strike'] / spot
        chain['norm_price'] = (chain['price_btc'] * spot) / spot # Normalized price

        return spot, chosen_days / 365.0, chain.sort_values('strike')


# --- 3. JUMP ESTIMATION (STATISTICAL) ---

class JumpEstimator:
    """
    Analyzes historical returns to estimate Merton Jump parameters:
    Lambda (frequency), Mu_J (mean jump), Sigma_J (jump vol).
    """
    @staticmethod
    def estimate(df: pd.DataFrame) -> Tuple[float, float, float]:
        # 1. Calculate Log Returns
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df = df.dropna()

        # 2. Statistics
        mean_ret = df['log_ret'].mean()
        std_ret = df['log_ret'].std()

        # 3. Identify Jumps (Thresholding)
        # We assume "Normal" volatility is within 3 standard deviations.
        # Anything outside 3 sigma is considered a "Jump".
        threshold = 3.0 * std_ret

        jumps = df[np.abs(df['log_ret'] - mean_ret) > threshold]['log_ret']

        # 4. Calculate Parameters
        # Time conversion: Data is hourly (resolution='60')
        hours_per_year = 24 * 365
        total_hours = len(df)

        # Lambda: Expected jumps per year
        # If we saw 5 jumps in 720 hours (30 days), lambda = 5 * (8760 / 720)
        n_jumps = len(jumps)

        if n_jumps == 0:
            # Fallback if market was quiet
            print("No jumps detected in 30d. Using nominal defaults.")
            return 0.1, 0.0, 0.01

        lambda_j = n_jumps * (hours_per_year / total_hours)

        # Jump Mean and Vol (Parameters of the Jump Distribution)
        # Note: 'jumps' contains the actual return values.
        # Merton assumes J ~ N(mu_j, sigma_j).
        mu_j = jumps.mean()
        sigma_j = jumps.std(ddof=1) if len(jumps) > 1 else std_ret * 2

        print(f"  -> Found {n_jumps} jumps in {total_hours} hours.")
        print(f"  -> Threshold: +/- {threshold:.4f} log-ret")

        return lambda_j, mu_j, sigma_j


# --- 4. CALIBRATION & EXECUTION ---

class BatesCalibrator:
    def __init__(self, currency: str = 'BTC'):
        self.data_manager = DeribitDataManager(currency)

    def calibrate(self) -> Tuple[Dict[str, float], float]:
        # A. GET OPTION CHAIN
        spot, T, chain = self.data_manager.get_liquid_chain()

        # B. GET JUMP PARAMETERS (Historical 30d)
        hist_data = self.data_manager.get_historical_data(days=30)
        lambda_j, mu_j, sigma_j = JumpEstimator.estimate(hist_data)

        print(f"Calibrating Bates: Spot=${spot:.2f} | T={T:.3f}y")
        print(f"Fixed Jump Params (30d hist): Lambda={lambda_j:.2f}, Mu_J={mu_j:.4f}, Sigma_J={sigma_j:.4f}")

        # C. CALIBRATE HESTON PART (Kappa, Theta, Sigma_V, Rho, V0)
        # We fix the Jump parameters and only optimize the Heston diffusion parameters.

        # Initial guess: [kappa, theta, sigma_v, rho, v0]
        initial_guess = [2.0, 0.4, 0.5, -0.5, 0.04]
        bounds = ((0.1, 10.0), (0.01, 2.0), (0.01, 5.0), (-0.99, 0.99), (0.01, 1.0))

        def objective(h_params):
            # Combine Heston params (variable) with Jump params (fixed)
            full_params = list(h_params) + [lambda_j, mu_j, sigma_j]
            sse = 0.0
            for _, row in chain.iterrows():
                # Pricing using S=1.0 for normalized calculation
                model_price = BatesModel.price_vanilla_call(1.0, row['norm_strike'], T, full_params)
                sse += (model_price - row['norm_price'])**2
            return sse

        print("Optimizing Heston Diffusion parameters...")
        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, tol=1e-5)

        p = result.x
        final_params = {
            'kappa': round(p[0], 4), 'theta': round(p[1], 4), 'sigma_v': round(p[2], 4),
            'rho': round(p[3], 4), 'v0': round(p[4], 4),
            'lambda_j': round(lambda_j, 4), 'mu_j': round(mu_j, 4), 'sigma_j': round(sigma_j, 4)
        }
        return final_params, spot

if __name__ == "__main__":
    try:
        calibrator = BatesCalibrator('SOLUSDC')
        params, current_spot = calibrator.calibrate()

        print("\n--- BATES MODEL CALIBRATED PARAMETERS ---")
        print(" (First 5 calibrated to Option Chain, Last 3 calculated from 30d History)")
        for k, v in params.items():
            print(f"{k:>10}: {v}")

        # --- PREDICTION / USAGE ---
        my_spot = 89711.99

        my_strike = 89871.04
        my_time = 2/(365*24*60)

        # Convert dict to list for the model function
        param_list = list(params.values())

        # Price a Binary Call (Probability of ITM)
        prob = BatesModel.price_binary_call(my_spot, my_strike, my_time, param_list)
        vanilla_price = BatesModel.price_vanilla_call(my_spot, my_strike, my_time, param_list)

        print(f"\n--- PREDICTION (7 Days) ---")
        print(f"Spot: ${my_spot:.2f} -> Strike: ${my_strike:.2f}")
        print(f"Bates Call Price: ${vanilla_price:.2f}")
        print(f"Prob ITM:         {prob*100:.2f}%")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()