import numpy as np
import requests
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional
import warnings

warnings.filterwarnings("ignore")

class BatesModel:
    """
    Implements the Bates Model, which extends the Heston Stochastic Volatility model
    by adding Merton Log-Normal Jumps to the spot price process.

    The Characteristic Function (CF) is the product of the Heston CF and the Merton Jump CF.
    """

    @staticmethod
    def _integrand(phi: float, K_norm: float, T: float, params: List[float], P_num: int) -> float:
        """
        Calculates the integrand for the Gil-Pelaez inversion formula.

        Parameters:
            phi (float): Integration variable (frequency domain).
            K_norm (float): Strike price normalized by Spot (K/S).
            T (float): Time to expiry in years.
            params (List[float]): Model parameters in the following order:
                - kappa: Mean reversion speed of volatility.
                - theta: Long-run variance.
                - sigma_v: Volatility of volatility (vol-of-vol).
                - rho: Correlation between spot and variance Brownian motions.
                - v0: Initial variance.
                - lambda_j: Mean number of jumps per year (intensity).
                - mu_j: Mean log-jump size.
                - sigma_j: Standard deviation of log-jump size.
            P_num (int): 1 or 2, corresponding to the two probabilities P1 and P2 in the pricing formula.

        Returns:
            float: Real part of the characteristic function adapted for integration.
                """
        if isinstance(params, dict):
            kappa   = params['kappa']
            theta   = params['theta']
            sigma_v = params['sigma_v']
            rho     = params['rho']
            v0      = params['v0']
            lambda_j= params['lambda_j']
            mu_j    = params['mu_j']
            sigma_j = params['sigma_j']
        else:
            # Handle Legacy List Input (strict order)
            kappa, theta, sigma_v, rho, v0, lambda_j, mu_j, sigma_j = params
        S_internal = 1.0
        i = 1j

        # Define Heston parameters u and b based on the probability measure (P1 or P2)
        if P_num == 1:
            u, b = 0.5, kappa - rho * sigma_v
        else:
            u, b = -0.5, kappa

        # Heston Characteristic Function Components
        a = kappa * theta
        d_sq = (rho * sigma_v * phi * i - b)**2 - sigma_v**2 * (2 * u * phi * i - phi**2)
        d = np.sqrt(d_sq)
        g = (b - rho * sigma_v * phi * i - d) / (b - rho * sigma_v * phi * i + d)

        C = (a / sigma_v**2) * ((b - rho * sigma_v * phi * i - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
        D = (b - rho * sigma_v * phi * i - d) / sigma_v**2 * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))

        heston_cf = np.exp(C + D * v0 + i * phi * np.log(S_internal))

        # Merton Jump Component
        # k is the compensator required to maintain the Martingale condition for the risk-neutral drift.
        k = np.exp(mu_j + 0.5 * sigma_j**2) - 1

        # Calculate the Jump Drift Correction and Randomness
        # The term -i * phi * lambda_j * k * T ensures the process is drift-consistent.
        val_jump = lambda_j * T * (np.exp(mu_j * i * phi - 0.5 * sigma_j**2 * phi**2) - 1) - i * phi * lambda_j * k * T

        # Full Bates CF
        full_cf = heston_cf * np.exp(val_jump)

        return np.real(np.exp(-i * phi * np.log(K_norm)) * full_cf / (i * phi))

    @staticmethod
    def price_vanilla_call(S: float, K: float, T: float, params: List[float]) -> float:
        """
        Prices a European Call option using the Gil-Pelaez formula.
        """
        K_norm = K / S
        # Integration limit set to 200 for speed; adequate for smooth Heston integrands.
        P1 = 0.5 + (1 / np.pi) * quad(BatesModel._integrand, 1e-8, 200, args=(K_norm, T, params, 1))[0]
        P2 = 0.5 + (1 / np.pi) * quad(BatesModel._integrand, 1e-8, 200, args=(K_norm, T, params, 2))[0]
        return S * P1 - K * P2

    @staticmethod
    def price_binary_call(S: float, K: float, T: float, params: List[float]) -> float:
        """
        Prices a Binary Call (Cash-or-Nothing). Returns probability of ending ITM (P2).
        """
        K_norm = K / S
        P2 = 0.5 + (1 / np.pi) * quad(BatesModel._integrand, 1e-8, 5000, args=(K_norm, T, params, 2))[0]
        return P2


class DeribitDataManager:
    """
    Handles interaction with the Deribit public API to fetch spot prices,
    historical OHLC data, and option chains.
    """
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

    def get_historical_data(self, days: int = 5) -> pd.DataFrame:
        """
        Fetches hourly historical data to estimate jump parameters.
        Returns a DataFrame with 'timestamp' and 'close'.
        """
        print(f"Fetching last {days} days of historical data for Jump Estimation...")
        end_ts = int(datetime.now().timestamp() * 1000)
        start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

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

    def get_liquid_chain(self, target_days: int = 30) -> Tuple[float, float, pd.DataFrame]:
        """
        Fetches the Option Chain for a specific expiry target.
        Filters for liquidity and near-the-money strikes (80% to 120% of spot).
        Returns Spot, Time-to-Expiry (years), and the Chain DataFrame.
        """
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

        chain = chain[(chain['strike'] > spot * 0.80) & (chain['strike'] < spot * 1.20)]
        chain['norm_strike'] = chain['strike'] / spot
        chain['norm_price'] = (chain['price_btc'] * spot) / spot

        return spot, chosen_days / 365.0, chain.sort_values('strike')


class JumpEstimator:
    """
    Estimates Jump Diffusion parameters using historical time-series statistical analysis.
    Uses a thresholding method: returns > 3 standard deviations are classified as jumps.
    """
    @staticmethod
    def estimate(df: pd.DataFrame) -> Tuple[float, float, float]:
        """
        Returns:
            lambda_j: Annualized frequency of jumps.
            mu_j: Mean size of log-jumps.
            sigma_j: Volatility of log-jumps.
        """
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df = df.dropna()

        mean_ret = df['log_ret'].mean()
        std_ret = df['log_ret'].std()

        # Threshold for identifying a jump event
        threshold = 3.0 * std_ret

        jumps = df[np.abs(df['log_ret'] - mean_ret) > threshold]['log_ret']

        hours_per_year = 24 * 365
        total_hours = len(df)
        n_jumps = len(jumps)

        if n_jumps == 0:
            print("No jumps detected in 30d. Using nominal defaults.")
            return 0.1, 0.0, 0.01

        lambda_j = n_jumps * (hours_per_year / total_hours)
        mu_j = jumps.mean()
        sigma_j = jumps.std(ddof=1) if len(jumps) > 1 else std_ret * 2

        print(f"  -> Found {n_jumps} jumps in {total_hours} hours.")
        print(f"  -> Threshold: +/- {threshold:.4f} log-ret")

        return lambda_j, mu_j, sigma_j


class BatesCalibrator:
    """
    Orchestrates the calibration process:
    1. Fetches Option Chain and History.
    2. Estimates Jump parameters historically (fixed during calibration).
    3. Minimizes Sum of Squared Errors (SSE) to find optimal Heston diffusion parameters.
    """
    def __init__(self, currency: str = 'BTC'):
        self.data_manager = DeribitDataManager(currency)

    def calibrate(self) -> Tuple[Dict[str, float], float]:
        """
        Performs calibration using Nelder-Mead optimization.
        Returns a dictionary of calibrated parameters and the current spot price.
        """
        spot, T, chain = self.data_manager.get_liquid_chain()
        hist_data = self.data_manager.get_historical_data(days=90)
        lambda_j, mu_j, sigma_j = JumpEstimator.estimate(hist_data)

        print(f"Calibrating Bates: Spot=${spot:.2f} | T={T:.3f}y")
        print(f"Fixed Jump Params (30d hist): Lambda={lambda_j:.2f}, Mu_J={mu_j:.4f}, Sigma_J={sigma_j:.4f}")

        # Initial guesses suitable for crypto markets (high vol, mean reverting)
        # kappa, theta, sigma_v, rho, v0
        initial_guess = [2.0, 0.25, 1.0, -0.5, 0.25]

        # Bounds are enforced via penalty function in 'objective' because Nelder-Mead
        # does not support strict bounds natively, but it is robust for this error surface.
        def objective(h_params):
            # 1. Parameter Constraints (Soft Penalty)
            # kappa, theta, sigma_v, rho, v0
            if not (0.1 <= h_params[0] <= 15.0 and    # kappa (Increased upper bound for short-expiry sensitivity)
                    0.001 <= h_params[1] <= 1.0 and   # theta
                    0.01 <= h_params[2] <= 8.0 and    # sigma_v
                    -0.99 <= h_params[3] <= 0.99 and  # rho
                    0.001 <= h_params[4] <= 1.0):     # v0
                return 1e12

            full_params = list(h_params) + [lambda_j, mu_j, sigma_j]
            total_error = 0.0

            for _, row in chain.iterrows():
                market_price = row['norm_price']
                strike_norm = row['norm_strike']

                # Calculate Model Price
                model_price = BatesModel.price_vanilla_call(1.0, strike_norm, T, full_params)

                diff = model_price - market_price

                # --- WEIGHTING STRATEGY ---

                # 1. Relative Weighting:
                # This ensures we care about the 0.0001 price differences in OTM options
                # Adding 1e-5 prevents division by zero for deep OTM
                weight = 1.0 / (market_price**2 + 1e-5)

                # 2. Asymmetric "Hope" Penalty (Crucial for Binary Sensitivity):
                # If Model > Market (Model thinks there's a chance, Market says NO),
                # we penalize this error 5x harder. This kills "fat tails" that don't exist.
                if diff > 0 and strike_norm > 1.05: # Only applied to OTM calls
                    weight *= 5.0

                total_error += weight * (diff**2)

            # Regularization: Penalize excessive Vol-of-Vol (sigma_v)
            # High sigma_v keeps OTM options alive too long.
            total_error += 0.1 * (h_params[2] ** 2)

            return total_error
        print("Optimizing Heston Diffusion parameters (Nelder-Mead)...")
        result = minimize(objective, initial_guess, method='Nelder-Mead', tol=1e-4, options={'maxiter': 200, 'disp': True})

        p = result.x
        final_params = {
            'kappa': round(p[0], 4),
            'theta': round(p[1], 4),
            'sigma_v': round(p[2], 4),
            'rho': round(p[3], 4),
            'v0': round(p[4], 4),
            'lambda_j': round(lambda_j, 4),
            'mu_j': round(mu_j, 4),
            'sigma_j': round(sigma_j, 4)
        }
        return final_params, spot


if __name__ == "__main__":
    currencies = ['BTC', 'ETH']
    all_results = []

    print("--- BATES MODEL MULTI-ASSET CALIBRATION ---")

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
        filename = "/home/mithil/PycharmProjects/PolymarketPred/data/bates_params.jsonl"
        df_results.to_json(filename, orient='records', lines=True)
        print(f"\n--- SUCCESS ---")
        print(f"Parameters for {len(all_results)} assets saved to '{filename}'.")
        print(df_results)
    else:
        print("\n--- NO RESULTS ---")
        print("Calibration failed for all assets.")