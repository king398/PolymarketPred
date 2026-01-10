import warnings
from typing import List, Union, Dict

import numpy as np
from scipy.integrate import quad

warnings.filterwarnings("ignore")

eps = 1e-12



class HestonModel:
    """
    Pure Heston stochastic volatility model.

    Implements:
      - vanilla call via P1/P2 (Gil-Pelaez)
      - digital call = P2 (risk-neutral Prob(ST > K) when r=0)
      - optional short-expiry "crush" adjustment (same idea as your Bates code)
    """

    @staticmethod
    def _adjust_params_for_expiry(T: float, initial_T: float, params) -> List[float]:
        """
        Always returns a flat list of 5 python floats:
        [kappa, theta, sigma_v, rho, v0]
        """
        # normalize input into a flat list
        if isinstance(params, dict):
            p = [
                params["kappa"], params["theta"], params["sigma_v"],
                params["rho"], params["v0"]
            ]
        else:
            p = list(params)

        # force numeric floats
        try:
            p = [float(x) for x in p]
        except Exception as e:
            raise TypeError(f"Non-numeric Heston params detected: {p}") from e

        # Protect against division by zero
        if initial_T <= 0:
            initial_T = T + 1e-9

        pct_remaining = max(0.0, min(1.0, T / initial_T))
        minutes_remaining = T * 365.0 * 24.0 * 60.0

        # Same idea: damp vol-of-vol and v0 near expiry
        kill_threshold = 0.2
        if pct_remaining < kill_threshold:
            zone_progress = pct_remaining / kill_threshold
            decay_factor = zone_progress ** 12
            p[2] *= decay_factor              # sigma_v
            p[4] = max(1e-6, p[4] * (zone_progress ** 3))  # v0

        # Same "locking factor" heuristic on kappa in last minutes
        if minutes_remaining < 5:
            locking_factor = float(np.exp((15.0 - minutes_remaining) / 3.0))
            p[0] *= locking_factor  # kappa

        return p

    @staticmethod
    def _cf(phi: float, T: float, params: List[float], P_num: int) -> complex:
        # params: [kappa, theta, sigma_v, rho, v0]
        kappa, theta, sigma_v, rho, v0 = params

        i = 1j
        S_internal = 1.0  # normalized S space

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

        return np.exp(C + D * v0 + i * phi * np.log(S_internal))

    @staticmethod
    def _p_integrand(phi: float, K_norm: float, T: float, params: List[float], P_num: int) -> float:
        i = 1j
        cf = HestonModel._cf(phi, T, params, P_num)
        return np.real(np.exp(-i * phi * np.log(K_norm)) * cf / (i * phi + eps))

    @staticmethod
    def _P(K_norm: float, T: float, params: List[float], P_num: int, limit: float = 400.0) -> float:
        val = 0.5 + (1.0 / np.pi) * quad(
            HestonModel._p_integrand, 1e-8, limit, args=(K_norm, T, params, P_num), limit=200
        )[0]
        return float(np.clip(val, 0.0, 1.0))

    @staticmethod
    def price_vanilla_call(S: float, K: float, T: float, params: List[float]) -> float:
        if T <= 0:
            return max(S - K, 0.0)

        K_norm = K / S
        P1 = HestonModel._P(K_norm, T, params, P_num=1, limit=400.0)
        P2 = HestonModel._P(K_norm, T, params, P_num=2, limit=400.0)
        return float(np.clip(S * P1 - K * P2, 0.0, 1e18))

    @staticmethod
    def price_binary_call(S: float, K: float, T: float, initial_T: float, params: List[float]) -> float:
        """
        Digital call paying 1 if ST > K (r≈0).
        """
        if T <= 0:
            return 1.0 if S > K else 0.0

        adj_params = HestonModel._adjust_params_for_expiry(T, initial_T, params)
        K_norm = K / S

        try:
            P2 = HestonModel._P(K_norm, T, adj_params, P_num=2, limit=2000.0)
            return float(np.clip(P2, 0.0, 1.0))
        except Exception:
            return 1.0 if S > K else 0.0



class FastHestonModel:
    """
    Optimized Heston Model using Vectorized Gauss-Legendre Quadrature.
    Significant speedup over scipy.integrate.quad.
    """

    # Pre-compute Legendre nodes/weights for standard interval [-1, 1]
    # 64 points is usually sufficient for Heston, 100 is very safe.
    _gl_deg = 100
    _gl_x, _gl_w = np.polynomial.legendre.leggauss(_gl_deg)

    @staticmethod
    def _adjust_params_for_expiry(T: float, initial_T: float, params) -> List[float]:
        # (Same logic as original, just ensuring type safety)
        if isinstance(params, dict):
            p = [params["kappa"], params["theta"], params["sigma_v"], params["rho"], params["v0"]]
        else:
            p = list(params)

        try:
            p = [float(x) for x in p]
        except Exception as e:
            raise TypeError(f"Non-numeric params: {p}") from e

        if initial_T <= 0: initial_T = T + 1e-9

        pct = max(0.0, min(1.0, T / initial_T))
        mins = T * 525600.0 # 365*24*60

        kill_threshold = 0.2
        if pct < kill_threshold:
            progress = pct / kill_threshold
            decay = progress ** 12
            p[2] *= decay
            p[4] = max(1e-6, p[4] * (progress ** 3))

        if mins < 5:
            lock = float(np.exp((15.0 - mins) / 3.0))
            p[0] *= lock

        return p

    @staticmethod
    def _cf_vectorized(phi: np.ndarray, T: float, params: List[float], P_num: int) -> np.ndarray:
        """
        Vectorized Characteristic Function.
        phi is a numpy array, returns a numpy array of complex values.
        """
        kappa, theta, sigma_v, rho, v0 = params
        i = 1j

        # P1 vs P2 logic
        if P_num == 1:
            u = 0.5
            b = kappa - rho * sigma_v
        else:
            u = -0.5
            b = kappa

        a = kappa * theta
        sig_v2 = sigma_v * sigma_v

        # Vectorized math
        d_sq = (rho * sigma_v * phi * i - b)**2 - sig_v2 * (2 * u * phi * i - phi**2)
        d = np.sqrt(d_sq)

        g_num = b - rho * sigma_v * phi * i - d
        g_den = b - rho * sigma_v * phi * i + d
        g = g_num / (g_den + eps)

        exp_dt = np.exp(-d * T)

        # Heston CF main formula
        C = (a / (sig_v2 + eps)) * (g_num * T - 2.0 * np.log((1 - g * exp_dt) / (1 - g + eps)))
        D = (g_num / (sig_v2 + eps)) * ((1 - exp_dt) / (1 - g * exp_dt + eps))

        # S_internal is 1.0, so log(S) term is 0. Removed for speed.
        return np.exp(C + D * v0)

    @staticmethod
    def _integrate_vectorized(K_norm: float, T: float, params: List[float], P_num: int, limit: float) -> float:
        """
        Performs integration using mapped Gauss-Legendre nodes.
        Integral from 0 to limit.
        """
        # Map [-1, 1] (standard GL) to [0, limit]
        # Change of variables: x = (limit/2) * (gl_x + 1)
        # dx = (limit/2) * d_gl_x

        mid = limit / 2.0
        # Broadcast nodes to actual integration domain
        phi_grid = mid * (FastHestonModel._gl_x + 1)

        # Calculate CF for all phi points at once
        cf_vals = FastHestonModel._cf_vectorized(phi_grid, T, params, P_num)

        # Calculate integrand: Real( exp(-i*phi*log(K))*CF / (i*phi) )
        i = 1j
        integrand = np.real(np.exp(-i * phi_grid * np.log(K_norm)) * cf_vals / (i * phi_grid + eps))

        # Dot product with weights
        integral_val = np.dot(integrand, FastHestonModel._gl_w) * mid

        return 0.5 + (1.0 / np.pi) * integral_val

    @staticmethod
    def price_vanilla_call(S: float, K: float, T: float, params: List[float]) -> float:
        if T <= 0: return max(S - K, 0.0)

        K_norm = K / S
        # Same limit as your original code
        limit = 400.0

        P1 = FastHestonModel._integrate_vectorized(K_norm, T, params, 1, limit)
        P2 = FastHestonModel._integrate_vectorized(K_norm, T, params, 2, limit)

        # Clip probabilities to [0,1] to ensure stability
        P1 = np.clip(P1, 0.0, 1.0)
        P2 = np.clip(P2, 0.0, 1.0)

        return float(np.clip(S * P1 - K * P2, 0.0, 1e18))

    @staticmethod
    def price_binary_call(S: float, K: float, T: float, initial_T: float, params: List[float]) -> float:
        if T <= 0: return 1.0 if S > K else 0.0

        adj_params = FastHestonModel._adjust_params_for_expiry(T, initial_T, params)
        K_norm = K / S
        limit = 500 # Matching your binary limit

        try:
            P2 = FastHestonModel._integrate_vectorized(K_norm, T, adj_params, 2, limit)
            return float(np.clip(P2, 0.0, 1.0))
        except Exception:
            return 1.0 if S > K else 0.0
