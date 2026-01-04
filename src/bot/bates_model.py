import warnings
from typing import List

import numpy as np
from scipy.integrate import quad

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



