import asyncio
import websockets
import json
import os
import math
from datetime import datetime
import pytz
from scipy.stats import norm  # Required for Black-Scholes CDF
import time

DATA_DIR = os.path.join(os.getcwd(), "data")
MARKET_ID_FILE = os.path.join(DATA_DIR, "clob_options_token_ids.jsonl")
ET = pytz.timezone("US/Eastern")
RISK_FREE_RATE = 0.045  # Assumed 4.5% risk-free rate, adjust as needed


def load_option_targets(file_path):
    targets = []
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return []

    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            targets.append(obj)
    return targets


def et_time_to_expiry_years(market_end_str: str) -> float:
    ET = pytz.timezone("US/Eastern")
    end = datetime.fromisoformat(market_end_str).astimezone(ET)
    now = datetime.now(ET)
    return max((end - now).total_seconds(), 0) / (365.25 * 24 * 3600)


def get_digital_bs_price(S, K, T, r, sigma):
    """
    Calculates the Price of a Digital (Binary) Cash-or-Nothing Option.
    Pays 1.00 if ITM.
    """
    if T <= 0:
        return 0.0
    if sigma <= 0:
        return 0.0

    # Calculate d2
    # d2 = (ln(S/K) + (r - 0.5 * sigma^2) * T) / (sigma * sqrt(T))
    try:
        d2 = (math.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    except ValueError:
        return 0.0  # Handle math domain errors (e.g. S=0)

    discount_factor = math.exp(-r * T)

    # Digital Call = e^(-rT) * N(d2)
    price = discount_factor * norm.cdf(d2)

    return price


def parse_instrument_details(instrument_name):
    """
    Parses Deribit format: BTC-29DEC23-50000-C
    Returns (Strike, OptionType)
    """
    try:
        parts = instrument_name.split('-')
        strike = float(parts[-2])
        opt_type = parts[-1]  # 'C' or 'P'
        return strike, opt_type
    except Exception as e:
        print(f"Error parsing {instrument_name}: {e}")
        return None, None


async def stream_iv(targets):
    print("Finding ATM Option...")
    print(f"Target Acquired: {len(targets)} targets loaded.")
    print("-" * 40)

    target_list = [f"ticker.{t['related_option']}.100ms" for t in targets]
    option_map = {t['related_option']: t for t in targets}

    async with websockets.connect("wss://www.deribit.com/ws/api/v2") as ws:
        msg = {
            "jsonrpc": "2.0",
            "method": "public/subscribe",
            "params": {"channels": target_list},
            "id": 100
        }
        await ws.send(json.dumps(msg))

        while True:
            response = json.loads(await ws.recv())

            if 'params' in response and 'data' in response['params']:
                data = response['params']['data']
                # Extract Data
                # Deribit sends mark_iv as percent (e.g. 50.0 for 50%), so divide by 100?
                # Actually, check Deribit docs: "mark_iv: Implied volatility in percent".
                # Usually standard BS requires decimal (0.50).
                mark_iv_percent = data.get('mark_iv', 0)
                sigma = mark_iv_percent / 100.0

                underlying_price = data.get('index_price', 0)
                instrument_name = data.get('instrument_name', 'Unknown')
                timestamp = data.get('timestamp', 0)
                latency = (time.time() * 1000) - timestamp
                if instrument_name in option_map:
                    time_remaining = et_time_to_expiry_years(option_map[instrument_name]['market_end'])

                    strike, opt_type = parse_instrument_details(instrument_name)

                    if strike and opt_type and time_remaining > 0 and underlying_price > 0:
                        digital_price = get_digital_bs_price(
                            S=underlying_price,
                            K=strike,
                            T=time_remaining,
                            r=RISK_FREE_RATE,
                            sigma=sigma,
                        )

                        print(
                            f"{instrument_name:<25} | "
                            f"IV: {mark_iv_percent:>5.2f}% | "
                            f"S: {underlying_price:>7.2f} | "
                            f"K: {strike:>7.0f} | "
                            f"Digital Price: ${digital_price:.4f}"
                            f" | Latency: {latency :.1f} ms"
                        )
                    else:
                        print(f"Skipping calc for {instrument_name} (Missing data or expired)")


if __name__ == "__main__":
    try:
        targets = load_option_targets(MARKET_ID_FILE)
        if targets:
            asyncio.run(stream_iv(targets))
        else:
            print("No targets found.")
    except KeyboardInterrupt:
        print("\nStream stopped by user.")
"""ETH-18JAN26-3400-C        | IV: 45.23% | S: 3360.59 | K:    3400 | Digital Price: $0.3645 | Latency: 420.0 ms"""