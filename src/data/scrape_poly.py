import requests
import time
import json
import os
import pandas as pd  # Core library for data handling
from datetime import datetime, timedelta


def fetch_market_details(limit=50):
    """
    Fetches closed markets to get their start/end dates and token IDs.
    """
    url = "https://gamma-api.polymarket.com/events"

    params = {
        "limit": limit,
        "closed": "true",
        "order": "volume",
        "ascending": "false",
        "endDateMin": "2024-01-01",
        "endDateMax": "2025-06-01",
        "offset": "10",
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        valid_markets = []
        for event in data:
            if not event.get('markets'): continue
            market = event['markets'][0]

            # Get token ID
            raw_clob_ids = market.get('clobTokenIds', [])
            if isinstance(raw_clob_ids, str):
                try:
                    raw_clob_ids = json.loads(raw_clob_ids.replace("'", '"'))
                except:
                    continue

            if isinstance(raw_clob_ids, list) and len(raw_clob_ids) >= 2:
                # We need start/end dates for pagination
                start_iso = market.get('startDate') or market.get('creationDate')
                end_iso = market.get('endDate')

                if start_iso and end_iso:
                    valid_markets.append({
                        "question": market.get('question'),
                        "start_date": start_iso,
                        "end_date": end_iso,
                        "token_id": raw_clob_ids[1]  # YES token
                    })
        return valid_markets
    except Exception as e:
        print(f"Error fetching markets: {e}")
        return []


def fetch_1m_data_chunked(token_id, start_iso, end_iso):
    """
    Loops through the date range in 7-day chunks to get 1-minute data.
    """
    url = "https://clob.polymarket.com/prices-history"
    full_history = []

    try:
        start_dt = datetime.fromisoformat(start_iso.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_iso.replace('Z', '+00:00'))
    except ValueError:
        return []

    current_start = start_dt
    chunk_size = timedelta(days=7)

    print(f"  > Paginating from {start_dt.date()} to {end_dt.date()}...")

    while current_start < end_dt:
        current_end = min(current_start + chunk_size, end_dt)
        ts_start = int(current_start.timestamp())
        ts_end = int(current_end.timestamp())

        params = {
            "market": token_id,
            "startTs": ts_start,
            "endTs": ts_end,
            "fidelity": 1
        }

        try:
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            chunk_history = data.get('history', [])

            if chunk_history:
                full_history.extend(chunk_history)
                print(f"    + Fetched {len(chunk_history)} points ({current_start.date()} -> {current_end.date()})")
            else:
                print(f"    . No data for {current_start.date()} -> {current_end.date()}")

        except Exception as e:
            print(f"    ! Error fetching chunk: {e}")

        current_start = current_end
        time.sleep(0.5)

    return full_history


def main():
    # Create a directory to keep things organized
    output_dir = "data/polymarket_parquet"
    os.makedirs(output_dir, exist_ok=True)

    print("--- 1. Fetching Markets ---")
    markets = fetch_market_details(limit=2)

    if not markets:
        print("No markets found.")
        return
    print(f"Fetched {len(markets)} markets.")

    print("--- 2. Fetching Granular History ---")
    for market in markets:
        print(f"\nMarket: {market['question']}")

        history = fetch_1m_data_chunked(
            market['token_id'],
            market['start_date'],
            market['end_date']
        )

        if history:
            print(f"  > Processing {len(history)} records...")

            # --- OPTIMIZED STORAGE IMPLEMENTATION ---

            # 1. Create DataFrame directly from the list of dicts
            df = pd.DataFrame(history)

            # 2. Rename columns to be descriptive (t -> timestamp, p -> price)
            df = df.rename(columns={'t': 'timestamp', 'p': 'price'})

            # 3. Optimize Data Types
            # Convert Unix timestamp (int) to true datetime objects
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

            # Downcast price to float32 (saves 50% memory vs float64, plenty of precision for prices)
            df['price'] = df['price'].astype('float32')

            # 4. Generate a clean filename
            # Remove special characters from question to make it filesystem-safe
            safe_question = "".join([c if c.isalnum() else "_" for c in market['question']])
            # Truncate filename to avoid OS limits, append Token ID for uniqueness
            filename = f"{safe_question[:50]}_{market['token_id']}.parquet"
            file_path = os.path.join(output_dir, filename)

            # 5. Save as Parquet with ZSTD compression
            df.to_parquet(file_path, engine='pyarrow', compression='zstd', index=False)

            print(f"  > SUCCESS: Saved to {file_path}")
            print(f"  > First: {df['timestamp'].iloc[0]} | Last: {df['timestamp'].iloc[-1]}")

        else:
            print("  > No history found.")


if __name__ == "__main__":
    main()