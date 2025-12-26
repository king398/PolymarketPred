import os
import time
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
import uuid

PRICES_HISTORY_URL = "https://clob.polymarket.com/prices-history"


def parse_iso_datetime(iso_str: str):
    """
    Parses ISO timestamps like '2025-06-01T00:00:00Z' or '2025-06-01T00:00:00+00:00'
    Returns timezone-aware datetime or None.
    """
    if not iso_str or not isinstance(iso_str, str):
        return None
    try:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except ValueError:
        return None


def fetch_1m_data_chunked(session: requests.Session, token_id: str, start_iso: str, end_iso: str,
                          chunk_days: int = 30, sleep_s: float = 0.25, timeout_s: int = 30):
    """
    Fetch minute-level history for one token_id from start_iso to end_iso using 7-day chunks.
    Returns list of dicts like {'t': unix_ts, 'p': price}.
    """
    start_dt = parse_iso_datetime(start_iso)

    end_dt = parse_iso_datetime(end_iso)
    if start_dt is None or end_dt is None or start_dt >= end_dt:
        return []

    full_history = []
    current_start = start_dt
    chunk_size = timedelta(days=chunk_days)

    while current_start < end_dt:
        current_end = min(current_start + chunk_size, end_dt)
        ts_start = int(current_start.timestamp())
        ts_end = int(current_end.timestamp())

        params = {
            "market": token_id,
            "startTs": ts_start,
            "endTs": ts_end,
            "fidelity": 1,  # 5-minute
        }

        try:
            resp = session.get(PRICES_HISTORY_URL, params=params, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()
            chunk_history = data.get("history", [])
            if chunk_history:
                full_history.extend(chunk_history)
        except Exception:
            print(f"  ! Error fetching token_id={token_id} chunk {current_start} to {current_end}")
            pass

        current_start = current_end
        ##time.sleep(sleep_s)

    return full_history


def safe_filename(text: str, max_len: int = 80):
    if not text:
        return "market"
    s = "".join(c if c.isalnum() else "_" for c in text)
    s = s.strip("_")
    return (s[:max_len] if len(s) > max_len else s) or "market"


def main():
    # --- INPUT: your metadata parquet from Script 1 ---
    metadata_parquet = "/home/mithil/PycharmProjects/PolymarketPred/data/polymarket_parquet/closed_markets_metadata.parquet"

    # --- OUTPUT directory for minute-level series ---
    out_dir = "/home/mithil/PycharmProjects/PolymarketPred/data/polymarket_minute_parquet"
    os.makedirs(out_dir, exist_ok=True)

    # Choose which tokens to fetch:
    FETCH_YES = True
    FETCH_NO = False  # set True if you also want NO token histories

    # Load metadata
    meta = pd.read_parquet(metadata_parquet)[-100:]

    # Keep only rows that have an end_date and at least one token id
    meta = meta.copy()
    meta["start_iso"] = meta["start_date"].fillna(meta["creation_date"])
    meta = meta[meta["end_date"].notna()]
    mapping_path = os.path.join(out_dir, "token_uuid_map.json")

    # Load existing mapping if present (resume-safe)
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            uuid_map = json.load(f)
    else:
        uuid_map = {}

    # Build list of jobs: (token_id, question, start_iso, end_iso, market_id, event_id)
    jobs = []
    for _, row in meta.iterrows():
        q = row.get("question")
        start_iso = row.get("start_iso")
        end_iso = row.get("end_date")
        market_id = row.get("market_id")
        event_id = row.get("event_id")

        if FETCH_YES and pd.notna(row.get("yes_token_id")):
            jobs.append(("YES", str(row["yes_token_id"]), q, start_iso, end_iso, market_id, event_id))

        if FETCH_NO and pd.notna(row.get("no_token_id")):
            jobs.append(("NO", str(row["no_token_id"]), q, start_iso, end_iso, market_id, event_id))

    total_jobs = len(jobs)
    if total_jobs == 0:
        print("No token jobs found (check yes_token_id/no_token_id/end_date).")
        return

    session = requests.Session()

    done = 0
    skipped = 0
    failed = 0

    try:
        for side, token_id, question, start_iso, end_iso, market_id, event_id in jobs:
            done += 1

            # output file per token
            # Create deterministic key for this token
            token_key = f"{side}:{token_id}"

            # Reuse UUID if already assigned
            if token_key in uuid_map:
                token_uuid = uuid_map[token_key]["uuid"]
            else:
                token_uuid = str(uuid.uuid4())
                uuid_map[token_key] = {
                    "uuid": token_uuid,
                    "side": side,
                    "token_id": token_id,
                    "market_id": market_id,
                    "event_id": event_id,
                    "question": question,
                    "created_at": datetime.now().isoformat() + "Z",
                }

            out_path = os.path.join(out_dir, f"{token_uuid}.parquet")

            # Skip if already downloaded
            if os.path.exists(out_path):
                skipped += 1
                print(
                    f"\rMinute history | done {done}/{total_jobs} | skipped {skipped} | failed {failed}",
                    end="",
                    flush=True,
                )
                continue

            history = fetch_1m_data_chunked(session, token_id, start_iso, end_iso)

            if not history:
                failed += 1
                print(
                    f"\rMinute history | done {done}/{total_jobs} | skipped {skipped} | failed {failed}",
                    end="",
                    flush=True,
                )
                continue

            df = pd.DataFrame(history).rename(columns={"t": "timestamp", "p": "price"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df["price"] = df["price"]
            df = df.sort_values("timestamp").reset_index(drop=True)
            # add identifiers so each file is self-describing
            df["token_id"] = token_id
            df["side"] = side
            df["market_id"] = market_id
            df["event_id"] = event_id
            df["question"] = question

            df.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
            print(
                f"\rMinute history | done {done}/{total_jobs} | skipped {skipped} | failed {failed}",
                end="",
                flush=True,
            )
            # Persist mapping safely after each successful save
            with open(mapping_path, "w") as f:
                json.dump(uuid_map, f, indent=2)

    finally:
        session.close()

    print()  # newline
    print(
        f"DONE ✅ tokens attempted={total_jobs} | skipped={skipped} | failed={failed} | saved={total_jobs - skipped - failed}")
    print(f"Output folder: {out_dir}")


if __name__ == "__main__":
    main()
