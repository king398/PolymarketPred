import os
import json
import time
import uuid
import random
import requests
import pandas as pd
from datetime import datetime, timedelta
from joblib import Parallel, delayed
from tqdm import tqdm
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


def fetch_1m_data_chunked(
        session: requests.Session,
        token_id: str,
        start_iso: str,
        end_iso: str,
        chunk_days: int = 7,
        sleep_s: float = 0.0,
        timeout_s: int = 30,
):
    """
    Fetch minute-level history for one token_id from start_iso to end_iso using chunked windows.
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
            "fidelity": 1,  # 1-minute
        }

        try:
            resp = session.get(PRICES_HISTORY_URL, params=params, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()
            chunk_history = data.get("history", [])
            if chunk_history:
                full_history.extend(chunk_history)
        except Exception:
            # keep going; caller can decide how to count failures
            pass

        current_start = current_end
        if sleep_s and sleep_s > 0:
            time.sleep(sleep_s)

    return full_history


def _fetch_one_token_job(
        *,
        side: str,
        token_id: str,
        question: str,
        start_iso: str,
        end_iso: str,
        market_id,
        event_id,
        out_path: str,
        chunk_days: int,
        timeout_s: int,
        per_chunk_sleep_s: float,
        max_retries: int = 2,
):
    """
    Worker: fetch a single token_id and write a parquet to out_path.
    Returns a tuple: (status, out_path, token_id, msg)
    status in {"saved","skipped","failed"}
    """
    if os.path.exists(out_path):
        return ("skipped", out_path, token_id, "already exists")

    # small jitter to avoid thundering herd
    time.sleep(random.uniform(0.0, 0.2))

    for attempt in range(max_retries + 1):
        session = requests.Session()
        try:
            history = fetch_1m_data_chunked(
                session,
                token_id,
                start_iso,
                end_iso,
                chunk_days=chunk_days,
                sleep_s=per_chunk_sleep_s,
                timeout_s=timeout_s,
            )

            if not history:
                raise RuntimeError("empty history")

            df = pd.DataFrame(history).rename(columns={"t": "timestamp", "p": "price"})
            # ensure UTC timestamps
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)

            # identifiers so each file is self-describing
            df["token_id"] = token_id
            df["side"] = side
            df["market_id"] = market_id
            df["event_id"] = event_id
            df["question"] = question

            tmp_path = out_path + ".tmp"
            df.to_parquet(tmp_path, engine="pyarrow", compression="zstd", index=False)
            os.replace(tmp_path, out_path)  # atomic-ish on same filesystem

            return ("saved", out_path, token_id, "ok")

        except Exception as e:
            # backoff then retry
            if attempt < max_retries:
                time.sleep(0.5 * (2 ** attempt) + random.uniform(0.0, 0.3))
            else:
                return ("failed", out_path, token_id, f"{type(e).__name__}: {e}")
        finally:
            try:
                session.close()
            except Exception:
                pass


def main():
    # --- INPUT: your metadata parquet from Script 1 ---
    metadata_parquet = "/home/mithil/PycharmProjects/PolymarketPred/data/polymarket_parquet/closed_crypto_market_ids.parquet"

    # --- OUTPUT directory for minute-level series ---
    out_dir = "/home/mithil/PycharmProjects/PolymarketPred/data/polymarket_minute_parquet"
    os.makedirs(out_dir, exist_ok=True)

    # Choose which tokens to fetch:
    FETCH_YES = True
    FETCH_NO = False  # set True if you also want NO token histories

    # Parallelism / throttling
    MAX_WORKERS = 64
    PER_CHUNK_SLEEP_S = 0.00
    CHUNK_DAYS = 15
    TIMEOUT_S = 30

    # Load metadata
    meta = pd.read_parquet(metadata_parquet)
    print(meta)
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

    # Build jobs (side, token_id, question, start_iso, end_iso, market_id, event_id)
    raw_jobs = []
    for _, row in meta.iterrows():
        q = row.get("question")
        start_iso = row.get("start_iso")
        end_iso = row.get("end_date")
        market_id = row.get("market_id")
        event_id = row.get("event_id")

        if FETCH_YES and pd.notna(row.get("yes_token_id")):
            raw_jobs.append(("YES", str(row["yes_token_id"]), q, start_iso, end_iso, market_id, event_id))
        if FETCH_NO and pd.notna(row.get("no_token_id")):
            raw_jobs.append(("NO", str(row["no_token_id"]), q, start_iso, end_iso, market_id, event_id))

    if not raw_jobs:
        print("No token jobs found (check yes_token_id/no_token_id/end_date).")
        return

    # Pre-assign UUIDs (single-threaded) and build concrete tasks with out_path
    tasks = []
    newly_assigned = 0

    for side, token_id, question, start_iso, end_iso, market_id, event_id in raw_jobs:
        token_key = f"{side}:{token_id}"

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
            newly_assigned += 1

        out_path = os.path.join(out_dir, f"{token_uuid}.parquet")

        # only enqueue if not already downloaded
        if not os.path.exists(out_path):
            tasks.append(
                dict(
                    side=side,
                    token_id=token_id,
                    question=question,
                    start_iso=start_iso,
                    end_iso=end_iso,
                    market_id=market_id,
                    event_id=event_id,
                    out_path=out_path,
                    chunk_days=CHUNK_DAYS,
                    timeout_s=TIMEOUT_S,
                    per_chunk_sleep_s=PER_CHUNK_SLEEP_S,
                )
            )

    # Persist mapping once (safe)
    if newly_assigned > 0 or not os.path.exists(mapping_path):
        with open(mapping_path, "w") as f:
            json.dump(uuid_map, f, indent=2)

    total = len(raw_jobs)
    to_fetch = len(tasks)
    already = total - to_fetch
    print(f"Total tokens in metadata: {total}")
    print(f"Already present (skipped): {already}")
    print(f"To fetch in parallel: {to_fetch}")
    print(f"Workers: {MAX_WORKERS} | chunk_days={CHUNK_DAYS} | per_chunk_sleep={PER_CHUNK_SLEEP_S}s")

    if to_fetch == 0:
        print("Nothing to do ✅")
        return

    # Run parallel fetches
    results = Parallel(n_jobs=MAX_WORKERS, backend="threading", prefer="threads")(
        delayed(_fetch_one_token_job)(**t) for t in tqdm(tasks)
    )

    saved = sum(1 for r in results if r[0] == "saved")
    skipped = already + sum(1 for r in results if r[0] == "skipped")
    failed = sum(1 for r in results if r[0] == "failed")

    # Optional: print failures
    if failed:
        print("\nFailures:")
        for status, out_path, token_id, msg in results:
            if status == "failed":
                print(f" - token_id={token_id} -> {msg} ({out_path})")

    print("\nDONE ✅")
    print(f"attempted_total={total} | saved_now={saved} | skipped_total={skipped} | failed_now={failed}")
    print(f"Output folder: {out_dir}")
    print(f"UUID map: {mapping_path}")


if __name__ == "__main__":
    main()
