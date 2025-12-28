import os
import time
import json
import re
import requests
import pandas as pd
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq

GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"

# ---- your crypto pattern ----
CRYPTO_PATTERN = re.compile(r"\b(bitcoin|ethereum|solana|xrp|btc|eth)\b", re.IGNORECASE)


def _safe_json_loads_maybe(s):
    """
    clobTokenIds sometimes appears as a stringified python list with single quotes.
    Try to parse robustly.
    """
    if s is None:
        return None
    if isinstance(s, list):
        return s
    if isinstance(s, str):
        s2 = s.strip()
        if not s2:
            return None
        try:
            return json.loads(s2)
        except Exception:
            try:
                return json.loads(s2.replace("'", '"'))
            except Exception:
                return None
    return None


def _first_nonempty(*vals):
    """Return the first value that is a non-empty string (or a non-None non-str)."""
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str):
            if v.strip():
                return v.strip()
        else:
            return v
    return None


def fetch_all_closed_events(
        session: requests.Session,
        limit: int = 100,
        sleep_s: float = 0.25,
        end_date_min: str | None = None,
        end_date_max: str | None = None,
        start_date_min: str | None = None,
):
    """
    Generator yielding events from Gamma API by paginating offset until exhaustion.
    """
    offset = 0
    while True:
        params = {
            "limit": limit,
            "offset": offset,
            "closed": "true",
            "order": "volume",
            "ascending": "false",
        }
        if end_date_min:
            params["end_date_min"] = end_date_min
        if end_date_max:
            params["end_date_max"] = end_date_max
        if start_date_min:
            params["start_date_min"] = start_date_min

        resp = session.get(GAMMA_EVENTS_URL, params=params, timeout=30)
        resp.raise_for_status()
        events = resp.json()

        if not events:
            break

        for ev in events:
            yield ev

        got = len(events)
        offset += got
        if got < limit:
            break

        if sleep_s:
            time.sleep(sleep_s)


def flatten_event_to_market_rows(
        event: dict,
        *,
        crypto_pattern: re.Pattern = CRYPTO_PATTERN,
        match_event_title_fallback: bool = True,
        min_event_volume: float = 25_000.0,
) -> list[dict]:
    """
    Convert one event -> rows, filtered to markets whose question (or optionally event title)
    matches crypto_pattern.

    Writes IDs + fields needed for downstream minute fetch:
      - start_date, creation_date, end_date
    """
    rows = []
    markets = event.get("markets") or []
    if not markets:
        return rows

    event_id = event.get("id")
    event_title = event.get("title") or event.get("name") or ""
    event_volume = event.get("volume")

    # --- IMPORTANT: add these for Script 2 ---
    # Gamma docs use camelCase: startDate, creationDate, endDate
    # But sometimes you might see snake_case in other dumps, so handle both.
    start_date = _first_nonempty(event.get("startDate"), event.get("start_date"))
    creation_date = _first_nonempty(event.get("creationDate"), event.get("creation_date"))
    end_date = _first_nonempty(event.get("endDate"), event.get("end_date"))

    # keep your event_volume filter (optional)
    if not isinstance(event_volume, (int, float)) or float(event_volume) < float(min_event_volume):
        return rows

    # optional fallback: if event title is crypto-related, include markets even if question missing match
    event_is_crypto = bool(crypto_pattern.search(event_title)) if match_event_title_fallback else False

    for m in markets:
        market_id = m.get("id")
        question = (m.get("question") or "").strip()
        condition_id = m.get("conditionId")

        # match against question primarily
        question_is_crypto = bool(crypto_pattern.search(question))
        if not (question_is_crypto or event_is_crypto):
            continue

        raw_clob = m.get("clobTokenIds")
        clob_ids = _safe_json_loads_maybe(raw_clob) or []

        # common convention: [NO, YES]
        no_token_id = clob_ids[0] if isinstance(clob_ids, list) and len(clob_ids) >= 1 else None
        yes_token_id = clob_ids[1] if isinstance(clob_ids, list) and len(clob_ids) >= 2 else None

        rows.append({
            "event_id": event_id,
            "market_id": market_id,
            "condition_id": condition_id,
            "yes_token_id": yes_token_id,
            "no_token_id": no_token_id,
            "question": question,
            "event_title": event_title,
            "event_volume": float(event_volume) if event_volume is not None else None,

            # ✅ needed by Script 2
            "start_date": start_date,
            "creation_date": creation_date,
            "end_date": end_date,
        })

    return rows


def write_closed_crypto_markets_ids_to_parquet(
        output_path: str,
        limit: int = 500,
        batch_size_rows: int = 10_000,
        sleep_s: float = 0.2,
        end_date_min: str | None = None,
        end_date_max: str | None = None,
        start_date_min: str | None = None,
):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    session = requests.Session()
    buffer: list[dict] = []
    writer = None

    total_rows = 0
    total_events = 0
    total_markets_kept = 0
    t0 = time.time()

    try:
        for event in fetch_all_closed_events(
                session=session,
                limit=limit,
                sleep_s=sleep_s,
                end_date_min=end_date_min,
                end_date_max=end_date_max,
                start_date_min=start_date_min,
        ):
            total_events += 1
            rows = flatten_event_to_market_rows(event)

            if rows:
                buffer.extend(rows)
                total_markets_kept += len(rows)

            print(
                f"\rScraping Closed Crypto Markets | "
                f"Events: {total_events} | "
                f"Crypto markets kept: {total_markets_kept} | "
                f"Rows written: {total_rows}",
                end="",
                flush=True
            )

            if len(buffer) >= batch_size_rows:
                df = pd.DataFrame(buffer)
                table = pa.Table.from_pandas(df, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")

                writer.write_table(table)
                total_rows += len(buffer)
                buffer = []

        if buffer:
            df = pd.DataFrame(buffer)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
            total_rows += len(buffer)

        dt = time.time() - t0
        print()
        print(
            f"DONE ✅ Events scanned: {total_events} | "
            f"Crypto markets kept: {total_markets_kept} | "
            f"Rows written: {total_rows} | "
            f"Time: {dt:.1f}s"
        )

    finally:
        if writer is not None:
            writer.close()
        session.close()


if __name__ == "__main__":
    output_file = "/home/mithil/PycharmProjects/PolymarketPred/data/polymarket_parquet/closed_crypto_market_ids.parquet"

    write_closed_crypto_markets_ids_to_parquet(
        output_path=output_file,
        limit=500,
        batch_size_rows=10_000,
        sleep_s=0.0,
        end_date_max=datetime.now().strftime("%Y-%m-%d"),
        start_date_min="2025-10-01",
    )
