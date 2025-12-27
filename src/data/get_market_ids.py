import os
import time
import json
import requests
import pandas as pd
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq

GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"


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
        # try normal JSON first
        try:
            return json.loads(s2)
        except Exception:
            # try replace single quotes -> double quotes
            try:
                return json.loads(s2.replace("'", '"'))
            except Exception:
                return None
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

        # If returned fewer than limit, likely end of list
        if got < limit:
            break



def flatten_event_to_market_rows(event: dict, verbose: bool = True) -> list[dict]:
    """
    Convert one event to many rows, one per market in event['markets'].
    Includes event-level + market-level columns.
    """
    rows = []
    markets = event.get("markets") or []
    if not markets:
        return rows

    # --- event-level fields (best-effort; Gamma fields can vary) ---
    event_id = event.get("id")
    event_title = event.get("title") or event.get("name")
    event_slug = event.get("slug")
    event_category = event.get("category")
    event_tags = event.get("tags")
    event_created_at = event.get("createdAt") or event.get("creationDate")
    event_start_date = event.get("startDate")
    event_end_date = event.get("endDate")
    event_volume = event.get("volume")
    event_liquidity = event.get("liquidity")

    for m in markets:
        # market-level fields
        market_id = m.get("id")
        question = m.get("question")
        condition_id = m.get("conditionId")
        market_slug = m.get("slug")

        creation_date = m.get("creationDate")
        start_date = m.get("startDate") or creation_date
        end_date = m.get("endDate")

        volume = m.get("volume")
        liquidity = m.get("liquidity")
        closed = m.get("closed")
        active = m.get("active")

        # clob token ids
        raw_clob = m.get("clobTokenIds")
        clob_ids = _safe_json_loads_maybe(raw_clob) or []

        # best-effort YES/NO interpretation:
        # common convention: [NO, YES] -> index 0 NO, index 1 YES
        no_token_id = None
        yes_token_id = None
        if isinstance(clob_ids, list):
            if len(clob_ids) >= 1:
                no_token_id = clob_ids[0]
            if len(clob_ids) >= 2:
                yes_token_id = clob_ids[1]



        # include full list as JSON string (Parquet-friendly, reproducible)
        clob_ids_json = json.dumps(clob_ids, ensure_ascii=False)

        if not isinstance(event_volume,float) or event_volume < 25000:
            print(f"Skipping low-volume event {event_id} ({event_volume})")
            continue
        rows.append({
            # event-level
            "event_id": event_id,
            "event_title": event_title,
            "event_slug": event_slug,
            "event_category": event_category,
            "event_tags": json.dumps(event_tags, ensure_ascii=False) if event_tags is not None else None,
            "event_created_at": event_created_at,
            "event_start_date": event_start_date,
            "event_end_date": event_end_date,
            "event_volume": event_volume,
            "event_liquidity": event_liquidity,

            # market-level
            "market_id": market_id,
            "market_slug": market_slug,
            "question": question,
            "condition_id": condition_id,
            "creation_date": creation_date,
            "start_date": start_date,
            "end_date": end_date,
            "volume": volume,
            "liquidity": liquidity,
            "closed": closed,
            "active": active,

            # tokens
            "clob_token_ids_json": clob_ids_json,
            "no_token_id": no_token_id,
            "yes_token_id": yes_token_id,
        })

    return rows


def write_closed_markets_to_parquet(
        output_path: str,
        limit: int = 100,
        batch_size_rows: int = 50_000,
        sleep_s: float = 0.25,
        end_date_min: str | None = None,
        end_date_max: str | None = None,
        start_date_min: str | None = None,
):
    """
    Fetch all closed events -> flatten -> stream-write to one Parquet file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    total_markets = 0

    session = requests.Session()
    buffer = []
    writer = None
    total_rows = 0
    total_events = 0
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
                total_markets += len(rows)

                print(
                    f"\rScraping Polymarket | "
                    f"Events: {total_events} | "
                    f"Markets: {total_markets} | "
                    f"Rows written: {total_rows}",
                    end="",
                    flush=True
                )


            # flush when buffer big enough
            if len(buffer) >= batch_size_rows:
                df = pd.DataFrame(buffer)

                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                writer.write_table(table)

                total_rows += len(buffer)
                buffer = []

                print(
                    f"\rScraping Polymarket | "
                    f"Events: {total_events} | "
                    f"Markets: {total_markets} | "
                    f"Rows written: {total_rows}",
                    end="",
                    flush=True
                )

        # final flush
        if buffer:
            df = pd.DataFrame(buffer)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
            total_rows += len(buffer)
            buffer = []

        dt = time.time() - t0
        print()
        print(    f"DONE ✅ Events: {total_events} | "
                  f"Markets: {total_markets} | "
                  f"Rows written: {total_rows} | "
                  f"Time: {dt:.1f}s")

    finally:
        if writer is not None:
            writer.close()
        session.close()


if __name__ == "__main__":
    # You can remove end_date_min/end_date_max to fetch "everything closed" the API returns.
    # Keeping them can reduce load if you only want a date window.
    output_file = "/home/mithil/PycharmProjects/PolymarketPred/data/polymarket_parquet/closed_markets_metadata.parquet"

    write_closed_markets_to_parquet(
        output_path=output_file,
        limit=500,  # Gamma API page size
        batch_size_rows=10000,  # controls memory use
        sleep_s=0.2,  # be nice to the API
        end_date_max="2025-11-01",
        start_date_min="2025-09-01",
    )
