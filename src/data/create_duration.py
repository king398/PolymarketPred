#!/usr/bin/env python3
"""
Derive market window purely from UUID parquet files.

Input JSON format (example):
{
  "YES:<token_id>": {
     "uuid": "...",
     "side": "YES",
     "token_id": "...",
     "market_id": "...",
     "event_id": "...",
     "question": "...",
     "created_at": "..."
  },
  ...
}

For each uuid:
  reads {parquet_dir}/{uuid}.parquet
  uses min(timestamp), max(timestamp) as start/end
  computes duration

Usage:
  python derive_windows_from_parquet.py \
      --json_path markets.json \
      --parquet_dir /home/mithil/PycharmProjects/PolymarketPred/data/polymarket_minute_parquet \
      --out_csv market_windows.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_read_parquet_timestamps(pq_path: Path) -> Optional[pd.Series]:
    """
    Returns a tz-aware UTC datetime series if possible, else None.
    Only reads 'timestamp' column for speed.
    """
    if not pq_path.exists():
        return None

    # Read only timestamp column (fast)
    df = pd.read_parquet(pq_path, columns=["timestamp"])
    if "timestamp" not in df.columns or df.empty:
        return None

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    ts = ts.dropna()
    if ts.empty:
        return None
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_path", required=True, type=str)
    ap.add_argument("--parquet_dir", required=True, type=str)
    ap.add_argument("--out_csv", required=True, type=str)
    ap.add_argument("--out_parquet", default='', type=str, help="Optional: also write parquet output")
    args = ap.parse_args()

    json_path = Path(args.json_path)
    parquet_dir = Path(args.parquet_dir)
    out_csv = Path(args.out_csv)
    out_parquet = Path(args.out_parquet) if args.out_parquet else None

    data = read_json(json_path)

    rows: List[Dict[str, Any]] = []

    for key, obj in data.items():
        uuid = obj.get("uuid")
        pq_path = parquet_dir / f"{uuid}.parquet" if uuid else None

        base = {
            "key": key,
            "uuid": uuid,
            "side": obj.get("side"),
            "token_id": obj.get("token_id"),
            "market_id": obj.get("market_id"),
            "event_id": obj.get("event_id"),
            "question": obj.get("question"),
            "created_at": obj.get("created_at"),
            "parquet_path": str(pq_path) if pq_path else None,
        }

        if not uuid or pq_path is None:
            base.update(
                {
                    "status": "missing_uuid",
                    "start_ts_utc": pd.NaT,
                    "end_ts_utc": pd.NaT,
                    "duration_seconds": None,
                    "duration_minutes": None,
                    "duration_hours": None,
                }
            )
            rows.append(base)
            continue

        ts = safe_read_parquet_timestamps(pq_path)
        if ts is None:
            base.update(
                {
                    "status": "parquet_missing_or_empty",
                    "start_ts_utc": pd.NaT,
                    "end_ts_utc": pd.NaT,
                    "duration_seconds": None,
                    "duration_minutes": None,
                    "duration_hours": None,
                }
            )
            rows.append(base)
            continue

        start_ts = ts.min()
        end_ts = ts.max()

        dur_s = (end_ts - start_ts).total_seconds()
        base.update(
            {
                "status": "ok",
                "start_ts_utc": start_ts,
                "end_ts_utc": end_ts,
                "duration_seconds": dur_s,
                "duration_minutes": dur_s / 60.0,
                "duration_hours": dur_s / 3600.0,
            }
        )
        rows.append(base)

    out_df = pd.DataFrame(rows)

    # Nice ordering
    preferred_cols = [
        "status",
        "uuid",
        "market_id",
        "event_id",
        "side",
        "token_id",
        "question",
        "created_at",
        "start_ts_utc",
        "end_ts_utc",
        "duration_seconds",
        "duration_minutes",
        "duration_hours",
        "parquet_path",
        "key",
    ]
    cols = [c for c in preferred_cols if c in out_df.columns] + [c for c in out_df.columns if c not in preferred_cols]
    out_df = out_df[cols]

    out_df.to_csv(out_csv, index=False)
    print(f"Wrote CSV: {out_csv} (rows={len(out_df)})")

    if out_parquet:
        out_df.to_parquet(out_parquet, index=False)
        print(f"Wrote Parquet: {out_parquet}")


if __name__ == "__main__":
    main()

