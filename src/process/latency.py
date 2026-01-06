#!/usr/bin/env python3
import argparse
import json
import socket
import statistics as stats
import time
from typing import List, Optional

EXCLUDE_DEFAULT = {"XRP", "SOL", "BTC", "ETH"}


def percentile(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    xs2 = sorted(xs)
    k = int(round((p / 100.0) * (len(xs2) - 1)))
    return xs2[k]


def ms_now() -> int:
    return int(time.time() * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--exclude", default="XRP,SOL,BTC,ETH",
                    help="Comma-separated asset_ids to exclude")
    ap.add_argument("--timeout", type=float, default=0.25,
                    help="Socket recv timeout (seconds)")
    args = ap.parse_args()

    exclude = {x.strip() for x in args.exclude.split(",") if x.strip()}

    # Stats
    lat_ms: List[int] = []  # recv_ms - ts_ms
    gap_ms: List[int] = []  # recv_ms - prev_recv_ms (inter-arrival)
    total = 0
    kept = 0
    parse_errors = 0
    missing_ts = 0

    prev_recv: Optional[int] = None
    buf = b""

    end_time = time.monotonic() + args.seconds

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(args.timeout)
    s.connect((args.host, args.port))

    try:
        while time.monotonic() < end_time:
            try:
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
            except socket.timeout:
                continue

            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line:
                    continue

                recv = ms_now()
                if prev_recv is not None:
                    gap_ms.append(recv - prev_recv)
                prev_recv = recv

                total += 1

                try:
                    msg = json.loads(line.decode("utf-8", errors="replace"))
                except Exception:
                    parse_errors += 1
                    continue

                aid = msg.get("asset_id")
                if aid in exclude:
                    continue

                kept += 1
                ts_ms = msg.get("ts_ms")
                if isinstance(ts_ms, (int, float)):
                    lat_ms.append(recv - int(ts_ms))
                else:
                    missing_ts += 1

    finally:
        try:
            s.close()
        except Exception:
            pass

    def summarize(name: str, xs: List[int]):
        if not xs:
            print(f"{name}: n=0")
            return
        xs_f = [float(x) for x in xs]
        print(
            f"{name}: n={len(xs_f)} "
            f"min={min(xs_f):.0f}ms "
            f"p50={percentile(xs_f, 50):.0f}ms "
            f"p95={percentile(xs_f, 95):.0f}ms "
            f"p99={percentile(xs_f, 99):.0f}ms "
            f"max={max(xs_f):.0f}ms "
            f"mean={stats.mean(xs_f):.2f}ms "
            f"stdev={stats.pstdev(xs_f):.2f}ms"
        )

    print(f"Connected to {args.host}:{args.port} for {args.seconds:.1f}s")
    print(f"Total lines seen: {total}")
    print(f"Excluded asset_ids: {sorted(exclude)}")
    print(f"Kept (non-excluded): {kept}")
    print(f"JSON parse errors: {parse_errors}")
    print(f"Missing/invalid ts_ms (kept msgs): {missing_ts}")
    summarize("E2E latency (recv_ms - ts_ms)", lat_ms)
    summarize("Inter-arrival gap (recv_ms - prev_recv_ms)", gap_ms)


if __name__ == "__main__":
    main()
