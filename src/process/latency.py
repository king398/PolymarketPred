#!/usr/bin/env python3
import argparse
import zmq
import struct
import statistics as stats
import time
from typing import List, Optional

# Constants matching the Feed Handler
STRUCT_FMT = '<qff'  # Little-endian: int64 (ts), float32 (bid), float32 (ask)
STRUCT_SIZE = struct.calcsize(STRUCT_FMT)

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
    ap.add_argument("--addr", default="tcp://127.0.0.1:5567", help="ZMQ Publisher Address")
    ap.add_argument("--seconds", type=float, default=30.0, help="Duration to measure")
    ap.add_argument("--exclude", default="", help="Comma-separated asset_ids to exclude (e.g. BTC,ETH)")

    args = ap.parse_args()

    # Parse exclusions
    exclude = {x.strip() for x in args.exclude.split(",") if x.strip()}

    # Stats containers
    lat_ms: List[int] = []   # (recv_time - payload_timestamp)
    gap_ms: List[int] = []   # (current_recv - prev_recv)

    total_msgs = 0
    kept_msgs = 0
    decode_errors = 0

    prev_recv_ts: Optional[int] = None

    # Setup ZMQ
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(args.addr)
    sub.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all topics

    # Use Poller for precise timeout handling in the loop
    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    print(f"[Profiler] Connected to {args.addr}")
    print(f"[Profiler] Measuring for {args.seconds} seconds...")
    print(f"[Profiler] Excluded topics: {exclude if exclude else 'None'}")

    end_time = time.monotonic() + args.seconds

    try:
        while time.monotonic() < end_time:
            # Poll with 100ms timeout to check end_time frequently
            socks = dict(poller.poll(100))

            if sub in socks and socks[sub] == zmq.POLLIN:
                try:
                    # Receive Multipart: [Topic (AssetID), Payload (Binary)]
                    msg_parts = sub.recv_multipart()
                    recv_ts = ms_now() # Capture arrival time immediately

                    if len(msg_parts) != 2:
                        decode_errors += 1
                        continue

                    topic_bytes, payload = msg_parts
                    asset_id = topic_bytes.decode('utf-8', errors='ignore')

                    total_msgs += 1

                    # 1. Filter Exclusions
                    if asset_id in exclude:
                        continue

                    # 2. Unpack Binary Data
                    if len(payload) != STRUCT_SIZE:
                        decode_errors += 1
                        continue

                    # Unpack: timestamp (int64), bid (float), ask (float)
                    ts_payload, bid, ask = struct.unpack(STRUCT_FMT, payload)

                    # 3. Record Stats
                    kept_msgs += 1

                    # E2E Latency: Now - Timestamp inside packet
                    # Note: If running locally, this is processing lag.
                    # If remote, this includes clock skew.
                    latency = recv_ts - ts_payload
                    lat_ms.append(latency)

                    # Inter-arrival gap
                    if prev_recv_ts is not None:
                        gap_ms.append(recv_ts - prev_recv_ts)
                    prev_recv_ts = recv_ts

                except Exception as e:
                    decode_errors += 1

    except KeyboardInterrupt:
        print("\nStopping early...")
    finally:
        sub.close()
        ctx.term()

    # --- REPORTING ---
    def summarize(name: str, xs: List[int]):
        if not xs:
            print(f"\n{name}: No data collected.")
            return

        # Convert to float for stats
        xs_f = [float(x) for x in xs]

        print(f"\n{name}")
        print(f"{'-'*len(name)}")
        print(f"Count : {len(xs_f)}")
        print(f"Min   : {min(xs_f):.2f} ms")
        print(f"Mean  : {stats.mean(xs_f):.2f} ms")
        print(f"Max   : {max(xs_f):.2f} ms")
        print(f"StDev : {stats.pstdev(xs_f):.2f} ms")
        print(f"----------------")
        print(f"P50   : {percentile(xs_f, 50):.2f} ms")
        print(f"P95   : {percentile(xs_f, 95):.2f} ms")
        print(f"P99   : {percentile(xs_f, 99):.2f} ms")

    print(f"\n{'='*30}")
    print(f"RESULTS ({args.seconds}s)")
    print(f"{'='*30}")
    print(f"Total Messages : {total_msgs}")
    print(f"Kept Messages  : {kept_msgs}")
    print(f"Decode Errors  : {decode_errors}")

    summarize("End-to-End Latency (Recv - Payload_TS)", lat_ms)
    summarize("Inter-Arrival Gap (Jitter)", gap_ms)

if __name__ == "__main__":
    main()