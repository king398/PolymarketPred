import asyncio
import json
import time
import threading
from queue import Queue, Empty
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

HOST = "127.0.0.1"
PORT = 9000

# ---- plotting controls ----
MAX_POINTS = 600          # keep last N points in memory (e.g., 10 min @ 1 Hz)
PLOT_FPS = 30             # UI refresh rate (not network rate)


def _extract_point(msg: Dict[str, Any]) -> Optional[Tuple[float, Optional[float], Optional[float]]]:
    """
    Return (t_seconds, bid, ask, last) for plotting.
    Adjust these keys to match your server output.
    """
    # time axis
    ts_ms = msg.get("ts_ms")
    if isinstance(ts_ms, (int, float)):
        t = float(ts_ms) / 1000.0
    else:
        t = time.time()

    # common field name variants
    bid = msg.get("best_bid", msg.get("bid"))
    ask = msg.get("best_ask", msg.get("ask"))

    def _to_float(x):
        if x is None:
            return None
        try:
            return float(x)
        except Exception:
            return None

    return (t, _to_float(bid), _to_float(ask))


async def tcp_reader(out_q: Queue):
    reader, writer = await asyncio.open_connection(HOST, PORT)
    print(f"Connected to tcp://{HOST}:{PORT}")

    buffer = b""
    try:
        while True:
            chunk = await reader.read(4096)
            if not chunk:
                raise ConnectionError("Server closed connection.")
            buffer += chunk

            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue

                pt = _extract_point(msg)
                if pt is not None:
                    out_q.put(pt)
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


def run_async_client(out_q: Queue):
    asyncio.run(tcp_reader(out_q))


def main():
    q: Queue = Queue()

    # Start asyncio TCP client in background thread
    t = threading.Thread(target=run_async_client, args=(q,), daemon=True)
    t.start()

    # Live plot state
    xs, bids, asks = [], [], []

    fig, ax = plt.subplots()
    line_bid, = ax.plot([], [], label="best_bid")
    line_ask, = ax.plot([], [], label="best_ask")
    ax.set_title("Live Market Stream")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Price / Probability")
    ax.legend(loc="upper left")

    def init():
        ax.relim()
        ax.autoscale_view()
        return line_bid, line_ask

    def update(_frame):
        # Drain queue (non-blocking)
        drained = 0
        while True:
            try:
                t_sec, bid, ask = q.get_nowait()
            except Empty:
                break

            xs.append(t_sec)
            bids.append(bid)
            asks.append(ask)
            drained += 1

        if drained == 0:
            return line_bid, line_ask

        # Keep last MAX_POINTS
        if len(xs) > MAX_POINTS:
            xs[:] = xs[-MAX_POINTS:]
            bids[:] = bids[-MAX_POINTS:]
            asks[:] = asks[-MAX_POINTS:]

        # Convert Nones to NaNs so matplotlib breaks the line cleanly
        import math
        def _nanify(arr):
            return [x if (x is not None) else math.nan for x in arr]

        line_bid.set_data(xs, _nanify(bids))
        line_ask.set_data(xs, _nanify(asks))

        ax.relim()
        ax.autoscale_view()

        # Optional: show only recent window in X
        ax.set_xlim(xs[0], xs[-1])

        return line_bid, line_ask

    ani = FuncAnimation(fig, update, init_func=init, interval=int(1000 / PLOT_FPS), blit=False)
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
