import zmq
import numpy as np
from joblib.testing import param
from market_websocket import ASSET_ID_FILE
from herston import BatesModel
import json

ZMQ_SUB = "tcp://127.0.0.1:5567"

params_map = {}
with open("/home/mithil/PycharmProjects/PolymarketPred/data/bates_params.jsonl", "r") as f:
    for line in f:
        p = json.loads(line)
        params_map[p['currency']] = p
with open(ASSET_ID_FILE, "r") as f:
    for line in f:
        if line.strip():
            try:
                obj = json.loads(line)

            except:
                pass
# Must match publisher dtype exactly
tick_dtype = np.dtype([
    ("ts_ms", np.int64),
    ("bid", np.float32),
    ("ask", np.float32),
])
asset_ticks = {}


def main():
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(ZMQ_SUB)

    # Subscribe to ALL assets
    sub.setsockopt(zmq.SUBSCRIBE, b"")

    print("ZMQ subscriber connected. Waiting for ticks...")

    while True:
        try:
            aid_bytes, raw = sub.recv_multipart()
            aid = aid_bytes.decode()

            arr = np.frombuffer(raw, dtype=tick_dtype)

            # Example: print last tick
            asset_ticks[aid] = arr
            if aid == "BTC":

                spot = (arr[-1]['ask'] + arr[-1]['bid']) / 2
                strike = 90013.89
                params = params_map['BTC']
                T = 55 / (365* 24*60)
                price = BatesModel.price_binary_call(
                    S=spot,
                    K=strike,
                    T=T,
                    params=params
                )
                print(f"BTC Spot: {spot:.2f}, Strike: {strike}, 4h Binary Call Price: {price:.4f}")



        except KeyboardInterrupt:
            print("\nSubscriber stopped.")
            break


if __name__ == "__main__":
    main()
