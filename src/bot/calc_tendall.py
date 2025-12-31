import zmq
import numpy as np

tick_dtype = np.dtype([
    ("ts_ms", np.int64),
    ("bid",   np.float32),
    ("ask",   np.float32),
])

ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect("tcp://127.0.0.1:5567")


# subscribe to everything
sub.setsockopt(zmq.SUBSCRIBE, b"")

while True:
    aid, payload = sub.recv_multipart()
    arr = np.frombuffer(payload, dtype=tick_dtype)
    print(aid.decode(), arr.shape)






