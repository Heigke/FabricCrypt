"""
Remote half-reservoir daemon. Runs on daedalus.
Holds W_d (a chunk of reservoir recurrent weights and input weights for
"daedalus-half" neurons). Receives full state vector x_t over a TCP socket,
returns its half's pre-activation contribution.

Protocol (length-prefixed pickled dict):
  {"cmd": "init", "N": 200, "seed": 1234, "corrupt": false}      -> {"ok": True}
  {"cmd": "step", "x": np.ndarray (N,), "u": float}              -> {"y": np.ndarray (N_d,)}
  {"cmd": "corrupt"}                                              -> {"ok": True}  # randomize weights
  {"cmd": "set_random_weights"}                                   -> sanity-check mode
  {"cmd": "ping"}                                                 -> {"ok": True}
  {"cmd": "shutdown"}                                             -> closes
"""
from __future__ import annotations
import socket, struct, pickle, sys
import numpy as np

HOST = "0.0.0.0"
PORT = 47011


class Half:
    def __init__(self):
        self.W = None
        self.Win = None
        self.idx = None  # which neurons (indices) belong to this half
        self.N = None
        self.spectral_radius = 0.9
        self.input_scale = 0.5

    def init(self, N, seed):
        rng = np.random.default_rng(seed + 1)  # different seed than ikaros half
        self.N = N
        # daedalus half = odd-indexed neurons
        self.idx = np.arange(1, N, 2)
        nd = len(self.idx)
        W = rng.standard_normal((nd, N)) / np.sqrt(N)
        # rescale so each half contributes ~spectral_radius / 2 of dynamics
        s = np.max(np.abs(np.linalg.eigvals(rng.standard_normal((nd, nd)) / np.sqrt(nd))))
        W = (self.spectral_radius / max(s, 1e-6)) * W
        self.W = W
        self.Win = self.input_scale * rng.standard_normal(nd)

    def corrupt(self, mode="random"):
        if self.W is None:
            return
        rng = np.random.default_rng(99999)
        if mode == "random":
            # replace with random weights of similar scale
            self.W = rng.standard_normal(self.W.shape) * np.std(self.W)
            self.Win = rng.standard_normal(self.Win.shape) * np.std(self.Win)
        elif mode == "zero":
            self.W = np.zeros_like(self.W)
            self.Win = np.zeros_like(self.Win)

    def step(self, x, u):
        # contribute pre-activation for daedalus-half neurons
        return self.W @ x + self.Win * u


def recv_all(sock, n):
    buf = b""
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            return None
        buf += c
    return buf


def recv_msg(sock):
    hdr = recv_all(sock, 4)
    if hdr is None:
        return None
    (l,) = struct.unpack("!I", hdr)
    body = recv_all(sock, l)
    if body is None:
        return None
    return pickle.loads(body)


def send_msg(sock, obj):
    body = pickle.dumps(obj)
    sock.sendall(struct.pack("!I", len(body)) + body)


def serve():
    half = Half()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(1)
    print(f"[daemon] listening on {HOST}:{PORT}", flush=True)
    while True:
        conn, addr = s.accept()
        print(f"[daemon] accept {addr}", flush=True)
        try:
            while True:
                msg = recv_msg(conn)
                if msg is None:
                    break
                cmd = msg.get("cmd")
                if cmd == "init":
                    half.init(int(msg["N"]), int(msg["seed"]))
                    send_msg(conn, {"ok": True, "n_half": len(half.idx), "idx": half.idx})
                elif cmd == "step":
                    y = half.step(msg["x"], float(msg["u"]))
                    send_msg(conn, {"y": y})
                elif cmd == "corrupt":
                    half.corrupt(msg.get("mode", "random"))
                    send_msg(conn, {"ok": True})
                elif cmd == "ping":
                    send_msg(conn, {"ok": True})
                elif cmd == "shutdown":
                    send_msg(conn, {"ok": True})
                    conn.close()
                    return
                else:
                    send_msg(conn, {"err": f"unknown cmd {cmd}"})
        except Exception as e:
            print(f"[daemon] err: {e}", flush=True)
        finally:
            conn.close()


if __name__ == "__main__":
    serve()
