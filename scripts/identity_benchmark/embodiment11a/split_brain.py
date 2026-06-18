"""
TASK J — Split-brain co-dependence.

ONE reservoir (N=200) whose recurrent weights are split:
  - ikaros holds even-indexed neurons (W_i, Win_i)
  - daedalus holds odd-indexed neurons (W_d, Win_d) via remote TCP daemon

Forward step:
  x_{t+1}[even] = tanh( W_i @ x_t + Win_i * u_t )
  x_{t+1}[odd]  = tanh( W_d @ x_t + Win_d * u_t )   (computed remotely)

Train ridge-regression output W_out on NARMA-10. Eval NRMSE.

Conditions:
  BOTH_ALIVE   — full model
  IKAROS_CORRUPT  — randomize even-indexed half (local corrupt)
  DAEDALUS_CORRUPT — daemon randomizes its half
  BOTH_RANDOM  — both halves randomized

Pre-reg: BOTH_ALIVE NRMSE < 0.5,  any single-half-corrupt NRMSE >= 2.0.

Usage:
  # on daedalus first:  python split_brain_daemon.py
  # then on ikaros:     python split_brain.py
"""
from __future__ import annotations
import json, socket, struct, pickle, time, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment11a"
OUT.mkdir(parents=True, exist_ok=True)

DAEDALUS_HOST = os.environ.get("DAEDALUS_HOST", "daedalus.local")
PORT = 47011
N = 200
T_TRAIN = 1500
T_WASH = 200
T_TEST = 500
SEED = 20260601

RNG = np.random.default_rng(SEED)


# ---------- RPC ----------
class RemoteHalf:
    def __init__(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.sock.settimeout(30.0)

    def _send(self, obj):
        body = pickle.dumps(obj)
        self.sock.sendall(struct.pack("!I", len(body)) + body)

    def _recv(self):
        hdr = b""
        while len(hdr) < 4:
            c = self.sock.recv(4 - len(hdr))
            if not c:
                raise IOError("conn closed")
            hdr += c
        (l,) = struct.unpack("!I", hdr)
        body = b""
        while len(body) < l:
            c = self.sock.recv(min(65536, l - len(body)))
            if not c:
                raise IOError("conn closed")
            body += c
        return pickle.loads(body)

    def init(self, N, seed):
        self._send({"cmd": "init", "N": N, "seed": seed})
        return self._recv()

    def step(self, x, u):
        self._send({"cmd": "step", "x": x.astype(np.float32), "u": float(u)})
        return self._recv()["y"]

    def corrupt(self, mode="random"):
        self._send({"cmd": "corrupt", "mode": mode})
        return self._recv()

    def ping(self):
        self._send({"cmd": "ping"})
        return self._recv()

    def shutdown(self):
        try:
            self._send({"cmd": "shutdown"})
            _ = self._recv()
        except Exception:
            pass
        self.sock.close()


# ---------- Ikaros half ----------
class LocalHalf:
    def __init__(self, N, seed, spectral_radius=0.9, input_scale=0.5):
        rng = np.random.default_rng(seed)
        self.N = N
        self.idx = np.arange(0, N, 2)  # even
        nd = len(self.idx)
        W = rng.standard_normal((nd, N)) / np.sqrt(N)
        s = np.max(np.abs(np.linalg.eigvals(rng.standard_normal((nd, nd)) / np.sqrt(nd))))
        self.W = (spectral_radius / max(s, 1e-6)) * W
        self.Win = input_scale * rng.standard_normal(nd)
        self._saveW = (self.W.copy(), self.Win.copy())

    def restore(self):
        self.W, self.Win = self._saveW[0].copy(), self._saveW[1].copy()

    def corrupt(self):
        rng = np.random.default_rng(123)
        self.W = rng.standard_normal(self.W.shape) * np.std(self.W)
        self.Win = rng.standard_normal(self.Win.shape) * np.std(self.Win)

    def step(self, x, u):
        return self.W @ x + self.Win * u


# ---------- Task ----------
def narma10(T, rng):
    u = rng.uniform(0, 0.5, T)
    y = np.zeros(T)
    for t in range(10, T):
        y[t] = 0.3 * y[t - 1] + 0.05 * y[t - 1] * np.sum(y[t - 10 : t]) + 1.5 * u[t - 10] * u[t - 1] + 0.1
    return u, y


def run_episode(local: LocalHalf, remote: RemoteHalf, idx_local, idx_remote, u_train, y_train, u_test, y_test, label=""):
    N = local.N
    # collect states
    def collect(u_seq):
        T = len(u_seq)
        x = np.zeros(N)
        X = np.zeros((T, N))
        for t in range(T):
            pre = np.zeros(N)
            pre[idx_local] = local.step(x, u_seq[t])
            pre[idx_remote] = remote.step(x, u_seq[t])
            x = np.tanh(pre)
            X[t] = x
        return X
    Xtr = collect(u_train)
    Xte = collect(u_test)
    # ridge
    Xb = Xtr[T_WASH:]; yb = y_train[T_WASH:]
    A = Xb.T @ Xb + 1e-3 * np.eye(N)
    b = Xb.T @ yb
    Wout = np.linalg.solve(A, b)
    pred = Xte @ Wout
    err = pred[T_WASH:] - y_test[T_WASH:]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    nrmse = rmse / max(np.std(y_test[T_WASH:]), 1e-9)
    print(f"[J] {label}: NRMSE={nrmse:.4f}  RMSE={rmse:.4f}")
    return nrmse, rmse


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED + 7)
    print(f"[J] connecting to {DAEDALUS_HOST}:{PORT} ...")
    remote = RemoteHalf(DAEDALUS_HOST, PORT)
    remote.ping()
    info = remote.init(N=N, seed=SEED)
    idx_remote = np.array(info["idx"])
    print(f"[J] daemon init ok, remote-half size={info['n_half']}")

    local = LocalHalf(N=N, seed=SEED)
    idx_local = local.idx
    assert set(idx_local).isdisjoint(set(idx_remote))
    assert len(idx_local) + len(idx_remote) == N

    u_train, y_train = narma10(T_TRAIN, rng)
    u_test, y_test = narma10(T_TEST, rng)

    results = {}

    # ----- BOTH_ALIVE -----
    nr, rm = run_episode(local, remote, idx_local, idx_remote, u_train, y_train, u_test, y_test, "BOTH_ALIVE")
    results["BOTH_ALIVE"] = {"nrmse": nr, "rmse": rm}

    # ----- IKAROS_CORRUPT -----
    local.corrupt()
    nr, rm = run_episode(local, remote, idx_local, idx_remote, u_train, y_train, u_test, y_test, "IKAROS_CORRUPT")
    results["IKAROS_CORRUPT"] = {"nrmse": nr, "rmse": rm}
    local.restore()

    # ----- DAEDALUS_CORRUPT -----
    remote.corrupt("random")
    nr, rm = run_episode(local, remote, idx_local, idx_remote, u_train, y_train, u_test, y_test, "DAEDALUS_CORRUPT")
    results["DAEDALUS_CORRUPT"] = {"nrmse": nr, "rmse": rm}
    # re-init remote so it's restored
    remote.init(N=N, seed=SEED)

    # ----- BOTH_RANDOM -----
    local.corrupt(); remote.corrupt("random")
    nr, rm = run_episode(local, remote, idx_local, idx_remote, u_train, y_train, u_test, y_test, "BOTH_RANDOM")
    results["BOTH_RANDOM"] = {"nrmse": nr, "rmse": rm}

    # ----- BASELINE: monolithic ESN on ikaros only (sanity reference) -----
    rng2 = np.random.default_rng(SEED)
    W = rng2.standard_normal((N, N)) / np.sqrt(N)
    s = np.max(np.abs(np.linalg.eigvals(W)))
    W *= 0.9 / max(s, 1e-6)
    Win = 0.5 * rng2.standard_normal(N)
    def mono_collect(u_seq):
        x = np.zeros(N); X = np.zeros((len(u_seq), N))
        for t in range(len(u_seq)):
            x = np.tanh(W @ x + Win * u_seq[t]); X[t] = x
        return X
    Xtr = mono_collect(u_train); Xte = mono_collect(u_test)
    Xb = Xtr[T_WASH:]; yb = y_train[T_WASH:]
    Wout = np.linalg.solve(Xtr[T_WASH:].T @ Xtr[T_WASH:] + 1e-3 * np.eye(N), Xb.T @ yb)
    pred = Xte @ Wout
    err = pred[T_WASH:] - y_test[T_WASH:]
    mono_nrmse = float(np.sqrt(np.mean(err ** 2)) / max(np.std(y_test[T_WASH:]), 1e-9))
    results["MONO_BASELINE_LOCAL"] = {"nrmse": mono_nrmse}
    print(f"[J] MONO_BASELINE_LOCAL: NRMSE={mono_nrmse:.4f}")

    summary = {
        "task": "J_split_brain",
        "N": N,
        "T_train": T_TRAIN, "T_test": T_TEST, "T_washout": T_WASH,
        "daedalus_host": DAEDALUS_HOST,
        "results": results,
        "prereg_full_lt_0p5": bool(results["BOTH_ALIVE"]["nrmse"] < 0.5),
        "prereg_ikaros_corrupt_ge_2p0": bool(results["IKAROS_CORRUPT"]["nrmse"] >= 2.0),
        "prereg_daedalus_corrupt_ge_2p0": bool(results["DAEDALUS_CORRUPT"]["nrmse"] >= 2.0),
        "prereg_both_random_ge_2p0": bool(results["BOTH_RANDOM"]["nrmse"] >= 2.0),
        "prereg_PASS": bool(
            results["BOTH_ALIVE"]["nrmse"] < 0.5
            and results["IKAROS_CORRUPT"]["nrmse"] >= 2.0
            and results["DAEDALUS_CORRUPT"]["nrmse"] >= 2.0
        ),
        "elapsed_s": round(time.time() - t0, 2),
    }
    with open(OUT / "task_j_split_brain.json", "w") as f:
        json.dump(summary, f, indent=2)
    remote.shutdown()
    print(f"[J] wrote {OUT/'task_j_split_brain.json'}  PASS={summary['prereg_PASS']}")


if __name__ == "__main__":
    main()
