"""Phase 9 Task B — CONSTITUTIVE live-substrate coupling.

The leak rate alpha[t] and gain[t] of a reservoir are computed at EVERY
forward-pass step from live substrate sensors. This is NOT a feature
appended to x; it IS a parameter of the recurrent update. Two hosts with
different thermal trajectories therefore run DIFFERENT dynamical systems.

Experiment matrix (transplant test):
    A — ikaros-trained,  evaluated with ikaros-recorded substrate trajectory
    B — daedalus-trained, evaluated with ikaros-recorded substrate trajectory
    C — random (alpha=0.5 constant), evaluated on ikaros data
    D — SHUFFLE: ikaros-trained model, evaluated with daedalus's substrate replayed
        (tests whether substrate-trajectory specificity matters)

Pre-reg:
    A − B  ≥ 10% NRMSE (own substrate beats transplant)
    A − D  ≥ 5%  NRMSE (own-trajectory beats alien-trajectory replay)
    A − C  ≥ 5%  NRMSE (substrate coupling beats no-coupling)

Substrate trajectories are RECORDED first (so we can replay on either host
for offline reproducible eval). Source: /sys/class/thermal/thermal_zone0/temp
and amdgpu hwmon power1_average. Sampled at the same step rate as the model
runs (10 Hz nominal, no GPU stress to keep APU < 70 C).
"""
from __future__ import annotations
import json
import socket
import time
import argparse
from pathlib import Path
import numpy as np

HOST = socket.gethostname()
ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
DATA_DIR = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment5"
OUT_DIR = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment9"
OUT_DIR.mkdir(parents=True, exist_ok=True)

THERMAL_PATH = Path("/sys/class/thermal/thermal_zone0/temp")


def find_amdgpu_hwmon() -> Path | None:
    for h in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (h / "name").read_text().strip() == "amdgpu":
                return h
        except Exception:
            pass
    return None


def read_apu_temp_c() -> float:
    try:
        return int(THERMAL_PATH.read_text().strip()) / 1000.0
    except Exception:
        return 50.0


def read_gpu_power_w(hwmon: Path | None) -> float:
    if hwmon is None:
        return 15.0
    try:
        p = (hwmon / "power1_average").read_text().strip()
        return int(p) / 1_000_000.0
    except Exception:
        try:
            p = (hwmon / "power1_input").read_text().strip()
            return int(p) / 1_000_000.0
        except Exception:
            return 15.0


def record_substrate(n_steps: int, dt: float = 0.1, light_load: bool = True) -> np.ndarray:
    """Record a live substrate trajectory (n_steps, 2) = [T_c, P_w].

    `light_load`: do a small numpy op each step so the substrate actually
    moves. Capped to keep APU < 70 C.
    """
    hwmon = find_amdgpu_hwmon()
    out = np.zeros((n_steps, 2), dtype=np.float32)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((128, 128)).astype(np.float32)
    for t in range(n_steps):
        if light_load:
            _ = A @ A  # ~50 us, harmless
        T = read_apu_temp_c()
        P = read_gpu_power_w(hwmon)
        out[t] = (T, P)
        if T > 70.0:
            time.sleep(0.5)  # thermal safety
        time.sleep(dt)
    return out


# ---------------------------------------------------------------------------
# Constitutive reservoir
# ---------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class ConstitutiveReservoir:
    """Reservoir with leak rate alpha[t] and gain[t] computed from LIVE
    substrate at every step. Substrate trajectory is supplied externally
    (recorded once per host then replayed for reproducibility).
    """

    def __init__(self, din: int, dout: int, dim: int = 128, seed: int = 0,
                 coupling_mode: str = "constitutive"):
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.din = din
        self.dout = dout
        self.W_in = rng.standard_normal((din, dim)).astype(np.float32) / np.sqrt(din)
        self.W_rec = rng.standard_normal((dim, dim)).astype(np.float32) / np.sqrt(dim) * 0.9
        self.bias = rng.standard_normal(dim).astype(np.float32) * 0.1
        self.W_out = None
        self.b_out = None
        self.coupling_mode = coupling_mode  # 'constitutive', 'control_const'

    def step(self, h, x, sub_T, sub_P):
        if self.coupling_mode == "control_const":
            alpha = 0.5
            gain = 1.0
        else:
            alpha = float(sigmoid((sub_T - 50.0) / 10.0) * 0.5 + 0.25)
            gain = float(1.0 + 0.1 * (sub_P / 15.0 - 1.0))
        pre = gain * (x @ self.W_in + h @ self.W_rec + self.bias)
        return (1.0 - alpha) * h + alpha * np.tanh(pre)

    def features(self, X, substrate):
        """X: (n, T, din); substrate: (T, 2) replayed for all n.
        Returns final reservoir state (n, dim).
        """
        n, T, _ = X.shape
        h = np.zeros((n, self.dim), dtype=np.float32)
        for t in range(T):
            sub_T = substrate[t, 0]
            sub_P = substrate[t, 1]
            h = self.step(h, X[:, t, :], sub_T, sub_P)
        return h

    def fit(self, X, Y, substrate, lam: float = 1e-3):
        H = self.features(X, substrate)
        Yf = Y.reshape(len(X), -1)
        A = H.T @ H + lam * np.eye(self.dim, dtype=np.float32)
        B = H.T @ Yf
        self.W_out = np.linalg.solve(A, B)
        self.b_out = Yf.mean(0) - H.mean(0) @ self.W_out

    def predict(self, X, substrate, horizon: int, dout: int):
        H = self.features(X, substrate)
        Yf = H @ self.W_out + self.b_out
        return Yf.reshape(-1, horizon, dout)


def load_data(host: str) -> np.ndarray:
    if host == "ikaros":
        return np.load(DATA_DIR / "c1_ikaros" / "c1_ikaros_data.npy")
    return np.load(DATA_DIR / f"c1_{host}_data.npy")


def make_windows(data, n, hist, horizon, seed):
    rng = np.random.default_rng(seed)
    T = len(data)
    starts = rng.choice(T - hist - horizon, n, replace=False)
    X = np.stack([data[s:s+hist] for s in starts]).astype(np.float32)
    Y = np.stack([data[s+hist:s+hist+horizon] for s in starts]).astype(np.float32)
    return X, Y


def nrmse(yt, yp):
    err = ((yt - yp) ** 2).mean(axis=(0, 1))
    var = yt.var(axis=(0, 1)) + 1e-8
    return float(np.sqrt(err / var).mean())


def bootstrap_ci(arr, n=2000, alpha=0.05):
    arr = np.asarray(arr)
    rng = np.random.default_rng(0)
    bs = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                   for _ in range(n)])
    return float(np.percentile(bs, 100*alpha/2)), float(np.percentile(bs, 100*(1-alpha/2)))


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------
HIST = 100
HORIZON = 10
D = 5
N_TRAIN = 400
N_TEST = 120
N_SEEDS = 30


def run_cell(model_train_host: str, eval_substrate_host: str, mode: str,
             substrates: dict, n_seeds: int = N_SEEDS) -> dict:
    """One ABCD cell.

    model_train_host: 'ikaros' / 'daedalus' / 'random_init' (untrained dummy)
    eval_substrate_host: which substrate trajectory to replay at INFERENCE
    mode: 'constitutive' | 'control_const'
    """
    # We always train on the SAME-HOST data + SAME-HOST substrate as
    # model_train_host (the "natural" pairing). The transplant is at
    # inference time: eval data is the local eval host's data; eval
    # substrate is eval_substrate_host's recorded trajectory.
    if model_train_host == "random_init":
        # no fit — control C: constant alpha untrained
        nrmses = []
        for seed in range(n_seeds):
            data_eval = load_data(HOST)
            _, _ = make_windows(data_eval, N_TRAIN, HIST, HORIZON, seed)
            Xte, Yte = make_windows(data_eval, N_TEST, HIST, HORIZON, seed + 9001)
            mu = data_eval.mean(axis=(0,), keepdims=True)
            sd = data_eval.std(axis=(0,), keepdims=True) + 1e-6
            Xte = (Xte - mu) / sd; Yte_n = (Yte - mu) / sd
            sub = substrates[eval_substrate_host]
            sub = sub[np.linspace(0, len(sub)-1, HIST).astype(int)]
            model = ConstitutiveReservoir(D, D, seed=seed, coupling_mode="control_const")
            # untrained: predict zeros (Y is normalized → mean 0 is best baseline)
            Ypred = np.zeros_like(Yte_n)
            nrmses.append(nrmse(Yte_n, Ypred))
        return {"nrmse_per_seed": nrmses}

    data_train = load_data(model_train_host)
    data_eval = load_data(HOST)
    sub_train = substrates[model_train_host]
    sub_eval = substrates[eval_substrate_host]
    # Resample substrate to HIST steps
    sub_train_h = sub_train[np.linspace(0, len(sub_train)-1, HIST).astype(int)]
    sub_eval_h = sub_eval[np.linspace(0, len(sub_eval)-1, HIST).astype(int)]

    mu_tr = data_train.mean(axis=(0,), keepdims=True)
    sd_tr = data_train.std(axis=(0,), keepdims=True) + 1e-6

    nrmses = []
    for seed in range(n_seeds):
        Xtr, Ytr = make_windows(data_train, N_TRAIN, HIST, HORIZON, seed)
        Xte, Yte = make_windows(data_eval, N_TEST, HIST, HORIZON, seed + 9001)
        Xtr = (Xtr - mu_tr) / sd_tr; Ytr = (Ytr - mu_tr) / sd_tr
        Xte = (Xte - mu_tr) / sd_tr; Yte_n = (Yte - mu_tr) / sd_tr
        model = ConstitutiveReservoir(D, D, seed=seed, coupling_mode=mode)
        model.fit(Xtr, Ytr, sub_train_h)
        Ypred = model.predict(Xte, sub_eval_h, HORIZON, D)
        nrmses.append(nrmse(Yte_n, Ypred))
    return {"nrmse_per_seed": nrmses}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record-substrate", action="store_true",
                    help="record a fresh substrate trajectory on this host")
    ap.add_argument("--n-steps", type=int, default=300)
    args = ap.parse_args()

    sub_path = OUT_DIR / f"substrate_{HOST}.npy"
    if args.record_substrate or not sub_path.exists():
        print(f"[{HOST}] recording substrate ({args.n_steps} steps @ 10 Hz)...")
        sub = record_substrate(args.n_steps, dt=0.1)
        np.save(sub_path, sub)
        print(f"[{HOST}] substrate saved: mean T={sub[:,0].mean():.1f}C "
              f"P={sub[:,1].mean():.2f}W, std T={sub[:,0].std():.2f}")
        return

    # Load substrates from both hosts (must have been recorded already)
    substrates = {}
    for host in ("ikaros", "daedalus"):
        p = OUT_DIR / f"substrate_{host}.npy"
        if not p.exists():
            print(f"[!] missing substrate {p} — run --record-substrate on {host}")
            return
        substrates[host] = np.load(p)
        print(f"[{host}] substrate: T={substrates[host][:,0].mean():.1f}C "
              f"P={substrates[host][:,1].mean():.2f}W")

    # ABCD on this eval host
    cells = {
        "A_own_own":    run_cell(HOST,           HOST,            "constitutive", substrates),
        "B_other_own":  run_cell("daedalus" if HOST=="ikaros" else "ikaros", HOST, "constitutive", substrates),
        "C_no_coupling":run_cell(HOST,           HOST,            "control_const", substrates),
        "D_shuffle":    run_cell(HOST,           "daedalus" if HOST=="ikaros" else "ikaros", "constitutive", substrates),
    }

    results = {"host": HOST, "n_seeds": N_SEEDS, "cells": {}}
    for name, c in cells.items():
        arr = np.array(c["nrmse_per_seed"])
        lo, hi = bootstrap_ci(arr)
        results["cells"][name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "ci95": [lo, hi],
            "per_seed": c["nrmse_per_seed"],
        }
        print(f"  {name:18s}  NRMSE = {arr.mean():.4f} ± {arr.std():.4f}  CI95=[{lo:.4f}, {hi:.4f}]")

    # Pre-reg gates
    A = np.array(cells["A_own_own"]["nrmse_per_seed"])
    B = np.array(cells["B_other_own"]["nrmse_per_seed"])
    C = np.array(cells["C_no_coupling"]["nrmse_per_seed"])
    D = np.array(cells["D_shuffle"]["nrmse_per_seed"])

    def rel_advantage(other, own):
        return float((other.mean() - own.mean()) / other.mean())

    gates = {
        "A_vs_B_pct": rel_advantage(B, A),  # positive = A better (lower NRMSE)
        "A_vs_D_pct": rel_advantage(D, A),
        "A_vs_C_pct": rel_advantage(C, A),
        "gate_A_minus_B_ge_10pct": rel_advantage(B, A) >= 0.10,
        "gate_A_minus_D_ge_5pct":  rel_advantage(D, A) >= 0.05,
        "gate_A_minus_C_ge_5pct":  rel_advantage(C, A) >= 0.05,
    }
    results["gates"] = gates
    print(f"\n  Gates: {gates}")

    out_path = OUT_DIR / f"constitutive_{HOST}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
