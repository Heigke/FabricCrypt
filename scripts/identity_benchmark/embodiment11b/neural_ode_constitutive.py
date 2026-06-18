"""Phase 11B — Neural ODE / Spiking / Ridge ESN A/B/C/D ablation matrix.

Hypothesis: ridge ESN architecture washes out substrate signal because the
ridge readout dominates. Neural ODE where substrate IS the vector field
should preserve it. We test 3 architectures on the same data + ablation matrix.

Substrate sources (4 channels, z-scored from Phase 8 rich .npz):
    APU temp  (hwmon_acpitz_temp1_input, ch 0)
    RAPL/GPU power (hwmon_amdgpu_power1_input, ch 13)
    GPU freq (hwmon_amdgpu_freq1_input, ch 10)
    GPU temp (hwmon_amdgpu_temp1_input, ch 14) -- proxy for RTN-rate

A: ikaros substrate (own chassi)
B: zero/constant substrate (no chassi info)
C: daedalus substrate (wrong-chassi)
D: random noise matched amplitude

Task: NARMA-10 (standard reservoir benchmark).
30 seeds, bootstrap 95% CI on A-B and A-C deltas.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

sys.path.insert(0, str(Path(__file__).parent))
from _thermal import wait_cool, read_apu_c

REPO = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
PHASE8_DIR = REPO / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment8"
OUT_DIR = REPO / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment11b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUBSTRATE_CH = [0, 13, 10, 14]  # APU temp, power, GPU freq, GPU temp
SUB_NAMES = ["apu_temp", "power", "gpu_freq", "gpu_temp"]


# ---------------------------------------------------------------------------
# Data: NARMA-10 task + substrate trajectories
# ---------------------------------------------------------------------------
def narma10(T: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, T + 200).astype(np.float32)
    y = np.zeros(T + 200, dtype=np.float32)
    for t in range(10, T + 199):
        y[t + 1] = (0.3 * y[t]
                    + 0.05 * y[t] * y[t-9:t+1].sum()
                    + 1.5 * u[t-9] * u[t]
                    + 0.1)
    return u[200:200+T], y[200:200+T]


def load_substrate(host: str) -> np.ndarray:
    """Return (T, 4) z-scored substrate trajectory."""
    z = np.load(PHASE8_DIR / f"{host}_rich.npz", allow_pickle=True)
    data = z["data"][:, SUBSTRATE_CH].astype(np.float32)
    # Z-score per channel using ikaros stats consistently (so chassi-difference shows up)
    # but to make B/D bounded we do per-host zscore; the cross-chassi C test still
    # carries timing/spectral differences.
    mu = data.mean(axis=0, keepdims=True)
    sd = data.std(axis=0, keepdims=True) + 1e-6
    return ((data - mu) / sd).astype(np.float32)


def build_substrate_variants(T: int, seed: int) -> dict[str, np.ndarray]:
    """Build A/B/C/D substrate trajectories of length T."""
    ik = load_substrate("ikaros")
    da = load_substrate("daedalus")
    rng = np.random.default_rng(seed + 7919)
    # Random crops length T from each chassi
    def crop(arr, name):
        if len(arr) <= T:
            reps = T // len(arr) + 2
            arr = np.tile(arr, (reps, 1))
        s = rng.integers(0, len(arr) - T)
        return arr[s:s+T]
    A = crop(ik, "ik")
    C = crop(da, "da")
    B = np.zeros((T, 4), dtype=np.float32)
    # D: random noise matched in std (z-scored => std≈1) to A
    D = rng.standard_normal((T, 4)).astype(np.float32) * A.std(axis=0, keepdims=True)
    return {"A": A, "B": B, "C": C, "D": D}


# ---------------------------------------------------------------------------
# Architecture 1: Ridge ESN with substrate as extra input
# ---------------------------------------------------------------------------
class RidgeESN:
    def __init__(self, structure_seed: int, n_res: int = 64, spectral_radius: float = 0.9):
        rng = np.random.default_rng(structure_seed)
        # 1 input (u) + 4 substrate channels; substrate columns scaled smaller
        W_in = rng.standard_normal((5, n_res)).astype(np.float32)
        W_in[0] *= 0.5            # u input gain
        W_in[1:] *= 0.1           # substrate input gain (smaller — distractor)
        self.W_in = W_in
        W = rng.standard_normal((n_res, n_res)).astype(np.float32)
        # Scale spectral radius
        evals = np.linalg.eigvals(W)
        W = W * (spectral_radius / max(abs(evals).max(), 1e-6))
        self.W_rec = W.astype(np.float32)
        self.bias = (rng.standard_normal(n_res) * 0.1).astype(np.float32)
        self.n_res = n_res
        self.W_out = None
        self.b_out = 0.0

    def states(self, u: np.ndarray, sub: np.ndarray) -> np.ndarray:
        T = len(u)
        h = np.zeros(self.n_res, dtype=np.float32)
        H = np.zeros((T, self.n_res), dtype=np.float32)
        for t in range(T):
            inp = np.concatenate([[u[t]], sub[t]])
            h = np.tanh(inp @ self.W_in + h @ self.W_rec + self.bias)
            H[t] = h
        return H

    def fit(self, u: np.ndarray, sub: np.ndarray, y: np.ndarray, lam: float = 1e-3, washout: int = 100):
        H = self.states(u, sub)
        Hf = H[washout:]
        yf = y[washout:]
        A = Hf.T @ Hf + lam * np.eye(self.n_res, dtype=np.float32)
        b = Hf.T @ yf
        self.W_out = np.linalg.solve(A, b)
        self.b_out = float(yf.mean() - Hf.mean(axis=0) @ self.W_out)

    def predict(self, u: np.ndarray, sub: np.ndarray) -> np.ndarray:
        H = self.states(u, sub)
        return H @ self.W_out + self.b_out


# ---------------------------------------------------------------------------
# Architecture 2: Neural ODE with substrate-parameterised vector field
# ---------------------------------------------------------------------------
class NeuralODEFunc(nn.Module):
    """dh/dt = -alpha*h + tanh( W1 h + W2 u(t) + g(s(t)) * h + W3 s(t) ).

    Substrate enters as:
      - multiplicative gate g(s) on hidden (constitutive coupling)
      - additive bias W3 s(t)
    """
    def __init__(self, n_hidden: int = 16):
        super().__init__()
        self.n_hidden = n_hidden
        self.W1 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.W2 = nn.Linear(1, n_hidden, bias=True)        # input u
        self.W3 = nn.Linear(4, n_hidden, bias=False)        # substrate additive
        self.Wg = nn.Linear(4, n_hidden, bias=True)         # substrate gate
        self.alpha = nn.Parameter(torch.tensor(1.0))
        # Trajectories injected per-forward
        self.u_traj = None  # (T,)
        self.s_traj = None  # (T, 4)
        self.dt = 1.0       # time per step

    def set_traj(self, u: torch.Tensor, s: torch.Tensor, dt: float = 1.0):
        self.u_traj = u
        self.s_traj = s
        self.dt = dt

    def forward(self, t, h):
        # t is scalar (ODE integration time). Map to discrete index.
        idx = torch.clamp((t / self.dt).long(), 0, self.u_traj.shape[0] - 1)
        u_t = self.u_traj[idx].unsqueeze(-1)
        s_t = self.s_traj[idx]
        gate = torch.sigmoid(self.Wg(s_t))
        drive = torch.tanh(self.W1(h) + self.W2(u_t) + self.W3(s_t) + gate * h)
        return -torch.abs(self.alpha) * h + drive


class NeuralODEModel(nn.Module):
    def __init__(self, n_hidden: int = 16):
        super().__init__()
        self.func = NeuralODEFunc(n_hidden)
        self.readout = nn.Linear(n_hidden, 1)
        self.n_hidden = n_hidden

    def forward(self, u: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        T = u.shape[0]
        h0 = torch.zeros(self.n_hidden, device=u.device)
        self.func.set_traj(u, s, dt=1.0)
        t_grid = torch.arange(T, dtype=torch.float32, device=u.device)
        # rtol/atol relaxed for speed; fixed-step solver = stable + fast
        H = odeint(self.func, h0, t_grid, method="rk4", options={"step_size": 1.0})
        return self.readout(H).squeeze(-1)


# ---------------------------------------------------------------------------
# Architecture 3: Spiking NN (LIF), substrate modulates threshold
# ---------------------------------------------------------------------------
class SpikingLIFNet(nn.Module):
    """Surrogate-gradient LIF. Substrate modulates per-neuron threshold and leak.

    v_{t+1} = (1-leak(s)) * v_t * (1-spike_t) + W_in u + W_rec spike_t
    spike  = surrogate( v - theta(s) )
    """
    def __init__(self, n_hidden: int = 32):
        super().__init__()
        self.n_hidden = n_hidden
        self.W_in = nn.Linear(1, n_hidden, bias=True)
        self.W_rec = nn.Linear(n_hidden, n_hidden, bias=False)
        self.W_thresh = nn.Linear(4, n_hidden, bias=True)
        self.W_leak = nn.Linear(4, n_hidden, bias=True)
        self.readout = nn.Linear(n_hidden, 1)

    def forward(self, u: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        T = u.shape[0]
        v = torch.zeros(self.n_hidden, device=u.device)
        sp = torch.zeros(self.n_hidden, device=u.device)
        ys = []
        for t in range(T):
            theta = 0.5 + 0.3 * torch.tanh(self.W_thresh(s[t]))
            leak = torch.sigmoid(self.W_leak(s[t])) * 0.3 + 0.1
            v = (1 - leak) * v * (1 - sp) + self.W_in(u[t:t+1]).squeeze(0) + self.W_rec(sp)
            # Surrogate gradient (fast sigmoid)
            diff = v - theta
            sp = (diff > 0).float() + (torch.sigmoid(4 * diff) - torch.sigmoid(4 * diff).detach())
            ys.append(self.readout(sp).squeeze(-1))
        return torch.stack(ys)


# ---------------------------------------------------------------------------
# Train / eval helpers
# ---------------------------------------------------------------------------
def nrmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = ((y_true - y_pred) ** 2).mean()
    var = y_true.var() + 1e-12
    return float(np.sqrt(err / var))


def split_train_test(u, y, sub, train_frac=0.7):
    T = len(u)
    nt = int(T * train_frac)
    return (u[:nt], y[:nt], sub[:nt]), (u[nt:], y[nt:], sub[nt:])


def run_ridge_esn(u_tr, y_tr, sub_tr, u_te, y_te, sub_te, struct_seed):
    m = RidgeESN(struct_seed)
    m.fit(u_tr, sub_tr, y_tr)
    pred = m.predict(u_te, sub_te)
    return nrmse_np(y_te[100:], pred[100:])  # skip washout


def run_neural_ode(u_tr, y_tr, sub_tr, u_te, y_te, sub_te, struct_seed, epochs=40):
    torch.manual_seed(struct_seed)
    m = NeuralODEModel(n_hidden=16).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    u_t = torch.tensor(u_tr, device=DEVICE)
    y_t = torch.tensor(y_tr, device=DEVICE)
    s_t = torch.tensor(sub_tr, device=DEVICE)
    chunk = 200
    for ep in range(epochs):
        # train in chunks for memory
        starts = list(range(0, len(u_tr) - chunk, chunk))
        np.random.shuffle(starts)
        for st in starts[:6]:  # 6 chunks/epoch
            uc = u_t[st:st+chunk]
            yc = y_t[st:st+chunk]
            sc = s_t[st:st+chunk]
            pred = m(uc, sc)
            loss = ((pred[50:] - yc[50:]) ** 2).mean()  # skip transient
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
    m.eval()
    with torch.no_grad():
        u_e = torch.tensor(u_te, device=DEVICE)
        s_e = torch.tensor(sub_te, device=DEVICE)
        pred = m(u_e, s_e).cpu().numpy()
    return nrmse_np(y_te[100:], pred[100:])


def run_spiking(u_tr, y_tr, sub_tr, u_te, y_te, sub_te, struct_seed, epochs=40):
    torch.manual_seed(struct_seed)
    m = SpikingLIFNet(n_hidden=32).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    u_t = torch.tensor(u_tr, device=DEVICE)
    y_t = torch.tensor(y_tr, device=DEVICE)
    s_t = torch.tensor(sub_tr, device=DEVICE)
    chunk = 200
    for ep in range(epochs):
        starts = list(range(0, len(u_tr) - chunk, chunk))
        np.random.shuffle(starts)
        for st in starts[:6]:
            uc = u_t[st:st+chunk]
            yc = y_t[st:st+chunk]
            sc = s_t[st:st+chunk]
            pred = m(uc, sc)
            loss = ((pred[50:] - yc[50:]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
    m.eval()
    with torch.no_grad():
        u_e = torch.tensor(u_te, device=DEVICE)
        s_e = torch.tensor(sub_te, device=DEVICE)
        pred = m(u_e, s_e).cpu().numpy()
    return nrmse_np(y_te[100:], pred[100:])


ARCHS = {
    "ridge_esn":  run_ridge_esn,
    "neural_ode": run_neural_ode,
    "spiking":    run_spiking,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_experiment(n_seeds: int = 30, T: int = 2000, archs: list[str] | None = None,
                   epochs: int = 40, log_every: int = 1):
    if archs is None:
        archs = list(ARCHS.keys())

    results = {a: {cond: [] for cond in "ABCD"} for a in archs}
    meta = {
        "n_seeds": n_seeds, "T": T, "archs": archs, "epochs": epochs,
        "task": "NARMA-10", "T_train_frac": 0.7,
        "substrate_channels": SUB_NAMES,
        "start": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(DEVICE),
    }

    t0 = time.time()
    for seed in range(n_seeds):
        # Generate NARMA task ONCE per seed (shared across archs & ablations)
        u, y = narma10(T, seed=seed)
        # Per-seed substrate variants
        sub_variants = build_substrate_variants(T, seed=seed)

        for arch in archs:
            for cond in "ABCD":
                wait_cool(tag=f"s{seed}/{arch}/{cond}")
                sub = sub_variants[cond]
                (u_tr, y_tr, s_tr), (u_te, y_te, s_te) = split_train_test(u, y, sub)
                struct_seed = hash((seed, arch, cond)) & 0xFFFFFFFF
                fn = ARCHS[arch]
                t_start = time.time()
                if arch == "ridge_esn":
                    err = fn(u_tr, y_tr, s_tr, u_te, y_te, s_te, struct_seed)
                else:
                    err = fn(u_tr, y_tr, s_tr, u_te, y_te, s_te, struct_seed, epochs=epochs)
                dt = time.time() - t_start
                results[arch][cond].append(err)
                if seed % log_every == 0:
                    print(f"[s{seed:02d} {arch:10s} {cond}] NRMSE={err:.4f}  "
                          f"dt={dt:.1f}s  APU={read_apu_c():.1f}C  "
                          f"total={(time.time()-t0)/60:.1f}min", flush=True)

        # Incremental save every seed
        meta["seeds_done"] = seed + 1
        meta["elapsed_min"] = (time.time() - t0) / 60.0
        with open(OUT_DIR / "abcd_neuralode.json", "w") as fout:
            json.dump({"meta": meta, "results": results}, fout, indent=2)

    meta["end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["elapsed_min"] = (time.time() - t0) / 60.0

    summary = compute_summary(results)
    payload = {"meta": meta, "results": results, "summary": summary}
    with open(OUT_DIR / "abcd_neuralode.json", "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def bootstrap_ci(arr: np.ndarray, n_boot: int = 2000, alpha: float = 0.05):
    rng = np.random.default_rng(0)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), len(arr))
        means.append(arr[idx].mean())
    means = np.array(means)
    return float(np.quantile(means, alpha/2)), float(np.quantile(means, 1-alpha/2))


def compute_summary(results: dict) -> dict:
    out = {}
    for arch, conds in results.items():
        A = np.array(conds["A"]); B = np.array(conds["B"])
        C = np.array(conds["C"]); D = np.array(conds["D"])
        if len(A) < 3:
            continue
        # NRMSE: lower = better. Relative improvement (B-A)/B (positive => substrate helps)
        delta_AB = (B - A) / (B + 1e-9)   # positive if A < B (substrate helps)
        delta_AC = (C - A) / (C + 1e-9)   # positive if A < C (own-chassi beats wrong-chassi)
        delta_AD = (D - A) / (D + 1e-9)
        out[arch] = {
            "n": len(A),
            "nrmse_A_mean": float(A.mean()), "nrmse_A_std": float(A.std(ddof=1)),
            "nrmse_B_mean": float(B.mean()), "nrmse_B_std": float(B.std(ddof=1)),
            "nrmse_C_mean": float(C.mean()), "nrmse_C_std": float(C.std(ddof=1)),
            "nrmse_D_mean": float(D.mean()), "nrmse_D_std": float(D.std(ddof=1)),
            "rel_AminusB_mean": float(delta_AB.mean()),
            "rel_AminusB_CI95": bootstrap_ci(delta_AB),
            "rel_AminusC_mean": float(delta_AC.mean()),
            "rel_AminusC_CI95": bootstrap_ci(delta_AC),
            "rel_AminusD_mean": float(delta_AD.mean()),
            "rel_AminusD_CI95": bootstrap_ci(delta_AD),
        }
        s = out[arch]
        s["gate_AB_15pct"] = bool(s["rel_AminusB_mean"] >= 0.15 and s["rel_AminusB_CI95"][0] > 0)
        s["gate_AC_10pct"] = bool(s["rel_AminusC_mean"] >= 0.10 and s["rel_AminusC_CI95"][0] > 0)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--T", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--archs", type=str, default="ridge_esn,neural_ode,spiking",
                    help="comma-separated subset")
    ap.add_argument("--smoke", action="store_true", help="quick 2-seed sanity run")
    args = ap.parse_args()

    if args.smoke:
        args.seeds = 2
        args.T = 800
        args.epochs = 10

    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    payload = run_experiment(n_seeds=args.seeds, T=args.T, archs=archs, epochs=args.epochs)
    print("\n=== SUMMARY ===")
    print(json.dumps(payload["summary"], indent=2))
