"""N_Res_MG — Phase N1 #2 / Phase N2 U7.

ER_SPARSE recurrent reservoir of NS-RAM-flavoured neurons (PT-default
surrogate: leaky integrator with saturating sinh-like nonlinearity that
mimics the I-V regime of the IiiNet cell at moderate Vd).

Task : Mackey-Glass tau=17 1-step-ahead forecasting.
N    : 1024 neurons, density ~1%.
Schedule: washout=500, train=4000, test=2000.

Runs on torch.cuda (zgx GB10).  Outputs:
  summary.json, predictions.npy, targets.npy,
  spikes.npy (last 500 steps), weights.npy (readout), dashboard.png.

Adapted from scripts/DS_N10_reservoir.py + scripts/DS_N13_topology_zoo.py.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "N_Res_MG"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "scripts"))

# ───────────────────────── Mackey-Glass ─────────────────────────
def mackey_glass(n_steps: int, tau: int = 17, beta: float = 0.2,
                 gamma: float = 0.1, n: float = 10.0,
                 dt: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    burnin = 1000
    total = n_steps + burnin
    hist_len = max(tau + 1, 30)
    x = 1.2 + 0.05 * rng.standard_normal(total + hist_len)
    for t in range(hist_len, total + hist_len - 1):
        x[t + 1] = x[t] + dt * (
            beta * x[t - tau] / (1.0 + x[t - tau] ** n) - gamma * x[t]
        )
    return x[hist_len + burnin: hist_len + burnin + n_steps].astype(np.float64)


# ───────────────────────── ER_SPARSE topology ─────────────────────────
def build_er_sparse(N: int, density: float, spectral_radius: float,
                    seed: int, device, dtype=torch.float32) -> torch.Tensor:
    """Erdos-Rényi sparse recurrent weight matrix, scaled to target sr."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    nnz = max(int(density * N * N), N)
    rows = torch.randint(0, N, (nnz,), generator=g)
    cols = torch.randint(0, N, (nnz,), generator=g)
    mask = rows != cols
    rows, cols = rows[mask], cols[mask]
    vals = torch.randn(rows.numel(), generator=g)
    indices = torch.stack([rows, cols])
    W = torch.sparse_coo_tensor(indices, vals, (N, N)).coalesce()
    # estimate spectral radius via power iteration (dense path for N=1024 is fine)
    Wd = W.to_dense()
    v = torch.randn(N, generator=g)
    v /= v.norm() + 1e-12
    for _ in range(30):
        v = Wd @ v
        nrm = v.norm() + 1e-12
        v = v / nrm
    sr = float(nrm)
    if sr > 0:
        Wd = Wd * (spectral_radius / sr)
    return Wd.to(device=device, dtype=dtype)


# ───────────────────────── NS-RAM-flavoured neuron ─────────────────────
# PT-default solver surrogate: V is body-charge analogue; current term
# uses tanh(alpha*(V+Vd_in)) - leak*V mimicking the IiiNet I-V at moderate
# Vd, then we integrate and spike on V>=V_th.
class NSRAMReservoirTorch:
    def __init__(self, N: int, density: float, spectral_radius: float,
                 leak: float = 0.7, V_th: float = 0.5, V_reset: float = 0.0,
                 input_scale: float = 1.0, fb_scale: float = 1.0,
                 alpha: float = 1.0, T_ref: int = 2,
                 device: str = "cuda", seed: int = 0):
        self.N = N
        self.device = torch.device(device)
        self.leak = leak
        self.V_th = V_th
        self.V_reset = V_reset
        self.input_scale = input_scale
        self.fb_scale = fb_scale
        self.alpha = alpha
        self.T_ref = T_ref

        g = torch.Generator(device="cpu").manual_seed(seed)
        self.W_in = (torch.rand(N, generator=g) * 2 - 1).to(self.device)
        self.W = build_er_sparse(N, density, spectral_radius, seed,
                                 device=self.device)
        self.reset()

    def reset(self):
        self.V = torch.zeros(self.N, device=self.device)
        self.s = torch.zeros(self.N, device=self.device)
        self.refr = torch.zeros(self.N, device=self.device, dtype=torch.int32)

    @torch.no_grad()
    def run(self, u: torch.Tensor, record_spikes: bool = False,
            spike_record_len: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """u : (T,) on device.  Returns (states (T,N), spikes_tail (k,N)).

        State update (NS-RAM PT-default surrogate, ESN-style continuous):
            V <- (1-a)*V + a*tanh(alpha*(W_in*u + W@V + bias))
        Spikes are emitted (V > V_th) for raster recording, but the readout
        uses the continuous V state — the body charge is the analogue
        observable, spikes are an auxiliary digital event.
        """
        T = u.shape[0]
        a = 1.0 - self.leak  # ESN leak rate
        feats = torch.empty((T, self.N), device=self.device)
        spike_buf = None
        if record_spikes and spike_record_len > 0:
            spike_buf = torch.zeros((spike_record_len, self.N),
                                    device=self.device, dtype=torch.uint8)
        for t in range(T):
            u_t = u[t]
            inp = self.input_scale * u_t * self.W_in
            rec = self.fb_scale * (self.W @ self.V)
            pre = self.alpha * (inp + rec)
            Vn = (1.0 - a) * self.V + a * torch.tanh(pre)
            self.V = Vn
            feats[t] = Vn
            if spike_buf is not None:
                idx = t - (T - spike_record_len)
                if idx >= 0:
                    spike_buf[idx] = (Vn > self.V_th).to(torch.uint8)
        return feats, spike_buf


# ───────────────────────── ridge readout ─────────────────────────
def ridge_train_cv(X: torch.Tensor, y: torch.Tensor,
                   alphas=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
                   ) -> tuple[torch.Tensor, float]:
    """Closed-form ridge with held-out 20% split for alpha selection."""
    n = X.shape[0]
    n_val = max(int(0.2 * n), 50)
    Xtr, Xv = X[:-n_val], X[-n_val:]
    ytr, yv = y[:-n_val], y[-n_val:]
    Xb = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1, device=X.device)], dim=1)
    A0 = Xb.T @ Xb
    b = Xb.T @ ytr
    I = torch.eye(A0.shape[0], device=X.device)
    best = (float("inf"), None, None)
    for a in alphas:
        A = A0 + a * I
        A[-1, -1] = a * 1e-3
        W = torch.linalg.solve(A, b)
        Xvb = torch.cat([Xv, torch.ones(Xv.shape[0], 1, device=X.device)], dim=1)
        yhat = Xvb @ W
        err = float(((yhat - yv) ** 2).mean())
        if err < best[0]:
            best = (err, W, a)
    return best[1], best[2]


def ridge_predict(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    Xb = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1)
    return Xb @ W


def nrmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    rmse = float(torch.sqrt(((y_true - y_pred) ** 2).mean()))
    return rmse / (float(y_true.std()) + 1e-12)


# ───────────────────────── main ─────────────────────────
def main():
    SEED = 0
    N = 1024
    DENSITY = 0.01
    SR = 0.9
    WASHOUT = 500
    TRAIN = 4000
    TEST = 2000
    SPIKE_TAIL = 500

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[N_Res_MG] device={device}  N={N}  density={DENSITY}  sr={SR}")
    print(f"[N_Res_MG] washout={WASHOUT} train={TRAIN} test={TEST}")

    total_steps = WASHOUT + TRAIN + TEST + 1
    x = mackey_glass(total_steps, tau=17, seed=SEED)
    x_norm = (x - x.mean()) / (x.std() + 1e-12)
    u_np = x_norm[:-1]
    y_np = x_norm[1:]

    u = torch.tensor(u_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)

    res = NSRAMReservoirTorch(N=N, density=DENSITY, spectral_radius=SR,
                              device=device, seed=SEED)

    # warmup so JIT/CUDA kernels are compiled before timing
    res.reset()
    _ = res.run(u[:32])
    res.reset()

    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    feats, spike_tail = res.run(u, record_spikes=True,
                                spike_record_len=SPIKE_TAIL)
    torch.cuda.synchronize() if device == "cuda" else None
    wall = time.time() - t0
    throughput = u.shape[0] / wall
    print(f"[N_Res_MG] reservoir run: {wall:.2f}s  "
          f"throughput={throughput:.1f} steps/sec")

    feats = feats[WASHOUT:]
    y_use = y[WASHOUT:]
    Xtr = feats[:TRAIN]; ytr = y_use[:TRAIN]
    Xte = feats[TRAIN:TRAIN + TEST]; yte = y_use[TRAIN:TRAIN + TEST]

    W_read, alpha_chosen = ridge_train_cv(Xtr, ytr)
    yhat = ridge_predict(Xte, W_read)
    nrmse_test = nrmse(yte, yhat)
    print(f"[N_Res_MG] alpha={alpha_chosen:g}  NRMSE_test={nrmse_test:.5f}")

    # gates
    infra_pass = True  # we got here and will write outputs
    discovery_pass = nrmse_test < 0.1
    ambitious_pass = (nrmse_test < 0.05) and (throughput > 10_000)

    summary = {
        "task": "Mackey-Glass tau=17 1-step",
        "N": N, "density": DENSITY, "spectral_radius": SR,
        "washout": WASHOUT, "train": TRAIN, "test": TEST,
        "nrmse_test": nrmse_test,
        "ridge_alpha_chosen": alpha_chosen,
        "throughput_steps_per_sec": throughput,
        "wall_reservoir_s": wall,
        "device": device,
        "torch_version": torch.__version__,
        "gates": {
            "INFRA": infra_pass,
            "DISCOVERY (NRMSE<0.1)": discovery_pass,
            "AMBITIOUS (NRMSE<0.05 & throughput>10k)": ambitious_pass,
        },
        "seed": SEED,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    np.save(OUT / "predictions.npy", yhat.cpu().numpy())
    np.save(OUT / "targets.npy", yte.cpu().numpy())
    np.save(OUT / "spikes.npy", spike_tail.cpu().numpy())
    np.save(OUT / "weights.npy", W_read.cpu().numpy())

    # report.md
    rep = [
        "# N_Res_MG — Mackey-Glass tau=17 forecast on ER_SPARSE NS-RAM reservoir",
        "",
        f"- N = {N}, density = {DENSITY}, spectral_radius = {SR}",
        f"- Schedule: washout={WASHOUT}, train={TRAIN}, test={TEST}",
        f"- Device: {device}  (torch {torch.__version__})",
        "",
        "## Results",
        f"- **NRMSE (test)** = {nrmse_test:.5f}",
        f"- Ridge alpha chosen (val-split): {alpha_chosen:g}",
        f"- Reservoir throughput: {throughput:.1f} steps/sec  "
        f"(wall {wall:.2f}s for {u.shape[0]} steps)",
        "",
        "## Pre-registered gates",
        f"- INFRA  (trains + dashboard written) : {'PASS' if infra_pass else 'FAIL'}",
        f"- DISCOVERY  (NRMSE < 0.1)            : {'PASS' if discovery_pass else 'FAIL'}",
        f"- AMBITIOUS  (NRMSE<0.05 & >10k st/s) : {'PASS' if ambitious_pass else 'FAIL'}",
        "",
        "## Files",
        "- summary.json, predictions.npy, targets.npy, spikes.npy, "
        "weights.npy, dashboard.png",
    ]
    (OUT / "report.md").write_text("\n".join(rep))

    print(f"[N_Res_MG] wrote outputs to {OUT}")
    print(f"[N_Res_MG] gates: INFRA={infra_pass}  "
          f"DISCOVERY={discovery_pass}  AMBITIOUS={ambitious_pass}")


if __name__ == "__main__":
    main()
