#!/usr/bin/env python3
"""
z261 — N1b: real-MNIST 28×28 LIF baseline with faithful "Poisson-trained
weights, LIF inference, no readout refit" pipeline (Sebas slide 12.39).

Stage A — Poisson reference network:
  N=128 rate-coded units, sigmoid(W_in @ x + W @ prev) iterated for a
  few steps OR a single-step continuous readout (slide describes
  "Poisson neurons" as instantaneous rate units). We use the canonical
  reservoir-style single-step: rate = sigmoid(W_in @ x). Train ridge
  readout 128 → 10. Expect ~85% on MNIST.

Stage B — LIF inference with frozen Poisson weights:
  SAME W, W_in trained in Stage A. NOT retrained.
  Replace rate units with LIF (tau=10, V_TH=1, dt=1, T=100, Poisson
  rate-encoded inputs). Mean spike rate per neuron over T steps is
  fed to the FROZEN Stage-A readout. Expect ~72%.

5 seeds, mean ± std + 95% bootstrap CI.

Pre-registered gates (locked in research_plan/01_LOG.md, 2026-05-11
BEFORE first training run):
  Gate A  (Poisson)      : mean ∈ [80%, 90%]
  Gate B  (LIF strict)   : mean ∈ [70%, 74%]
  Gate C  (drop P → LIF) : drop ∈ [10, 16] pp

NO-CHEAT: no tuning of tau, V_TH, ridge α, T, N after first run.

GPU: AMD Radeon 8060S via ROCm with HSA_OVERRIDE_GFX_VERSION=11.0.0.
Thermal: APU 99°C trip → forced shutdown. Pause at 75°C, abort at 90°C.
"""

from __future__ import annotations

# CRITICAL: must precede torch import for gfx1151
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import json
import time
from pathlib import Path

import numpy as np
import torch

# ------------------------------------------------------------------
# Config (locked by spec — DO NOT tune to hit gates)
# ------------------------------------------------------------------
N_NEURONS = 128
TAU = 10.0
V_THRESH = 1.0
V_RESET = 0.0
DT = 1.0
T_STEPS = 100
N_SEEDS = 5
SEEDS = [0, 1, 2, 3, 4]
BOOTSTRAP_N = 1000

RESERVOIR_GAIN = 0.3
W_IN_SCALE = 0.3
SPECTRAL_RADIUS = 0.9
SPARSITY = 0.1
N_EXC_FRAC = 0.8
SYN_DECAY = 0.8
RIDGE_LAMBDA = 1e-3

THERMAL_ZONE = Path("/sys/class/thermal/thermal_zone0/temp")
THERMAL_PAUSE_C = 75.0
THERMAL_RESUME_C = 50.0
THERMAL_ABORT_C = 90.0
THERMAL_CHECK_EVERY = 30  # batches

RESULTS_DIR = (
    Path(__file__).resolve().parent.parent / "results" / "z261_n1b_lif_mnist"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def apu_temp_c() -> float:
    try:
        return float(THERMAL_ZONE.read_text().strip()) / 1000.0
    except Exception:
        return -1.0


def thermal_guard():
    t = apu_temp_c()
    if t >= THERMAL_ABORT_C:
        raise RuntimeError(f"APU thermal abort at {t:.1f}C")
    if t >= THERMAL_PAUSE_C:
        print(
            f"  [thermal] APU={t:.1f}C — pausing until <{THERMAL_RESUME_C}C",
            flush=True,
        )
        while apu_temp_c() > THERMAL_RESUME_C:
            time.sleep(5)
        print(f"  [thermal] resumed at APU={apu_temp_c():.1f}C", flush=True)
    return t


# ------------------------------------------------------------------
# MNIST loader (torchvision -> fetch_openml fallback)
# ------------------------------------------------------------------
def load_mnist():
    """Returns (X_train, y_train, X_test, y_test, source) all normalized [0,1]."""
    # Try torchvision first
    try:
        import torchvision  # noqa
        from torchvision import datasets

        t0 = time.time()
        train_ds = datasets.MNIST("/tmp/mnist", train=True, download=True)
        test_ds = datasets.MNIST("/tmp/mnist", train=False, download=True)
        dt = time.time() - t0
        if dt > 300:
            raise TimeoutError("torchvision MNIST took >5 min")

        X_train = train_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        y_train = train_ds.targets.numpy().astype(np.int64)
        X_test = test_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        y_test = test_ds.targets.numpy().astype(np.int64)
        return X_train, y_train, X_test, y_test, "torchvision_mnist"
    except Exception as e:
        print(f"[loader] torchvision failed: {e!r} — falling back to fetch_openml",
              flush=True)

    # Fallback: sklearn fetch_openml
    from sklearn.datasets import fetch_openml

    t0 = time.time()
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    dt = time.time() - t0
    if dt > 300:
        raise TimeoutError("fetch_openml MNIST took >5 min")
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int64)
    # standard split: first 60k train, last 10k test
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test, "fetch_openml_mnist_784"


# ------------------------------------------------------------------
# Reservoir construction (shared between Stage A and Stage B)
# ------------------------------------------------------------------
def build_reservoir(N: int, n_in: int, seed: int, device):
    rng = np.random.RandomState(seed)
    mask = (rng.rand(N, N) < SPARSITY).astype(np.float32)
    np.fill_diagonal(mask, 0)
    W = rng.randn(N, N).astype(np.float32) * mask
    n_exc = int(N * N_EXC_FRAC)
    signs = np.ones(N, dtype=np.float32)
    signs[n_exc:] = -1
    W = np.abs(W) * signs[:, None]
    eigs = np.abs(np.linalg.eigvals(W))
    if eigs.max() > 0:
        W *= SPECTRAL_RADIUS / eigs.max()
    W_in = rng.randn(N, n_in).astype(np.float32) * W_IN_SCALE
    return (
        torch.from_numpy(W).to(device),
        torch.from_numpy(W_in).to(device),
    )


# ------------------------------------------------------------------
# Stage A — Poisson (rate-coded) features
# ------------------------------------------------------------------
@torch.no_grad()
def run_rate_batch(X_batch, W, W_in):
    """
    X_batch: (B, n_in) intensities in [0,1]
    Iterated rate update for T_STEPS to give recurrence a chance.
    Returns mean rate per neuron: (B, N)
    """
    B = X_batch.shape[0]
    N = W.shape[0]
    r = torch.zeros(B, N, device=X_batch.device)
    rate_sum = torch.zeros(B, N, device=X_batch.device)
    Wt = W.t()
    Win_t = W_in.t()  # (n_in, N)
    for _ in range(T_STEPS):
        I = X_batch @ Win_t + (r @ Wt) * RESERVOIR_GAIN
        r = torch.sigmoid(I)
        rate_sum += r
    return rate_sum / T_STEPS


# ------------------------------------------------------------------
# Stage B — LIF inference with frozen Stage-A weights
# ------------------------------------------------------------------
@torch.no_grad()
def run_lif_batch(spikes_in, W, W_in):
    """
    spikes_in: (B, T, n_in) binary spikes (Poisson encoded pixels)
    Returns mean spike rate per neuron: (B, N)
    """
    B, T, n_in = spikes_in.shape
    N = W.shape[0]
    alpha = float(np.exp(-DT / TAU))

    v = torch.zeros(B, N, device=spikes_in.device)
    syn = torch.zeros(B, N, device=spikes_in.device)
    spike_sum = torch.zeros(B, N, device=spikes_in.device)

    Wt = W.t()
    Win_t = W_in.t()

    for t in range(T):
        I = spikes_in[:, t] @ Win_t + (syn @ Wt) * RESERVOIR_GAIN
        v = alpha * v + (1 - alpha) * I
        fired = (v >= V_THRESH).float()
        spike_sum += fired
        v = torch.where(fired.bool(), torch.full_like(v, V_RESET), v)
        syn = SYN_DECAY * syn + fired
    return spike_sum / T


def poisson_encode_torch(X_batch: torch.Tensor, T: int, generator) -> torch.Tensor:
    """X_batch: (B, n_in) in [0,1]. Returns (B, T, n_in) binary."""
    B, n_in = X_batch.shape
    p = X_batch.unsqueeze(1).expand(B, T, n_in)
    u = torch.rand(B, T, n_in, generator=generator, device=X_batch.device)
    return (u < p).float()


# ------------------------------------------------------------------
# Featurize helpers
# ------------------------------------------------------------------
def featurize_rate(X_np, W, W_in, device, batch=512):
    out = np.zeros((X_np.shape[0], N_NEURONS), dtype=np.float32)
    nb = 0
    for i in range(0, X_np.shape[0], batch):
        chunk = torch.from_numpy(X_np[i:i + batch]).to(device)
        feats = run_rate_batch(chunk, W, W_in)
        out[i:i + batch] = feats.detach().cpu().numpy()
        nb += 1
        if nb % THERMAL_CHECK_EVERY == 0:
            thermal_guard()
    return out


def featurize_lif(X_np, W, W_in, device, seed: int, batch=256):
    """LIF with Poisson-encoded input. seed controls Poisson noise."""
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) * 1000 + 7)
    out = np.zeros((X_np.shape[0], N_NEURONS), dtype=np.float32)
    nb = 0
    for i in range(0, X_np.shape[0], batch):
        chunk = torch.from_numpy(X_np[i:i + batch]).to(device)
        spikes = poisson_encode_torch(chunk, T_STEPS, gen)
        feats = run_lif_batch(spikes, W, W_in)
        out[i:i + batch] = feats.detach().cpu().numpy()
        nb += 1
        if nb % THERMAL_CHECK_EVERY == 0:
            thermal_guard()
    return out


# ------------------------------------------------------------------
# Ridge readout fit + apply
# ------------------------------------------------------------------
def fit_ridge_readout(F_train: np.ndarray, y_train: np.ndarray, n_classes: int):
    Y = np.zeros((len(y_train), n_classes), dtype=np.float32)
    Y[np.arange(len(y_train)), y_train] = 1.0
    F_b = np.hstack([F_train, np.ones((len(F_train), 1), dtype=np.float32)])
    A = F_b.T @ F_b + RIDGE_LAMBDA * np.eye(F_b.shape[1], dtype=np.float32)
    b = F_b.T @ Y
    Wro = np.linalg.solve(A, b)
    return Wro  # (N+1, n_classes)


def apply_readout(F: np.ndarray, Wro: np.ndarray) -> np.ndarray:
    F_b = np.hstack([F, np.ones((len(F), 1), dtype=np.float32)])
    return (F_b @ Wro).argmax(axis=1)


# ------------------------------------------------------------------
# Single-seed run: Stage A then Stage B with frozen readout
# ------------------------------------------------------------------
def run_seed(seed, X_train, y_train, X_test, y_test, device):
    t0 = time.time()
    print(f"[seed {seed}] start  APU={apu_temp_c():.1f}C", flush=True)
    torch.manual_seed(seed)

    n_in = X_train.shape[1]
    W, W_in = build_reservoir(N_NEURONS, n_in, seed, device)

    # --- Stage A: Poisson (rate) ---
    Fa_tr = featurize_rate(X_train, W, W_in, device)
    Fa_te = featurize_rate(X_test, W, W_in, device)
    n_classes = int(max(y_train.max(), y_test.max())) + 1
    Wro = fit_ridge_readout(Fa_tr, y_train, n_classes)

    poisson_train_acc = float((apply_readout(Fa_tr, Wro) == y_train).mean())
    poisson_test_acc = float((apply_readout(Fa_te, Wro) == y_test).mean())
    t_after_A = time.time() - t0
    print(
        f"[seed {seed}] STAGE A (Poisson)  train={poisson_train_acc:.4f}  "
        f"test={poisson_test_acc:.4f}  ({t_after_A:.1f}s, APU={apu_temp_c():.1f}C)",
        flush=True,
    )
    thermal_guard()

    # --- Stage B: LIF inference with FROZEN W, W_in, Wro ---
    Fb_tr = featurize_lif(X_train, W, W_in, device, seed=seed)
    Fb_te = featurize_lif(X_test, W, W_in, device, seed=seed + 100000)

    lif_train_acc = float((apply_readout(Fb_tr, Wro) == y_train).mean())
    lif_test_acc = float((apply_readout(Fb_te, Wro) == y_test).mean())
    dt_total = time.time() - t0
    print(
        f"[seed {seed}] STAGE B (LIF)      train={lif_train_acc:.4f}  "
        f"test={lif_test_acc:.4f}  ({dt_total:.1f}s, APU={apu_temp_c():.1f}C)",
        flush=True,
    )

    return {
        "seed": seed,
        "poisson_train_acc": poisson_train_acc,
        "poisson_test_acc": poisson_test_acc,
        "lif_train_acc": lif_train_acc,
        "lif_test_acc": lif_test_acc,
        "wall_s": dt_total,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def bootstrap_ci(arr, n=BOOTSTRAP_N, seed=12345):
    rng = np.random.RandomState(seed)
    a = np.asarray(arr)
    boots = np.array(
        [rng.choice(a, size=len(a), replace=True).mean() for _ in range(n)]
    )
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    print(f"torch {torch.__version__}  cuda_avail={torch.cuda.is_available()}",
          flush=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        device = torch.device("cpu")
        print("device: CPU (ROCm unavailable)", flush=True)

    print("[loader] loading MNIST 28×28...", flush=True)
    X_train, y_train, X_test, y_test, dataset = load_mnist()
    print(
        f"dataset: {dataset}  train={X_train.shape}  test={X_test.shape}",
        flush=True,
    )

    t_global = time.time()
    thermal_peak = apu_temp_c()
    per_seed = []
    for s in SEEDS:
        r = run_seed(s, X_train, y_train, X_test, y_test, device)
        per_seed.append(r)
        thermal_peak = max(thermal_peak, apu_temp_c())
        thermal_guard()

    poisson_accs = np.array([r["poisson_test_acc"] for r in per_seed])
    lif_accs = np.array([r["lif_test_acc"] for r in per_seed])

    poisson_mean = float(poisson_accs.mean())
    poisson_std = float(poisson_accs.std(ddof=1))
    poisson_ci = bootstrap_ci(poisson_accs)

    lif_mean = float(lif_accs.mean())
    lif_std = float(lif_accs.std(ddof=1))
    lif_ci = bootstrap_ci(lif_accs)

    drop_pp = (poisson_mean - lif_mean) * 100.0

    verdict_A = "PASS" if (0.80 <= poisson_mean <= 0.90) else "FAIL"
    verdict_B = "PASS" if (0.70 <= lif_mean <= 0.74) else "FAIL"
    verdict_C = "PASS" if (10.0 <= drop_pp <= 16.0) else "FAIL"

    wall = time.time() - t_global
    summary = {
        "experiment": "z261_n1b_lif_mnist",
        "date": "2026-05-11",
        "dataset": dataset,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_classes": 10,
        "config": {
            "N_neurons": N_NEURONS,
            "tau": TAU,
            "v_thresh": V_THRESH,
            "v_reset": V_RESET,
            "dt": DT,
            "T_steps": T_STEPS,
            "reservoir_gain": RESERVOIR_GAIN,
            "w_in_scale": W_IN_SCALE,
            "spectral_radius": SPECTRAL_RADIUS,
            "sparsity": SPARSITY,
            "syn_decay": SYN_DECAY,
            "readout": "ridge_lstsq_FROZEN_from_stage_A",
            "ridge_lambda": RIDGE_LAMBDA,
        },
        "n_seeds": N_SEEDS,
        "seeds": SEEDS,
        "per_seed": per_seed,
        "poisson_accuracy_mean": poisson_mean,
        "poisson_accuracy_std": poisson_std,
        "poisson_ci95": list(poisson_ci),
        "lif_accuracy_mean": lif_mean,
        "lif_accuracy_std": lif_std,
        "lif_ci95": list(lif_ci),
        "drop_pp": drop_pp,
        "gates": {
            "A_poisson_85pct": {"lo": 0.80, "hi": 0.90, "target": 0.85},
            "B_lif_72pct":     {"lo": 0.70, "hi": 0.74, "target": 0.72},
            "C_drop_13pp":     {"lo": 10.0, "hi": 16.0, "target": 13.0},
        },
        "stage_a_verdict_85pct": verdict_A,
        "stage_b_verdict_72pct": verdict_B,
        "drop_pp_verdict_13pp": verdict_C,
        "source": "Sebas slide 12.39 — LIF=72%, Poisson=85%, drop=13pp",
        "wall_s_total": wall,
        "thermal_peak_c": thermal_peak,
        "device": str(device),
        "torch_version": torch.__version__,
    }

    out = RESULTS_DIR / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWROTE {out}", flush=True)
    print(
        f"Poisson: mean={poisson_mean:.4f}  std={poisson_std:.4f}  "
        f"CI95={poisson_ci[0]:.4f}..{poisson_ci[1]:.4f}",
        flush=True,
    )
    print(
        f"LIF    : mean={lif_mean:.4f}  std={lif_std:.4f}  "
        f"CI95={lif_ci[0]:.4f}..{lif_ci[1]:.4f}",
        flush=True,
    )
    print(f"drop   : {drop_pp:.2f} pp", flush=True)
    print(
        f"GATES: A(85%±5)={verdict_A}  B(72%±2)={verdict_B}  C(13pp±3)={verdict_C}",
        flush=True,
    )
    print(
        f"wall={wall:.1f}s  thermal_peak={thermal_peak:.1f}C  device={device}",
        flush=True,
    )


if __name__ == "__main__":
    main()
