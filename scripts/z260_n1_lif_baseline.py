#!/usr/bin/env python3
"""
z260 — N1: LIF SNN baseline reproduction.

Goal: reproduce Sebas's slide 12.33/12.39 LIF=72% result with a plain
PyTorch LIF (NO NS-RAM). 5 seeds, lstsq readout, Poisson rate-encoded
input.

Pre-registered gate (locked in research_plan/01_LOG.md, 2026-05-11):
  PASS = 5-seed mean accuracy ∈ [70%, 74%]
  FAIL = anywhere outside [70%, 74%].

NO-CHEAT: tau / V_TH / N fixed by task spec, no post-hoc tuning.

GPU: AMD Radeon 8060S via ROCm with HSA_OVERRIDE_GFX_VERSION=11.0.0.
Concurrency: M3 BBO is running on CPU; we stay on GPU and only touch
APU thermal monitor on disk.
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
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# Config (locked by spec — DO NOT tune to hit 72%)
# ------------------------------------------------------------------
N_NEURONS = 128
TAU = 10.0
V_THRESH = 1.0
V_RESET = 0.0
DT = 1.0
T_STEPS = 100        # Poisson rate-encode each pixel over T steps
N_SEEDS = 5
SEEDS = [0, 1, 2, 3, 4]
BOOTSTRAP_N = 1000
RESERVOIR_GAIN = 0.3        # recurrent connectivity scale (matches PLIFNetwork)
W_IN_SCALE = 0.3            # input projection scale (matches PLIFNetwork)
SPECTRAL_RADIUS = 0.9
SPARSITY = 0.1
N_EXC_FRAC = 0.8
SYN_DECAY = 0.8

THERMAL_ZONE = Path("/sys/class/thermal/thermal_zone0/temp")
THERMAL_PAUSE_C = 75.0
THERMAL_RESUME_C = 50.0
THERMAL_ABORT_C = 90.0

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "z260_n1_lif_baseline"
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
        print(f"  [thermal] APU={t:.1f}C — pausing until <{THERMAL_RESUME_C}C", flush=True)
        while apu_temp_c() > THERMAL_RESUME_C:
            time.sleep(5)
        print(f"  [thermal] resumed at APU={apu_temp_c():.1f}C", flush=True)
    return t


# ------------------------------------------------------------------
# LIF reservoir (fixed tau; no learning of recurrent weights)
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
    return (torch.from_numpy(W).to(device),
            torch.from_numpy(W_in).to(device))


@torch.no_grad()
def run_lif_batch(spikes_in, W, W_in, device):
    """
    spikes_in: (B, T, n_in) float32 binary spikes (Poisson encoded pixels)
    Returns mean spike rate per neuron: (B, N)
    """
    B, T, n_in = spikes_in.shape
    N = W.shape[0]
    alpha = float(np.exp(-DT / TAU))

    v = torch.zeros(B, N, device=device)
    syn = torch.zeros(B, N, device=device)
    spike_sum = torch.zeros(B, N, device=device)

    Wt = W.t()            # (N, N) but used as syn @ W.T equivalent
    Win_t = W_in.t()      # (n_in, N): spikes_in[:,t] @ Win_t -> (B, N)

    for t in range(T):
        I = spikes_in[:, t] @ Win_t + (syn @ Wt) * RESERVOIR_GAIN
        v = alpha * v + (1 - alpha) * I
        fired = (v >= V_THRESH).float()
        spike_sum += fired
        # reset
        v = torch.where(fired.bool(), torch.full_like(v, V_RESET), v)
        syn = SYN_DECAY * syn + fired
    return spike_sum / T   # mean spike rate per neuron


def poisson_encode(X: np.ndarray, T: int, rng: np.random.RandomState) -> np.ndarray:
    """
    X: (B, n_in) intensities in [0,1]
    Returns: (B, T, n_in) binary spikes with per-step prob = X
    """
    B, n_in = X.shape
    # broadcast (T, B, n_in) then permute to (B, T, n_in)
    p = X[None, :, :].repeat(T, axis=0)  # (T, B, n_in)
    u = rng.rand(T, B, n_in).astype(np.float32)
    spikes = (u < p).astype(np.float32)
    return spikes.transpose(1, 0, 2)


# ------------------------------------------------------------------
# Single-seed run
# ------------------------------------------------------------------
def run_seed(seed: int, X_train, y_train, X_test, y_test, device) -> dict:
    print(f"[seed {seed}] start  APU={apu_temp_c():.1f}C", flush=True)
    t0 = time.time()
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    n_in = X_train.shape[1]
    W, W_in = build_reservoir(N_NEURONS, n_in, seed, device)

    # Encode train/test
    spikes_train = poisson_encode(X_train, T_STEPS, rng)
    spikes_test = poisson_encode(X_test, T_STEPS, rng)

    # Run LIF in mini-batches to avoid GPU OOM
    BATCH = 256

    def featurize(spikes_np):
        out = np.zeros((spikes_np.shape[0], N_NEURONS), dtype=np.float32)
        for i in range(0, spikes_np.shape[0], BATCH):
            chunk = torch.from_numpy(spikes_np[i:i + BATCH]).to(device)
            feats = run_lif_batch(chunk, W, W_in, device)
            out[i:i + BATCH] = feats.detach().cpu().numpy()
            thermal_guard()
        return out

    F_train = featurize(spikes_train)
    F_test = featurize(spikes_test)

    # One-hot targets + lstsq readout (closed-form, no backprop)
    n_classes = int(max(y_train.max(), y_test.max())) + 1
    Y = np.zeros((len(y_train), n_classes), dtype=np.float32)
    Y[np.arange(len(y_train)), y_train] = 1.0

    # bias column
    F_train_b = np.hstack([F_train, np.ones((len(F_train), 1), dtype=np.float32)])
    F_test_b = np.hstack([F_test, np.ones((len(F_test), 1), dtype=np.float32)])

    # ridge for numerical stability (lambda small — does not move accuracy meaningfully)
    lam = 1e-3
    A = F_train_b.T @ F_train_b + lam * np.eye(F_train_b.shape[1], dtype=np.float32)
    b = F_train_b.T @ Y
    Wro = np.linalg.solve(A, b)  # (N+1, n_classes)

    train_pred = (F_train_b @ Wro).argmax(axis=1)
    test_pred = (F_test_b @ Wro).argmax(axis=1)
    train_acc = float((train_pred == y_train).mean())
    test_acc = float((test_pred == y_test).mean())
    dt_s = time.time() - t0
    print(f"[seed {seed}] train={train_acc:.4f}  test={test_acc:.4f}  "
          f"({dt_s:.1f}s, APU={apu_temp_c():.1f}C)", flush=True)
    return {
        "seed": seed,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "wall_s": dt_s,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print(f"torch {torch.__version__}  cuda_avail={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        device = torch.device("cpu")
        print("device: CPU (ROCm unavailable)", flush=True)

    # Dataset — sklearn digits 8x8 (1797 samples). Faster than MNIST, fixed by task spec.
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0  # normalize to [0,1]
    y = digits.target.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    dataset_name = "sklearn_digits_8x8"
    print(f"dataset: {dataset_name}  train={X_train.shape}  test={X_test.shape}",
          flush=True)

    t_global = time.time()
    thermal_peak = apu_temp_c()
    per_seed = []
    for s in SEEDS:
        r = run_seed(s, X_train, y_train, X_test, y_test, device)
        per_seed.append(r)
        thermal_peak = max(thermal_peak, apu_temp_c())
        thermal_guard()

    accs = np.array([r["test_acc"] for r in per_seed])
    mean = float(accs.mean())
    std = float(accs.std(ddof=1))

    # bootstrap CI on the mean over seeds
    rng = np.random.RandomState(12345)
    boots = np.array([
        rng.choice(accs, size=len(accs), replace=True).mean()
        for _ in range(BOOTSTRAP_N)
    ])
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

    verdict = "PASS" if (0.70 <= mean <= 0.74) else "FAIL"
    fail_reason = None
    if verdict == "FAIL":
        if mean > 0.74:
            fail_reason = f"accuracy {mean:.4f} above gate upper bound 0.74 — setup mismatch (easier task, larger feature space, or stronger readout than Sebas's Brian2)"
        else:
            fail_reason = f"accuracy {mean:.4f} below gate lower bound 0.70 — encoding or LIF dynamics underperforming"

    wall = time.time() - t_global
    summary = {
        "experiment": "z260_n1_lif_baseline",
        "date": "2026-05-11",
        "dataset": dataset_name,
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
            "readout": "ridge_lstsq",
            "ridge_lambda": 1e-3,
        },
        "n_seeds": N_SEEDS,
        "seeds": SEEDS,
        "per_seed": per_seed,
        "accuracy_mean": mean,
        "accuracy_std": std,
        "accuracy_ci95": list(ci),
        "gate": {
            "lower": 0.70,
            "upper": 0.74,
            "target": 0.72,
            "source": "Sebas slide 12.33/12.39 LIF=72%",
        },
        "verdict": verdict,
        "fail_reason": fail_reason,
        "wall_s_total": wall,
        "thermal_peak_c": thermal_peak,
        "device": str(device),
        "torch_version": torch.__version__,
    }

    out = RESULTS_DIR / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWROTE {out}", flush=True)
    print(f"VERDICT: {verdict}  mean={mean:.4f}  std={std:.4f}  "
          f"CI95={ci[0]:.4f}..{ci[1]:.4f}  wall={wall:.1f}s  "
          f"thermal_peak={thermal_peak:.1f}C", flush=True)
    if fail_reason:
        print(f"FAIL_REASON: {fail_reason}", flush=True)


if __name__ == "__main__":
    main()
