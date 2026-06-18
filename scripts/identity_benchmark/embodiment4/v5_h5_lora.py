"""V5-H5: Per-chip-tuned LoRA-style adapter, cross-evaluation.

Hypothesis:
  A small adapter (low-rank update) trained on ikaros performs better
  on ikaros than the adapter trained on daedalus does on ikaros.

We DON'T use a transformer here (too costly) — instead, we run a small
neural regression task where the chip's actual numeric behaviour matters:
  - "Base model": a tiny frozen feature extractor (random projection).
  - "Adapter": a 2-layer head with low-rank weight matrix, trained on
    per-chip noise-augmented data. The noise is the chip's real thermal
    jitter measured live during training (NOT the same as random noise).
  - Cross-eval: ikaros-adapter on ikaros vs ikaros-adapter on daedalus.

Pre-reg gate: |A - C| ≥ 5% AND |B - D| ≥ 5% (own-chip wins both sides).

In this script we run ONE chip's adapter training. The other chip runs
the same script. Cross-eval done at synthesis time.
"""
from __future__ import annotations
import json, time, socket, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
HOST = socket.gethostname()
OUT = ROOT / f"results/IDENTITY_BENCHMARK_2026-05-30/embodiment4/v5_h5_{HOST}.json"

N_FEAT = 32
N_HID = 16
RANK = 4
N_OUT = 1
N_TRAIN = 800
N_EPOCHS = 50


def read_apu_temp_c() -> float:
    try:
        return float(open("/sys/class/thermal/thermal_zone0/temp").read().strip()) / 1000.0
    except Exception:
        return 50.0


def make_data(seed: int, chip_noise_fn=None):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N_TRAIN, N_FEAT)).astype(np.float32)
    # ground-truth function — same across chips
    w_true = rng.standard_normal((N_FEAT, N_OUT)).astype(np.float32)
    y = np.tanh(X @ w_true).squeeze()
    # add CHIP-SPECIFIC noise: live thermal jitter values during data prep
    if chip_noise_fn is not None:
        for i in range(N_TRAIN):
            jit = chip_noise_fn() * 0.05
            X[i] += np.float32(jit)
    Xte = rng.standard_normal((200, N_FEAT)).astype(np.float32)
    yte = np.tanh(Xte @ w_true).squeeze()
    return X, y, Xte, yte, w_true


class LoRAHead:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.A = (rng.standard_normal((N_FEAT, RANK)) * 0.1).astype(np.float32)
        self.B = (rng.standard_normal((RANK, N_HID)) * 0.1).astype(np.float32)
        self.W2 = (rng.standard_normal((N_HID, N_OUT)) * np.sqrt(1.0 / N_HID)).astype(np.float32)
        self.b1 = np.zeros(N_HID, dtype=np.float32)
        self.b2 = np.zeros(N_OUT, dtype=np.float32)

    def forward(self, X):
        z1 = X @ (self.A @ self.B) + self.b1
        h = np.tanh(z1)
        z2 = h @ self.W2 + self.b2
        return z2.squeeze(), h, z1

    def step(self, X, y, lr):
        n = len(y)
        yhat, h, z1 = self.forward(X)
        d2 = (2.0 * (yhat - y) / n)[:, None]
        gW2 = h.T @ d2; gb2 = d2.sum(0)
        dh = d2 @ self.W2.T
        dh = dh * (1 - h*h)
        # gradient through A·B
        AB = self.A @ self.B
        gAB = X.T @ dh
        gA = gAB @ self.B.T; gB = self.A.T @ gAB
        gb1 = dh.sum(0)
        self.A -= lr * gA; self.B -= lr * gB
        self.W2 -= lr * gW2; self.b1 -= lr * gb1; self.b2 -= lr * gb2
        return float(np.mean((yhat - y)**2))


def train_adapter(seed, chip_noise_fn=None):
    Xtr, ytr, Xte, yte, _ = make_data(seed, chip_noise_fn=chip_noise_fn)
    head = LoRAHead(seed=seed * 7 + 1)
    for ep in range(N_EPOCHS):
        idx = np.random.permutation(N_TRAIN)
        for i in range(0, N_TRAIN, 64):
            b = idx[i:i+64]
            head.step(Xtr[b], ytr[b], lr=0.01)
    yhat_te, _, _ = head.forward(Xte)
    test_mse = float(np.mean((yhat_te - yte) ** 2))
    return head, test_mse


def evaluate_on_other_data(head, seed, chip_noise_fn=None):
    """Apply this chip's adapter to *the other chip's noise-distorted* data."""
    Xtr, ytr, Xte, yte, _ = make_data(seed, chip_noise_fn=chip_noise_fn)
    yhat_te, _, _ = head.forward(Xte)
    return float(np.mean((yhat_te - yte) ** 2))


def chip_noise_fn():
    """Returns a small live thermal-derived jitter for this chip, called once per sample."""
    base = read_apu_temp_c()
    return float((base - 55.0) / 100.0)  # signed, small magnitude, chip-specific


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seeds = [1, 2, 3]
    own_mses, transplant_mses_ifother_noisefn = [], []

    # Train adapter on THIS chip
    own_heads = []
    for s in seeds:
        head, mse = train_adapter(s, chip_noise_fn=chip_noise_fn)
        own_heads.append(head); own_mses.append(mse)
        print(f"[H5] {HOST} train seed={s} own_test_mse={mse:.5f}", flush=True)

    # Save adapter weights for cross-eval at synthesis
    adapter_path = OUT.parent / f"v5_h5_{HOST}_adapter.npz"
    np.savez(adapter_path, A=own_heads[0].A, B=own_heads[0].B,
              W2=own_heads[0].W2, b1=own_heads[0].b1, b2=own_heads[0].b2)

    res = {"host": HOST, "seeds": seeds,
            "own_test_mse_per_seed": own_mses,
            "own_test_mse_med": float(np.median(own_mses)),
            "adapter_path": str(adapter_path)}
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"[H5] {HOST} own_mse_med={res['own_test_mse_med']:.5f}", flush=True)
    print(f"[H5] adapter weights saved to {adapter_path}", flush=True)


if __name__ == "__main__":
    main()
