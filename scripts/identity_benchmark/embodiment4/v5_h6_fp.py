"""V5-H6: Chip-calibrated FP16 model vs generic FP16.

Hypothesis:
  Per-chip subnormal handling, rounding-mode quirks, and CPU FMA ordering
  differ. A model whose weights are quantized/rounded to match THIS chip's
  exact FP16 behaviour should achieve marginally higher accuracy at FP16
  than a generic model rounded chip-agnostically.

Method:
  - Train a small float32 regression model on a fixed task.
  - "Generic FP16": cast weights to fp16 using numpy default rounding.
  - "Chip-calibrated FP16": cast → compute residual using THIS CPU's actual
    matmul (which includes its FMA order / SIMD path) → train a small
    residual correction layer in fp16 to compensate.
  - Compare test accuracy.

Gate: chip-calibrated FP16 model gives ≥1% higher accuracy than generic.
"""
from __future__ import annotations
import json, time, socket
from pathlib import Path
import numpy as np

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
HOST = socket.gethostname()
OUT = ROOT / f"results/IDENTITY_BENCHMARK_2026-05-30/embodiment4/v5_h6_{HOST}.json"

N_FEAT = 32
N_HID = 64
N_TRAIN = 1000
N_TEST = 300
N_EPOCHS = 80


def make_task(seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N_TRAIN + N_TEST, N_FEAT)).astype(np.float32)
    Wt = rng.standard_normal((N_FEAT, N_HID)).astype(np.float32)
    Wt2 = rng.standard_normal((N_HID, 1)).astype(np.float32)
    y = np.tanh(X @ Wt) @ Wt2
    y = y.squeeze()
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]


def train_fp32(seed):
    Xtr, ytr, Xte, yte = make_task(seed)
    rng = np.random.default_rng(seed * 7 + 1)
    W1 = (rng.standard_normal((N_FEAT, N_HID)) * np.sqrt(2.0 / N_FEAT)).astype(np.float32)
    b1 = np.zeros(N_HID, dtype=np.float32)
    W2 = (rng.standard_normal((N_HID, 1)) * np.sqrt(2.0 / N_HID)).astype(np.float32)
    b2 = np.zeros(1, dtype=np.float32)
    for ep in range(N_EPOCHS):
        idx = np.random.permutation(N_TRAIN)
        for i in range(0, N_TRAIN, 64):
            b = idx[i:i+64]
            X, y = Xtr[b], ytr[b]
            z1 = X @ W1 + b1; h = np.tanh(z1)
            z2 = h @ W2 + b2; yhat = z2.squeeze()
            d2 = (2.0 * (yhat - y) / len(b)).astype(np.float32)[:, None]
            gW2 = h.T @ d2; gb2 = d2.sum(0)
            dh = (d2 @ W2.T) * (1 - h*h)
            gW1 = X.T @ dh; gb1 = dh.sum(0)
            W1 -= 0.01 * gW1; b1 -= 0.01 * gb1
            W2 -= 0.01 * gW2; b2 -= 0.01 * gb2
    z1 = Xte @ W1 + b1; h = np.tanh(z1)
    z2 = h @ W2 + b2; yhat = z2.squeeze()
    mse_fp32 = float(np.mean((yhat - yte)**2))
    return (W1, b1, W2, b2), (Xtr, ytr, Xte, yte), mse_fp32


def eval_fp16(weights, data):
    W1, b1, W2, b2 = weights
    Xtr, ytr, Xte, yte = data
    Wf1 = W1.astype(np.float16); bf1 = b1.astype(np.float16)
    Wf2 = W2.astype(np.float16); bf2 = b2.astype(np.float16)
    Xf = Xte.astype(np.float16)
    z1 = (Xf @ Wf1 + bf1).astype(np.float16)
    h = np.tanh(z1.astype(np.float32)).astype(np.float16)
    z2 = (h @ Wf2 + bf2).astype(np.float16)
    yhat = z2.astype(np.float32).squeeze()
    return float(np.mean((yhat - yte)**2)), (Wf1, bf1, Wf2, bf2)


def calibrate_fp16(generic_w16, weights_fp32, data):
    """Train a tiny residual to compensate for FP16 round-off on THIS CPU.

    Idea: predict residual = (fp32_pred - fp16_pred) from h_fp16,
    using a fp16 linear correction. The correction's quality depends on
    chip-specific FP16 sequence (np matmul uses BLAS, BLAS uses chip SIMD).
    """
    W1, b1, W2, b2 = weights_fp32
    Wf1, bf1, Wf2, bf2 = generic_w16
    Xtr, ytr, _, _ = data
    Xtr_f = Xtr.astype(np.float16)
    # Compute fp16 hidden using THIS CPU's matmul
    z1f = (Xtr_f @ Wf1 + bf1)
    hf = np.tanh(z1f.astype(np.float32)).astype(np.float16)
    z2f = (hf @ Wf2 + bf2).astype(np.float32).squeeze()
    # fp32 hidden
    z1 = Xtr @ W1 + b1; h = np.tanh(z1)
    z2 = (h @ W2 + b2).squeeze()
    residual = (z2 - z2f).astype(np.float32)
    # train a 1-layer linear correction: y_calib = z2f + c * (chip_residual_signal),
    # where chip_residual_signal = (h_fp16 - h_fp32) summed → scalar per sample
    chip_signal = (hf.astype(np.float32) - h).sum(axis=1)
    # least squares for coefficient c so residual ≈ c * chip_signal
    denom = float(np.sum(chip_signal**2)) + 1e-9
    c = float(np.sum(chip_signal * residual) / denom)
    return c


def eval_calibrated_fp16(weights_fp32, w16, data, c):
    W1, b1, W2, b2 = weights_fp32
    Wf1, bf1, Wf2, bf2 = w16
    _, _, Xte, yte = data
    Xf = Xte.astype(np.float16)
    z1f = (Xf @ Wf1 + bf1)
    hf = np.tanh(z1f.astype(np.float32)).astype(np.float16)
    z2f = (hf @ Wf2 + bf2).astype(np.float32).squeeze()
    z1 = Xte @ W1 + b1; h = np.tanh(z1)
    chip_signal = (hf.astype(np.float32) - h).sum(axis=1)
    yhat_cal = z2f + c * chip_signal
    mse = float(np.mean((yhat_cal - yte) ** 2))
    return mse


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seeds = [1, 2, 3, 4, 5]
    mses = []
    for s in seeds:
        wf, data, mse_fp32 = train_fp32(s)
        mse_fp16_gen, w16 = eval_fp16(wf, data)
        c = calibrate_fp16(w16, wf, data)
        mse_fp16_cal = eval_calibrated_fp16(wf, w16, data, c)
        print(f"[H6] {HOST} seed={s} fp32={mse_fp32:.6f} fp16_generic={mse_fp16_gen:.6f} fp16_calibrated={mse_fp16_cal:.6f} c={c:+.4f}", flush=True)
        mses.append({"seed": s, "fp32": mse_fp32, "fp16_generic": mse_fp16_gen,
                      "fp16_calibrated": mse_fp16_cal, "c": c})
    arr_gen = np.array([m["fp16_generic"] for m in mses])
    arr_cal = np.array([m["fp16_calibrated"] for m in mses])
    # Lower MSE → higher accuracy. "1% higher accuracy" → mse_cal ≤ 0.99 × mse_gen
    res = {"host": HOST, "per_seed": mses,
            "generic_fp16_mse_med": float(np.median(arr_gen)),
            "calibrated_fp16_mse_med": float(np.median(arr_cal)),
            "accuracy_improvement_pct": 100.0 * (float(np.median(arr_gen)) - float(np.median(arr_cal))) / float(np.median(arr_gen))}
    res["WIN"] = res["accuracy_improvement_pct"] >= 1.0
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"[H6] gen={res['generic_fp16_mse_med']:.6f} cal={res['calibrated_fp16_mse_med']:.6f} gain={res['accuracy_improvement_pct']:+.2f}% WIN={res['WIN']}", flush=True)


if __name__ == "__main__":
    main()
