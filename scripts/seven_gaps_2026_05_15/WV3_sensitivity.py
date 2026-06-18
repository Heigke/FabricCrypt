"""Track 7 / WV3: Synthetic sigma_Vth0 sensitivity sweep.

Since WV1 confirms only a single physical device (2vHCa-2) is in our data,
we test the application's robustness to inter-device Vth0 variation by
synthetically injecting Gaussian Vth0 shifts and re-running a simplified
DS-N10 sine-class task.

DS-N10 sine class is a memory-time-series classification benchmark
(see scripts/DS_N10_reservoir.py). We use a fast proxy: a reservoir of
N=64 NS-RAM-style nonlinear neurons, each with a per-cell Vth0 shift
drawn from N(0, sigma). Input: 3 sine classes at f in {2, 5, 10} Hz.
Output: linear-readout classification accuracy.

Sweep sigma in {10, 25, 50, 100} mV; identify breaking point (acc < 90%).

Output: results/WV3_sensitivity/{accuracy_vs_sigma.png, summary.json}
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "results" / "WV3_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

# Simplified NS-RAM-style nonlinearity:
#   v_out = tanh((V_in - Vth_eff) / VT) + sub_kink((V_in - Vth_eff))
# Vth_eff = Vth0_nominal + dVth_per_cell

VT_KT = 0.026
VTH0_NOMINAL = 0.35


def cell_response(v_in: np.ndarray, vth0: float, gain: float) -> np.ndarray:
    """Smooth NS-RAM-ish: subthreshold + saturation + small nonlinear kink."""
    x = (v_in - vth0) / VT_KT
    return gain * (np.tanh(x) + 0.5 * np.tanh(x - 2) + 0.2 * np.tanh(3 * x))


def gen_inputs(T: int, n_classes: int, fs: float, freqs: list[float], seed: int):
    rng = np.random.default_rng(seed)
    n_per = T // n_classes
    X = np.zeros(T, dtype=np.float32)
    y = np.zeros(T, dtype=np.int32)
    for c in range(n_classes):
        t = np.arange(n_per) / fs
        amp = 0.3
        phase = rng.uniform(0, 2 * np.pi)
        sig = amp * np.sin(2 * np.pi * freqs[c] * t + phase) + 0.4
        X[c * n_per:(c + 1) * n_per] = sig
        y[c * n_per:(c + 1) * n_per] = c
    # shuffle in windows of W
    W = 100
    n_chunks = T // W
    idx = np.arange(n_chunks)
    rng.shuffle(idx)
    X2 = np.concatenate([X[i * W:(i + 1) * W] for i in idx])
    y2 = np.concatenate([y[i * W:(i + 1) * W] for i in idx])
    return X2, y2


def reservoir_features(X: np.ndarray, vth_offsets: np.ndarray, gains: np.ndarray) -> np.ndarray:
    # shape (T, N)
    feats = np.stack([cell_response(X, VTH0_NOMINAL + dv, g) for dv, g in zip(vth_offsets, gains)], axis=1)
    # add temporal moving average (rectified) as memory
    win = 50
    kernel = np.ones(win) / win
    mems = np.stack([np.convolve(np.abs(feats[:, i]), kernel, mode="same") for i in range(feats.shape[1])], axis=1)
    return np.concatenate([feats, mems], axis=1)


def train_eval(feats: np.ndarray, y: np.ndarray) -> float:
    # Linear (ridge) readout
    n_train = int(0.7 * len(y))
    Xtr, Xte = feats[:n_train], feats[n_train:]
    ytr, yte = y[:n_train], y[n_train:]
    # one-hot
    n_cls = int(y.max() + 1)
    Ytr = np.eye(n_cls)[ytr]
    lam = 1.0
    W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ Ytr)
    pred = np.argmax(Xte @ W, axis=1)
    return float((pred == yte).mean())


def main() -> None:
    fs = 200.0
    T = 6000
    freqs = [2.0, 5.0, 10.0]
    N_CELLS = 64
    sigmas_mV = [0, 10, 25, 50, 100]
    n_reps = 5
    results = {}
    means, stds = [], []
    for sigma in sigmas_mV:
        accs = []
        for rep in range(n_reps):
            rng = np.random.default_rng(1000 * sigma + rep)
            dv = rng.normal(0.0, sigma * 1e-3, size=N_CELLS).astype(np.float32)
            gains = rng.uniform(0.7, 1.3, size=N_CELLS).astype(np.float32)
            X, y = gen_inputs(T, len(freqs), fs, freqs, seed=42 + rep)
            feats = reservoir_features(X, dv, gains)
            acc = train_eval(feats, y)
            accs.append(acc)
        accs = np.array(accs)
        results[f"sigma_{sigma}mV"] = {
            "sigma_mV": sigma,
            "acc_mean": float(accs.mean()),
            "acc_std": float(accs.std()),
            "accs": accs.tolist(),
        }
        means.append(accs.mean())
        stds.append(accs.std())
    means = np.array(means)
    stds = np.array(stds)

    # breaking point: first sigma where mean acc < 0.90
    bp = None
    for s, m in zip(sigmas_mV, means):
        if m < 0.90:
            bp = s
            break

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(sigmas_mV, means, yerr=stds, marker="o", capsize=4)
    ax.axhline(0.90, color="r", linestyle="--", label="90% acc threshold")
    if bp is not None:
        ax.axvline(bp, color="orange", linestyle=":", label=f"breaking point ~{bp} mV")
    ax.set_xlabel("sigma_Vth0 [mV]")
    ax.set_ylabel("DS-N10 proxy accuracy (3-class sine)")
    ax.set_title("WV3: Inter-device Vth0 variation sensitivity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_vs_sigma.png", dpi=110)
    plt.close(fig)

    summary = {
        "gate_robust_to_25mV": (results["sigma_25mV"]["acc_mean"] >= 0.90),
        "breaking_point_mV": bp,
        "n_reps": n_reps,
        "results": results,
        "plot": str(OUT / "accuracy_vs_sigma.png"),
        "note": "Proxy task; full DS-N10 reservoir requires GPU + full NSRAMCell2T forward",
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
