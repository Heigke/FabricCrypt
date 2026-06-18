"""V5-H5 cross-eval: A,B,C,D matrix.

A = ikaros-adapter eval on ikaros (data made on ikaros)
B = daedalus-adapter eval on daedalus
C = ikaros-adapter eval on daedalus (transplant)
D = daedalus-adapter eval on ikaros (transplant)

Since cross-eval requires the OTHER chip's noise-distorted data, we must
ship per-chip evaluation data files. To keep it simple here we use a
ZERO-NOISE evaluation surface (same task across chips). The 'transplant'
test then is essentially: same adapter, same task. Differences come from
the chip-specific TRAINING data noise that shaped the adapter.

A and B come from each chip's own train-script run.
C is: load ikaros adapter, eval on a chip-noise-augmented daedalus task (per the daedalus seed).
D is: load daedalus adapter, eval on a chip-noise-augmented ikaros task.

We approximate by running this on ikaros only with both adapter files,
using ikaros and daedalus signature data to construct each chip's noise.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from v5_h5_lora import LoRAHead, make_data, N_FEAT, N_HID, RANK
ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT_DIR = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment4"
SIGS = OUT_DIR / "signatures"


def load_adapter(path: Path) -> LoRAHead:
    d = np.load(path)
    h = LoRAHead.__new__(LoRAHead)
    h.A = d["A"]; h.B = d["B"]; h.W2 = d["W2"]
    h.b1 = d["b1"]; h.b2 = d["b2"]
    return h


def sig_to_noise(sig_path: Path, n_samples: int) -> np.ndarray:
    """Use sig dynamic bins to derive deterministic noise vector for that chip."""
    sig = json.loads(sig_path.read_text())
    # collect median values of per_signal sensors
    per = sig.get("per_signal", {})
    vals = []
    for k in sorted(per.keys()):
        arr = [v for v in per[k] if v == v]  # not nan
        if arr:
            vals.append(float(np.median(arr)))
    arr = np.array(vals[:n_samples], dtype=np.float32) if vals else np.zeros(n_samples, dtype=np.float32)
    # normalize to small noise
    if arr.std() > 0:
        arr = (arr - arr.mean()) / (arr.std() + 1e-9) * 0.05
    if len(arr) < n_samples:
        arr = np.concatenate([arr, np.zeros(n_samples - len(arr), dtype=np.float32)])
    return arr


def make_chip_data(seed: int, chip_noise_vec: np.ndarray):
    rng = np.random.default_rng(seed)
    N_TRAIN = 800
    X = rng.standard_normal((N_TRAIN, N_FEAT)).astype(np.float32)
    w_true = rng.standard_normal((N_FEAT, 1)).astype(np.float32)
    y = np.tanh(X @ w_true).squeeze()
    for i in range(N_TRAIN):
        X[i] += np.float32(chip_noise_vec[i % len(chip_noise_vec)])
    Xte = rng.standard_normal((200, N_FEAT)).astype(np.float32)
    yte = np.tanh(Xte @ w_true).squeeze()
    return X, y, Xte, yte


def main():
    ikaros_adapter = load_adapter(OUT_DIR / "v5_h5_ikaros_adapter.npz")
    daedalus_adapter = load_adapter(OUT_DIR / "v5_h5_daedalus_adapter.npz")
    # Use ikaros_v2a_t0 + daedalus_prereboot as chip-noise sources
    ik_noise = sig_to_noise(ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment3/signatures/ikaros_v2a_t0.json", 800)
    da_noise = sig_to_noise(SIGS / "daedalus_prereboot.json", 800)

    seeds = [1, 2, 3]
    mses = {"A_ik_on_ik": [], "B_da_on_da": [], "C_ik_on_da": [], "D_da_on_ik": []}
    for s in seeds:
        _, _, Xte_ik, yte_ik = make_chip_data(s, ik_noise)
        _, _, Xte_da, yte_da = make_chip_data(s, da_noise)
        yhat, _, _ = ikaros_adapter.forward(Xte_ik); mses["A_ik_on_ik"].append(float(np.mean((yhat - yte_ik)**2)))
        yhat, _, _ = daedalus_adapter.forward(Xte_da); mses["B_da_on_da"].append(float(np.mean((yhat - yte_da)**2)))
        yhat, _, _ = ikaros_adapter.forward(Xte_da); mses["C_ik_on_da"].append(float(np.mean((yhat - yte_da)**2)))
        yhat, _, _ = daedalus_adapter.forward(Xte_ik); mses["D_da_on_ik"].append(float(np.mean((yhat - yte_ik)**2)))

    summary = {k: float(np.median(v)) for k, v in mses.items()}
    print(f"[H5 cross] A(ik→ik)={summary['A_ik_on_ik']:.5f}  B(da→da)={summary['B_da_on_da']:.5f}", flush=True)
    print(f"[H5 cross] C(ik→da)={summary['C_ik_on_da']:.5f}  D(da→ik)={summary['D_da_on_ik']:.5f}", flush=True)
    # Gates: A < C by ≥5% AND B < D by ≥5% (own-chip wins)
    gain_A_vs_C = 100.0 * (summary["C_ik_on_da"] - summary["A_ik_on_ik"]) / max(1e-9, summary["C_ik_on_da"])
    gain_B_vs_D = 100.0 * (summary["D_da_on_ik"] - summary["B_da_on_da"]) / max(1e-9, summary["D_da_on_ik"])
    res = {"per_seed": mses, "summary": summary,
            "gain_A_vs_C_pct": gain_A_vs_C, "gain_B_vs_D_pct": gain_B_vs_D,
            "WIN": gain_A_vs_C >= 5.0 and gain_B_vs_D >= 5.0}
    print(f"[H5 cross] gain(A vs C)={gain_A_vs_C:+.2f}% gain(B vs D)={gain_B_vs_D:+.2f}% WIN={res['WIN']}", flush=True)
    (OUT_DIR / "v5_h5_crosseval.json").write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
