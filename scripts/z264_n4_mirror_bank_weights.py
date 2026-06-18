#!/usr/bin/env python3
"""
z264 — N4: mirror-bank weight encoding test.

Tests NS-RAM in a DIFFERENT architectural role from N2/N2b: not as
input/compute neuron, but as analog WEIGHT MEMORY in the readout layer
(Sebas slide 13.33: "weight comes from the mirror bank generating the
bias voltage").

Architecture
------------
1. INPUT FEATURES: N1b Stage A Poisson rate features, 128-dim.
2. IDEAL READOUT: y = softmax(W_ideal @ phi(x) + b), W_ideal in R^{10x128},
   trained by ridge-lstsq (same code as z261).
3. NS-RAM WEIGHT ENCODING (per weight w_ij):
     w_norm  = w_ij / max(|W_ideal|)            in [-1, +1]
     V_G2_ij = 0.25 - 0.25 * w_norm             in [0, 0.5] V
     I_d_ij  = surrogate(V_G1=0.3, V_G2=V_G2_ij, V_d=1.0)  (linear scale)
     I_d_mid = surrogate(0.3, 0.25, 1.0)
     w_raw   = I_d_ij - I_d_mid
     W_eff   = w_raw * (max(|W_ideal|) / max(|w_raw|))     (rescale)
4. NS-RAM READOUT: y = softmax(W_eff @ phi(x) + b). b unchanged.
5. MNIST 28x28 test set, 5 seeds.

Pre-registered gates (locked 2026-05-11 in research_plan/01_LOG.md
BEFORE first training run):
  Baseline = N1b Poisson mean = 0.8465
  PASS CONSERVATIVE: nsram mean >= 0.8265 (within 2 pp of ideal),
                     with non-overlap CI vs PASS bar.
  PASS AMBITIOUS   : nsram mean > 0.8465 with non-overlap CI vs ideal.
  FAIL             : nsram mean < 0.8265 — V_G2-as-weight is too lossy.

NO-CHEAT
--------
- V_G2 range locked at [0, 0.5] V.
- V_G1=0.3, V_d=1.0 locked.
- Bias b not tuned (uses ridge-lstsq's solved b).
- 5 seeds, same SEEDS list as N1b.
- Surrogate OOD clipping is logged.
"""

from __future__ import annotations

import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import json
import time
import importlib.util
from pathlib import Path

import numpy as np
import torch

# --- Reuse N1b helpers ---------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "z261_n1b", SCRIPT_DIR / "z261_n1b_lif_mnist.py"
)
z261 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(z261)

# Import surrogate
spec_surr = importlib.util.spec_from_file_location(
    "nsram_surrogate", SCRIPT_DIR / "nsram_surrogate.py"
)
nsram_surrogate = importlib.util.module_from_spec(spec_surr)
spec_surr.loader.exec_module(nsram_surrogate)
NSRAMSurrogate = nsram_surrogate.NSRAMSurrogate
# nsram_surrogate sets torch float64 as default; restore float32 for N1b ops
torch.set_default_dtype(torch.float32)

# --- Locked config (NO TUNING) -----------------------------------------
SEEDS = [0, 1, 2, 3, 4]
N_SEEDS = len(SEEDS)
BOOTSTRAP_N = 1000

# Encoding constants — LOCKED
VG2_LO = 0.0
VG2_HI = 0.5
VG2_MID = 0.25                    # corresponds to w_norm = 0
VG1_OP = 0.3
VD_OP = 1.0

# Surrogate axis bounds (from nsram_surrogate.py)
VG1_BOUNDS = nsram_surrogate.VG1_RANGE     # (0.10, 0.80)
VG2_BOUNDS = nsram_surrogate.VG2_RANGE     # (-0.10, 0.60)
VD_BOUNDS  = nsram_surrogate.VD_RANGE      # (0.10, 2.20)

# N1b baseline (poisson_accuracy_mean from results/z261_n1b_lif_mnist/summary.json)
BASELINE_N1B_POISSON = 0.8465

# Gate thresholds (pre-registered)
PASS_CONSERVATIVE = 0.8265        # baseline - 2 pp
PASS_AMBITIOUS    = BASELINE_N1B_POISSON  # 0.8465

RESULTS_DIR = (
    Path(__file__).resolve().parent.parent / "results"
    / "z264_n4_mirror_bank_weights"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --- Encoding ----------------------------------------------------------
def encode_weights_via_nsram(W_ideal: np.ndarray, surr: NSRAMSurrogate):
    """
    W_ideal: (N_in, n_classes) float32  (NOTE: this is the orientation
             returned by z261.fit_ridge_readout WITHOUT the bias row.)
    Returns:
      W_eff      : same shape as W_ideal, encoded->decoded weights
      info       : dict with diagnostics
    """
    Wmax = float(np.abs(W_ideal).max())
    if Wmax < 1e-12:
        raise RuntimeError("W_ideal is ~zero; cannot normalize")

    w_norm = W_ideal / Wmax                       # in [-1, +1]
    # Clip just-in-case; ridge-lstsq output should already be in [-1, 1].
    w_norm_clipped = np.clip(w_norm, -1.0, 1.0)
    n_norm_clip = int((np.abs(w_norm) > 1.0).sum())

    VG2 = VG2_MID - 0.25 * w_norm_clipped         # in [0, 0.5]
    # Clip to surrogate bounds (V_G2 range is [-0.10, 0.60], so safe)
    VG2_clipped = np.clip(VG2, VG2_BOUNDS[0], VG2_BOUNDS[1])
    n_vg2_clip = int(((VG2 < VG2_BOUNDS[0]) | (VG2 > VG2_BOUNDS[1])).sum())

    VG1_arr = np.full_like(VG2_clipped, VG1_OP)
    Vd_arr  = np.full_like(VG2_clipped, VD_OP)

    # surrogate returns log10|I_d|; convert to linear I_d
    log_Id = surr.eval(VG1_arr, VG2_clipped, Vd_arr)
    Id = 10.0 ** log_Id                           # >0 always

    # Reference at midpoint (w=0)
    log_Id_mid = surr.eval(
        np.array(VG1_OP), np.array(VG2_MID), np.array(VD_OP)
    )
    Id_mid = float(10.0 ** log_Id_mid)

    w_raw = Id - Id_mid
    w_raw_max = float(np.abs(w_raw).max())
    if w_raw_max < 1e-30:
        raise RuntimeError(
            "Encoded weight dynamic range collapsed (max|w_raw|~0). "
            "Check V_G2 sensitivity."
        )

    W_eff = w_raw * (Wmax / w_raw_max)

    info = {
        "Wmax_ideal": Wmax,
        "Id_mid_A": Id_mid,
        "Id_min_A": float(Id.min()),
        "Id_max_A": float(Id.max()),
        "w_raw_min": float(w_raw.min()),
        "w_raw_max": float(w_raw.max()),
        "w_raw_abs_max": w_raw_max,
        "W_eff_abs_max": float(np.abs(W_eff).max()),
        "n_w_norm_clipped": n_norm_clip,
        "n_vg2_clipped": n_vg2_clip,
        "n_weights_total": int(W_ideal.size),
        "ood_clip_rate": float(
            (n_norm_clip + n_vg2_clip) / W_ideal.size
        ),
        # Approximate linearity diagnostic: corr(w_norm, w_eff_norm)
        "corr_ideal_eff": float(
            np.corrcoef(W_ideal.ravel(), W_eff.ravel())[0, 1]
        ),
        # Quantization-step diagnostic: number of unique discretized
        # values vs total (not applicable here — continuous surrogate),
        # so we report std-of-step proxy:
        "rel_l2_error": float(
            np.linalg.norm(W_eff - W_ideal) / (np.linalg.norm(W_ideal) + 1e-30)
        ),
    }
    return W_eff.astype(np.float32), info


# --- Apply readout with (W_NoBias, bias) split --------------------------
def apply_readout_split(F: np.ndarray, W_nobias: np.ndarray,
                        b: np.ndarray) -> np.ndarray:
    """
    F        : (B, N_in)
    W_nobias : (N_in, n_classes)
    b        : (n_classes,)
    Returns argmax over classes.
    """
    logits = F @ W_nobias + b[None, :]
    return logits.argmax(axis=1)


# --- Per-seed runner ----------------------------------------------------
def run_seed(seed, X_train, y_train, X_test, y_test, device, surr):
    t0 = time.time()
    print(f"[seed {seed}] start  APU={z261.apu_temp_c():.1f}C", flush=True)
    torch.manual_seed(seed)

    n_in = X_train.shape[1]
    W_res, W_in = z261.build_reservoir(z261.N_NEURONS, n_in, seed, device)

    # --- N1b Stage A features (frozen) ---
    Fa_tr = z261.featurize_rate(X_train, W_res, W_in, device)
    Fa_te = z261.featurize_rate(X_test, W_res, W_in, device)
    z261.thermal_guard()

    n_classes = int(max(y_train.max(), y_test.max())) + 1
    Wro = z261.fit_ridge_readout(Fa_tr, y_train, n_classes)
    # Wro shape = (N+1, n_classes); last row is bias
    W_ideal_nb = Wro[:-1, :].astype(np.float32)   # (N_in, n_classes)
    b_ideal    = Wro[-1, :].astype(np.float32)    # (n_classes,)

    # --- IDEAL readout (sanity vs N1b) ---
    yhat_ideal = apply_readout_split(Fa_te, W_ideal_nb, b_ideal)
    ideal_acc = float((yhat_ideal == y_test).mean())

    # --- NS-RAM weight-encoded readout ---
    W_eff, enc_info = encode_weights_via_nsram(W_ideal_nb, surr)
    yhat_nsram = apply_readout_split(Fa_te, W_eff, b_ideal)
    nsram_acc = float((yhat_nsram == y_test).mean())

    dt = time.time() - t0
    delta_pp = (nsram_acc - ideal_acc) * 100.0
    print(
        f"[seed {seed}] IDEAL test={ideal_acc:.4f}  "
        f"NSRAM-W test={nsram_acc:.4f}  delta={delta_pp:+.2f}pp  "
        f"({dt:.1f}s, APU={z261.apu_temp_c():.1f}C, "
        f"OOD={enc_info['ood_clip_rate']:.4f})",
        flush=True,
    )

    return {
        "seed": seed,
        "ideal_acc": ideal_acc,
        "nsram_weight_acc": nsram_acc,
        "delta_pp": delta_pp,
        "encoding_info": enc_info,
        "wall_s": dt,
    }


# --- Main ---------------------------------------------------------------
def main():
    print(
        f"torch {torch.__version__}  cuda_avail={torch.cuda.is_available()}",
        flush=True,
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        device = torch.device("cpu")
        print("device: CPU", flush=True)

    print("[surrogate] loading NS-RAM surrogate...", flush=True)
    surr = NSRAMSurrogate.build_or_load(grid_size=(20, 20, 25))

    # Quick midpoint diagnostic
    Id_mid = 10.0 ** float(
        surr.eval(np.array(VG1_OP), np.array(VG2_MID), np.array(VD_OP))
    )
    Id_lo  = 10.0 ** float(
        surr.eval(np.array(VG1_OP), np.array(VG2_LO), np.array(VD_OP))
    )
    Id_hi  = 10.0 ** float(
        surr.eval(np.array(VG1_OP), np.array(VG2_HI), np.array(VD_OP))
    )
    print(
        f"[surrogate] I_d at V_G1=0.3, V_d=1.0: "
        f"V_G2=0.0 -> {Id_lo:.3e} A | "
        f"V_G2=0.25 -> {Id_mid:.3e} A | "
        f"V_G2=0.5 -> {Id_hi:.3e} A",
        flush=True,
    )

    print("[loader] loading MNIST 28x28...", flush=True)
    X_train, y_train, X_test, y_test, dataset = z261.load_mnist()
    print(
        f"dataset: {dataset}  train={X_train.shape}  test={X_test.shape}",
        flush=True,
    )

    t_global = time.time()
    thermal_peak = z261.apu_temp_c()
    per_seed = []
    for s in SEEDS:
        r = run_seed(s, X_train, y_train, X_test, y_test, device, surr)
        per_seed.append(r)
        thermal_peak = max(thermal_peak, z261.apu_temp_c())
        z261.thermal_guard()

    ideal_accs = np.array([r["ideal_acc"] for r in per_seed])
    nsram_accs = np.array([r["nsram_weight_acc"] for r in per_seed])

    ideal_mean = float(ideal_accs.mean())
    ideal_std  = float(ideal_accs.std(ddof=1))
    ideal_ci   = z261.bootstrap_ci(ideal_accs)

    nsram_mean = float(nsram_accs.mean())
    nsram_std  = float(nsram_accs.std(ddof=1))
    nsram_ci   = z261.bootstrap_ci(nsram_accs)

    delta_vs_ideal_in_run = nsram_mean - ideal_mean  # architectural cost

    # Verdicts (pre-registered)
    # Consistency: ideal_mean within +/-1 pp of 0.8465
    verdict_consistency = "PASS" if (
        abs(ideal_mean - BASELINE_N1B_POISSON) <= 0.01
    ) else "FAIL"

    # Conservative: nsram_mean >= ideal_mean - 2 pp AND non-overlap CI
    # Non-overlap CI here means nsram_ci_lo > (ideal_mean - 2pp lower bound).
    # We interpret "non-overlap CI" as: lower bound of nsram CI is above
    # the conservative bar (PASS_CONSERVATIVE = 0.8265).
    conservative_bar = max(PASS_CONSERVATIVE, ideal_mean - 0.02)
    verdict_conservative = (
        "PASS" if (nsram_mean >= conservative_bar
                   and nsram_ci[0] >= conservative_bar)
        else "FAIL"
    )

    # Ambitious: nsram_mean >= ideal_mean AND non-overlap CI vs ideal
    # Non-overlap means nsram_ci[0] > ideal_ci[1].
    verdict_ambitious = (
        "PASS" if (nsram_mean >= ideal_mean and nsram_ci[0] > ideal_ci[1])
        else "FAIL"
    )

    wall = time.time() - t_global

    # Aggregate OOD clip stats
    ood_total = sum(
        r["encoding_info"]["n_w_norm_clipped"]
        + r["encoding_info"]["n_vg2_clipped"]
        for r in per_seed
    )
    n_w_total = sum(r["encoding_info"]["n_weights_total"] for r in per_seed)
    ood_rate_overall = float(ood_total / max(1, n_w_total))

    summary = {
        "experiment": "z264_n4_mirror_bank_weights",
        "date": "2026-05-11",
        "dataset": dataset,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_classes": 10,
        "config": {
            "n_neurons_features": z261.N_NEURONS,
            "vg2_lo": VG2_LO,
            "vg2_hi": VG2_HI,
            "vg2_mid": VG2_MID,
            "vg1_op": VG1_OP,
            "vd_op": VD_OP,
            "ridge_lambda": z261.RIDGE_LAMBDA,
            "readout_bias_b": "unchanged from ridge-lstsq",
            "surrogate_grid_size": list(surr.meta.get("grid_size", [])),
        },
        "n_seeds": N_SEEDS,
        "seeds": SEEDS,
        "per_seed": per_seed,
        "ideal_accuracy_mean": ideal_mean,
        "ideal_accuracy_std": ideal_std,
        "ideal_ci95": list(ideal_ci),
        "nsram_weight_accuracy_mean": nsram_mean,
        "nsram_weight_accuracy_std": nsram_std,
        "nsram_weight_ci95": list(nsram_ci),
        "delta_vs_ideal_in_run_pp":
            float(delta_vs_ideal_in_run * 100.0),
        "baseline_n1b_poisson": BASELINE_N1B_POISSON,
        "gates": {
            "pass_conservative_bar": PASS_CONSERVATIVE,
            "pass_ambitious_bar": PASS_AMBITIOUS,
        },
        "verdict_consistency_with_n1b": verdict_consistency,
        "verdict_conservative_vs_ideal": verdict_conservative,
        "verdict_ambitious_vs_ideal": verdict_ambitious,
        "conservative_bar_effective": conservative_bar,
        "ood_clip_rate_overall": ood_rate_overall,
        "wall_s_total": wall,
        "thermal_peak_c": thermal_peak,
        "device": str(device),
        "torch_version": torch.__version__,
    }

    out = RESULTS_DIR / "summary.json"
    out.write_text(json.dumps(summary, indent=2))

    print(f"\nWROTE {out}", flush=True)
    print(
        f"IDEAL : mean={ideal_mean:.4f}  std={ideal_std:.4f}  "
        f"CI95={ideal_ci[0]:.4f}..{ideal_ci[1]:.4f}",
        flush=True,
    )
    print(
        f"NSRAM : mean={nsram_mean:.4f}  std={nsram_std:.4f}  "
        f"CI95={nsram_ci[0]:.4f}..{nsram_ci[1]:.4f}",
        flush=True,
    )
    print(
        f"delta(nsram-ideal): {delta_vs_ideal_in_run*100:+.2f} pp",
        flush=True,
    )
    print(
        f"baseline N1b Poisson: {BASELINE_N1B_POISSON:.4f}",
        flush=True,
    )
    print(
        f"VERDICTS:\n"
        f"  consistency_with_n1b  (ideal within +/-1pp of 0.8465): "
        f"{verdict_consistency}\n"
        f"  conservative_vs_ideal (nsram >= max(0.8265, ideal-2pp), "
        f"non-overlap): {verdict_conservative}\n"
        f"  ambitious_vs_ideal    (nsram >= ideal, CI non-overlap):  "
        f"{verdict_ambitious}",
        flush=True,
    )
    print(
        f"OOD clip rate: {ood_rate_overall:.6f} ({ood_total}/{n_w_total})",
        flush=True,
    )
    print(
        f"wall={wall:.1f}s  thermal_peak={thermal_peak:.1f}C  device={device}",
        flush=True,
    )


if __name__ == "__main__":
    main()
