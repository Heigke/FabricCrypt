#!/usr/bin/env python3
"""
z265 — N4b: differential-pair NS-RAM weight encoding.

Rescue test for N4 (z264). N4 used SINGLE-ENDED V_G2 mapping and failed
(22% accuracy) because the V_G2 transfer curve at V_G1=0.3 is asymmetric
around the V_G2=0.25 midpoint (positive weights -> flat dead zone,
negative weights amplified ~73x).

N4b encodes each weight as a DIFFERENTIAL PAIR of two NS-RAM cells:

    if w_ij >= 0:
        V_G2_pos = 0.50 * (1 - |w_norm|)   # high w -> low V_G2 -> high I_d
        V_G2_neg = 0.25                    # reference midpoint
    else:
        V_G2_pos = 0.25
        V_G2_neg = 0.50 * (1 - |w_norm|)
    w_eff_raw = I_d(V_G2_pos) - I_d(V_G2_neg)

V_G1=0.3, V_d=1.0, V_b=0 — LOCKED, no tuning.

After encoding, rescale globally so max(|W_eff|) == max(|W_ideal|),
then evaluate y = softmax(W_eff @ phi(x) + b) where b is the same ideal
ridge-lstsq bias as N4.

Pre-registered gates (reaffirmed before run):
  baseline_n1b           = 0.8465  (consistency check)
  baseline_n4_single     = 0.2191  (intermediate gate)
  verdict_consistency       : PASS if ideal within +/-1 pp of 0.8465
  verdict_conservative      : PASS if nsram_diff_mean >= ideal_mean - 2 pp
                              AND CI lower bound above that bar
  verdict_ambitious         : PASS if nsram_diff_mean >= ideal_mean AND
                              CI non-overlap with ideal CI
  verdict_intermediate      : PASS if nsram_diff_mean > 0.2191 AND CI lower
                              bound > 0.2191

NO-CHEAT
--------
- Encoding rule locked above.
- V_G1=0.3, V_d=1.0, V_b=0 locked.
- Bias b unchanged.
- 5 seeds.
- Linearity score logged regardless of outcome.
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

# --- Reuse N1b / N4 helpers ---------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "z261_n1b", SCRIPT_DIR / "z261_n1b_lif_mnist.py"
)
z261 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(z261)

spec_surr = importlib.util.spec_from_file_location(
    "nsram_surrogate", SCRIPT_DIR / "nsram_surrogate.py"
)
nsram_surrogate = importlib.util.module_from_spec(spec_surr)
spec_surr.loader.exec_module(nsram_surrogate)
NSRAMSurrogate = nsram_surrogate.NSRAMSurrogate
torch.set_default_dtype(torch.float32)

# --- Locked config ------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]
N_SEEDS = len(SEEDS)

VG1_OP = 0.3
VD_OP = 1.0
VG2_MID = 0.25
VG2_HI = 0.50   # high-current end (low V_G2 not used; formula gives [0,0.5])

BASELINE_N1B_POISSON = 0.8465
BASELINE_N4_SINGLE = 0.2191

VG2_BOUNDS = nsram_surrogate.VG2_RANGE

RESULTS_DIR = (
    Path(__file__).resolve().parent.parent / "results"
    / "z265_n4b_differential_weights"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --- Surrogate helper ---------------------------------------------------
def Id_at(surr: NSRAMSurrogate, vg2_arr: np.ndarray) -> np.ndarray:
    """Return linear-scale I_d at V_G1=0.3, V_d=1.0 for an array of V_G2."""
    vg1 = np.full_like(vg2_arr, VG1_OP, dtype=np.float64)
    vd  = np.full_like(vg2_arr, VD_OP, dtype=np.float64)
    vg2_clipped = np.clip(vg2_arr, VG2_BOUNDS[0], VG2_BOUNDS[1])
    log_Id = surr.eval(vg1, vg2_clipped, vd)
    return 10.0 ** log_Id


# --- Differential-pair encoding (LOCKED) --------------------------------
def encode_weights_differential(W_ideal: np.ndarray, surr: NSRAMSurrogate):
    """
    W_ideal: (N_in, n_classes) float32
    Each weight -> two NS-RAM cells; w_eff = I_d(V_G2_pos) - I_d(V_G2_neg).
    Then rescale globally so max(|W_eff|) == max(|W_ideal|).
    """
    Wmax = float(np.abs(W_ideal).max())
    if Wmax < 1e-12:
        raise RuntimeError("W_ideal is ~zero; cannot normalize")

    w_norm = W_ideal / Wmax                      # in [-1, +1]
    w_norm_clipped = np.clip(w_norm, -1.0, 1.0)
    n_norm_clip = int((np.abs(w_norm) > 1.0).sum())

    abs_wn = np.abs(w_norm_clipped)
    pos_mask = (w_norm_clipped >= 0.0)

    # V_G2_pos and V_G2_neg per locked rule
    Vg2_pos = np.where(pos_mask, 0.50 * (1.0 - abs_wn), 0.25)
    Vg2_neg = np.where(pos_mask, 0.25, 0.50 * (1.0 - abs_wn))

    Id_pos = Id_at(surr, Vg2_pos)
    Id_neg = Id_at(surr, Vg2_neg)
    w_raw = Id_pos - Id_neg

    w_raw_max = float(np.abs(w_raw).max())
    if w_raw_max < 1e-30:
        raise RuntimeError("differential dynamic range collapsed")

    W_eff = w_raw * (Wmax / w_raw_max)

    info = {
        "Wmax_ideal": Wmax,
        "Id_pos_min_A": float(Id_pos.min()),
        "Id_pos_max_A": float(Id_pos.max()),
        "Id_neg_min_A": float(Id_neg.min()),
        "Id_neg_max_A": float(Id_neg.max()),
        "w_raw_min": float(w_raw.min()),
        "w_raw_max": float(w_raw.max()),
        "w_raw_abs_max": w_raw_max,
        "W_eff_abs_max": float(np.abs(W_eff).max()),
        "n_w_norm_clipped": n_norm_clip,
        "n_weights_total": int(W_ideal.size),
        "n_positive": int(pos_mask.sum()),
        "n_negative": int((~pos_mask).sum()),
        "corr_ideal_eff": float(
            np.corrcoef(W_ideal.ravel(), W_eff.ravel())[0, 1]
        ),
        "rel_l2_error": float(
            np.linalg.norm(W_eff - W_ideal)
            / (np.linalg.norm(W_ideal) + 1e-30)
        ),
    }
    return W_eff.astype(np.float32), info


# --- Diagnostics: transfer curve + diff-pair linearity ------------------
def compute_transfer_diagnostic(surr: NSRAMSurrogate):
    vg2_points = np.linspace(0.0, 0.5, 11)
    Id_points = Id_at(surr, vg2_points)
    transfer = [
        {"vg2": float(v), "Id_A": float(i)}
        for v, i in zip(vg2_points, Id_points)
    ]

    # Differential pair sweep w_norm in {-1,-.75,...,1}
    w_norm_sweep = np.array(
        [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    )
    abs_wn = np.abs(w_norm_sweep)
    pos = (w_norm_sweep >= 0.0)
    vg2_pos = np.where(pos, 0.50 * (1.0 - abs_wn), 0.25)
    vg2_neg = np.where(pos, 0.25, 0.50 * (1.0 - abs_wn))
    Id_p = Id_at(surr, vg2_pos)
    Id_n = Id_at(surr, vg2_neg)
    w_eff_raw = Id_p - Id_n

    diff_sweep = [
        {
            "w_norm": float(wn),
            "vg2_pos": float(vp),
            "vg2_neg": float(vn),
            "Id_pos_A": float(ip),
            "Id_neg_A": float(in_),
            "w_eff_raw_A": float(we),
        }
        for wn, vp, vn, ip, in_, we in zip(
            w_norm_sweep, vg2_pos, vg2_neg, Id_p, Id_n, w_eff_raw
        )
    ]

    # Linear fit and R^2 of w_eff_raw vs w_norm
    A = np.vstack([w_norm_sweep, np.ones_like(w_norm_sweep)]).T
    slope, intercept = np.linalg.lstsq(A, w_eff_raw, rcond=None)[0]
    pred = slope * w_norm_sweep + intercept
    ss_res = float(np.sum((w_eff_raw - pred) ** 2))
    ss_tot = float(np.sum((w_eff_raw - w_eff_raw.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    max_dev = float(np.max(np.abs(w_eff_raw - pred)))

    return {
        "transfer_curve_vg2_sweep": transfer,
        "differential_pair_sweep": diff_sweep,
        "linear_fit_slope": float(slope),
        "linear_fit_intercept": float(intercept),
        "diff_pair_max_deviation_from_linear_A": max_dev,
        "diff_pair_linearity_R2": float(r2),
    }


# --- Apply readout ------------------------------------------------------
def apply_readout_split(F, W_nobias, b):
    logits = F @ W_nobias + b[None, :]
    return logits.argmax(axis=1)


# --- Per-seed runner ----------------------------------------------------
def run_seed(seed, X_train, y_train, X_test, y_test, device, surr):
    t0 = time.time()
    print(f"[seed {seed}] start  APU={z261.apu_temp_c():.1f}C", flush=True)
    torch.manual_seed(seed)

    n_in = X_train.shape[1]
    W_res, W_in = z261.build_reservoir(z261.N_NEURONS, n_in, seed, device)
    Fa_tr = z261.featurize_rate(X_train, W_res, W_in, device)
    Fa_te = z261.featurize_rate(X_test, W_res, W_in, device)
    z261.thermal_guard()

    n_classes = int(max(y_train.max(), y_test.max())) + 1
    Wro = z261.fit_ridge_readout(Fa_tr, y_train, n_classes)
    W_ideal_nb = Wro[:-1, :].astype(np.float32)
    b_ideal    = Wro[-1, :].astype(np.float32)

    yhat_ideal = apply_readout_split(Fa_te, W_ideal_nb, b_ideal)
    ideal_acc = float((yhat_ideal == y_test).mean())

    W_eff, enc_info = encode_weights_differential(W_ideal_nb, surr)
    yhat_nsram = apply_readout_split(Fa_te, W_eff, b_ideal)
    nsram_acc = float((yhat_nsram == y_test).mean())

    dt = time.time() - t0
    delta_pp = (nsram_acc - ideal_acc) * 100.0
    print(
        f"[seed {seed}] IDEAL={ideal_acc:.4f}  "
        f"NSRAM-DIFF={nsram_acc:.4f}  delta={delta_pp:+.2f}pp  "
        f"({dt:.1f}s, APU={z261.apu_temp_c():.1f}C, "
        f"corr={enc_info['corr_ideal_eff']:.3f})",
        flush=True,
    )
    return {
        "seed": seed,
        "ideal_acc": ideal_acc,
        "nsram_diff_acc": nsram_acc,
        "delta_pp": delta_pp,
        "encoding_info": enc_info,
        "wall_s": dt,
    }


def main():
    print(
        "PRE-REGISTERED GATES (reaffirmed):\n"
        f"  baseline_n1b           = {BASELINE_N1B_POISSON}\n"
        f"  baseline_n4_single     = {BASELINE_N4_SINGLE}\n"
        f"  encoding rule          = LOCKED (differential pair, V_G1=0.3, V_d=1.0)\n",
        flush=True,
    )
    print(
        f"torch {torch.__version__}  cuda_avail={torch.cuda.is_available()}",
        flush=True,
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        device = torch.device("cpu")

    print("[surrogate] loading...", flush=True)
    surr = NSRAMSurrogate.build_or_load(grid_size=(20, 20, 25))

    diag = compute_transfer_diagnostic(surr)
    print(
        f"[diag] diff_pair_linearity_R^2 = {diag['diff_pair_linearity_R2']:.4f}  "
        f"max_dev = {diag['diff_pair_max_deviation_from_linear_A']:.3e} A",
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
    nsram_accs = np.array([r["nsram_diff_acc"] for r in per_seed])

    ideal_mean = float(ideal_accs.mean())
    ideal_std  = float(ideal_accs.std(ddof=1))
    ideal_ci   = z261.bootstrap_ci(ideal_accs)

    nsram_mean = float(nsram_accs.mean())
    nsram_std  = float(nsram_accs.std(ddof=1))
    nsram_ci   = z261.bootstrap_ci(nsram_accs)

    delta_vs_ideal_in_run = nsram_mean - ideal_mean
    delta_vs_single_ended = nsram_mean - BASELINE_N4_SINGLE

    # Verdicts
    verdict_consistency = (
        "PASS" if abs(ideal_mean - BASELINE_N1B_POISSON) <= 0.01 else "FAIL"
    )
    conservative_bar = max(0.8265, ideal_mean - 0.02)
    verdict_conservative = (
        "PASS"
        if (nsram_mean >= conservative_bar and nsram_ci[0] >= conservative_bar)
        else "FAIL"
    )
    verdict_ambitious = (
        "PASS"
        if (nsram_mean >= ideal_mean and nsram_ci[0] > ideal_ci[1])
        else "FAIL"
    )
    verdict_intermediate = (
        "PASS"
        if (nsram_mean > BASELINE_N4_SINGLE and nsram_ci[0] > BASELINE_N4_SINGLE)
        else "FAIL"
    )

    wall = time.time() - t_global

    summary = {
        "experiment": "z265_n4b_differential_weights",
        "date": "2026-05-11",
        "dataset": dataset,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_classes": 10,
        "config": {
            "n_neurons_features": z261.N_NEURONS,
            "vg1_op": VG1_OP,
            "vd_op": VD_OP,
            "vg2_mid": VG2_MID,
            "encoding_rule": (
                "if w>=0: vg2_pos=0.5*(1-|w_norm|), vg2_neg=0.25; "
                "else:    vg2_pos=0.25,            vg2_neg=0.5*(1-|w_norm|). "
                "w_eff = Id(vg2_pos) - Id(vg2_neg). "
                "Global rescale so max|W_eff|=max|W_ideal|."
            ),
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
        "nsram_diff_accuracy_mean": nsram_mean,
        "nsram_diff_accuracy_std": nsram_std,
        "nsram_diff_ci95": list(nsram_ci),
        "delta_vs_ideal_in_run_pp": float(delta_vs_ideal_in_run * 100.0),
        "delta_vs_single_ended_pp": float(delta_vs_single_ended * 100.0),
        "baseline_n1b": BASELINE_N1B_POISSON,
        "baseline_n4_single": BASELINE_N4_SINGLE,
        "verdict_consistency_with_n1b": verdict_consistency,
        "verdict_conservative_vs_ideal": verdict_conservative,
        "verdict_ambitious_vs_ideal": verdict_ambitious,
        "verdict_intermediate_vs_single": verdict_intermediate,
        "conservative_bar_effective": conservative_bar,
        "transfer_diagnostic": diag,
        "diff_pair_linearity_R2": diag["diff_pair_linearity_R2"],
        "wall_s_total": wall,
        "thermal_peak_c": thermal_peak,
        "device": str(device),
        "torch_version": torch.__version__,
    }

    out = RESULTS_DIR / "summary.json"
    out.write_text(json.dumps(summary, indent=2))

    print(f"\nWROTE {out}", flush=True)
    print(
        f"IDEAL      : mean={ideal_mean:.4f}  std={ideal_std:.4f}  "
        f"CI95={ideal_ci[0]:.4f}..{ideal_ci[1]:.4f}",
        flush=True,
    )
    print(
        f"NSRAM-DIFF : mean={nsram_mean:.4f}  std={nsram_std:.4f}  "
        f"CI95={nsram_ci[0]:.4f}..{nsram_ci[1]:.4f}",
        flush=True,
    )
    print(
        f"delta vs ideal  : {delta_vs_ideal_in_run*100:+.2f} pp\n"
        f"delta vs single : {delta_vs_single_ended*100:+.2f} pp",
        flush=True,
    )
    print(
        f"diff_pair_linearity_R^2 = {diag['diff_pair_linearity_R2']:.4f}",
        flush=True,
    )
    print(
        f"VERDICTS:\n"
        f"  consistency_with_n1b      : {verdict_consistency}\n"
        f"  conservative_vs_ideal     : {verdict_conservative}\n"
        f"  ambitious_vs_ideal        : {verdict_ambitious}\n"
        f"  intermediate_vs_single    : {verdict_intermediate}",
        flush=True,
    )
    print(
        f"wall={wall:.1f}s  thermal_peak={thermal_peak:.1f}C  device={device}",
        flush=True,
    )


if __name__ == "__main__":
    main()
