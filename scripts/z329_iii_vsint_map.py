"""z329 — R-12b: empirical map of Iii_M1(Vsint) for all 33 Sebas IV biases.

Independent of R-12 solver-fix. Tells us what Vsint range gives M1 ON
and Iii non-zero — i.e. where the solver should converge.

For each (VG1, VG2) bias (33 total), V_d = 2.0V (standard Sebas OP):
  Sweep Vsint in [0, V_d] over 50 steps. At each Vsint, evaluate M1's Iii
  FORCED (not solver-converged). Record Iii, Ids, Vdseff.

Gates:
  INFRA: >=30/33 biases have non-trivial Vsint_transition (Iii > 1e-25 at some Vsint)
  PASS:  median Vsint_transition < 0.7 * V_d

Outputs:
  results/z329_iii_vsint_map/summary.json
  results/z329_iii_vsint_map/iii_heatmap.png
  results/z329_iii_vsint_map/vsint_transition_hist.png
"""
from __future__ import annotations
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import importlib.util
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
SCRIPTS = ROOT / "scripts"
OUT_DIR = ROOT / "results/z329_iii_vsint_map"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY = OUT_DIR / "summary.json"
HEATMAP_PNG = OUT_DIR / "iii_heatmap.png"
HIST_PNG = OUT_DIR / "vsint_transition_hist.png"

sys.path.insert(0, str(ROOT / "nsram"))

DEVICE = torch.device("cpu")
DTYPE = torch.float64

ALPHA0_CONST = 7.842e-5
V_D = 2.0           # Standard Sebas operating point
N_VSINT = 50
III_FLOOR = 1e-25   # Iii < this => M1 effectively OFF


def _load_module(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


@contextmanager
def patch_sd_scaled(sd, overrides):
    if not overrides:
        yield
        return
    saved = {}
    try:
        for k, v in overrides.items():
            saved[k] = sd.scaled.get(k, None)
            sd.scaled[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sd.scaled.pop(k, None)
            else:
                sd.scaled[k] = v


def main():
    t0 = time.time()
    print(f"[z329] start at {time.strftime('%H:%M:%S')}", flush=True)

    z326 = _load_module("z326_solver_fix", SCRIPTS / "z326_solver_fix.py")
    z304 = _load_module("z304_sebas_three_branch_refit",
                        SCRIPTS / "z304_sebas_three_branch_refit.py")
    z91f, cfg, M1, M2, sd_M1, sd_M2, forward_2t = z304.build_models_once()
    sebas_rows = z304.load_sebas_params()
    print(f"[z329] models built ({time.time()-t0:.1f}s), {len(sebas_rows)} biases", flush=True)

    from nsram.bsim4_port.dc import compute_dc
    from nsram.bsim4_port.leak import compute_iimpact

    vsint_grid = np.linspace(0.0, V_D, N_VSINT)
    bias_results = []
    iii_matrix = np.full((len(sebas_rows), N_VSINT), np.nan, dtype=np.float64)

    for bi, row in enumerate(sebas_rows):
        VG1 = float(row["VG1"]); VG2 = float(row["VG2"])
        if math.isnan(row.get("K1", float("nan"))):
            print(f"[z329] bias {bi:02d} VG1={VG1:.2f} VG2={VG2:.2f}: K1 nan, skip", flush=True)
            bias_results.append({
                "idx": bi, "VG1": VG1, "VG2": VG2, "V_d": V_D,
                "valid": False, "reason": "K1 nan",
                "Vsint_transition": None, "Iii_max": None,
                "Iii_at_Vsint0": None,
            })
            continue

        z326.configure_v5b_postfix(cfg, VG1)
        P_M1, P_M2 = z304.make_row_overrides(
            row, ALPHA0_CONST, z91f.M2_STATIC_OVERRIDES)

        iii_vals = np.zeros(N_VSINT, dtype=np.float64)
        ids_vals = np.zeros(N_VSINT, dtype=np.float64)
        with torch.no_grad(), patch_sd_scaled(sd_M1, P_M1):
            for vi, vsint in enumerate(vsint_grid):
                vds = V_D - vsint
                vgs = VG1 - vsint
                vbs = 0.0 - vsint
                dc = compute_dc(M1, sd_M1,
                                Vgs=torch.tensor(vgs, dtype=DTYPE),
                                Vds=torch.tensor(vds, dtype=DTYPE),
                                Vbs=torch.tensor(vbs, dtype=DTYPE))
                iii = float(compute_iimpact(M1, sd_M1, dc,
                                            Vds=torch.tensor(vds, dtype=DTYPE)))
                iii_vals[vi] = max(iii, 0.0)
                ids_vals[vi] = float(dc.Ids)

        iii_matrix[bi, :] = iii_vals
        # Vsint_transition: smallest Vsint where Iii drops below floor (M1 ON->OFF).
        # If Iii never above floor, transition = 0. If Iii never below floor, transition = V_d.
        above = iii_vals > III_FLOOR
        if not above.any():
            v_trans = 0.0
            non_trivial = False
        elif above.all():
            v_trans = float(V_D)
            non_trivial = True
        else:
            # find first idx where above->below after first above
            first_on = int(np.argmax(above))
            after_first = np.where(~above[first_on:])[0]
            if len(after_first) == 0:
                v_trans = float(V_D)
            else:
                v_trans = float(vsint_grid[first_on + after_first[0]])
            non_trivial = True

        bias_results.append({
            "idx": bi, "VG1": VG1, "VG2": VG2, "V_d": V_D,
            "valid": True, "non_trivial": bool(non_trivial),
            "Vsint_transition": v_trans,
            "Iii_max": float(iii_vals.max()),
            "Iii_at_Vsint0": float(iii_vals[0]),
            "Iii_at_Vsint_Vd": float(iii_vals[-1]),
            "Ids_at_Vsint0": float(ids_vals[0]),
        })
        print(f"[z329] {bi:02d} VG1={VG1:+.2f} VG2={VG2:+.2f}: "
              f"Iii(0)={iii_vals[0]:.2e} Iii_max={iii_vals.max():.2e} "
              f"V_trans={v_trans:.3f}V non_trivial={non_trivial}", flush=True)

    # ------------ Aggregate gates ------------
    valid = [b for b in bias_results if b.get("valid")]
    non_trivial = [b for b in valid if b.get("non_trivial")]
    v_trans_vals = [b["Vsint_transition"] for b in non_trivial]
    n_nontrivial = len(non_trivial)
    median_v_trans = float(np.median(v_trans_vals)) if v_trans_vals else float("nan")
    mean_v_trans = float(np.mean(v_trans_vals)) if v_trans_vals else float("nan")

    infra_pass = n_nontrivial >= 30
    pass_gate = (not math.isnan(median_v_trans)) and (median_v_trans < 0.7 * V_D)

    # ------------ Heatmap PNG ------------
    fig, ax = plt.subplots(figsize=(10, 8))
    with np.errstate(divide="ignore"):
        log_iii = np.log10(np.maximum(iii_matrix, 1e-50))
    im = ax.imshow(log_iii, aspect="auto", origin="lower",
                   extent=[vsint_grid[0], vsint_grid[-1], 0, len(sebas_rows)],
                   cmap="viridis", vmin=-50, vmax=-3)
    ax.set_xlabel("Vsint [V]")
    ax.set_ylabel("Sebas bias index (sorted by CSV order)")
    ax.set_title(f"z329 — log10(Iii_M1) vs Vsint, V_d={V_D}V, 33 Sebas biases")
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("log10(Iii [A])")
    # Overlay transition Vsint as a white line per bias
    for b in valid:
        if b["non_trivial"]:
            ax.plot(b["Vsint_transition"], b["idx"] + 0.5, "wx", markersize=5)
    plt.tight_layout()
    plt.savefig(HEATMAP_PNG, dpi=120)
    plt.close(fig)

    # ------------ Histogram PNG ------------
    fig, ax = plt.subplots(figsize=(8, 5))
    if v_trans_vals:
        ax.hist(v_trans_vals, bins=np.linspace(0, V_D, 21),
                edgecolor="black", color="steelblue")
        ax.axvline(median_v_trans, color="red", linestyle="--",
                   label=f"median={median_v_trans:.3f}V")
        ax.axvline(0.7 * V_D, color="green", linestyle=":",
                   label=f"0.7·V_d={0.7*V_D:.2f}V (PASS threshold)")
        ax.legend()
    ax.set_xlabel("Vsint_transition [V] (where Iii drops < 1e-25 A)")
    ax.set_ylabel("Bias count")
    ax.set_title(f"z329 — Vsint_transition histogram (N={len(v_trans_vals)}/{len(sebas_rows)})")
    plt.tight_layout()
    plt.savefig(HIST_PNG, dpi=120)
    plt.close(fig)

    # ------------ Summary ------------
    summary = {
        "script": "z329_iii_vsint_map",
        "device": str(DEVICE),
        "V_d": V_D,
        "n_vsint_steps": N_VSINT,
        "iii_floor": III_FLOOR,
        "alpha0_const": ALPHA0_CONST,
        "n_biases_total": len(sebas_rows),
        "n_biases_valid": len(valid),
        "n_biases_non_trivial": n_nontrivial,
        "median_Vsint_transition": median_v_trans,
        "mean_Vsint_transition": mean_v_trans,
        "min_Vsint_transition": float(np.min(v_trans_vals)) if v_trans_vals else None,
        "max_Vsint_transition": float(np.max(v_trans_vals)) if v_trans_vals else None,
        "gates": {
            "INFRA_n_nontrivial_ge_30": bool(infra_pass),
            "PASS_median_lt_0p7_Vd": bool(pass_gate),
        },
        "pass_threshold_V": 0.7 * V_D,
        "per_bias": bias_results,
        "heatmap_png": str(HEATMAP_PNG),
        "hist_png": str(HIST_PNG),
        "elapsed_s": time.time() - t0,
    }
    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[z329] === SUMMARY ===", flush=True)
    print(f"  N biases: {len(sebas_rows)}, valid: {len(valid)}, non-trivial: {n_nontrivial}", flush=True)
    print(f"  median Vsint_transition: {median_v_trans:.4f} V", flush=True)
    print(f"  mean   Vsint_transition: {mean_v_trans:.4f} V", flush=True)
    print(f"  INFRA gate (>=30 non-trivial): {infra_pass}", flush=True)
    print(f"  PASS  gate (median < 0.7*V_d={0.7*V_D}): {pass_gate}", flush=True)
    print(f"  heatmap: {HEATMAP_PNG}", flush=True)
    print(f"  hist:    {HIST_PNG}", flush=True)
    print(f"  summary: {SUMMARY}", flush=True)
    print(f"  elapsed: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
