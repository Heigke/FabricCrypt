"""z305_aggregate — collect 12 (Bf, Rs) corrective job outputs, compute
cell-wide median log-RMSE, evaluate the pre-registered O49 gate, compare
to z304 best-per-branch baselines.

Output: results/z305_corrective/summary.json

Pre-registered gates (O49-locked):
  PASS-conservative : cell-wide median forward log-RMSE < 0.5 dec
  AMBITIOUS         : cell-wide median < 0.3 AND |signed_dec| < 0.1
  FALSIFICATION     : if conservative PASS → topology-gap claim is wrong
"""
from __future__ import annotations
import json, math
from pathlib import Path

import numpy as np

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT = ROOT / "results/z305_corrective"

# z304 best per-branch (median log-RMSE) — for comparison only.
# From results/z304_sebas_refit/summary.json.
Z304_BEST_PER_BRANCH = {
    0.2: {"bf": 500, "rs": 0, "median_log_rmse": 2.061},
    0.4: {"bf": 50,  "rs": 1.0e10, "median_log_rmse": 1.405},
    0.6: {"bf": 9000, "rs": 1.0e10, "median_log_rmse": 0.70},  # spec-cited
}


def main():
    cells = []
    sources = []
    for jf in sorted(OUT.glob("corrective_bf_*_rs_*.json")):
        try:
            d = json.loads(jf.read_text())
        except Exception as e:
            print(f"skip {jf.name}: {e}")
            continue
        sources.append(jf.name)

        rows = d.get("rows", [])
        per_vg1 = {}
        all_log_rmses = []
        all_signed = []
        for r in rows:
            vg1 = round(float(r["vg1"]), 2)
            per_vg1[vg1] = {
                "median_log_rmse": r["median_log_rmse"],
                "signed_dec_median": r["signed_dec_median"],
                "p90_log_rmse": r.get("p90_log_rmse"),
                "n_finite": r.get("n_finite"),
                "n_total": r.get("n_total"),
            }
            for pc in r.get("per_curve", []):
                lr = pc.get("log_rmse")
                sg = pc.get("signed_dec")
                if lr is not None and math.isfinite(lr):
                    all_log_rmses.append(lr)
                if sg is not None and math.isfinite(sg):
                    all_signed.append(sg)

        cellwide_med = float(np.median(all_log_rmses)) if all_log_rmses else float("inf")
        cellwide_signed = float(np.median(all_signed)) if all_signed else float("nan")
        worst_branch = max((per_vg1[v]["median_log_rmse"]
                             for v in per_vg1
                             if math.isfinite(per_vg1[v]["median_log_rmse"])),
                             default=float("inf"))

        cells.append({
            "bf": d["bf"],
            "rs": d["rs"],
            "alpha0": d.get("alpha0"),
            "cellwide_median_log_rmse": cellwide_med,
            "cellwide_signed_dec_median": cellwide_signed,
            "worst_branch_median": worst_branch,
            "n_curves_cellwide": len(all_log_rmses),
            "per_vg1": per_vg1,
        })

    print(f"loaded {len(cells)} cells from {len(sources)} files")

    # Find best (lowest cell-wide median log-RMSE)
    finite_cells = [c for c in cells if math.isfinite(c["cellwide_median_log_rmse"])]
    if not finite_cells:
        print("no finite cells; aborting")
        return
    best = min(finite_cells, key=lambda c: c["cellwide_median_log_rmse"])

    # Also rank by worst-branch (max over 3 V_G1 medians)
    by_worst = sorted(finite_cells, key=lambda c: c["worst_branch_median"])
    best_worst = by_worst[0]

    # Pre-registered gates against the BEST cell
    bw_med = best["cellwide_median_log_rmse"]
    bw_signed = best["cellwide_signed_dec_median"]
    gates = {
        "PASS_conservative":  bw_med < 0.5,
        "AMBITIOUS":          (bw_med < 0.3 and abs(bw_signed) < 0.1),
        "cellwide_median_log_rmse": bw_med,
        "cellwide_signed_dec_median": bw_signed,
        "threshold_conservative": 0.5,
        "threshold_ambitious_med": 0.3,
        "threshold_ambitious_bias": 0.1,
    }

    # Diff vs z304 per-branch
    diff_vs_z304 = {}
    for vg1, ref in Z304_BEST_PER_BRANCH.items():
        c_med = best["per_vg1"].get(vg1, {}).get("median_log_rmse")
        diff_vs_z304[str(vg1)] = {
            "z304_best_med": ref["median_log_rmse"],
            "z304_best_bf": ref["bf"], "z304_best_rs": ref["rs"],
            "z305_at_best_cell_med": c_med,
            "delta": (c_med - ref["median_log_rmse"]) if c_med is not None and math.isfinite(c_med) else None,
        }

    # Verdict
    if gates["PASS_conservative"]:
        verdict = ("CONSERVATIVE PASS: O49 clipped-parameterization hypothesis "
                    "HOLDS. SA1-canonical (per-V_G1 K1/mbjt/BETA0/ETAB) enables "
                    "cell-wide fit < 0.5 dec. Topology-gap-mandatory claim is "
                    "REFUTED. v4.4 path is back open.")
    else:
        verdict = ("CONSERVATIVE FAIL: O49 clipped-parameterization hypothesis "
                    "REJECTED. Even with SA1-canonical per-branch overrides, "
                    "cell-wide median exceeds 0.5 dec. SA3 topology-gap "
                    "narrative is supported.")

    summary = {
        "script": "z305_aggregate",
        "n_cells": len(cells),
        "n_finite_cells": len(finite_cells),
        "n_source_files": len(sources),
        "best_cell": best,
        "best_by_worst_branch": best_worst,
        "all_cells_ranked": sorted(finite_cells, key=lambda c: c["cellwide_median_log_rmse"]),
        "gates_at_best": gates,
        "diff_vs_z304_per_branch": diff_vs_z304,
        "verdict": verdict,
    }
    out_path = OUT / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {out_path}")
    print(f"\nBEST cell:  Bf={best['bf']}  Rs={best['rs']}")
    print(f"  cell-wide median log-RMSE = {bw_med:.3f}")
    print(f"  cell-wide signed bias     = {bw_signed:+.3f}")
    print(f"  per-V_G1: 0.2={best['per_vg1'].get(0.2,{}).get('median_log_rmse','-'):.3f}  "
          f"0.4={best['per_vg1'].get(0.4,{}).get('median_log_rmse','-'):.3f}  "
          f"0.6={best['per_vg1'].get(0.6,{}).get('median_log_rmse','-'):.3f}")
    print(f"\nGATE PASS-conservative (<0.5): {gates['PASS_conservative']}")
    print(f"GATE AMBITIOUS         (<0.3 & |b|<0.1): {gates['AMBITIOUS']}")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
