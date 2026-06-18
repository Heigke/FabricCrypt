"""z305b_aggregate — aggregate 12-cell sweep with per-branch ETAB.

Pre-registered gates (oracle O50):
  PASS-bug-confirmed : V_G1=0.2 log-RMSE ≤ 2.30 dec at best cell
  BONUS PASS         : cell-wide median < 0.5 dec at best cell
  FAIL               : V_G1=0.2 > 3.0 dec (real physics, not bug)
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT = ROOT / "results/z305b_etab_perbranch"

# z304 best per-branch baseline
Z304_BEST_PER_BRANCH = {
    0.2: {"bf": 500, "rs": 0,     "median_log_rmse": 2.061},
    0.4: {"bf": 50,  "rs": 1.0e10, "median_log_rmse": 1.405},
    0.6: {"bf": 9000, "rs": 1.0e10, "median_log_rmse": 0.70},
}

# z305 corrective (with ETAB bug) baseline — for delta-comparison
Z305_REF = ROOT / "results/z305_corrective/summary.json"


def main():
    cells = []
    sources = []
    for jf in sorted(OUT.glob("corrective_bf_*_rs_*.json")):
        try:
            d = json.loads(jf.read_text())
        except Exception as e:
            print(f"skip {jf.name}: {e}"); continue
        sources.append(jf.name)
        rows = d.get("rows", [])
        per_vg1 = {}
        all_log_rmses, all_signed = [], []
        for r in rows:
            vg1 = round(float(r["vg1"]), 2)
            per_vg1[vg1] = {
                "median_log_rmse": r["median_log_rmse"],
                "signed_dec_median": r["signed_dec_median"],
                "etab_used": r.get("etab_used"),
                "p90_log_rmse": r.get("p90_log_rmse"),
                "n_finite": r.get("n_finite"),
                "n_total": r.get("n_total"),
            }
            for pc in r.get("per_curve", []):
                lr = pc.get("log_rmse"); sg = pc.get("signed_dec")
                if lr is not None and math.isfinite(lr): all_log_rmses.append(lr)
                if sg is not None and math.isfinite(sg): all_signed.append(sg)
        cellwide_med = float(np.median(all_log_rmses)) if all_log_rmses else float("inf")
        cellwide_signed = float(np.median(all_signed)) if all_signed else float("nan")
        worst = max((per_vg1[v]["median_log_rmse"] for v in per_vg1
                       if math.isfinite(per_vg1[v]["median_log_rmse"])),
                       default=float("inf"))
        cells.append({
            "bf": d["bf"], "rs": d["rs"], "alpha0": d.get("alpha0"),
            "cellwide_median_log_rmse": cellwide_med,
            "cellwide_signed_dec_median": cellwide_signed,
            "worst_branch_median": worst,
            "n_curves_cellwide": len(all_log_rmses),
            "per_vg1": per_vg1,
        })
    print(f"loaded {len(cells)} cells from {len(sources)} files")
    finite_cells = [c for c in cells if math.isfinite(c["cellwide_median_log_rmse"])]
    if not finite_cells:
        print("no finite cells; aborting"); return

    # Best by cell-wide median
    best = min(finite_cells, key=lambda c: c["cellwide_median_log_rmse"])
    # Best by V_G1=0.2 (the focus of this bug-fix experiment)
    best_for_02 = min(
        (c for c in finite_cells
         if math.isfinite(c["per_vg1"].get(0.2, {}).get("median_log_rmse", float("inf")))),
        key=lambda c: c["per_vg1"][0.2]["median_log_rmse"],
        default=None,
    )

    bw_med = best["cellwide_median_log_rmse"]
    bw_signed = best["cellwide_signed_dec_median"]
    v02_at_best = best["per_vg1"].get(0.2, {}).get("median_log_rmse", float("inf"))
    v02_at_best_for_02 = (best_for_02["per_vg1"][0.2]["median_log_rmse"]
                           if best_for_02 else float("inf"))

    gates = {
        "PASS_bug_confirmed_at_best_for_02": v02_at_best_for_02 <= 2.30,
        "PASS_bug_confirmed_at_cellwide_best": v02_at_best <= 2.30,
        "BONUS_cellwide_median_lt_0p5": bw_med < 0.5,
        "FAIL_v02_above_3dec": v02_at_best_for_02 > 3.0,
        "v02_log_rmse_at_best_for_02": v02_at_best_for_02,
        "v02_log_rmse_at_cellwide_best": v02_at_best,
        "cellwide_median_log_rmse_at_best": bw_med,
        "cellwide_signed_at_best": bw_signed,
    }

    diff_vs_z304 = {}
    for vg1, ref in Z304_BEST_PER_BRANCH.items():
        c_med = best["per_vg1"].get(vg1, {}).get("median_log_rmse")
        diff_vs_z304[str(vg1)] = {
            "z304_best_med": ref["median_log_rmse"],
            "z305b_at_best_cell_med": c_med,
            "delta": (c_med - ref["median_log_rmse"])
                       if c_med is not None and math.isfinite(c_med) else None,
        }

    # Compare to z305 (bug)
    z305_compare = None
    if Z305_REF.exists():
        z305d = json.loads(Z305_REF.read_text())
        z305_best = z305d.get("best_cell", {})
        z305_per_vg1 = z305_best.get("per_vg1", {})
        z305_compare = {
            "z305_best_cell_bf": z305_best.get("bf"),
            "z305_best_cell_rs": z305_best.get("rs"),
            "z305_cellwide_med": z305_best.get("cellwide_median_log_rmse"),
            "z305_per_vg1_med": {k: z305_per_vg1.get(k, {}).get("median_log_rmse")
                                   if isinstance(z305_per_vg1, dict) else None
                                   for k in ["0.2", "0.4", "0.6"]},
            "z305b_minus_z305_v02": (
                v02_at_best_for_02 -
                (z305_per_vg1.get("0.2", {}).get("median_log_rmse", float("nan"))
                 if isinstance(z305_per_vg1, dict) else float("nan"))
            ),
        }

    if gates["BONUS_cellwide_median_lt_0p5"]:
        verdict = ("BONUS PASS: cell-wide median < 0.5 dec with per-branch ETAB. "
                   "O49 conservative gate HIT — v4.4 readiness changed significantly. "
                   "Topology-gap-mandatory narrative REFUTED.")
    elif gates["PASS_bug_confirmed_at_best_for_02"]:
        verdict = ("PASS-bug-confirmed: V_G1=0.2 log-RMSE recovers to ≤2.30 dec "
                   "after per-branch ETAB fix. z305 regression was the ETAB bug, "
                   "not physics. v4.4 path returns to z304 baseline.")
    elif gates["FAIL_v02_above_3dec"]:
        verdict = ("FAIL: V_G1=0.2 stays >3.0 dec even with per-branch ETAB. "
                   "The regression is REAL PHYSICS, not the ETAB bug. "
                   "SA3 topology-gap narrative is supported.")
    else:
        verdict = ("PARTIAL: V_G1=0.2 between 2.30 and 3.0 dec — ETAB fix helps "
                   "but does not fully recover z304 baseline. Mixed evidence.")

    summary = {
        "script": "z305b_aggregate",
        "n_cells": len(cells), "n_finite_cells": len(finite_cells),
        "n_source_files": len(sources),
        "best_cell_by_cellwide": best,
        "best_cell_by_v02": best_for_02,
        "all_cells_ranked_cellwide": sorted(finite_cells,
                                              key=lambda c: c["cellwide_median_log_rmse"]),
        "all_cells_ranked_v02": sorted(
            (c for c in finite_cells
             if math.isfinite(c["per_vg1"].get(0.2, {}).get("median_log_rmse", float("inf")))),
            key=lambda c: c["per_vg1"][0.2]["median_log_rmse"]),
        "gates": gates,
        "diff_vs_z304_per_branch": diff_vs_z304,
        "diff_vs_z305_bug": z305_compare,
        "verdict": verdict,
    }
    out_path = OUT / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {out_path}")
    print(f"\nBEST CELL (cell-wide): Bf={best['bf']}  Rs={best['rs']}")
    print(f"  cell-wide med = {bw_med:.3f}  signed = {bw_signed:+.3f}")
    p02 = best["per_vg1"].get(0.2, {}).get("median_log_rmse", float("nan"))
    p04 = best["per_vg1"].get(0.4, {}).get("median_log_rmse", float("nan"))
    p06 = best["per_vg1"].get(0.6, {}).get("median_log_rmse", float("nan"))
    print(f"  per V_G1: 0.2={p02:.3f}  0.4={p04:.3f}  0.6={p06:.3f}")
    if best_for_02:
        print(f"\nBEST CELL for V_G1=0.2: Bf={best_for_02['bf']} Rs={best_for_02['rs']}")
        print(f"  V_G1=0.2 med = {v02_at_best_for_02:.3f}")
    print(f"\nGate PASS-bug-confirmed (V_G1=0.2 ≤ 2.30): {gates['PASS_bug_confirmed_at_best_for_02']}")
    print(f"Gate BONUS cell-wide < 0.5:                {gates['BONUS_cellwide_median_lt_0p5']}")
    print(f"Gate FAIL V_G1=0.2 > 3.0:                  {gates['FAIL_v02_above_3dec']}")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
