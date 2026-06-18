"""z304_aggregate — collect per-(V_G1, Bf) job outputs, find best per branch
and cell-wide compromise, evaluate gates, write summary.json.

Output: results/z304_sebas_refit/summary.json

Gates:
  PASS-conservative : per-branch median log-RMSE < 0.7 dec
  AMBITIOUS         : per-branch median < 0.3 AND |signed_dec| < 0.1
  SAFETY            : per-branch median < 1.5 dec
  SHIP (AMBITIOUS all 3 branches): this is the v4.4 baseline
"""
from __future__ import annotations
import json, math
from pathlib import Path

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT = ROOT / "results/z304_sebas_refit"

DA3_REF = 0.99  # current pyport median (Bf=3000)


def is_dominated(cell, others):
    """Pareto: minimize median_log_rmse and |signed_dec|."""
    a, b = cell["median_log_rmse"], abs(cell["signed_dec_median"])
    for o in others:
        if o is cell:
            continue
        oa, ob = o["median_log_rmse"], abs(o["signed_dec_median"])
        if oa <= a and ob <= b and (oa < a or ob < b):
            return True
    return False


def main():
    rows_all = []
    src_files = []
    for jf in sorted(OUT.glob("refit_vg1_*_bf_*.json")):
        try:
            d = json.loads(jf.read_text())
            rows_all.extend(d.get("rows", []))
            src_files.append(jf.name)
        except Exception as e:
            print(f"skip {jf.name}: {e}")

    print(f"Loaded {len(rows_all)} cells from {len(src_files)} files")

    # Drop infs
    finite_rows = [r for r in rows_all
                    if math.isfinite(r.get("median_log_rmse", float("inf")))
                    and math.isfinite(r.get("signed_dec_median", float("nan")))]
    print(f"Finite cells: {len(finite_rows)}")

    # Best per branch (minimize median_log_rmse)
    by_vg1 = {}
    for vg1 in [0.2, 0.4, 0.6]:
        branch_rows = [r for r in finite_rows
                        if abs(r["vg1"] - vg1) < 1e-3]
        if not branch_rows:
            by_vg1[vg1] = None
            continue
        best = min(branch_rows, key=lambda r: r["median_log_rmse"])
        # Pareto set
        pareto = [r for r in branch_rows if not is_dominated(r, branch_rows)]
        pareto.sort(key=lambda r: r["median_log_rmse"])
        by_vg1[vg1] = {
            "best": {k: best[k] for k in ("vg1", "bf", "alpha0", "rs",
                                            "median_log_rmse",
                                            "signed_dec_median",
                                            "p90_log_rmse", "n_finite")},
            "pareto": [{k: r[k] for k in ("bf", "alpha0", "rs",
                                            "median_log_rmse",
                                            "signed_dec_median")}
                        for r in pareto[:6]],
            "n_branch_cells": len(branch_rows),
        }

    # Cell-wide compromise: find single (Bf, alpha0, Rs) cell that minimizes
    # MAX over branches (worst-branch).
    by_global = {}
    for r in finite_rows:
        key = (r["bf"], r["alpha0"], r["rs"])
        by_global.setdefault(key, {})[r["vg1"]] = r
    cell_wide = []
    for key, perbr in by_global.items():
        if len(perbr) < 3:
            continue
        meds = [perbr[vg1]["median_log_rmse"] for vg1 in [0.2, 0.4, 0.6]]
        signs = [perbr[vg1]["signed_dec_median"] for vg1 in [0.2, 0.4, 0.6]]
        cell_wide.append({
            "bf": key[0], "alpha0": key[1], "rs": key[2],
            "vg1_02_med": meds[0], "vg1_04_med": meds[1], "vg1_06_med": meds[2],
            "worst_branch_med": max(meds),
            "median_across_branches": float(sorted(meds)[1]),
            "max_abs_signed": max(abs(s) for s in signs),
        })
    cell_wide.sort(key=lambda r: r["worst_branch_med"])
    best_compromise = cell_wide[0] if cell_wide else None

    # Gates
    gates = {}
    for vg1 in [0.2, 0.4, 0.6]:
        b = by_vg1.get(vg1)
        if b is None:
            gates[f"vg1_{vg1}"] = {"PASS": False, "AMBITIOUS": False,
                                     "SAFETY": False, "note": "no data"}
            continue
        med = b["best"]["median_log_rmse"]
        signed = b["best"]["signed_dec_median"]
        gates[f"vg1_{vg1}"] = {
            "PASS_conservative": med < 0.7,
            "AMBITIOUS": (med < 0.3 and abs(signed) < 0.1),
            "SAFETY": med < 1.5,
            "median_log_rmse": med,
            "signed_dec_median": signed,
        }
    ship = all(gates[f"vg1_{v}"].get("AMBITIOUS", False) for v in [0.2, 0.4, 0.6])
    all_pass = all(gates[f"vg1_{v}"].get("PASS_conservative", False) for v in [0.2, 0.4, 0.6])
    all_safe = all(gates[f"vg1_{v}"].get("SAFETY", False) for v in [0.2, 0.4, 0.6])
    cellwide_pass = (best_compromise is not None
                      and best_compromise["worst_branch_med"] < DA3_REF)

    summary = {
        "script": "z304_aggregate",
        "n_cells_loaded": len(rows_all),
        "n_finite_cells": len(finite_rows),
        "n_source_files": len(src_files),
        "by_vg1": by_vg1,
        "best_cellwide_compromise": best_compromise,
        "top_5_cellwide": cell_wide[:5],
        "gates": gates,
        "verdict": {
            "ALL_PASS_conservative": all_pass,
            "ALL_AMBITIOUS_SHIP_v4.4": ship,
            "ALL_SAFETY": all_safe,
            "CELLWIDE_BEATS_DA3": cellwide_pass,
        },
        "da3_reference_median": DA3_REF,
    }
    out_path = OUT / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nwrote {out_path}")
    for vg1 in [0.2, 0.4, 0.6]:
        if by_vg1[vg1] is None:
            print(f"  V_G1={vg1}: NO DATA"); continue
        b = by_vg1[vg1]["best"]
        print(f"  V_G1={vg1}: best Bf={b['bf']} alpha0={b['alpha0']:.0e} Rs={b['rs']} "
              f"→ med={b['median_log_rmse']:.3f}  signed={b['signed_dec_median']:+.3f}")
    if best_compromise:
        bc = best_compromise
        print(f"  cell-wide: Bf={bc['bf']} alpha0={bc['alpha0']:.0e} Rs={bc['rs']} "
              f"→ worst={bc['worst_branch_med']:.3f}  median={bc['median_across_branches']:.3f}")
    print(f"  gates: SHIP={ship}  ALL_PASS={all_pass}  ALL_SAFE={all_safe}  CW_BEATS_DA3={cellwide_pass}")


if __name__ == "__main__":
    main()
