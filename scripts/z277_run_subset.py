"""z277 MEP-1 driver: run 16-cell subset with quadrilinear interp and
aggregate vs. the nearest-neighbor baseline (results/sweep_v2_aggregated).

Output: results/z277_mep1_trilinear/summary.json
"""
from __future__ import annotations
import json, time, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts" / "z277_mep1_trilinear_z272.py"
OUT_DIR = REPO / "results" / "z277_mep1_trilinear"
PER_CELL = OUT_DIR / "per_cell"
BASELINE = REPO / "results" / "sweep_v2_aggregated"

# 16-cell subset: D1 best (d115) + top-4, 8 stratified mids, 3 bottom.
SUBSET = [
    ("d115",  8.0, 0.35, 1e-07, 0.8),
    ("d179", 10.0, 0.35, 1e-07, 0.8),
    ("d051",  5.0, 0.35, 1e-07, 0.8),
    ("d050",  5.0, 0.35, 1e-07, 0.4),
    ("d243", 14.0, 0.35, 1e-07, 0.8),
    ("d131", 10.0, 0.05, 1e-07, 0.8),
    ("d053",  5.0, 0.35, 5e-07, 0.2),
    ("d276", 20.0, 0.15, 5e-07, 0.1),
    ("d039",  5.0, 0.25, 5e-07, 0.8),
    ("d235", 14.0, 0.25, 1e-06, 0.8),
    ("d258", 20.0, 0.05, 1e-07, 0.4),
    ("d194", 14.0, 0.05, 1e-07, 0.4),
    ("d282", 20.0, 0.15, 1e-06, 0.4),
    ("d047",  5.0, 0.25, 5e-06, 0.8),
    ("d078",  8.0, 0.05, 5e-06, 0.4),
    ("d094",  8.0, 0.15, 5e-06, 0.4),
]


def run_cell(cell_id, C_b_fF, V_G2, dt_s, g_in):
    cmd = [
        sys.executable, str(RUNNER),
        "--cell_id", cell_id,
        "--C_b_fF", str(C_b_fF),
        "--V_G2_bias", str(V_G2),
        "--dt_s", str(dt_s),
        "--g_in", str(g_in),
        "--seeds", "0", "1", "2", "3",
        "--subsample", "10000", "2000",
        "--out_dir", str(PER_CELL),
    ]
    print(f"[run] cell {cell_id} C_b={C_b_fF} VG2={V_G2} dt={dt_s} g_in={g_in}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
    return wall


def main():
    PER_CELL.mkdir(parents=True, exist_ok=True)
    t_global = time.time()
    cell_walls = {}
    for spec in SUBSET:
        cell_walls[spec[0]] = run_cell(*spec)
    total_wall = time.time() - t_global

    # Aggregate
    rows = []
    max_delta = -1e9
    max_delta_cell = None
    d115_delta = None
    for cell_id, *_ in SUBSET:
        base_p = BASELINE / f"cell_{cell_id}" / "summary.json"
        tri_p  = PER_CELL  / f"cell_{cell_id}" / "summary.json"
        if not base_p.exists() or not tri_p.exists():
            rows.append({"cell_id": cell_id, "error": "missing summary",
                         "base_exists": base_p.exists(),
                         "tri_exists": tri_p.exists()})
            continue
        base = json.load(open(base_p))
        tri  = json.load(open(tri_p))
        nn = base.get("mean_acc")
        tl = tri.get("mean_acc")
        if nn is None or tl is None:
            rows.append({"cell_id": cell_id, "error": "no mean_acc"})
            continue
        delta_pp = (tl - nn) * 100.0
        if delta_pp > max_delta:
            max_delta = delta_pp
            max_delta_cell = cell_id
        if cell_id == "d115":
            d115_delta = delta_pp
        rows.append({
            "cell_id": cell_id,
            "C_b_fF": base.get("C_b_fF"),
            "V_G2_bias": base.get("V_G2_bias"),
            "dt_s": base.get("dt_s"),
            "g_in": base.get("g_in"),
            "nearest_mean_acc": nn,
            "trilinear_mean_acc": tl,
            "delta_pp": delta_pp,
            "nearest_std_acc": base.get("std_acc"),
            "trilinear_std_acc": tri.get("std_acc"),
            "trilinear_ci95": tri.get("ci95"),
            "nearest_ci95": base.get("ci95"),
        })

    # Gate verdict (pre-registered)
    verdict = "UNDETERMINED"
    reason = []
    if d115_delta is None:
        verdict = "FAIL"
        reason.append("d115 missing")
    else:
        if abs(d115_delta) > 1.0:
            verdict = "FAIL"
            reason.append(f"d115 delta_pp={d115_delta:.2f} > +/-1pp")
        else:
            if max_delta >= 0.5:
                verdict = "PASS"
                reason.append(f"d115 within 1pp ({d115_delta:.2f}); "
                              f"max_delta={max_delta:.2f}pp >= 0.5")
            else:
                verdict = "INFORMATIVE_PASS"
                reason.append(f"d115 within 1pp ({d115_delta:.2f}); "
                              f"max_delta={max_delta:.2f}pp < 0.5 (no improvement)")

    summary = {
        "experiment": "z277_mep1_trilinear",
        "interp": "quadrilinear (4D)",
        "baseline_source": str(BASELINE.relative_to(REPO)),
        "n_cells": len(SUBSET),
        "rows": rows,
        "d115_delta_pp": d115_delta,
        "max_delta_pp": max_delta,
        "max_delta_cell": max_delta_cell,
        "verdict": verdict,
        "reason": reason,
        "wall_s_total": total_wall,
        "wall_s_per_cell": cell_walls,
    }
    out = OUT_DIR / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n=== z277 MEP-1 ===")
    print(f"d115 delta_pp: {d115_delta}")
    print(f"max delta_pp: {max_delta} (cell {max_delta_cell})")
    print(f"VERDICT: {verdict}")
    print(f"Total wall: {total_wall:.1f}s")
    print(f"Summary written: {out}")


if __name__ == "__main__":
    main()
