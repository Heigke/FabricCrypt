"""MEP-2: Denser 4D body-state surrogate v3 with extended V_G2 and V_b axes.

Builds a 24000-point 4D (V_G1, V_G2, V_d, V_b) surrogate via pyport
Newton solve, reusing the _solve_at_fixed_vb routine from
scripts/nsram_surrogate_4d.py. Parallelized across CPU workers.

Adapted from scripts/z271_pmp3_dense_surrogate.py. New axes (locked):
  V_G1: 10 vals in [0.10, 0.80]
  V_G2: 15 vals in [-0.10, 0.60]   (extended past 0.45 surrogate edge)
  V_d : 8 vals in [0.25, 3.00]
  V_b : 20 vals in [0.00, 1.00]
  = 10 × 15 × 8 × 20 = 24000 op points (5.3× denser than z271)

Output: results/z278_mep2_surrogate_v3/surrogate_4d_v3.npz

Pre-reg gates (research_plan/MEP_DS_PLAN_2026-05-12.md):
  PASS ≥95% conv; INFORMATIVE-PASS 80-94%; FAIL <80%.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import json
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# MEP-2 dense axes (LOCKED — do not tweak post-run)
VG1_AXIS = np.array([0.10, 0.18, 0.26, 0.34, 0.42, 0.50, 0.58, 0.65, 0.72, 0.80],
                    dtype=np.float64)
VG2_AXIS = np.array([-0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25,
                     0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
                    dtype=np.float64)
VD_AXIS  = np.array([0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 2.50, 3.00],
                    dtype=np.float64)
VB_AXIS  = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                     0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
                     0.80, 0.85, 0.90, 1.00],
                    dtype=np.float64)
# 10 × 15 × 8 × 20 = 24000

# Per-worker globals (initialized in init_worker)
_CFG = _M1 = _M2 = _BJT = None
_SOLVE = None


def init_worker():
    """Build pyport models once per worker process."""
    global _CFG, _M1, _M2, _BJT, _SOLVE
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ns4d", ROOT / "scripts/nsram_surrogate_4d.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _CFG, _M1, _M2, _BJT = mod._build_pyport_models()
    _SOLVE = mod._solve_at_fixed_vb


def solve_point(args):
    i, j, k, l, vg1, vg2, vd, vb = args
    try:
        out = _SOLVE(_CFG, _M1, _M2, _BJT, vd, vg1, vg2, vb)
        return (i, j, k, l, out["Id"], out["Iii_in"], out["Ileak_out"],
                bool(out["converged"]), None)
    except Exception as e:
        return (i, j, k, l, np.nan, 0.0, 0.0, False, str(e))


def main(n_workers: int = 8):
    NG1, NG2, NVD, NVB = (len(VG1_AXIS), len(VG2_AXIS),
                          len(VD_AXIS), len(VB_AXIS))
    n_total = NG1 * NG2 * NVD * NVB
    print(f"[z278/mep2] grid = {NG1}x{NG2}x{NVD}x{NVB} = {n_total} pts; "
          f"workers={n_workers}")

    tasks = []
    for i, vg1 in enumerate(VG1_AXIS):
        for j, vg2 in enumerate(VG2_AXIS):
            for k, vd in enumerate(VD_AXIS):
                for l, vb in enumerate(VB_AXIS):
                    tasks.append((i, j, k, l,
                                  float(vg1), float(vg2),
                                  float(vd),  float(vb)))

    Id_grid    = np.full((NG1, NG2, NVD, NVB), np.nan, dtype=np.float64)
    Iii_grid   = np.zeros_like(Id_grid)
    Ileak_grid = np.zeros_like(Id_grid)
    conv_grid  = np.zeros((NG1, NG2, NVD, NVB), dtype=bool)
    err_count  = {}

    t0 = time.time()
    n_done = 0
    with Pool(n_workers, initializer=init_worker) as pool:
        for res in pool.imap_unordered(solve_point, tasks, chunksize=8):
            i, j, k, l, Id, Iii, Ileak, conv, err = res
            Id_grid[i, j, k, l]    = Id
            Iii_grid[i, j, k, l]   = Iii
            Ileak_grid[i, j, k, l] = Ileak
            conv_grid[i, j, k, l]  = conv
            if err is not None:
                err_count[err] = err_count.get(err, 0) + 1
            n_done += 1
            if n_done % 1000 == 0:
                wall = time.time() - t0
                eta = wall / n_done * (n_total - n_done)
                print(f"  {n_done}/{n_total} ({100*n_done/n_total:.0f}%); "
                      f"wall={wall:.0f}s eta={eta:.0f}s "
                      f"conv_so_far={int(conv_grid.sum())}")

    wall = time.time() - t0
    n_conv = int(conv_grid.sum())
    conv_rate = n_conv / n_total
    print(f"\n[z278/mep2] done in {wall:.0f}s; converged {n_conv}/{n_total} "
          f"({100*conv_rate:.1f}%)")

    out_dir = ROOT / "results/z278_mep2_surrogate_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "surrogate_4d_v3.npz"
    np.savez(out_path,
             Id=Id_grid, Iii=Iii_grid, Ileak=Ileak_grid,
             converged=conv_grid,
             vg1_axis=VG1_AXIS, vg2_axis=VG2_AXIS,
             vd_axis=VD_AXIS,  vb_axis=VB_AXIS)

    # Convergence breakdown along each axis
    conv_by_vg1 = {float(VG1_AXIS[i]): float(conv_grid[i].mean())
                   for i in range(NG1)}
    conv_by_vg2 = {float(VG2_AXIS[j]): float(conv_grid[:, j].mean())
                   for j in range(NG2)}
    conv_by_vd  = {float(VD_AXIS[k]):  float(conv_grid[:, :, k].mean())
                   for k in range(NVD)}
    conv_by_vb  = {float(VB_AXIS[l]):  float(conv_grid[:, :, :, l].mean())
                   for l in range(NVB)}

    # Locate non-convergence concentration: top 20 worst (vg1, vg2) cells
    fail_2d = (~conv_grid).sum(axis=(2, 3))  # NG1 x NG2
    flat = [(fail_2d[i, j], float(VG1_AXIS[i]), float(VG2_AXIS[j]))
            for i in range(NG1) for j in range(NG2)]
    flat.sort(reverse=True)
    worst = [{"vg1": v1, "vg2": v2, "n_fail": int(n)}
             for n, v1, v2 in flat[:20] if n > 0]

    # Also: which (vb, vd) slices are worst
    fail_vbvd = (~conv_grid).sum(axis=(0, 1))  # NVD x NVB
    flat_vbvd = [(fail_vbvd[k, l], float(VD_AXIS[k]), float(VB_AXIS[l]))
                 for k in range(NVD) for l in range(NVB)]
    flat_vbvd.sort(reverse=True)
    worst_vbvd = [{"vd": vd, "vb": vb, "n_fail": int(n)}
                  for n, vd, vb in flat_vbvd[:20] if n > 0]

    if conv_rate >= 0.95:
        verdict = "PASS"
    elif conv_rate >= 0.80:
        verdict = "INFORMATIVE-PASS"
    else:
        verdict = "FAIL"

    summary = {
        "task": "MEP-2 dense 4D surrogate v3 (extended V_G2 / V_b)",
        "verdict": verdict,
        "n_total": n_total,
        "n_converged": n_conv,
        "conv_rate": conv_rate,
        "wall_s": wall,
        "node": os.uname().nodename,
        "n_workers": n_workers,
        "grid_shape": [NG1, NG2, NVD, NVB],
        "axes": {
            "vg1": VG1_AXIS.tolist(), "vg2": VG2_AXIS.tolist(),
            "vd": VD_AXIS.tolist(),   "vb": VB_AXIS.tolist(),
        },
        "conv_by_vg1": conv_by_vg1,
        "conv_by_vg2": conv_by_vg2,
        "conv_by_vd":  conv_by_vd,
        "conv_by_vb":  conv_by_vb,
        "worst_vg1_vg2_cells": worst,
        "worst_vd_vb_cells":   worst_vbvd,
        "err_count": err_count,
        "out_path": str(out_path),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[z278/mep2] verdict = {verdict}; written {out_dir}/summary.json")
    return summary


if __name__ == "__main__":
    import sys
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    main(n_workers=nw)
