"""z303b — Bf sweep on Mario BJT topology to find optimal basin.

Mario topology: Vaf=40, Nf=0.9, Ne=1.5, Var=10, Is=1e-16.
Sweep Bf in {50, 100, 200, 500, 1000, 2000, 3000, 5000, 9000}.

Gates (locked):
- PASS-conservative: median fwd log-RMSE < 0.95 dec (beats DA3's 0.988)
- AMBITIOUS:         median < 0.5 dec AND V_G1=0.2 sub-3 dec
- SAFETY:            V_G1=0.6 median NOT > 1.02 dec (DA3+0.3)
"""
from __future__ import annotations
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import json
import re
import time
from pathlib import Path
import importlib.util

import numpy as np
import torch

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
DATA_ROOT = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z303b_mario_bf_sweep"
OUT.mkdir(parents=True, exist_ok=True)

# Reuse z303 module
def _load_module(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod

z303 = _load_module("z303", ROOT / "scripts/z303_mario_bjt_integration.py")

VG1_DIRS = {
    0.2: "2vHCa-2 I-Vs@VG2 VG1=0.2 vnwell=2",
    0.4: "2vHCa-2 I-Vs@VG2 VG1=0.4 vnwell=2",
    0.6: "2vHCa-2 I-Vs@VG2 VG1=0.6 vnwell=2",
}
VG2_RE = re.compile(r"VG2=(-?\d+\.\d+)")

BF_GRID = [50, 100, 200, 500, 1000, 2000, 3000, 5000, 9000]

# Reference DA3 numbers (from z303 results) for gate comparison
DA3_MEDIAN = 0.988
DA3_VG06 = 0.72  # approximate; safety gate uses 1.02 = DA3+0.3


def main():
    t0 = time.time()
    mep7 = _load_module("z294_z303b", ROOT / "scripts/z294_mep7_gpu_pyport.py")
    ns4d = mep7._load_cpu_ref()
    print(f"[z303b] loaded pyport. device check...")

    curve_meta = []
    for vg1, subdir in VG1_DIRS.items():
        d = DATA_ROOT / subdir
        for csv_path in sorted(d.glob("StandardIV*.csv")):
            m = VG2_RE.search(csv_path.name)
            if not m:
                continue
            vg2 = float(m.group(1))
            try:
                meas_vd, meas_id, meas_t = z303.load_one(csv_path)
            except Exception as e:
                print(f"[z303b] load fail {csv_path.name}: {e}")
                continue
            curve_meta.append({
                "vg1": vg1, "vg2": vg2, "file": csv_path.name,
                "meas_vd": meas_vd, "meas_id": meas_id, "meas_t": meas_t,
            })
    seg_idx = []; flat_vd = []; flat_vg1 = []; flat_vg2 = []
    for cm in curve_meta:
        s = len(flat_vd)
        flat_vd.extend(cm["meas_vd"].tolist())
        flat_vg1.extend([cm["vg1"]] * len(cm["meas_vd"]))
        flat_vg2.extend([cm["vg2"]] * len(cm["meas_vd"]))
        seg_idx.append((s, len(flat_vd)))
    all_vd  = np.asarray(flat_vd)
    all_vg1 = np.asarray(flat_vg1)
    all_vg2 = np.asarray(flat_vg2)
    print(f"[z303b] {len(curve_meta)} curves, {len(all_vd)} total points")

    rows = []
    for Bf in BF_GRID:
        t1 = time.time()
        print(f"\n[z303b] === Bf={Bf} (Mario topology) ===")
        r = z303.evaluate_config(
            mep7, ns4d, f"mario_bf{Bf}",
            Bf=float(Bf), Va=40.0, Is=1e-16,
            Nf=0.9, Ne=1.5, Var=10.0,
            all_vg1=all_vg1, all_vg2=all_vg2, all_vd=all_vd,
            curve_meta=curve_meta, seg_idx=seg_idx,
        )
        med_all = r["median_fwd_log_rmse_all"]
        signed = r["median_signed_dec_all"]
        v02 = r["by_vg1"].get(0.2, {}).get("median_fwd_log_rmse", None)
        v04 = r["by_vg1"].get(0.4, {}).get("median_fwd_log_rmse", None)
        v06 = r["by_vg1"].get(0.6, {}).get("median_fwd_log_rmse", None)
        elapsed = time.time() - t1
        print(f"  median_all={med_all:.3f} signed={signed:+.3f} "
              f"VG1=0.2:{v02:.3f}  0.4:{v04:.3f}  0.6:{v06:.3f}  ({elapsed:.1f}s)")
        rows.append({
            "Bf": Bf,
            "median_fwd_log_rmse_all": med_all,
            "median_signed_dec_all": signed,
            "vg1_02_median": v02,
            "vg1_04_median": v04,
            "vg1_06_median": v06,
            "n_curves": r["n_curves"],
        })

    # Find basin minimum
    best = min(rows, key=lambda x: x["median_fwd_log_rmse_all"])

    # Pareto-optimal: minimize (vg1_02, vg1_06) jointly. Non-dominated set.
    pareto = []
    for i, ri in enumerate(rows):
        dominated = False
        for j, rj in enumerate(rows):
            if i == j: continue
            if (rj["vg1_02_median"] <= ri["vg1_02_median"] and
                rj["vg1_06_median"] <= ri["vg1_06_median"] and
                (rj["vg1_02_median"] < ri["vg1_02_median"] or
                 rj["vg1_06_median"] < ri["vg1_06_median"])):
                dominated = True; break
        if not dominated:
            pareto.append(ri["Bf"])

    # Gate evaluation on best (overall median)
    gates = {
        "PASS_conservative": best["median_fwd_log_rmse_all"] < 0.95,
        "AMBITIOUS": (best["median_fwd_log_rmse_all"] < 0.5 and
                       best["vg1_02_median"] is not None and
                       best["vg1_02_median"] < 3.0),
        "SAFETY":  best["vg1_06_median"] is not None and
                    best["vg1_06_median"] <= 1.02,
    }

    summary = {
        "script": "z303b_mario_bf_sweep",
        "elapsed_s": time.time() - t0,
        "topology": {"Vaf": 40, "Nf": 0.9, "Ne": 1.5, "Var": 10, "Is": 1e-16},
        "bf_grid": BF_GRID,
        "rows": rows,
        "best_overall": best,
        "pareto_bf": pareto,
        "da3_reference": {"median_all": DA3_MEDIAN, "safety_ceiling_vg06": 1.02},
        "gates_on_best": gates,
    }
    out_path = OUT / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[z303b] wrote {out_path}")
    print(f"[z303b] basin min: Bf={best['Bf']} median={best['median_fwd_log_rmse_all']:.3f}")
    print(f"[z303b] pareto Bf: {pareto}")
    print(f"[z303b] gates: {gates}")


if __name__ == "__main__":
    main()
