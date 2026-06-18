"""R-6-lite z323 — Extended sweep beyond R-4b (z321).

R-4b swept Bf × Js × R_body.
R-1b found K1 LUT and mbjt step are *also* tunable.
This script adds those dimensions:

  knob 1: K1_LUT_scale ∈ {0.5, 0.8, 1.0, 1.2, 1.5}
          multiplier on Sebas's per-V_G1 LUT [0.55825, 0.53825, 0.41825]
  knob 2: mbjt_step_position ∈ {0.25, 0.30, 0.35, 0.40} V
          V_G1 below which mbjt=0.001 (M2/BJT off), at-or-above mbjt=1.0
  knob 3: Bf ∈ {100, 500, 3000}     (narrowed; R-4b covered wider)

  = 5 × 4 × 3 = 60 cells

Fixed (per task spec):
  Cb = 7 fF
  Adiode = 22 um^2
  ALPHA0 = 7.842e-5 (constant)
  body_pdiode_Js = PDIODE_JS_SEBAS = 2.44e4 A/m^2  (z321 default)
  R_body per-V_G1 (z321 table) — kept fixed; R-4b is sweeping it
  avalanche/Chynoweth lateral collector DISABLED (lat_BV high)
  TAT oracle values retained

Locked gates:
  PASS:       cell-wide median < 1.0 dec
  AMBITIOUS:  < 0.7 dec
  SAFETY:     V_G1=0.2 median < 1.5 dec  (vs. 4.7 catastrophe)

This script supports `--k1_scales` to split the 60-cell grid across multiple
queue submissions (one K1 scale per job → 12 cells, ~70 min wall).
After all sub-jobs complete, run with --aggregate to fuse them into
`results/z323_v5_extended/summary.json`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

# Resolve repo root from this script's location so it works on any worker host.
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data/sebas_2026_04_22"
OUT_DIR = ROOT / "results/z323_v5_extended"

# Node gate: this job requires nsram/ + data/sebas_2026_04_22/ (full repo).
# zgx workers have HAS_REPO=0; abort cleanly so the worker marks failed and we
# resubmit from master rather than wedging an inf-only partial file.
_NSRAM_DIR = ROOT / "nsram"
_DATA_DIR = DATA
if not _NSRAM_DIR.is_dir() or not _DATA_DIR.is_dir():
    import socket
    print(f"[z323] FATAL: missing nsram/ or data/sebas_2026_04_22/ at "
          f"{ROOT} (host={socket.gethostname()}). "
          f"This job requires the full repo (ikaros or daedalus).",
          flush=True)
    sys.exit(2)

OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "nsram"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

# ---- Fixed constants (per task spec) ----
ALPHA0_CONST = 7.842e-5
CBODY_FF = 7e-15
PDIODE_AREA = 2.2e-11
PDIODE_N = 1.0535
PDIODE_IS_TOTAL = 5.3675e-7
PDIODE_JS_SEBAS = PDIODE_IS_TOTAL / PDIODE_AREA   # 2.44e4 A/m^2

R_BODY_TABLE = {0.2: 1.0e10, 0.4: 1.0e9, 0.6: 1.0e8}

TAT_JTSS = 3.4e-7
TAT_NJTS = 20.0
TAT_VTSS = 10.0
TAT_XTSS = 0.02

LAT_BV_DISABLED = 1.0e6

# ---- Knob grids ----
K1_SCALE_GRID = [0.5, 0.8, 1.0, 1.2, 1.5]
MBJT_STEP_GRID = [0.25, 0.30, 0.35, 0.40]      # V_G1 threshold
BF_GRID = [100, 500, 3000]

# Gates (task spec)
GATE_PASS_DEC = 1.0
GATE_AMBITIOUS_DEC = 0.7
GATE_SAFETY_VG1_02_DEC = 1.5


def _load_module(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp)
    sys.modules[name] = mod
    sp.loader.exec_module(mod)
    return mod


def configure_v5_fixed(cfg, vg1):
    """Apply z321-style fixed config (avalanche off, R_body per V_G1)."""
    cfg.use_well_diode = False
    cfg.body_pdiode_to = "vnwell"
    if hasattr(cfg, "z310_enable_vnwell_diode"):
        cfg.z310_enable_vnwell_diode = False

    cfg.body_pdiode_area = PDIODE_AREA
    cfg.body_pdiode_n = PDIODE_N
    cfg.body_pdiode_Js = PDIODE_JS_SEBAS

    cfg.Cbody = CBODY_FF

    cfg.body_pdiode_Rs = float(R_BODY_TABLE.get(round(vg1, 2), 1.0e9))
    cfg.vnwell_Rs = 1.0e30

    cfg.use_lateral_collector = False
    cfg.lat_BV = LAT_BV_DISABLED
    cfg.lat_BV_max = LAT_BV_DISABLED * 1.1
    cfg.lat_N = 4.0
    cfg.lat_M_smooth_delta = 0.5

    cfg.enable_tat = True
    cfg.tat_jtss = TAT_JTSS
    cfg.tat_njts = TAT_NJTS
    cfg.tat_vtss = TAT_VTSS
    cfg.tat_xtss = TAT_XTSS

    if hasattr(cfg, "z313_enable_tat"):
        cfg.z313_enable_tat = False

    if hasattr(cfg, "invalidate"):
        cfg.invalidate()


def evaluate_cell_with_knobs(*, vg1, bf, k1_scale, mbjt_step,
                              curves, sebas_rows, z91f_mod,
                              cfg, M1, M2, sd_M1, sd_M2, forward_2t):
    """One (V_G1, Bf, K1_scale, mbjt_step) cell over all curves of that branch.

    Differs from z304.evaluate_cell:
      - K1 is taken from the per-V_G1 LUT and multiplied by k1_scale.
      - mbjt is set per row using the step function:
            mbjt = 0.001 if VG1 < mbjt_step else 1.0
        (overriding the CSV mbjt column)
      - alpha0 is ALPHA0_CONST (per task spec)
    """
    from nsram.bsim4_port.bjt import GummelPoonNPN
    # patch_sd_scaled inlined to avoid re-importing z304's contextmanager
    z304 = sys.modules["z304"]
    patch_sd_scaled = z304.patch_sd_scaled
    find_params = z304.find_params

    # K1 LUT (Sebas, per V_G1) — scale by knob
    K1_LUT = {0.2: 0.55825, 0.4: 0.53825, 0.6: 0.41825}
    k1_for_branch = K1_LUT[round(vg1, 2)] * float(k1_scale)

    # mbjt step (override CSV)
    mbjt_for_branch = 0.001 if vg1 < mbjt_step else 1.0

    log_eps = 1e-15
    per_curve = []
    for c in curves:
        sebas_row = find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None:
            continue
        # Build per-row overrides honoring the knobs
        P_M1 = {}
        if not math.isnan(sebas_row.get("ETAB", float("nan"))):
            P_M1["etab"] = torch.tensor(sebas_row["ETAB"], dtype=DTYPE)
        P_M1["k1"] = torch.tensor(k1_for_branch, dtype=DTYPE)
        P_M1["alpha0"] = torch.tensor(float(ALPHA0_CONST), dtype=DTYPE)
        if not math.isnan(sebas_row.get("BETA0", float("nan"))):
            P_M1["beta0"] = torch.tensor(sebas_row["BETA0"], dtype=DTYPE)
        P_M2 = {}
        if not math.isnan(sebas_row.get("NFACTOR", float("nan"))):
            P_M2["nfactor"] = torch.tensor(sebas_row["NFACTOR"], dtype=DTYPE)
        for k, v in z91f_mod.M2_STATIC_OVERRIDES.items():
            if k not in P_M2:
                P_M2[k] = torch.tensor(float(v), dtype=DTYPE)

        # BJT per row, with mbjt_step override
        bjt = GummelPoonNPN.from_sebas_card()
        if not math.isnan(sebas_row.get("IS", float("nan"))):
            bjt.Is = float(sebas_row["IS"])
        area = float(sebas_row.get("area", 1e-6))
        if math.isnan(area):
            area = 1e-6
        bjt.area = area * mbjt_for_branch
        bjt.Bf = float(bf)

        try:
            with torch.no_grad(), \
                  patch_sd_scaled(sd_M1, P_M1), \
                  patch_sd_scaled(sd_M2, P_M2):
                out = forward_2t(cfg, M1, bjt,
                                  c["Vd"], torch.tensor(c["VG1"]),
                                  torch.tensor(c["VG2"]),
                                  warm_start=True, use_homotopy=True)
            Id_pred = out["Id"].abs()
            conv = torch.tensor([bool(x) for x in out["converged"]])
        except Exception as e:
            per_curve.append({"VG2": float(c["VG2"]), "log_rmse": float("inf"),
                              "signed_dec": float("nan"), "n_conv": 0,
                              "err": str(e)[:120]})
            continue

        log_p = torch.log10(Id_pred + log_eps)
        log_m = torch.log10(c["Id"] + log_eps)
        if conv.any():
            mask = conv
            diff = (log_p[mask] - log_m[mask])
            rmse = float(torch.sqrt((diff ** 2).mean()))
            signed = float(torch.median(diff))
        else:
            rmse = float("inf")
            signed = float("nan")
        per_curve.append({"VG2": float(c["VG2"]), "log_rmse": rmse,
                          "signed_dec": signed, "n_conv": int(conv.sum())})

    finite = [pc for pc in per_curve if math.isfinite(pc["log_rmse"])]
    if finite:
        rmses = np.array([pc["log_rmse"] for pc in finite])
        signs = np.array([pc["signed_dec"] for pc in finite
                          if math.isfinite(pc["signed_dec"])])
        med = float(np.median(rmses))
        signed_med = float(np.median(signs)) if signs.size else float("nan")
        p90 = float(np.percentile(rmses, 90))
    else:
        med = float("inf")
        signed_med = float("nan")
        p90 = float("inf")
    return {
        "vg1": vg1, "bf": bf, "k1_scale": k1_scale, "mbjt_step": mbjt_step,
        "k1_used": k1_for_branch, "mbjt_used": mbjt_for_branch,
        "median_log_rmse": med, "signed_dec_median": signed_med,
        "p90_log_rmse": p90,
        "n_finite": len(finite), "n_total": len(per_curve),
    }


def run_sweep(k1_scales, mbjt_steps, bf_grid, out_path):
    """Run the (k1_scales × mbjt_steps × bf_grid) sub-grid sequentially."""
    t0 = time.time()
    print(f"[z323] device={DEVICE} k1_scales={k1_scales} "
          f"mbjt_steps={mbjt_steps} bf={bf_grid}", flush=True)

    z304 = _load_module("z304", SCRIPTS / "z304_sebas_three_branch_refit.py")
    z91f_built, cfg, M1, M2, sd_M1, sd_M2, forward_2t = z304.build_models_once()
    sebas_rows = z304.load_sebas_params()
    print(f"[z323] models built ({time.time()-t0:.1f}s)", flush=True)

    curves_by_vg1 = {vg1: z304.load_curves(vg1_filter=vg1) for vg1 in [0.2, 0.4, 0.6]}

    cells = {}
    n_total = len(k1_scales) * len(mbjt_steps) * len(bf_grid)
    cell_i = 0
    t_last_snap = time.time()

    for k1s in k1_scales:
        for mstep in mbjt_steps:
            for bf in bf_grid:
                cell_i += 1
                cell_key = f"k1s_{k1s}__mstep_{mstep}__bf_{bf}"
                print(f"\n=== [z323] cell {cell_i}/{n_total} {cell_key} ===",
                      flush=True)
                per_branch = {}
                all_rmses = []
                cell_failed = False
                for vg1 in [0.2, 0.4, 0.6]:
                    configure_v5_fixed(cfg, vg1)
                    curves = curves_by_vg1[vg1]
                    t_branch = time.time()
                    try:
                        r = evaluate_cell_with_knobs(
                            vg1=vg1, bf=bf, k1_scale=k1s, mbjt_step=mstep,
                            curves=curves, sebas_rows=sebas_rows,
                            z91f_mod=z91f_built, cfg=cfg,
                            M1=M1, M2=M2, sd_M1=sd_M1, sd_M2=sd_M2,
                            forward_2t=forward_2t,
                        )
                    except Exception as e:
                        print(f"[z323/{cell_key}] V_G1={vg1} EXC: "
                              f"{str(e)[:120]}", flush=True)
                        cell_failed = True
                        break
                    per_branch[str(vg1)] = {
                        "median_log_rmse": r["median_log_rmse"],
                        "signed_dec_median": r["signed_dec_median"],
                        "p90_log_rmse": r["p90_log_rmse"],
                        "n_finite": r["n_finite"], "n_total": r["n_total"],
                        "k1_used": r["k1_used"], "mbjt_used": r["mbjt_used"],
                    }
                    # Aggregate per-curve rmses for cell-wide median
                    # (use branch median * branch count -- but z321 used
                    # per_curve list; we approximate by repeating med across
                    # n_finite, which equals z321's flatten behavior IFF we
                    # had per_curve. Simpler: use branch medians directly.)
                    all_rmses.append(r["median_log_rmse"])

                cw = float(np.median(all_rmses)) if all_rmses else float("inf")
                cells[cell_key] = {
                    "bf": bf, "k1_scale": k1s, "mbjt_step": mstep,
                    "cell_wide_median_log_rmse": cw,
                    "per_branch": per_branch,
                    "failed": cell_failed,
                    "gate_PASS_lt_1_0": bool(cw < GATE_PASS_DEC),
                    "gate_AMBITIOUS_lt_0_7": bool(cw < GATE_AMBITIOUS_DEC),
                    "gate_SAFETY_vg1_0_2_lt_1_5": bool(
                        per_branch.get("0.2", {}).get("median_log_rmse",
                                                       math.inf)
                        < GATE_SAFETY_VG1_02_DEC),
                }
                print(f"[z323/{cell_key}] cw={cw:.3f} "
                      f"vg1=0.2:{per_branch.get('0.2',{}).get('median_log_rmse',float('nan')):.3f} "
                      f"vg1=0.4:{per_branch.get('0.4',{}).get('median_log_rmse',float('nan')):.3f} "
                      f"vg1=0.6:{per_branch.get('0.6',{}).get('median_log_rmse',float('nan')):.3f} "
                      f"({time.time()-t0:.0f}s total)", flush=True)

                # Periodic snapshot
                if time.time() - t_last_snap > 600:
                    out_path.write_text(json.dumps(
                        {"partial": True, "cells_done": cell_i,
                         "n_total": n_total, "cells": cells},
                        indent=2, default=float))
                    t_last_snap = time.time()

    # Best cell within this sub-grid
    valid = {k: v for k, v in cells.items()
             if math.isfinite(v["cell_wide_median_log_rmse"])}
    best_key = min(valid,
                   key=lambda k: valid[k]["cell_wide_median_log_rmse"]) \
        if valid else None

    summary = {
        "script": "z323_v5_extended_sweep",
        "elapsed_s": time.time() - t0,
        "device": str(DEVICE),
        "config": {
            "ALPHA0_CONST": ALPHA0_CONST,
            "CBODY_FF": CBODY_FF,
            "PDIODE_AREA": PDIODE_AREA,
            "PDIODE_JS": PDIODE_JS_SEBAS,
            "R_BODY_TABLE": R_BODY_TABLE,
            "K1_LUT_baseline": {0.2: 0.55825, 0.4: 0.53825, 0.6: 0.41825},
            "use_lateral_collector": False,
            "lat_BV_disabled": LAT_BV_DISABLED,
        },
        "grids": {"k1_scales": k1_scales, "mbjt_steps": mbjt_steps,
                  "bf": bf_grid},
        "gates": {"PASS_lt": GATE_PASS_DEC,
                  "AMBITIOUS_lt": GATE_AMBITIOUS_DEC,
                  "SAFETY_vg1_0_2_lt": GATE_SAFETY_VG1_02_DEC},
        "cells": cells,
        "best_cell": best_key,
        "best_cell_cw": (cells[best_key]["cell_wide_median_log_rmse"]
                         if best_key else None),
    }
    out_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[z323] wrote {out_path}  ({time.time()-t0:.0f}s)", flush=True)
    if best_key:
        print(f"[z323] best={best_key} cw="
              f"{cells[best_key]['cell_wide_median_log_rmse']:.3f}", flush=True)
    return 0


def aggregate(out_dir: Path, final_path: Path):
    """Fuse partial summaries (one per K1 scale) into one summary.json."""
    cells = {}
    sub_files = sorted(out_dir.glob("partial_*.json"))
    if not sub_files:
        print(f"[z323/agg] no partials in {out_dir}", flush=True)
        return 1
    elapsed_total = 0.0
    grids = {"k1_scales": set(), "mbjt_steps": set(), "bf": set()}
    config = None
    for f in sub_files:
        d = json.loads(f.read_text())
        cells.update(d.get("cells", {}))
        elapsed_total += float(d.get("elapsed_s", 0.0))
        for k in grids:
            for v in d.get("grids", {}).get(k, []):
                grids[k].add(v)
        if config is None:
            config = d.get("config")
    grids = {k: sorted(v) for k, v in grids.items()}

    valid = {k: v for k, v in cells.items()
             if math.isfinite(v["cell_wide_median_log_rmse"])}
    best_key = min(valid,
                   key=lambda k: valid[k]["cell_wide_median_log_rmse"]) \
        if valid else None
    summary = {
        "script": "z323_v5_extended_sweep",
        "aggregated_from": [f.name for f in sub_files],
        "elapsed_s_sum": elapsed_total,
        "config": config,
        "grids": grids,
        "gates": {"PASS_lt": GATE_PASS_DEC,
                  "AMBITIOUS_lt": GATE_AMBITIOUS_DEC,
                  "SAFETY_vg1_0_2_lt": GATE_SAFETY_VG1_02_DEC},
        "cells": cells,
        "n_cells_total": len(cells),
        "n_cells_valid": len(valid),
        "best_cell": best_key,
        "best_cell_cw": (cells[best_key]["cell_wide_median_log_rmse"]
                         if best_key else None),
        "best_per_branch": (cells[best_key]["per_branch"]
                            if best_key else None),
    }
    if best_key:
        b = cells[best_key]
        summary["best_gates"] = {
            "PASS_lt_1_0": b["gate_PASS_lt_1_0"],
            "AMBITIOUS_lt_0_7": b["gate_AMBITIOUS_lt_0_7"],
            "SAFETY_vg1_0_2_lt_1_5": b["gate_SAFETY_vg1_0_2_lt_1_5"],
        }
    final_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"[z323/agg] wrote {final_path}  "
          f"({len(cells)} cells, best={best_key})", flush=True)
    return 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k1_scales", type=float, nargs="+", default=None,
                   help="Subset of K1 scales to run. Default = all 5.")
    p.add_argument("--mbjt_steps", type=float, nargs="+", default=None)
    p.add_argument("--bf", type=int, nargs="+", default=None)
    p.add_argument("--out_tag", type=str, default=None,
                   help="Tag for partial output filename. "
                        "Default derived from k1_scales.")
    p.add_argument("--aggregate", action="store_true",
                   help="Merge results/z323_v5_extended/partial_*.json "
                        "into summary.json and exit.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.aggregate:
        return aggregate(OUT_DIR, OUT_DIR / "summary.json")

    k1s = args.k1_scales if args.k1_scales is not None else K1_SCALE_GRID
    ms = args.mbjt_steps if args.mbjt_steps is not None else MBJT_STEP_GRID
    bfs = args.bf if args.bf is not None else BF_GRID

    tag = args.out_tag
    if tag is None:
        tag = "k1s_" + "_".join(f"{x:g}" for x in k1s)
    out_path = OUT_DIR / f"partial_{tag}.json"
    return run_sweep(k1s, ms, bfs, out_path)


if __name__ == "__main__":
    sys.exit(main())
