"""z307 — pyport v2 topology refit on Sebas 33 IV curves.

Activates the three missing topology elements SA3 identified
(vnwell→Vb FULL diode with reverse-breakdown / avalanche, VB-VG2
coupling cap [transient-only], explicit Vb readout) plus drain-end
avalanche M(V_bc) coupled to channel, then re-evaluates against
Sebas's 33-curve 2Tcell_BSIM_param_DC IV set.

Initial v2 params per cron pre-reg:
  vnwell breakdown:  Vbr=2.5 V, Iav0=1e-7 A, n_av=2
  vnwell anode-vb fwd: Is=1e-14 A, n=1.0
  drain avalanche:   BV=6 V, N=4
  VB-VG2 cap:        Cgs_inj = 0.5 fF (transient-only; recorded only)

Locked gate:
  PASS-conservative : cell-wide median forward log-RMSE < 0.5 dec
  AMBITIOUS         : < 0.3 dec
  SAFETY            : V_G1=0.6 branch ≤ 0.73 dec (vs prior best 0.70)

Sweeps a small (Bf, alpha0, Rs) grid per branch with v2 enabled to
verify the new physics is not catastrophically off; reports best cell
per branch and cell-wide median.

Output: results/z307_pyport_v2/summary.json
"""
from __future__ import annotations
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse z304 helpers
def _load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m

z304 = _load("z304", ROOT / "scripts/z304_sebas_three_branch_refit.py")
from nsram_pyport_v2 import V2Params, enable_v2_topology, disable_v2_topology   # noqa: E402

OUT_DIR = ROOT / "results/z307_pyport_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Smaller sweep grid than z304 — we're testing v2 physics, not re-sweeping.
# Picks z304's best per-branch (Bf, Rs) and a tight alpha0 grid.
BF_GRID = [500, 3000, 9000]
ALPHA0_GRID = [1e-5, 1e-4, 1e-3]
RS_GRID = [1.0e9, 1.0e10]


def gate_verdict(cell_median, vg1_breakdown):
    verdict = {}
    verdict["pass_conservative"] = bool(cell_median < 0.5)
    verdict["ambitious"]        = bool(cell_median < 0.3)
    vg1_06 = vg1_breakdown.get(0.6, float("inf"))
    verdict["safety_vg1_06"]    = bool(vg1_06 <= 0.73)
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="One cell per branch (smoke test).")
    ap.add_argument("--out", type=str, default=str(OUT_DIR / "summary.json"))
    ap.add_argument("--mode", choices=["full", "ablation"], default="full",
                    help="full=v2 enabled only; ablation=4 conditions (baseline, "
                         "breakdown_only, drain_aval_only, full_v2).")
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[z307] device={device} start={time.strftime('%H:%M:%S')}",
          flush=True)

    # Build models (reuses z304 helper which already invokes z91f patches).
    z91f, cfg, M1, M2, sd_M1, sd_M2, forward_2t = z304.build_models_once()
    print(f"[z307] models built  ({time.time()-t0:.1f}s)", flush=True)

    # v2 topology params (applied per-condition below)
    v2p_full = V2Params()
    v2p_breakdown = V2Params(enable_drain_avalanche=False,
                              enable_vnwell_forward_anode_vb=False)
    v2p_drain = V2Params(enable_vnwell_breakdown=False,
                          enable_vnwell_forward_anode_vb=False)
    if args.mode == "full":
        conditions = [("full_v2", v2p_full)]
    else:
        conditions = [
            ("baseline", None),                # no v2
            ("breakdown_only", v2p_breakdown),
            ("drain_aval_only", v2p_drain),
            ("full_v2", v2p_full),
        ]
    print(f"[z307] mode={args.mode} conditions={[c[0] for c in conditions]}",
          flush=True)

    sebas_rows = z304.load_sebas_params()
    vg1_list = [0.2, 0.4, 0.6]
    curves_per_branch = {vg1: z304.load_curves(vg1_filter=vg1) for vg1 in vg1_list}
    for vg1, cs in curves_per_branch.items():
        print(f"[z307] branch V_G1={vg1}: {len(cs)} curves", flush=True)

    if args.quick:
        bf_grid, a0_grid, rs_grid = [3000], [1e-4], [1.0e10]
    else:
        bf_grid, a0_grid, rs_grid = BF_GRID, ALPHA0_GRID, RS_GRID

    all_rows = {}            # cond_name -> rows
    n_cells_per_cond = len(vg1_list) * len(bf_grid) * len(a0_grid) * len(rs_grid)
    for cond_name, v2p in conditions:
        rows = []
        if v2p is None:
            disable_v2_topology(cfg)
            print(f"[z307] === condition: {cond_name} (v2 OFF) ===", flush=True)
        else:
            enable_v2_topology(cfg, v2p)
            print(f"[z307] === condition: {cond_name} ===", flush=True)
            print(f"[z307]   vnwell_brk={v2p.enable_vnwell_breakdown} "
                  f"vnwell_fwd_vb={v2p.enable_vnwell_forward_anode_vb} "
                  f"drain_aval={v2p.enable_drain_avalanche}", flush=True)
        cell_i = 0
        for vg1 in vg1_list:
            curves = curves_per_branch[vg1]
            for bf in bf_grid:
                for a0 in a0_grid:
                    for rs in rs_grid:
                        cell_i += 1
                        tc = time.time()
                        r = z304.evaluate_cell(
                            vg1=vg1, bf=bf, alpha0=a0, rs=rs,
                            curves=curves, sebas_rows=sebas_rows,
                            z91f_mod=z91f, cfg=cfg, M1=M1, M2=M2,
                            sd_M1=sd_M1, sd_M2=sd_M2, forward_2t=forward_2t,
                        )
                        r["condition"] = cond_name
                        rows.append(r)
                        print(f"[z307] {cond_name} {cell_i}/{n_cells_per_cond}: "
                              f"vg1={vg1} bf={bf} a0={a0:.0e} Rs={rs:.0e} → "
                              f"med={r['median_log_rmse']:.3f} "
                              f"signed={r['signed_dec_median']:+.3f} "
                              f"({time.time()-tc:.1f}s, total {time.time()-t0:.0f}s)",
                              flush=True)
        all_rows[cond_name] = rows
    # For backward-compat with downstream aggregation, set `rows` to full_v2.
    rows = all_rows.get("full_v2", all_rows[conditions[-1][0]])
    n_cells = n_cells_per_cond

    # Aggregate per branch & cell-wide
    best_per_branch = {}
    for vg1 in vg1_list:
        rs_ = [r for r in rows if r["vg1"] == vg1 and math.isfinite(r["median_log_rmse"])]
        if not rs_:
            best_per_branch[vg1] = {"median_log_rmse": float("inf"),
                                     "params": None}
            continue
        rs_.sort(key=lambda r: r["median_log_rmse"])
        b = rs_[0]
        best_per_branch[vg1] = {
            "median_log_rmse": b["median_log_rmse"],
            "signed_dec_median": b["signed_dec_median"],
            "p90_log_rmse": b["p90_log_rmse"],
            "n_finite": b["n_finite"], "n_total": b["n_total"],
            "params": {"bf": b["bf"], "alpha0": b["alpha0"], "rs": b["rs"]},
        }

    # Cell-wide median = median across all 33 best per-curve log-RMSEs at
    # each branch's best cell.
    all_curve_rmses = []
    for vg1 in vg1_list:
        rs_ = [r for r in rows if r["vg1"] == vg1
               and math.isfinite(r["median_log_rmse"])]
        if not rs_:
            continue
        b = sorted(rs_, key=lambda r: r["median_log_rmse"])[0]
        for pc in b["per_curve"]:
            if math.isfinite(pc["log_rmse"]):
                all_curve_rmses.append(pc["log_rmse"])
    cell_median = float(np.median(all_curve_rmses)) if all_curve_rmses else float("inf")

    vg1_breakdown = {vg1: best_per_branch[vg1]["median_log_rmse"]
                      for vg1 in vg1_list}
    verdict = gate_verdict(cell_median, vg1_breakdown)

    # Per-condition aggregates
    cond_summary = {}
    for cond_name, cond_rows in all_rows.items():
        per_branch = {}
        all_rmses = []
        for vg1 in vg1_list:
            rs_ = [r for r in cond_rows if r["vg1"] == vg1
                    and math.isfinite(r["median_log_rmse"])]
            if not rs_:
                per_branch[vg1] = {"median_log_rmse": float("inf"),
                                    "params": None}
                continue
            rs_.sort(key=lambda r: r["median_log_rmse"])
            b = rs_[0]
            per_branch[vg1] = {
                "median_log_rmse": b["median_log_rmse"],
                "signed_dec_median": b["signed_dec_median"],
                "p90_log_rmse": b["p90_log_rmse"],
                "params": {"bf": b["bf"], "alpha0": b["alpha0"], "rs": b["rs"]},
            }
            for pc in b["per_curve"]:
                if math.isfinite(pc["log_rmse"]):
                    all_rmses.append(pc["log_rmse"])
        cell_med = float(np.median(all_rmses)) if all_rmses else float("inf")
        cond_summary[cond_name] = {
            "cell_wide_median_log_rmse": cell_med,
            "per_vg1_best": per_branch,
        }

    summary = {
        "script": "z307_pyport_v2",
        "mode": args.mode,
        "v2_params_full": {
            "vnwell_Vbr": v2p_full.vnwell_Vbr, "vnwell_Iav0": v2p_full.vnwell_Iav0,
            "vnwell_n_av": v2p_full.vnwell_n_av,
            "vnwell_Is_anode_vb": v2p_full.vnwell_Is_anode_vb,
            "vnwell_n_anode_vb": v2p_full.vnwell_n_anode_vb,
            "Cgs_inj_fF": v2p_full.Cgs_inj_fF,
            "drain_BV": v2p_full.drain_BV, "drain_N": v2p_full.drain_N,
        },
        "bf_grid": bf_grid, "alpha0_grid": a0_grid, "rs_grid": rs_grid,
        "n_cells_per_cond": n_cells_per_cond, "elapsed_s": time.time() - t0,
        "device": str(device),
        "conditions": list(all_rows.keys()),
        "per_condition": cond_summary,
        # full_v2 condition is the canonical gate
        "cell_wide_median_log_rmse": cell_median,
        "per_vg1_best": best_per_branch,
        "verdict": verdict,
        "rows": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[z307] wrote {out_path}", flush=True)
    print(f"[z307] cell-wide median log-RMSE = {cell_median:.4f} dec", flush=True)
    for vg1 in vg1_list:
        b = best_per_branch[vg1]
        print(f"[z307]   V_G1={vg1}: best median={b['median_log_rmse']:.4f} dec "
              f"(params={b['params']})", flush=True)
    print(f"[z307] verdict: {verdict}", flush=True)
    print(f"[z307] per-condition cell-wide median:", flush=True)
    for c, cs in cond_summary.items():
        per_vg1 = ", ".join(f"vg1={v}:{d['median_log_rmse']:.3f}"
                              for v, d in cs["per_vg1_best"].items())
        print(f"[z307]   {c}: median={cs['cell_wide_median_log_rmse']:.4f}  ({per_vg1})",
              flush=True)


if __name__ == "__main__":
    main()
