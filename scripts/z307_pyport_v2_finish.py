"""z307 follow-up: finish the ablation by running drain_aval_only and full_v2
on a TIGHT grid (best per-branch params only) since baseline + breakdown_only
were already shown to be IDENTICAL with initial v2 params (vnwell breakdown
threshold Vbr=2.5 doesn't fire when vnwell=2V and Vb<1V).

This script writes results/z307_pyport_v2/summary.json directly.
"""
from __future__ import annotations
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import importlib.util, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

def _load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m

z304 = _load("z304", ROOT / "scripts/z304_sebas_three_branch_refit.py")
from nsram_pyport_v2 import V2Params, enable_v2_topology, disable_v2_topology

OUT_DIR = ROOT / "results/z307_pyport_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Tight grid: per-branch best from z304 baseline
PER_BRANCH_BEST = {
    0.2: dict(bf=500,  alpha0=1e-5, rs=0),
    0.4: dict(bf=50,   alpha0=1e-5, rs=1e10),
    0.6: dict(bf=9000, alpha0=1e-5, rs=1e10),
}

# Conditions: baseline + 3 v2 variants
def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[z307b] device={device}", flush=True)
    z91f, cfg, M1, M2, sd_M1, sd_M2, forward_2t = z304.build_models_once()
    sebas_rows = z304.load_sebas_params()
    curves = {vg1: z304.load_curves(vg1_filter=vg1) for vg1 in PER_BRANCH_BEST}

    v2p_full = V2Params()
    v2p_brk = V2Params(enable_drain_avalanche=False, enable_vnwell_forward_anode_vb=False)
    v2p_drn = V2Params(enable_vnwell_breakdown=False, enable_vnwell_forward_anode_vb=False)
    conditions = [
        ("baseline", None),
        ("breakdown_only", v2p_brk),
        ("drain_aval_only", v2p_drn),
        ("full_v2", v2p_full),
    ]

    all_rows = {}
    for cond_name, v2p in conditions:
        print(f"[z307b] === {cond_name} ===", flush=True)
        if v2p is None:
            disable_v2_topology(cfg)
        else:
            enable_v2_topology(cfg, v2p)
        rows = []
        for vg1, p in PER_BRANCH_BEST.items():
            tc = time.time()
            r = z304.evaluate_cell(
                vg1=vg1, bf=p["bf"], alpha0=p["alpha0"], rs=p["rs"],
                curves=curves[vg1], sebas_rows=sebas_rows,
                z91f_mod=z91f, cfg=cfg, M1=M1, M2=M2,
                sd_M1=sd_M1, sd_M2=sd_M2, forward_2t=forward_2t,
            )
            r["condition"] = cond_name
            rows.append(r)
            print(f"[z307b] {cond_name} vg1={vg1}: med={r['median_log_rmse']:.4f} "
                  f"signed={r['signed_dec_median']:+.3f} ({time.time()-tc:.1f}s)",
                  flush=True)
        all_rows[cond_name] = rows

    # Aggregate
    cond_summary = {}
    for cond, rows in all_rows.items():
        per_branch = {}
        all_rmses = []
        for r in rows:
            per_branch[r["vg1"]] = {
                "median_log_rmse": r["median_log_rmse"],
                "signed_dec_median": r["signed_dec_median"],
                "p90_log_rmse": r["p90_log_rmse"],
                "n_finite": r["n_finite"], "n_total": r["n_total"],
                "params": {"bf": r["bf"], "alpha0": r["alpha0"], "rs": r["rs"]},
            }
            for pc in r["per_curve"]:
                if math.isfinite(pc["log_rmse"]):
                    all_rmses.append(pc["log_rmse"])
        cell_med = float(np.median(all_rmses)) if all_rmses else float("inf")
        cond_summary[cond] = {"cell_wide_median_log_rmse": cell_med,
                                "per_vg1_best": per_branch}

    full = cond_summary["full_v2"]
    vg1_breakdown = {v: d["median_log_rmse"]
                      for v, d in full["per_vg1_best"].items()}
    verdict = {
        "pass_conservative": bool(full["cell_wide_median_log_rmse"] < 0.5),
        "ambitious":         bool(full["cell_wide_median_log_rmse"] < 0.3),
        "safety_vg1_06":     bool(vg1_breakdown.get(0.6, float("inf")) <= 0.73),
    }

    summary = {
        "script": "z307_pyport_v2 (finish; tight grid at per-branch best)",
        "v2_params_full": {
            "vnwell_Vbr": v2p_full.vnwell_Vbr, "vnwell_Iav0": v2p_full.vnwell_Iav0,
            "vnwell_n_av": v2p_full.vnwell_n_av,
            "vnwell_Is_anode_vb": v2p_full.vnwell_Is_anode_vb,
            "Cgs_inj_fF": v2p_full.Cgs_inj_fF,
            "drain_BV": v2p_full.drain_BV, "drain_N": v2p_full.drain_N,
        },
        "per_branch_best_params": PER_BRANCH_BEST,
        "conditions": list(all_rows.keys()),
        "per_condition": cond_summary,
        "cell_wide_median_log_rmse": full["cell_wide_median_log_rmse"],
        "per_vg1_best": full["per_vg1_best"],
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
        "device": str(device),
        "rows": {c: r for c, r in all_rows.items()},
    }
    out_path = OUT_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[z307b] wrote {out_path}", flush=True)
    print(f"[z307b] cell-wide median (full_v2) = {full['cell_wide_median_log_rmse']:.4f} dec",
          flush=True)
    for cond, cs in cond_summary.items():
        per_vg1 = ", ".join(f"vg1={v}:{d['median_log_rmse']:.3f}"
                              for v, d in cs["per_vg1_best"].items())
        print(f"[z307b]   {cond}: median={cs['cell_wide_median_log_rmse']:.4f}  ({per_vg1})",
              flush=True)
    print(f"[z307b] verdict: {verdict}", flush=True)


if __name__ == "__main__":
    main()
