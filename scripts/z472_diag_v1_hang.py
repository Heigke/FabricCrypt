"""z472 — Diagnostic for z461 V1 hang on calibrated cell.

Runs the V1 (DC IV per-branch) inner loop with per-bias timing and PT-step
logging, plus a 'budget' wallclock cap per bias. Identifies WHICH (VG1,VG2,Vd)
points hit the budget without converging.

Output: results/z472_v1_fix/diag_hang.json
"""
from __future__ import annotations
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

import importlib.util as _ilu
def _load(name, path):
    sp = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m
z427 = _load("z427", ROOT / "scripts/z427_vsint_fix.py")
z429 = _load("z429", ROOT / "scripts/z429_multisolver_debug.py")

# Inline z461 config builders to avoid loading the dataclass-bound z461 module
V449B_BASE = {
    "use_vbic_for_q1": True,
    "vbic_AVC1": 0.5, "vbic_AVC2": 0.5,
    "Cbody": 1e-15,
    "body_pdiode_Cj0_per_area": 0.0,
}
SNAP_DEFAULT = dict(
    snap_BV=2.0, snap_n_avl=4.0, snap_Bf=417.0, snap_Va=0.90,
    snap_Is=6.0256e-9, snap_Nf=1.0,
    snap_Id_clamp=1e-2, snap_Iii_clamp=1e-2,
)
SNAP_HOT = dict(SNAP_DEFAULT)
SNAP_HOT["snap_BV"] = 2.0 * 0.6
SNAP_HOT["snap_Is"] = 4.5192e-12
SNAP_HOT["snap_Id_clamp"] = 1e-1
SNAP_HOT["snap_Iii_clamp"] = 1e-1

def make_config(name: str) -> dict:
    if name == "SB_OFF":
        return {**V449B_BASE, "use_snapback_sub": False}
    if name == "SB_HOT":
        return {**V449B_BASE, "use_snapback_sub": True, **SNAP_HOT}
    if name == "NX_1p8":
        return {**V449B_BASE, "use_snapback_sub": True, **SNAP_HOT,
                "snap_use_knee_gate": True,
                "snap_V_knee": 1.6, "snap_V_sharp": 0.05,
                "snap_npn_gate_mode": "current",
                "snap_npn_V_knee": 1.8, "snap_npn_V_sharp": 0.05,
                "snap_npn_V_BE_offset": 0.3}
    raise ValueError(name)


def main():
    out_dir = ROOT / "results/z472_v1_fix"
    out_dir.mkdir(parents=True, exist_ok=True)
    config_name = os.environ.get("Z472_CFG", "NX_1p8")
    per_bias_budget = float(os.environ.get("Z472_BUDGET", "5.0"))  # seconds per bias
    sharp_override = os.environ.get("Z472_NPN_SHARP", "")  # set "0.1", "0.25"

    cfg_flags = make_config(config_name)
    if sharp_override:
        cfg_flags["snap_npn_V_sharp"] = float(sharp_override)
        cfg_flags["snap_V_sharp"] = float(sharp_override)

    model_M1, model_M2 = z427.build_models()
    curves = z427.load_curves()
    sebas_rows = z427.load_sebas_params()
    print(f"[diag] config={config_name} V_sharp_override={sharp_override or 'default(0.05)'}", flush=True)
    print(f"[diag] {len(curves)} curves, per-bias budget={per_bias_budget}s", flush=True)

    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(cfg_flags))

    records = []
    panels = {0.2: 0, 0.4: 0, 0.6: 0}
    n_done = 0
    t0_all = time.time()
    for c in curves:
        if c["VG1"] not in panels:
            continue
        sebas_row = z427.find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        P_M1, P_M2 = z427.make_overrides(sebas_row)
        bjt = z427.make_bjt(sebas_row)
        Vd_arr = c["Vd"].numpy()
        Id_meas = c["Id"].numpy()
        order = np.argsort(Vd_arr)
        Vd_seq = Vd_arr[order]
        Id_meas_seq = Id_meas[order]
        rec_points = []
        curve_hang = False
        t_curve = time.time()
        try:
            with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), z427.patch_sd_scaled(sd_M2, P_M2):
                Vb_warm = 0.0
                for i, Vd_f in enumerate(Vd_seq):
                    t_bias = time.time()
                    r = z429.run_vsint_pinned(
                        cfg, model_M1, model_M2, bjt,
                        float(Vd_f), float(c["VG1"]), float(c["VG2"]),
                        Vsint_pin=0.0, Vb_init=Vb_warm)
                    dt = time.time() - t_bias
                    niter = r.get("niter", -1)
                    Id_val = abs(r["Id"]) if r.get("Id") is not None else 0.0
                    converged = bool(r["converged"])
                    if r["converged"]:
                        Vb_warm = r["Vb"]
                    else:
                        Vb_warm = 0.0
                    rec_points.append({
                        "Vd": float(Vd_f),
                        "Id_pred": float(Id_val),
                        "Id_meas": float(Id_meas_seq[i]),
                        "Vb": float(r["Vb"]),
                        "niter": int(niter),
                        "converged": converged,
                        "resid_RB": float(r["resid_RB"]),
                        "t_sec": dt,
                    })
                    if dt > per_bias_budget:
                        curve_hang = True
                        print(f"[diag][HANG] VG1={c['VG1']} VG2={c['VG2']:+.2f} "
                              f"Vd={Vd_f:.3f} t={dt:.2f}s niter={niter} "
                              f"conv={converged} Vb={r['Vb']:.4f} R_B={r['resid_RB']:.3g}",
                              flush=True)
        except Exception as e:
            print(f"[diag][EXC] VG1={c['VG1']} VG2={c['VG2']:+.2f}: {e}", flush=True)
        t_curve_total = time.time() - t_curve
        worst_t = max((p["t_sec"] for p in rec_points), default=0.0)
        n_unconv = sum(1 for p in rec_points if not p["converged"])
        print(f"[diag] VG1={c['VG1']} VG2={c['VG2']:+.2f}  curve_t={t_curve_total:.1f}s "
              f"worst_bias={worst_t:.2f}s n_unconv={n_unconv}/{len(rec_points)} hang={curve_hang}",
              flush=True)
        records.append({
            "VG1": float(c["VG1"]), "VG2": float(c["VG2"]),
            "n_pts": len(rec_points),
            "curve_t_sec": t_curve_total,
            "worst_bias_t_sec": worst_t,
            "n_unconverged": n_unconv,
            "hang_detected": curve_hang,
            "points": rec_points,
        })
        n_done += 1

    t_all = time.time() - t0_all
    summary = {
        "config": config_name,
        "npn_V_sharp_override": sharp_override or "0.05_default",
        "per_bias_budget_s": per_bias_budget,
        "wall_sec": t_all,
        "n_curves": n_done,
        "n_curves_with_hang": sum(1 for r in records if r["hang_detected"]),
        "curves": records,
    }
    out_path = out_dir / f"diag_hang_{config_name}_sharp{sharp_override or 'def'}.json"
    out_path.write_text(json.dumps(summary, indent=1, default=float))
    print(f"[diag] wrote {out_path}  total_wall={t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
