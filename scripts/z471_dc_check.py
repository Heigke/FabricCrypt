"""z471 — Fast DC sanity check for calibrated SNAP cell.

Compare DC RMSE for VG1=0.6 column (5-6 curves) between:
  - SB_OFF   (no snapback)
  - SNAP_CAL (snap_Is=4.5192e-12, lifted clamp)
  - SNAP_PRE (snap_Is=3.0128e-8, lifted clamp, pre-tune reference)

Per-Vd timeout: 5 s. Per-curve max 60 s. Total budget ~10 min.
"""
from __future__ import annotations
import json, math, sys, time, importlib.util as _ilu, signal
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"nsram")); sys.path.insert(0, str(ROOT/"scripts"))
OUT = ROOT / "results/z471_snap_calibrate"
LOG = OUT / "dc_check.log"
fh = open(LOG, "w")
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); fh.write(s + "\n"); fh.flush()

_spec = _ilu.spec_from_file_location("z454", ROOT/"scripts/z454_snapback_integration.py")
z454 = _ilu.module_from_spec(_spec); _spec.loader.exec_module(z454)
z449=z454.z449; z427=z454.z427; z429=z454.z429

log("loading models / curves / sebas...")
model_M1, model_M2 = z429.build_models()
curves = z429.load_curves() if hasattr(z429, "load_curves") else None
sebas_rows = z429.load_sebas_params()

# fallback for curves loader
if curves is None:
    # z429 typically exposes via build_models or load_curves; try other names
    for name in ("load_iv_curves", "load_dc_curves", "get_curves", "curves"):
        if hasattr(z429, name):
            v = getattr(z429, name)
            curves = v() if callable(v) else v
            break

if curves is None:
    # Last resort — import from z429 by inspecting source for curves loader
    import inspect
    src = inspect.getsource(z429)
    log("z429 module functions:", [n for n in dir(z429) if not n.startswith("_")][:40])

log(f"curves loaded: {len(curves) if curves is not None else 'NONE'}")

V449B_BASE = {"use_vbic_for_q1": True, "vbic_AVC1": 0.5, "vbic_AVC2": 0.5,
              "Cbody": 1e-15, "body_pdiode_Cj0_per_area": 0.0}
COMMON_AVL = dict(snap_BV=2.0*0.6, snap_n_avl=4.0,
                  snap_Id_clamp=1e-1, snap_Iii_clamp=1e-1,
                  snap_use_knee_gate=True,
                  snap_V_knee=1.6, snap_V_sharp=0.05)
def snap_cfg(snap_Is_val):
    return {**V449B_BASE, "use_snapback_sub": True, "snap_method": "snapback",
            **COMMON_AVL,
            "snap_Bf": 417.0, "snap_Va": 0.90, "snap_Is": float(snap_Is_val),
            "snap_Nf": 1.0,
            "snap_npn_gate_mode": "current",
            "snap_npn_V_knee": 1.8, "snap_npn_V_sharp": 0.05,
            "snap_npn_V_BE_offset": 0.3}

CONFIGS = {
    "SB_OFF":   {**V449B_BASE, "use_snapback_sub": False},
    "SNAP_CAL": snap_cfg(4.5192e-12),
    "SNAP_PRE": snap_cfg(3.0128e-8),
}

# Per-Vd timeout via signal alarm
class TimeOut(Exception): pass
def _ah(signum, frame): raise TimeOut()
signal.signal(signal.SIGALRM, _ah)

def dc_rmse_for_cfg(cfg_flags, curves, VG1_filter=0.6, per_solve_budget_s=4.0):
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(cfg_flags))
    rmse_per_curve = []
    n_ok = 0
    for c in curves:
        if abs(float(c["VG1"]) - VG1_filter) > 1e-6:
            continue
        sebas_row = z427.find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        P_M1, P_M2 = z427.make_overrides(sebas_row)
        bjt = z427.make_bjt(sebas_row)
        Vd_arr = c["Vd"].numpy() if hasattr(c["Vd"], "numpy") else np.asarray(c["Vd"])
        Id_meas = c["Id"].numpy() if hasattr(c["Id"], "numpy") else np.asarray(c["Id"])
        order = np.argsort(Vd_arr)
        Vd_seq = Vd_arr[order]; Id_meas_seq = Id_meas[order]
        Id_pred_seq = np.full_like(Vd_seq, float("nan"))
        t0 = time.time(); curve_ok = True
        try:
            with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), \
                 z427.patch_sd_scaled(sd_M2, P_M2):
                Vb_warm = 0.0
                for i, Vd_f in enumerate(Vd_seq):
                    signal.alarm(int(per_solve_budget_s))
                    try:
                        r = z429.run_vsint_pinned(cfg, model_M1, model_M2, bjt,
                            float(Vd_f), float(c["VG1"]), float(c["VG2"]),
                            Vsint_pin=0.0, Vb_init=Vb_warm)
                        Id_pred_seq[i] = abs(r["Id"]) if r.get("Id") is not None else 0.0
                        if r.get("converged"):
                            Vb_warm = r["Vb"]
                        else:
                            Vb_warm = 0.0
                    except TimeOut:
                        log(f"    TIMEOUT at Vd={Vd_f:.3f} after {per_solve_budget_s}s")
                        curve_ok = False; break
                    finally:
                        signal.alarm(0)
                    if time.time() - t0 > 90.0:
                        log(f"    CURVE BUDGET EXHAUSTED at Vd={Vd_f:.3f}")
                        curve_ok = False; break
        except Exception as e:
            log(f"    EXC curve VG1={c['VG1']} VG2={c['VG2']}: {e}")
            continue
        signal.alarm(0)
        if not curve_ok:
            continue
        eps = 1e-15
        lp = np.log10(Id_pred_seq + eps); lm = np.log10(Id_meas_seq + eps)
        rmse = float(np.sqrt(np.mean((lp - lm) ** 2)))
        rmse_per_curve.append({"VG2": float(c["VG2"]), "rmse_dec": rmse,
                               "n_points": int(len(Vd_seq)), "t_s": time.time()-t0})
        n_ok += 1
        log(f"    VG2={c['VG2']:+.2f}  RMSE={rmse:.3f}dec  n={len(Vd_seq)}  t={time.time()-t0:.1f}s")
    if not rmse_per_curve:
        return None, []
    quad = math.sqrt(sum(r["rmse_dec"]**2 for r in rmse_per_curve) / len(rmse_per_curve))
    return quad, rmse_per_curve


results = {}
for name, cfg in CONFIGS.items():
    log(f"\n--- {name} ---")
    t0 = time.time()
    quad, per = dc_rmse_for_cfg(cfg, curves, VG1_filter=0.6, per_solve_budget_s=4.0)
    log(f"  {name} quad RMSE = {quad}  (took {time.time()-t0:.1f}s)")
    results[name] = {"quad_rmse_dec": quad, "per_curve": per, "wall_s": time.time()-t0}

# Forward + backward sanity (single curve VG2=0)
log("\n--- fwd vs bwd (VG1=0.6, VG2=0) for SNAP_CAL ---")
for c in curves:
    if abs(float(c["VG1"])-0.6)<1e-6 and abs(float(c["VG2"]))<1e-6:
        Vd_arr = c["Vd"].numpy() if hasattr(c["Vd"], "numpy") else np.asarray(c["Vd"])
        Id_meas = c["Id"].numpy() if hasattr(c["Id"], "numpy") else np.asarray(c["Id"])
        sebas_row = z427.find_params(sebas_rows, c["VG1"], c["VG2"])
        P_M1, P_M2 = z427.make_overrides(sebas_row); bjt = z427.make_bjt(sebas_row)
        cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(CONFIGS["SNAP_CAL"]))
        order_fwd = np.argsort(Vd_arr); order_bwd = order_fwd[::-1]
        rmse_dir = {}
        for direction, order in [("fwd", order_fwd), ("bwd", order_bwd)]:
            Vd_seq = Vd_arr[order]; Id_meas_seq = Id_meas[order]
            Id_pred = np.full_like(Vd_seq, float("nan"))
            Vb_warm = 0.0
            t0 = time.time()
            try:
                with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), \
                     z427.patch_sd_scaled(sd_M2, P_M2):
                    for i, Vd_f in enumerate(Vd_seq):
                        signal.alarm(4)
                        try:
                            r = z429.run_vsint_pinned(cfg, model_M1, model_M2, bjt,
                                float(Vd_f), 0.6, 0.0, Vsint_pin=0.0, Vb_init=Vb_warm)
                            Id_pred[i] = abs(r["Id"]) if r.get("Id") is not None else 0.0
                            if r.get("converged"): Vb_warm = r["Vb"]
                            else: Vb_warm = 0.0
                        except TimeOut:
                            log(f"    {direction} timeout at Vd={Vd_f:.3f}")
                            break
                        finally:
                            signal.alarm(0)
                        if time.time()-t0 > 60.0:
                            log(f"    {direction} budget exhausted"); break
            except Exception as e:
                log(f"    {direction} EXC: {e}"); continue
            mask = np.isfinite(Id_pred)
            if mask.sum() < 3: rmse = float("nan")
            else:
                eps = 1e-15
                rmse = float(np.sqrt(np.mean((np.log10(Id_pred[mask]+eps) - np.log10(Id_meas_seq[mask]+eps))**2)))
            log(f"    {direction} RMSE={rmse:.3f} dec  (n={int(mask.sum())}/{len(Vd_seq)})")
            rmse_dir[direction] = rmse
        results["SNAP_CAL_fwd_bwd_VG1p6_VG0"] = rmse_dir
        break

(OUT / "dc_check.json").write_text(json.dumps(results, indent=2, default=str))
log("\n=== SUMMARY ===")
sb_off = results["SB_OFF"]["quad_rmse_dec"]
sn_cal = results["SNAP_CAL"]["quad_rmse_dec"]
sn_pre = results["SNAP_PRE"]["quad_rmse_dec"]
log(f"  SB_OFF   quad RMSE = {sb_off}")
log(f"  SNAP_PRE quad RMSE = {sn_pre}  (delta vs SB_OFF = {(sn_pre-sb_off) if sn_pre and sb_off else 'NA'})")
log(f"  SNAP_CAL quad RMSE = {sn_cal}  (delta vs SB_OFF = {(sn_cal-sb_off) if sn_cal and sb_off else 'NA'})")
log(f"  pre-reg: DC RMSE delta(CAL vs pre-tune) within 0.1 dec?")
if sn_cal is not None and sn_pre is not None:
    log(f"    |delta| = {abs(sn_cal - sn_pre):.3f} dec  PASS={abs(sn_cal - sn_pre) <= 0.1}")
log("DONE")
