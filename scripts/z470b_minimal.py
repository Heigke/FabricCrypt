"""z470b — Minimal resume of z470. Single bias VG1=0.6/VG2=0.0/Vd=2.0V.
Step 1: SNAP_DEFAULT vs SNAP_HOT (2 transients).
Step 2: thy_Gon sweep at 5/25/50 mS (3 transients).
"""
from __future__ import annotations
import json, math, sys, time, importlib.util as _ilu
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))

OUT = ROOT / "results" / "z470_clamp_lift"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run_z470b.log"
_logfh = open(LOG, "w")
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _logfh.write(s + "\n"); _logfh.flush()

log(f"[z470b] start  cwd={ROOT}")

_spec454 = _ilu.spec_from_file_location("z454", ROOT / "scripts/z454_snapback_integration.py")
z454 = _ilu.module_from_spec(_spec454); _spec454.loader.exec_module(z454)
z449 = z454.z449; z427 = z454.z427; z429 = z454.z429

from nsram.bsim4_port import transient_real_v2 as trv2
from nsram.bsim4_port.transient_real_v2 import integrate, TransientCfgV2, stim_fast_pulse

import inspect
assert "I_snap_d" in inspect.getsource(trv2._Id_from_comps), "z469 fix missing"
log("[z470b] z469 fix confirmed")

from nsram.bsim4_port.snapback_subcircuit import SnapbackParams
from nsram.bsim4_port.thyristor_pivot import ThyristorPivotParams
import dataclasses as _dc
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
_fields = {f.name: f for f in _dc.fields(NSRAMCell2TConfig)}
assert SnapbackParams().Id_extra_clamp >= 1e-1 - 1e-9
assert ThyristorPivotParams().Id_extra_clamp >= 1e-1 - 1e-9
assert _fields["snap_Id_clamp"].default >= 1e-1 - 1e-9
log("[z470b] clamp lift = 100 mA confirmed")

log("Loading models / curves / sebas rows...")
model_M1, model_M2 = z429.build_models()
sebas_rows = z429.load_sebas_params()

V449B_BASE = {
    "use_vbic_for_q1": True,
    "vbic_AVC1": 0.5, "vbic_AVC2": 0.5,
    "Cbody": 1e-15,
    "body_pdiode_Cj0_per_area": 0.0,
}
COMMON_AVL_LIFTED = dict(
    snap_BV=2.0 * 0.6, snap_n_avl=4.0,
    snap_Id_clamp=1e-1, snap_Iii_clamp=1e-1,
    snap_use_knee_gate=True,
    snap_V_knee=1.6, snap_V_sharp=0.05,
)

THY_DEFAULT = {
    **V449B_BASE,
    "use_snapback_sub": True,
    "snap_method": "thyristor",
    **COMMON_AVL_LIFTED,
    "snap_Bf": 417.0, "snap_Va": 0.90, "snap_Is": 6e-9, "snap_Nf": 1.0,
    "thy_Ipk": 2e-3, "thy_Vpk": 0.85, "thy_Wpk": 0.12,
    "thy_Gon": 5e-3, "thy_VH": 0.55, "thy_VT1": 1.00,
    "thy_K": 40.0, "thy_alpha": 2.0, "thy_tau_Q": 5e-9,
    "_R_body": 1e7, "_C_body": 1e-15,
}

SNAP_HOT = {
    **V449B_BASE,
    "use_snapback_sub": True,
    "snap_method": "snapback",
    **COMMON_AVL_LIFTED,
    "snap_Bf": 10000.0, "snap_Va": 100.0,
    "snap_Is": 5.0e-9, "snap_Nf": 1.0,
    "snap_npn_gate_mode": "current",
    "snap_npn_V_knee": 1.8, "snap_npn_V_sharp": 0.05,
    "snap_npn_V_BE_offset": 0.3,
    "_R_body": 1e7, "_C_body": 1e-15,
}
SNAP_DEFAULT = {**SNAP_HOT, "snap_Bf": 417.0, "snap_Va": 0.90,
                "snap_Is": 6.0256e-9 * 5.0}

VG1, VG2 = 0.6, 0.0
PULSE_T = stim_fast_pulse(V_hi=2.0, V_lo=0.0,
                          t_rise=100e-12, t_hold=5e-6, t_fall=100e-12,
                          t_pre=2e-9, t_post=200e-9, n_total=4000)


def run_pulse(cfg_flags, VG1, VG2):
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(cfg_flags))
    cfg.Cbody = float(cfg_flags.get("_C_body", 1e-15))
    tcfg = TransientCfgV2(
        C_B_const=float(cfg_flags.get("_C_body", 1e-15)),
        max_step=5e-9, first_step=1e-14,
        rtol=1e-5, atol=1e-14,
        R_body=float(cfg_flags.get("_R_body", 1e7)),
    )
    sebas_row = z427.find_params(sebas_rows, VG1, VG2)
    if sebas_row is None:
        return None
    P_M1, P_M2 = z427.make_overrides(sebas_row)
    bjt = z427.make_bjt(sebas_row)
    try:
        bjt.Bf = float(cfg_flags.get("snap_Bf", 417.0))
    except Exception:
        pass
    z449._VBIC_CTX["cfg"] = cfg
    z449._VBIC_CTX["bjt"] = bjt
    t, Vd = PULSE_T
    try:
        with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), \
             z427.patch_sd_scaled(sd_M2, P_M2):
            r = integrate(cfg, model_M1, model_M2, bjt,
                          t, Vd, VG1, VG2, tcfg=tcfg, Vb0=0.0)
    except Exception as e:
        log(f"  pulse EXC: {e}")
        return None
    finally:
        z449._VBIC_CTX["cfg"] = None
        z449._VBIC_CTX["bjt"] = None
    return r


def summarize(r):
    if r is None:
        return {k: float("nan") for k in
                ["Id_pk", "Vd_at_pk", "Vb_pk", "t_rise", "t_fall"]}
    Id = np.asarray(r["Id"], dtype=float)
    Vd = np.asarray(r.get("Vd", np.full_like(Id, np.nan)), dtype=float)
    Vb = np.asarray(r.get("Vb", np.full_like(Id, np.nan)), dtype=float)
    t = np.asarray(r.get("t",  np.arange(len(Id))*1e-9), dtype=float)
    mask = np.isfinite(Id) & np.isfinite(t)
    Id = Id[mask]; Vd = Vd[mask] if len(Vd)==len(mask) else Vd
    Vb = Vb[mask] if len(Vb)==len(mask) else Vb
    t = t[mask]
    if len(Id) == 0:
        return {k: float("nan") for k in
                ["Id_pk", "Vd_at_pk", "Vb_pk", "t_rise", "t_fall"]}
    aI = np.abs(Id)
    ipk = int(np.argmax(aI))
    Id_pk = float(aI[ipk])
    Vd_at = float(Vd[ipk]) if ipk < len(Vd) else float("nan")
    Vb_pk = float(np.max(np.abs(Vb))) if len(Vb) else float("nan")
    thr_lo = 0.1 * Id_pk; thr_hi = 0.9 * Id_pk
    pre = aI[:ipk+1]
    try:
        i_lo = int(np.argmax(pre >= thr_lo))
        i_hi = int(np.argmax(pre >= thr_hi))
        t_rise = float(t[i_hi] - t[i_lo]) if i_hi > i_lo else float("nan")
    except Exception:
        t_rise = float("nan")
    post = aI[ipk:]
    try:
        i_fhi = ipk + int(np.argmax(post <= thr_hi))
        i_flo = ipk + int(np.argmax(post <= thr_lo))
        t_fall = float(t[i_flo] - t[i_fhi]) if i_flo > i_fhi else float("nan")
    except Exception:
        t_fall = float("nan")
    return {"Id_pk": Id_pk, "Vd_at_pk": Vd_at, "Vb_pk": Vb_pk,
            "t_rise": t_rise, "t_fall": t_fall}


# Step 1
log("\n=== STEP 1: SNAP_DEFAULT vs SNAP_HOT (single bias) ===")
t0 = time.time()
log(f"  bias VG1={VG1} VG2={VG2}")
rd = run_pulse(SNAP_DEFAULT, VG1, VG2)
log(f"  DEFAULT done in {time.time()-t0:.1f}s")
t1 = time.time()
rh = run_pulse(SNAP_HOT, VG1, VG2)
log(f"  HOT done in {time.time()-t1:.1f}s")
sd = summarize(rd); sh = summarize(rh)
log(f"  DEFAULT Id_pk={sd['Id_pk']:.3e}A  Vd@pk={sd['Vd_at_pk']:.3f}V  Vb_pk={sd['Vb_pk']:.3f}V  trise={sd['t_rise']:.2e}s  tfall={sd['t_fall']:.2e}s")
log(f"  HOT     Id_pk={sh['Id_pk']:.3e}A  Vd@pk={sh['Vd_at_pk']:.3f}V  Vb_pk={sh['Vb_pk']:.3f}V  trise={sh['t_rise']:.2e}s  tfall={sh['t_fall']:.2e}s")

ratio_hot = sh["Id_pk"] / max(sd["Id_pk"], 1e-30)
mario_dec = math.log10(max(sd["Id_pk"], 1e-30) / 4.8e-3)
q4_confirmed = bool(ratio_hot > 1.5)
mario_within_05 = bool(abs(mario_dec) < 0.5)

step1 = {
    "bias": {"VG1": VG1, "VG2": VG2, "Vd_pulse": 2.0},
    "SNAP_DEFAULT": sd,
    "SNAP_HOT": sh,
    "MARIO_TARGET_A": 4.8e-3,
    "clamp_setting_A": 1e-1,
    "ratio_HOT_over_DEFAULT": float(ratio_hot),
    "mario_log10_gap_dec": float(mario_dec),
    "Q4_confirmed_HOT_gt_1p5x": q4_confirmed,
    "mario_within_0p5_dec": mario_within_05,
}
(OUT / "snap_default_vs_hot_single.json").write_text(json.dumps(step1, indent=2))
log(f"  ratio HOT/DEFAULT = {ratio_hot:.3f}    Mario log10-gap = {mario_dec:+.3f} dec")
log(f"  Q4 confirmed (>1.5x): {q4_confirmed}    Mario within 0.5 dec: {mario_within_05}")

# Step 2
log("\n=== STEP 2: thy_Gon sweep (5, 25, 50 mS) ===")
GON_VALUES = [5e-3, 25e-3, 50e-3]
thy_sweep = {"bias": {"VG1": VG1, "VG2": VG2}, "thy_Gon_S": GON_VALUES, "results": {}}
for gv in GON_VALUES:
    key = f"Gon_{gv*1000:.0f}mS"
    t2 = time.time()
    cfg_g = {**THY_DEFAULT, "thy_Gon": gv}
    r = run_pulse(cfg_g, VG1, VG2)
    s = summarize(r)
    thy_sweep["results"][key] = s
    log(f"  {key}  Id_pk={s['Id_pk']:.3e}A  Vb_pk={s['Vb_pk']:.3f}V  (took {time.time()-t2:.1f}s)")

ipk_vals = [thy_sweep["results"][f"Gon_{gv*1000:.0f}mS"]["Id_pk"] for gv in GON_VALUES]
ratio_10x = ipk_vals[2] / max(ipk_vals[0], 1e-30)
thy_sweep["ratio_50mS_over_5mS"] = float(ratio_10x)
thy_sweep["thy_Gon_is_binding"] = bool(ratio_10x > 1.5)
log(f"  ratio 50mS/5mS = {ratio_10x:.3f}    thy_Gon_binding = {thy_sweep['thy_Gon_is_binding']}")
(OUT / "thy_gon_sweep.json").write_text(json.dumps(thy_sweep, indent=2))

# Honest analysis
analysis = f"""# z470b honest analysis (resume of z470, minimal scope)

**Bias**: VG1={VG1}, VG2={VG2}, Vd_pulse=2.0V (single point).
**Clamp**: snap_Id_clamp / snap_Iii_clamp / thy.Id_extra_clamp lifted from 1e-2 to 1e-1 A (z470 patch in place).
**z469 fix**: I_snap_d added to _Id_from_comps in transient_real_v2 (confirmed in-place).

## Step 1: SNAP_DEFAULT vs SNAP_HOT

| Cell         | Id_pk (A)            | Vb_pk (V)         | t_rise (s)           |
|--------------|----------------------|-------------------|----------------------|
| SNAP_DEFAULT | {sd['Id_pk']:.3e}        | {sd['Vb_pk']:.3f}             | {sd['t_rise']:.2e}            |
| SNAP_HOT     | {sh['Id_pk']:.3e}        | {sh['Vb_pk']:.3f}             | {sh['t_rise']:.2e}            |

Ratio HOT/DEFAULT = **{ratio_hot:.3f}**.
Mario target = 4.8e-3 A. SNAP_DEFAULT log10-gap = **{mario_dec:+.3f} dec**.

**Q4 verdict**: {"CONFIRMED" if q4_confirmed else "FALSIFIED"} (gate: HOT > 1.5x DEFAULT).
**Mario realism**: {"PASS" if mario_within_05 else "FAIL"} (gate: |log10(Id_pk/4.8mA)| < 0.5).

## Step 2: thy_Gon sweep

| thy_Gon  | Id_pk (A)            | Vb_pk (V)         |
|----------|----------------------|-------------------|
| 5 mS     | {thy_sweep['results']['Gon_5mS']['Id_pk']:.3e}        | {thy_sweep['results']['Gon_5mS']['Vb_pk']:.3f}             |
| 25 mS    | {thy_sweep['results']['Gon_25mS']['Id_pk']:.3e}        | {thy_sweep['results']['Gon_25mS']['Vb_pk']:.3f}             |
| 50 mS    | {thy_sweep['results']['Gon_50mS']['Id_pk']:.3e}        | {thy_sweep['results']['Gon_50mS']['Vb_pk']:.3f}             |

Ratio 50mS / 5mS = **{ratio_10x:.3f}**. thy_Gon binding (>1.5x): **{thy_sweep['thy_Gon_is_binding']}**.

## Interpretation

With the safety clamps lifted to 100 mA, Id_pk now reports the underlying device physics
rather than the saturating clamp. The HOT/DEFAULT ratio of {ratio_hot:.2f} {"strongly supports" if q4_confirmed else "does NOT support"} the Q4
hypothesis that the Sebas-measured parasitic-NPN parameters (Bf=10000, Va=100) increase
regenerative kick by >1.5x once the topology/clamp ceiling is gone. SNAP_DEFAULT's raw Id_pk
of {sd['Id_pk']:.2e} A vs Mario's 4.8 mA target is {mario_dec:+.2f} decades off:
{"within the +/-0.5 dec realism band" if mario_within_05 else "outside the +/-0.5 dec realism band"}.
The thy_Gon sweep shows a 5x current ratio of {ratio_10x:.2f}, {"confirming" if thy_sweep['thy_Gon_is_binding'] else "rejecting"} z469's claim
that thy_Gon (not thy_Ipk) is the binding parameter for thyristor-cell Id_pk.
"""
(OUT / "honest_analysis.md").write_text(analysis)
log("\n[z470b] DONE. Outputs in results/z470_clamp_lift/")
log(f"  Q4_confirmed={q4_confirmed}  mario_within_0p5_dec={mario_within_05}  thy_Gon_binding={thy_sweep['thy_Gon_is_binding']}")
