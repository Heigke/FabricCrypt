"""z471 — Calibrate snap_Is so SNAP_DEFAULT Id_pk lands on Mario 4.8 mA target.

Step 1: 5-point snap_Is grid at VG1=0.6/VG2=0/Vd=2V.
Step 2: 4-bias verify (VG1=0.4/0.6 x VG2=0/-0.3) at calibrated snap_Is.
Step 3: DC sanity check (skipped here — captured via z461 V1 instead in step 4).
Step 4: Run z461 9-test scorecard at calibrated cell.

Outputs to results/z471_snap_calibrate/.
"""
from __future__ import annotations
import json, math, sys, time, importlib.util as _ilu
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results" / "z471_snap_calibrate"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run.log"
_logfh = open(LOG, "w")
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _logfh.write(s + "\n"); _logfh.flush()

log(f"[z471] start  cwd={ROOT}")

_spec454 = _ilu.spec_from_file_location("z454", ROOT / "scripts/z454_snapback_integration.py")
z454 = _ilu.module_from_spec(_spec454); _spec454.loader.exec_module(z454)
z449 = z454.z449; z427 = z454.z427; z429 = z454.z429

from nsram.bsim4_port import transient_real_v2 as trv2
from nsram.bsim4_port.transient_real_v2 import integrate, TransientCfgV2, stim_fast_pulse

import inspect
assert "I_snap_d" in inspect.getsource(trv2._Id_from_comps), "z469 fix missing"
log("[z471] z469 fix confirmed")

from nsram.bsim4_port.snapback_subcircuit import SnapbackParams
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
import dataclasses as _dc
_fields = {f.name: f for f in _dc.fields(NSRAMCell2TConfig)}
assert SnapbackParams().Id_extra_clamp >= 1e-1 - 1e-9
assert _fields["snap_Id_clamp"].default >= 1e-1 - 1e-9
log("[z471] clamp = 100 mA confirmed")

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

# Base SNAP_DEFAULT (snapback method) — same as z470b but snap_Is sweepable.
def make_snap_default(snap_Is_val):
    return {
        **V449B_BASE,
        "use_snapback_sub": True,
        "snap_method": "snapback",
        **COMMON_AVL_LIFTED,
        "snap_Bf": 417.0, "snap_Va": 0.90,
        "snap_Is": float(snap_Is_val), "snap_Nf": 1.0,
        "snap_npn_gate_mode": "current",
        "snap_npn_V_knee": 1.8, "snap_npn_V_sharp": 0.05,
        "snap_npn_V_BE_offset": 0.3,
        "_R_body": 1e7, "_C_body": 1e-15,
    }

# Reference z470b snap_Is = 6.0256e-9 * 5.0 = 3.0128e-8.
REF_SNAP_IS = 6.0256e-9 * 5.0

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
        return {"Id_pk": float("nan"), "Vb_pk": float("nan"), "Vd_at_pk": float("nan")}
    Id = np.asarray(r["Id"], dtype=float)
    Vd = np.asarray(r.get("Vd", np.full_like(Id, np.nan)), dtype=float)
    Vb = np.asarray(r.get("Vb", np.full_like(Id, np.nan)), dtype=float)
    mask = np.isfinite(Id)
    Id = Id[mask]
    if len(Id) == 0:
        return {"Id_pk": float("nan"), "Vb_pk": float("nan"), "Vd_at_pk": float("nan")}
    aI = np.abs(Id)
    ipk = int(np.argmax(aI))
    return {
        "Id_pk": float(aI[ipk]),
        "Vd_at_pk": float(Vd[ipk]) if ipk < len(Vd) else float("nan"),
        "Vb_pk": float(np.max(np.abs(Vb))) if len(Vb) else float("nan"),
    }


# ---------------- STEP 1: snap_Is grid at primary bias ----------------
log("\n=== STEP 1: snap_Is grid (VG1=0.6, VG2=0.0, Vd=2V) ===")
# Coarse 5-point (kept from spec) is run by ../z471_snap_calibrate_v1, results in
# snap_is_grid.json. Now use a refined fine grid around the [3,7] mA target:
# 2.27 mA at ×1e-4, 84.5 mA at ×1e-3 → target ~×2.1e-4 (snap_Is ~ 6.3e-12).
GRID_MULT = [3e-4, 2.5e-4, 2.0e-4, 1.5e-4, 1.0e-4]
grid_results = []
for m in GRID_MULT:
    snap_Is_val = REF_SNAP_IS * m
    t0 = time.time()
    r = run_pulse(make_snap_default(snap_Is_val), 0.6, 0.0)
    s = summarize(r)
    dec_gap_mario = math.log10(max(s["Id_pk"], 1e-30) / 4.8e-3)
    in_target = 3e-3 <= s["Id_pk"] <= 7e-3
    log(f"  snap_Is={snap_Is_val:.3e} (×{m:g})  Id_pk={s['Id_pk']:.3e}A  Vb_pk={s['Vb_pk']:.3f}V  "
        f"Mario_gap={dec_gap_mario:+.3f}dec  in[3,7]mA={in_target}  ({time.time()-t0:.1f}s)")
    grid_results.append({
        "multiplier": float(m),
        "snap_Is": float(snap_Is_val),
        "Id_pk_A": s["Id_pk"],
        "Vb_pk_V": s["Vb_pk"],
        "Vd_at_pk_V": s["Vd_at_pk"],
        "mario_log10_gap_dec": float(dec_gap_mario),
        "in_target_3_to_7_mA": bool(in_target),
    })

(OUT / "snap_is_grid.json").write_text(json.dumps({
    "primary_bias": {"VG1": 0.6, "VG2": 0.0, "Vd_pulse": 2.0},
    "ref_snap_Is": REF_SNAP_IS,
    "mario_target_A": 4.8e-3,
    "grid": grid_results,
}, indent=2))

# Choose calibration point: closest to Mario 4.8 mA in [3,7] mA, else closest by log10.
in_target = [g for g in grid_results if g["in_target_3_to_7_mA"]]
if in_target:
    chosen = min(in_target, key=lambda g: abs(math.log10(g["Id_pk_A"] / 4.8e-3)))
    log(f"\n[z471] Calibration point IN-TARGET: snap_Is={chosen['snap_Is']:.4e}  Id_pk={chosen['Id_pk_A']:.3e}A")
else:
    # No grid point landed [3,7] mA. Pick closest by |log10 gap|.
    chosen = min(grid_results, key=lambda g: abs(g["mario_log10_gap_dec"]) if math.isfinite(g["Id_pk_A"]) else 1e9)
    log(f"\n[z471] NO grid point in [3,7] mA. Closest by log10: snap_Is={chosen['snap_Is']:.4e}  Id_pk={chosen['Id_pk_A']:.3e}A  gap={chosen['mario_log10_gap_dec']:+.3f}dec")

CAL_SNAP_IS = chosen["snap_Is"]
landed_primary = bool(chosen["in_target_3_to_7_mA"])

# ---------------- STEP 2: 4-bias verify ----------------
log("\n=== STEP 2: 4-bias verify at snap_Is={:.4e} ===".format(CAL_SNAP_IS))
# NOTE: sebas rows only span VG2 ∈ [-0.2, 0.5]; task spec said VG2=-0.3
# (unavailable). Use VG2=-0.2 (closest available) instead.
BIASES = [(0.6, 0.0), (0.6, -0.2), (0.4, 0.0), (0.4, -0.2)]
verify_results = []
all_in_window = True
max_id = 0.0; min_id = 1e9
for vg1, vg2 in BIASES:
    t0 = time.time()
    r = run_pulse(make_snap_default(CAL_SNAP_IS), vg1, vg2)
    s = summarize(r)
    in_window = 1e-3 <= s["Id_pk"] <= 1e-2
    if not in_window:
        all_in_window = False
    if math.isfinite(s["Id_pk"]):
        max_id = max(max_id, s["Id_pk"])
        min_id = min(min_id, s["Id_pk"])
    log(f"  VG1={vg1} VG2={vg2}  Id_pk={s['Id_pk']:.3e}A  Vb_pk={s['Vb_pk']:.3f}V  "
        f"in[1,10]mA={in_window}  ({time.time()-t0:.1f}s)")
    verify_results.append({
        "VG1": vg1, "VG2": vg2,
        "Id_pk_A": s["Id_pk"], "Vb_pk_V": s["Vb_pk"], "Vd_at_pk_V": s["Vd_at_pk"],
        "in_window_1_to_10_mA": bool(in_window),
    })

dispersion_dec = math.log10(max(max_id, 1e-30) / max(min_id, 1e-30)) if max_id > 0 and min_id < 1e9 else float("nan")
(OUT / "four_bias_verify.json").write_text(json.dumps({
    "calibrated_snap_Is": CAL_SNAP_IS,
    "mario_target_A": 4.8e-3,
    "biases": verify_results,
    "all_in_window_1_to_10_mA": bool(all_in_window),
    "id_pk_dispersion_dec": float(dispersion_dec),
}, indent=2))
log(f"  ALL in [1,10] mA: {all_in_window}  dispersion={dispersion_dec:.3f} dec")

# ---------------- Plot: Id_pk vs bias overlay ----------------
fig, ax = plt.subplots(figsize=(7,4.5))
labels = [f"VG1={v1}\nVG2={v2}" for v1,v2 in BIASES]
ids = [v["Id_pk_A"]*1e3 for v in verify_results]  # mA
xs = np.arange(len(BIASES))
ax.bar(xs, ids, color="#3b7cd8", label=f"calibrated cell (snap_Is={CAL_SNAP_IS:.2e})")
ax.axhline(4.8, color="red", linestyle="--", label="Mario target 4.8 mA")
ax.axhspan(1.0, 10.0, color="green", alpha=0.1, label="acceptance window [1,10] mA")
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Id_pk [mA]")
ax.set_title(f"z471 Mario landing — snap_Is={CAL_SNAP_IS:.3e}")
ax.set_yscale("log"); ax.set_ylim(0.1, 200)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "mario_landed.png", dpi=120)
plt.close()
log(f"  wrote {OUT/'mario_landed.png'}")

# ---------------- Persist calibration summary ----------------
calibration_summary = {
    "calibrated_snap_Is": CAL_SNAP_IS,
    "calibration_multiplier_vs_z470b": CAL_SNAP_IS / REF_SNAP_IS,
    "landed_primary_3to7_mA": bool(landed_primary),
    "landed_4biases_1to10_mA": bool(all_in_window),
    "id_pk_dispersion_dec": float(dispersion_dec),
}
(OUT / "calibration_summary.json").write_text(json.dumps(calibration_summary, indent=2))

log(f"\n[z471] DONE step1+2.  calibrated snap_Is = {CAL_SNAP_IS:.4e}")
log(f"  primary in [3,7]mA: {landed_primary}")
log(f"  4-bias in [1,10]mA: {all_in_window}")
log(f"  dispersion: {dispersion_dec:.3f} dec")
