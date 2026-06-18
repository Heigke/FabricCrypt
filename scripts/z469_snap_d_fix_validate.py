"""z469 — Validate the 1-line fix in transient_real_v2._Id_from_comps.

Sequence:
  1) Smoke test: run one transient with VG1=0.6/VG2=0/V_d→2V and log comps.keys()
     + I_snap_d magnitude at first call. (Confirms key is present in comps.)
  2) THY_DEFAULT vs THY_STRONG (Ipk=2mA vs 10mA): if the bug-fix is real,
     Id_pk MUST now differ. Pre-bug they were identical (1.040e-06 A).
  3) SNAP_HOT canonical cell: snapback method with Bf=10000, Va=100 (Sebas-
     measured parasiticBJT). Compare Id_pk to Mario 4.8 mA across 4 biases.

Outputs into results/z469_snap_d_fix/:
  smoke_test.log, thy_compare.json, snap_hot_with_fix.json, mario_vs_us.png,
  honest_analysis.md
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

OUT = ROOT / "results" / "z469_snap_d_fix"
OUT.mkdir(parents=True, exist_ok=True)

LOG = OUT / "smoke_test.log"
_logfh = open(LOG, "w")
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _logfh.write(s + "\n"); _logfh.flush()

log(f"[z469] start  cwd={ROOT}")

# Reuse z467 machinery
_spec454 = _ilu.spec_from_file_location("z454", ROOT / "scripts/z454_snapback_integration.py")
z454 = _ilu.module_from_spec(_spec454); _spec454.loader.exec_module(z454)
z449 = z454.z449
z427 = z454.z427
z429 = z454.z429

from nsram.bsim4_port import transient_real_v2 as trv2
from nsram.bsim4_port.transient_real_v2 import integrate, TransientCfgV2, stim_fast_pulse

# ---------- Patch _Id_from_comps to log keys on first call ----------
_orig_Id_from_comps = trv2._Id_from_comps
_first_call = {"done": False, "keys": None, "I_snap_d": None}

def _patched_Id_from_comps(comps):
    if not _first_call["done"]:
        _first_call["done"] = True
        _first_call["keys"] = sorted(list(comps.keys()))
        try:
            isd = comps.get("I_snap_d")
            if isd is not None:
                _first_call["I_snap_d"] = float(isd.abs().item()) if hasattr(isd, "abs") else float(isd)
            else:
                _first_call["I_snap_d"] = None
        except Exception as e:
            _first_call["I_snap_d"] = f"err: {e}"
        log(f"[smoke] FIRST CALL comps.keys() = {_first_call['keys']}")
        log(f"[smoke] FIRST CALL |I_snap_d|  = {_first_call['I_snap_d']}")
    return _orig_Id_from_comps(comps)

trv2._Id_from_comps = _patched_Id_from_comps


# ---------- Build models / curves ----------
log("Loading models / curves / sebas rows...")
model_M1, model_M2 = z429.build_models()
sebas_rows = z429.load_sebas_params()


# ---------- Cell configs ----------
V449B_BASE = {
    "use_vbic_for_q1": True,
    "vbic_AVC1": 0.5, "vbic_AVC2": 0.5,
    "Cbody": 1e-15,
    "body_pdiode_Cj0_per_area": 0.0,
}
COMMON_AVL = dict(
    snap_BV=2.0 * 0.6, snap_n_avl=4.0,
    snap_Id_clamp=1e-2, snap_Iii_clamp=1e-2,
    snap_use_knee_gate=True,
    snap_V_knee=1.6, snap_V_sharp=0.05,
)
THY_DEFAULT = {
    **V449B_BASE,
    "use_snapback_sub": True,
    "snap_method": "thyristor",
    **COMMON_AVL,
    "snap_Bf": 417.0, "snap_Va": 0.90, "snap_Is": 6e-9, "snap_Nf": 1.0,
    "thy_Ipk": 2e-3, "thy_Vpk": 0.85, "thy_Wpk": 0.12,
    "thy_Gon": 5e-3, "thy_VH": 0.55, "thy_VT1": 1.00,
    "thy_K": 40.0, "thy_alpha": 2.0, "thy_tau_Q": 5e-9,
    "_R_body": 1e7, "_C_body": 1e-15,
}
THY_STRONG = {**THY_DEFAULT, "thy_Ipk": 1e-2}

# SNAP_HOT = parasitic-NPN snapback with Sebas's measured parasiticBJT
SNAP_HOT = {
    **V449B_BASE,
    "use_snapback_sub": True,
    "snap_method": "snapback",
    **COMMON_AVL,
    "snap_Bf": 10000.0, "snap_Va": 100.0,    # Sebas parasiticBJT.txt
    "snap_Is": 5.0e-9, "snap_Nf": 1.0,
    "snap_npn_gate_mode": "current",
    "snap_npn_V_knee": 1.8, "snap_npn_V_sharp": 0.05,
    "snap_npn_V_BE_offset": 0.3,
    "_R_body": 1e7, "_C_body": 1e-15,
}

# SNAP_DEFAULT = parasitic-NPN snapback with current Bf=417 defaults
SNAP_DEFAULT = {**SNAP_HOT, "snap_Bf": 417.0, "snap_Va": 0.90, "snap_Is": 6.0256e-9 * 5.0}


PULSE_BIASES = [
    ("VG1=0.6_VG2=0.0", 0.6, 0.0),
    ("VG1=0.6_VG2=0.2", 0.6, 0.2),
    ("VG1=0.6_VG2=0.4", 0.6, 0.4),
    ("VG1=0.4_VG2=0.0", 0.4, 0.0),
]
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
        log(f"  pulse EXC VG1={VG1} VG2={VG2}: {e}")
        return None
    finally:
        z449._VBIC_CTX["cfg"] = None
        z449._VBIC_CTX["bjt"] = None
    return r


def Id_peak(r):
    if r is None:
        return float("nan")
    Id = np.asarray(r["Id"], dtype=float)
    Id = Id[np.isfinite(Id)]
    if len(Id) == 0:
        return float("nan")
    return float(np.max(np.abs(Id)))


# ---------- Step 1: smoke test (one bias) ----------
log("\n=== STEP 1: SMOKE TEST (THY_DEFAULT, VG1=0.6, VG2=0) ===")
r_smoke = run_pulse(THY_DEFAULT, 0.6, 0.0)
log(f"smoke Id_pk = {Id_peak(r_smoke):.4e} A")
log(f"smoke comps.keys() captured = {_first_call['keys']}")
has_snap_d = (_first_call["keys"] is not None) and ("I_snap_d" in _first_call["keys"])
log(f"I_snap_d in comps? {has_snap_d}")


# ---------- Step 2: THY_DEFAULT vs THY_STRONG across all biases ----------
log("\n=== STEP 2: THY_DEFAULT vs THY_STRONG ===")
thy_compare = {"biases": [], "THY_DEFAULT": {}, "THY_STRONG": {}}
for tag, VG1, VG2 in PULSE_BIASES:
    log(f"  bias {tag}")
    rd = run_pulse(THY_DEFAULT, VG1, VG2)
    rs = run_pulse(THY_STRONG,  VG1, VG2)
    idp_d = Id_peak(rd); idp_s = Id_peak(rs)
    log(f"    THY_DEFAULT Id_pk={idp_d:.4e}A   THY_STRONG Id_pk={idp_s:.4e}A   ratio={idp_s/max(idp_d,1e-30):.3f}")
    thy_compare["biases"].append(tag)
    thy_compare["THY_DEFAULT"][tag] = idp_d
    thy_compare["THY_STRONG"][tag]  = idp_s

# Pre-bug observed: Id_pk_default == Id_pk_strong == 1.040e-06 A
thy_compare["pre_fix_observed"] = {tag: 1.040e-06 for tag, _, _ in PULSE_BIASES}
thy_compare["discovery_gate"] = {
    tag: (thy_compare["THY_STRONG"][tag] > 2.0 * max(thy_compare["THY_DEFAULT"][tag], 1e-30))
    for tag, _, _ in PULSE_BIASES
}
log(f"  DISCOVERY gate (THY_STRONG > 2× THY_DEFAULT) per bias: {thy_compare['discovery_gate']}")
(OUT / "thy_compare.json").write_text(json.dumps(thy_compare, indent=2))


# ---------- Step 3: SNAP_HOT (Bf=10000, Va=100) ----------
log("\n=== STEP 3: SNAP_HOT (Bf=10000, Va=100) vs SNAP_DEFAULT ===")
snap_hot = {"biases": [], "SNAP_DEFAULT": {}, "SNAP_HOT": {}, "MARIO_TARGET_A": 4.8e-3}
for tag, VG1, VG2 in PULSE_BIASES:
    log(f"  bias {tag}")
    rd = run_pulse(SNAP_DEFAULT, VG1, VG2)
    rh = run_pulse(SNAP_HOT,     VG1, VG2)
    idp_d = Id_peak(rd); idp_h = Id_peak(rh)
    log(f"    SNAP_DEFAULT Id_pk={idp_d:.4e}A   SNAP_HOT Id_pk={idp_h:.4e}A")
    snap_hot["biases"].append(tag)
    snap_hot["SNAP_DEFAULT"][tag] = idp_d
    snap_hot["SNAP_HOT"][tag] = idp_h

snap_hot["ambitious_gate_1mA_pass"] = sum(
    1 for tag in snap_hot["biases"] if snap_hot["SNAP_HOT"][tag] > 1e-3
)
snap_hot["breakthrough_gate_within_1dec_mario"] = sum(
    1 for tag in snap_hot["biases"]
    if 4.8e-4 <= snap_hot["SNAP_HOT"][tag] <= 4.8e-2
)
log(f"  AMBITIOUS gate (Id_pk > 1 mA): {snap_hot['ambitious_gate_1mA_pass']}/4 biases")
log(f"  BREAKTHROUGH gate (within 1 dec of 4.8 mA): {snap_hot['breakthrough_gate_within_1dec_mario']}/4")
(OUT / "snap_hot_with_fix.json").write_text(json.dumps(snap_hot, indent=2))


# ---------- Plot: Mario vs us ----------
log("\n=== PLOT: mario_vs_us.png ===")
fig, ax = plt.subplots(figsize=(9, 5))
xs = np.arange(len(PULSE_BIASES))
labels = [tag for tag, _, _ in PULSE_BIASES]
ax.axhline(4.8e-3, color="k", ls="--", lw=1, label="Mario 4.8 mA")
ax.axhline(1.040e-06, color="gray", ls=":", lw=1, label="z467 pre-fix (1.04 µA)")
ax.plot(xs, [thy_compare["THY_DEFAULT"][t] for t in labels], "o-", label="THY_DEFAULT (post-fix)")
ax.plot(xs, [thy_compare["THY_STRONG"][t]  for t in labels], "o-", label="THY_STRONG (post-fix, Ipk×5)")
ax.plot(xs, [snap_hot["SNAP_DEFAULT"][t]   for t in labels], "s-", label="SNAP_DEFAULT (Bf=417)")
ax.plot(xs, [snap_hot["SNAP_HOT"][t]       for t in labels], "s-", label="SNAP_HOT (Bf=10000, Va=100)")
ax.set_yscale("log")
ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("Id_pk  [A]"); ax.set_title("z469 — Id_pk per bias after I_snap_d fix")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(OUT / "mario_vs_us.png", dpi=120); plt.close(fig)


# ---------- Honest analysis ----------
log("\n=== WRITE honest_analysis.md ===")

def dec_gap(v):
    if v is None or not (v > 0):
        return float("nan")
    return math.log10(4.8e-3 / v)

best_pre = max([1.040e-06] * 4)
best_thy_default = max(thy_compare["THY_DEFAULT"].values())
best_thy_strong  = max(thy_compare["THY_STRONG"].values())
best_snap_default = max(snap_hot["SNAP_DEFAULT"].values())
best_snap_hot = max(snap_hot["SNAP_HOT"].values())

infra_pass = bool(has_snap_d)
discovery_pass = any(thy_compare["discovery_gate"].values())
ambitious_pass = snap_hot["ambitious_gate_1mA_pass"] >= 2
breakthrough_pass = snap_hot["breakthrough_gate_within_1dec_mario"] >= 1

md = f"""# z469 — I_snap_d Fix Validation: Honest Analysis

Date: 2026-05-17. Fix: 1-line addition of `+ comps.get("I_snap_d", ...)` in
`nsram/nsram/bsim4_port/transient_real_v2.py::_Id_from_comps`.

## Pre-registered gates

| Gate         | Criterion                                              | Result |
|--------------|--------------------------------------------------------|--------|
| INFRA        | I_snap_d present in `comps` at first call              | {"PASS" if infra_pass else "FAIL"} |
| DISCOVERY    | THY_STRONG Id_pk > 2× THY_DEFAULT (any bias)           | {"PASS" if discovery_pass else "FAIL"} |
| AMBITIOUS    | SNAP_HOT Id_pk > 1 mA on ≥2/4 biases                   | {"PASS" if ambitious_pass else "FAIL"} ({snap_hot['ambitious_gate_1mA_pass']}/4) |
| BREAKTHROUGH | SNAP_HOT Id_pk within 1 decade of Mario 4.8 mA (≥1/4)  | {"PASS" if breakthrough_pass else "FAIL"} ({snap_hot['breakthrough_gate_within_1dec_mario']}/4) |

## Smoke test

- `comps.keys()` at first transient call: `{_first_call['keys']}`
- First-call `|I_snap_d|`: `{_first_call['I_snap_d']}`

## THY_DEFAULT vs THY_STRONG (Q5: bug fix only)

Pre-fix (z467 log): both reported Id_pk = 1.040e-06 A identically.

| Bias | THY_DEFAULT Id_pk [A] | THY_STRONG Id_pk [A] | ratio |
|------|-----------------------|-----------------------|-------|
"""
for tag in [t for t, _, _ in PULSE_BIASES]:
    d = thy_compare["THY_DEFAULT"][tag]
    s = thy_compare["THY_STRONG"][tag]
    md += f"| {tag} | {d:.3e} | {s:.3e} | {s/max(d,1e-30):.3f} |\n"

md += f"""
Best THY_DEFAULT Id_pk (post-fix): **{best_thy_default:.3e} A**
Best THY_STRONG Id_pk (post-fix):  **{best_thy_strong:.3e} A**
Pre-fix observed (both):           **1.040e-06 A**

## SNAP_HOT (Q4: param fix on top of bug fix)

Sebas's measured parasiticBJT: `Bf=10000, Va=100, Is=5e-9`.
Default: `Bf=417, Va=0.9`.

| Bias | SNAP_DEFAULT Id_pk [A] | SNAP_HOT Id_pk [A] | log10(SNAP_HOT/SNAP_DEFAULT) |
|------|------------------------|---------------------|------------------------------|
"""
for tag in [t for t, _, _ in PULSE_BIASES]:
    d = snap_hot["SNAP_DEFAULT"][tag]
    h = snap_hot["SNAP_HOT"][tag]
    g = math.log10(max(h,1e-30) / max(d,1e-30))
    md += f"| {tag} | {d:.3e} | {h:.3e} | {g:+.2f} |\n"

md += f"""
Best SNAP_DEFAULT Id_pk: **{best_snap_default:.3e} A**
Best SNAP_HOT     Id_pk: **{best_snap_hot:.3e} A**
Mario target:            **4.8e-03 A**
Decade gap (best SNAP_HOT → Mario): **{dec_gap(best_snap_hot):+.2f} dec**

## Contribution of each fix

- Bug fix only (THY_STRONG ÷ THY_DEFAULT ratio): **{best_thy_strong/max(best_thy_default,1e-30):.2f}×**
  (pre-fix ratio = 1.000× — z467 reported identical Id_pk for both cells)
- Param fix only (SNAP_HOT ÷ SNAP_DEFAULT, post-bug-fix): **{best_snap_hot/max(best_snap_default,1e-30):.2f}×**

## Verdict

{"BUG REAL: the missing I_snap_d term DID cause z467's invariance under 5× Ipk scaling. THY_STRONG now reports {0:.2f}× more current than THY_DEFAULT (pre-fix 1.00×).".format(best_thy_strong/max(best_thy_default,1e-30)) if discovery_pass else "BUG NOT CONFIRMED: even after the fix, THY_STRONG ≈ THY_DEFAULT. z468 Q5 diagnosis was wrong. The thyristor compact-model output may not flow through I_snap_d, or another downstream path masks it. Need to instrument compute_thyristor_snap directly."}

{"PARAM FIX HELPS: SNAP_HOT closes the Mario gap from {0:.2f} to {1:.2f} decades.".format(dec_gap(best_snap_default), dec_gap(best_snap_hot)) if best_snap_hot > best_snap_default else "PARAM FIX DOES NOT HELP: Bf=10000+Va=100 yields no improvement over Bf=417. Other structural limit (likely the M2/Sint topology issue per z468 Q3) dominates."}
"""

(OUT / "honest_analysis.md").write_text(md)
log("Wrote honest_analysis.md")
log(f"\n[z469] done; outputs in {OUT}")
_logfh.close()
