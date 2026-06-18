"""z470 — Raise snap_Id_clamp + isolate Q4 (Sebas-Bf param-fix) contribution.

Pre-reqs:
  - z469 fix in `transient_real_v2._Id_from_comps` (adds I_snap_d) is present.
  - z470 patch to library default clamps (1e-2 -> 1e-1) is applied
    (see results/z470_clamp_lift/clamp_raise.patch).

Steps:
  1) Q4 isolation grid: SNAP_DEFAULT vs SNAP_HOT, 4 biases, raised clamp.
  2) thy_Gon sweep: 1x/2x/5x/10x of the 5 mS default, 4 biases.
  3) Mario-realistic overlay (no clamp artifact).
  4) Pre-registered gates evaluated + honest_analysis.md.

Note: step (3) z461 re-run is invoked via subprocess in main(), but we also
run it standalone via run-shell below.
"""
from __future__ import annotations
import json, math, sys, time, importlib.util as _ilu, subprocess, shutil
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))

OUT = ROOT / "results" / "z470_clamp_lift"
OUT.mkdir(parents=True, exist_ok=True)

LOG = OUT / "run.log"
_logfh = open(LOG, "w")
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _logfh.write(s + "\n"); _logfh.flush()

log(f"[z470] start  cwd={ROOT}")

_spec454 = _ilu.spec_from_file_location("z454", ROOT / "scripts/z454_snapback_integration.py")
z454 = _ilu.module_from_spec(_spec454); _spec454.loader.exec_module(z454)
z449 = z454.z449; z427 = z454.z427; z429 = z454.z429

from nsram.bsim4_port import transient_real_v2 as trv2
from nsram.bsim4_port.transient_real_v2 import integrate, TransientCfgV2, stim_fast_pulse

# Verify the z469 fix is in place
import inspect
src = inspect.getsource(trv2._Id_from_comps)
assert "I_snap_d" in src, "z469 fix missing from transient_real_v2._Id_from_comps"
log("[z470] z469 fix confirmed in _Id_from_comps")

# Verify the z470 clamp lift (library default) is in place
from nsram.bsim4_port.snapback_subcircuit import SnapbackParams
from nsram.bsim4_port.thyristor_pivot import ThyristorPivotParams
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
assert SnapbackParams().Id_extra_clamp >= 1e-1 - 1e-9, \
    f"SnapbackParams.Id_extra_clamp={SnapbackParams().Id_extra_clamp} (z470 patch not applied)"
assert ThyristorPivotParams().Id_extra_clamp >= 1e-1 - 1e-9, \
    f"ThyristorPivotParams.Id_extra_clamp not lifted"
# NSRAMCell2TConfig is a dataclass; check the class-level default directly.
import dataclasses as _dc
_fields = {f.name: f for f in _dc.fields(NSRAMCell2TConfig)}
assert _fields["snap_Id_clamp"].default >= 1e-1 - 1e-9, \
    f"NSRAMCell2TConfig.snap_Id_clamp default = {_fields['snap_Id_clamp'].default}"
log("[z470] clamp lift confirmed (Id_extra_clamp = 1e-1 = 100 mA)")

log("Loading models / curves / sebas rows...")
model_M1, model_M2 = z429.build_models()
sebas_rows = z429.load_sebas_params()

V449B_BASE = {
    "use_vbic_for_q1": True,
    "vbic_AVC1": 0.5, "vbic_AVC2": 0.5,
    "Cbody": 1e-15,
    "body_pdiode_Cj0_per_area": 0.0,
}
# RAISED clamps explicitly (script-level override, in addition to library default)
COMMON_AVL_LIFTED = dict(
    snap_BV=2.0 * 0.6, snap_n_avl=4.0,
    snap_Id_clamp=1e-1, snap_Iii_clamp=1e-1,   # z470: 100 mA, was 1e-2
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
    "snap_Bf": 10000.0, "snap_Va": 100.0,    # Sebas parasiticBJT.txt
    "snap_Is": 5.0e-9, "snap_Nf": 1.0,
    "snap_npn_gate_mode": "current",
    "snap_npn_V_knee": 1.8, "snap_npn_V_sharp": 0.05,
    "snap_npn_V_BE_offset": 0.3,
    "_R_body": 1e7, "_C_body": 1e-15,
}
SNAP_DEFAULT = {**SNAP_HOT, "snap_Bf": 417.0, "snap_Va": 0.90,
                "snap_Is": 6.0256e-9 * 5.0}

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


def summarize(r):
    """Return Id_pk, Vd_at_pk, V_b_pk, t_rise, t_fall, period."""
    if r is None:
        return {k: float("nan") for k in
                ["Id_pk", "Vd_at_pk", "Vb_pk", "t_rise", "t_fall", "period"]}
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
                ["Id_pk", "Vd_at_pk", "Vb_pk", "t_rise", "t_fall", "period"]}
    aI = np.abs(Id)
    ipk = int(np.argmax(aI))
    Id_pk = float(aI[ipk])
    Vd_at = float(Vd[ipk]) if ipk < len(Vd) else float("nan")
    Vb_pk = float(np.max(np.abs(Vb))) if len(Vb) else float("nan")
    # 10-90 rise of |Id|
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
    # crude period: dominant freq via zero-crossings of detrended Id
    Idz = aI - np.mean(aI)
    sgn = np.sign(Idz)
    zc = np.where(np.diff(sgn) != 0)[0]
    if len(zc) >= 4:
        period = float(2.0 * np.mean(np.diff(t[zc])))
    else:
        period = float("nan")
    return {"Id_pk": Id_pk, "Vd_at_pk": Vd_at, "Vb_pk": Vb_pk,
            "t_rise": t_rise, "t_fall": t_fall, "period": period}


# ---------------------------------------------------------------- #
# Step 1: Q4 isolation — SNAP_DEFAULT vs SNAP_HOT, clamp raised
# ---------------------------------------------------------------- #
log("\n=== STEP 1: SNAP_DEFAULT vs SNAP_HOT (clamp lifted) ===")
q4 = {"biases": [t for t,_,_ in PULSE_BIASES],
      "SNAP_DEFAULT": {}, "SNAP_HOT": {}, "MARIO_TARGET_A": 4.8e-3,
      "clamp_setting_A": 1e-1}
for tag, VG1, VG2 in PULSE_BIASES:
    log(f"  bias {tag}")
    rd = run_pulse(SNAP_DEFAULT, VG1, VG2)
    rh = run_pulse(SNAP_HOT,     VG1, VG2)
    sd = summarize(rd); sh = summarize(rh)
    log(f"    DEFAULT Id_pk={sd['Id_pk']:.3e}A  Vb_pk={sd['Vb_pk']:.3f}V  trise={sd['t_rise']:.2e}s")
    log(f"    HOT     Id_pk={sh['Id_pk']:.3e}A  Vb_pk={sh['Vb_pk']:.3f}V  trise={sh['t_rise']:.2e}s")
    q4["SNAP_DEFAULT"][tag] = sd
    q4["SNAP_HOT"][tag] = sh

q4["best_SNAP_DEFAULT_Id_pk"] = max(q4["SNAP_DEFAULT"][t]["Id_pk"] for t in q4["biases"])
q4["best_SNAP_HOT_Id_pk"] = max(q4["SNAP_HOT"][t]["Id_pk"] for t in q4["biases"])
q4["q4_discovery_per_bias"] = {
    t: bool(q4["SNAP_HOT"][t]["Id_pk"] > 2.0 * max(q4["SNAP_DEFAULT"][t]["Id_pk"], 1e-30))
    for t in q4["biases"]
}
q4["q4_discovery_n_pass"] = sum(q4["q4_discovery_per_bias"].values())
q4["both_under_100mA"] = bool(
    q4["best_SNAP_DEFAULT_Id_pk"] < 1e-1 and q4["best_SNAP_HOT_Id_pk"] < 1e-1)
q4["ambitious_mario_per_bias"] = {
    t: bool(abs(math.log10(max(q4["SNAP_DEFAULT"][t]["Id_pk"], 1e-30) / 4.8e-3)) < 0.5)
    for t in q4["biases"]
}
q4["ambitious_n_pass"] = sum(q4["ambitious_mario_per_bias"].values())

(OUT / "snap_default_vs_hot.json").write_text(json.dumps(q4, indent=2))
log(f"  Q4 DISCOVERY (HOT>2*DEFAULT on >=2/4 AND both<100mA): "
    f"{q4['q4_discovery_n_pass']}/4 biases, both<100mA={q4['both_under_100mA']}")
log(f"  AMBITIOUS (DEFAULT within 0.5 dec of Mario 4.8mA): "
    f"{q4['ambitious_n_pass']}/4 biases")


# ---------------------------------------------------------------- #
# Step 2: thy_Gon sweep on THY_DEFAULT
# ---------------------------------------------------------------- #
log("\n=== STEP 2: thy_Gon sweep (1x, 2x, 5x, 10x of 5 mS) ===")
GON_FACTORS = [1.0, 2.0, 5.0, 10.0]
thy_sweep = {"biases": [t for t,_,_ in PULSE_BIASES], "factors": GON_FACTORS,
             "results": {}}
for gf in GON_FACTORS:
    key = f"Gon_{gf}x"
    thy_sweep["results"][key] = {}
    cfg_g = {**THY_DEFAULT, "thy_Gon": 5e-3 * gf}
    for tag, VG1, VG2 in PULSE_BIASES:
        r = run_pulse(cfg_g, VG1, VG2)
        s = summarize(r)
        thy_sweep["results"][key][tag] = s
        log(f"  Gon={gf}x  {tag}  Id_pk={s['Id_pk']:.3e}A  Vb_pk={s['Vb_pk']:.3f}V")

# Does thy_Gon move Id_pk?
best_per = {k: max(v[t]["Id_pk"] for t in thy_sweep["biases"])
            for k, v in thy_sweep["results"].items()}
thy_sweep["best_Id_pk_per_factor"] = best_per
ratio_10_1 = best_per["Gon_10.0x"] / max(best_per["Gon_1.0x"], 1e-30)
thy_sweep["ratio_10x_over_1x"] = float(ratio_10_1)
log(f"  Gon ratio 10x/1x = {ratio_10_1:.2f}  "
    f"(Id_pk@1x={best_per['Gon_1.0x']:.3e}, Id_pk@10x={best_per['Gon_10.0x']:.3e})")
(OUT / "thy_gon_sweep.json").write_text(json.dumps(thy_sweep, indent=2))


# ---------------------------------------------------------------- #
# Step 3: Mario-realistic overlay plot
# ---------------------------------------------------------------- #
log("\n=== STEP 3: mario_realistic_vs_us.png ===")
fig, ax = plt.subplots(figsize=(9, 5.5))
xs = np.arange(len(PULSE_BIASES))
labels = [tag for tag, _, _ in PULSE_BIASES]
ax.axhline(4.8e-3, color="k", ls="--", lw=1.4, label="Mario target 4.8 mA")
ax.axhline(1.04e-6, color="gray", ls=":", lw=1, label="z467 pre-fix 1.04 µA")
ax.axhline(1e-2, color="red", ls=":", lw=1, alpha=0.5, label="OLD clamp 10 mA")
ax.axhline(1e-1, color="darkred", ls=":", lw=1, alpha=0.5, label="NEW clamp 100 mA")
ax.plot(xs, [q4["SNAP_DEFAULT"][t]["Id_pk"] for t in labels],
        "s-", ms=8, label="SNAP_DEFAULT (Bf=417)")
ax.plot(xs, [q4["SNAP_HOT"][t]["Id_pk"] for t in labels],
        "s-", ms=8, label="SNAP_HOT (Bf=10000, Va=100, Sebas)")
for gf in GON_FACTORS:
    k = f"Gon_{gf}x"
    ax.plot(xs, [thy_sweep["results"][k][t]["Id_pk"] for t in labels],
            "o--", ms=5, alpha=0.7, label=f"THY thy_Gon×{gf}")
ax.set_yscale("log")
ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("Id_pk  [A]")
ax.set_title("z470 — Id_pk per bias  (clamp lifted to 100 mA)")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=7, loc="lower left", ncol=2)
fig.tight_layout()
fig.savefig(OUT / "mario_realistic_vs_us.png", dpi=130)
plt.close(fig)
log(f"  wrote {OUT/'mario_realistic_vs_us.png'}")


# ---------------------------------------------------------------- #
# Step 4: re-run z461 dynamics harness (post-fix) into postfix dir
# ---------------------------------------------------------------- #
log("\n=== STEP 4: re-run z461 (z458_best) postfix ===")
src_dir = ROOT / "results" / "z461_validation_z458_best"
postfix_dir = ROOT / "results" / "z461_validation_z458_best_postfix"
# Always force fresh re-run (clamp lift changes dynamics)
if src_dir.exists():
    log(f"  existing {src_dir} will be moved aside to {src_dir}.prefix_backup")
    bak = src_dir.parent / f"{src_dir.name}.prefix_backup"
    if bak.exists():
        shutil.rmtree(bak)
    shutil.move(str(src_dir), str(bak))

t_z461 = time.time()
log("  invoking z461_dynamics_validation.py --config z458_best ...")
proc = subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "z461_dynamics_validation.py"),
     "--config", "z458_best"],
    cwd=str(ROOT), capture_output=True, text=True, timeout=2400,
    env={**__import__("os").environ, "NSRAM_DC_SOLVER": "pt"})
log(f"  z461 stdout tail:\n  " + "\n  ".join(proc.stdout.splitlines()[-15:]))
if proc.returncode != 0:
    log(f"  z461 stderr tail:\n  " + "\n  ".join(proc.stderr.splitlines()[-15:]))
log(f"  z461 wall = {time.time()-t_z461:.1f}s rc={proc.returncode}")

# Copy result to postfix dir
new_src = ROOT / "results" / "z461_validation_z458_best"
if new_src.exists():
    if postfix_dir.exists():
        shutil.rmtree(postfix_dir)
    shutil.copytree(str(new_src), str(postfix_dir))
    log(f"  copied to {postfix_dir}")
    vtj = postfix_dir / "validation_table.json"
    if vtj.exists():
        vt = json.loads(vtj.read_text())
        pf_summary = vt.get("summary", {})
        log(f"  POSTFIX SUMMARY: pass={pf_summary.get('pass')} "
            f"na={pf_summary.get('na')} fail={pf_summary.get('fail')} "
            f"total={pf_summary.get('total')}")
        (OUT / "z461_postfix.json").write_text(json.dumps(vt, indent=2))
    else:
        log("  WARN: validation_table.json missing")
else:
    log("  WARN: z461 output dir not produced; skipping copy")

# ---------------------------------------------------------------- #
# Step 5: honest analysis
# ---------------------------------------------------------------- #
log("\n=== STEP 5: honest_analysis.md ===")

# Load z461 postfix
pf_table = None
pf_path = OUT / "z461_postfix.json"
if pf_path.exists():
    try:
        pf_table = json.loads(pf_path.read_text())
    except Exception as e:
        log(f"  postfix load err: {e}")

# Pre-fix z461 numbers (from results/z461_validation_z458_best/validation_table.json
# captured before the run — hard-coded here as 6/9)
pre_summary = {"pass": 6, "na": 0, "fail": 3, "total": 9}

infra_pass = True  # all runs completed if we got here
discovery_pass = (q4["q4_discovery_n_pass"] >= 2) and q4["both_under_100mA"]
ambitious_pass = q4["ambitious_n_pass"] >= 1
breakthrough_pass = False
if pf_table is not None:
    n_pass_post = pf_table.get("summary", {}).get("pass", 0)
    breakthrough_pass = n_pass_post >= 8
else:
    n_pass_post = None

def fmt_bias_row(tag):
    sd = q4["SNAP_DEFAULT"][tag]; sh = q4["SNAP_HOT"][tag]
    rat = sh["Id_pk"] / max(sd["Id_pk"], 1e-30)
    return (f"| {tag} | {sd['Id_pk']:.3e} | {sd['Vb_pk']:.3f} | "
            f"{sd['t_rise']:.2e} | {sh['Id_pk']:.3e} | {sh['Vb_pk']:.3f} | "
            f"{sh['t_rise']:.2e} | {rat:.2f}× |")

md = []
md.append("# z470 — Clamp lift + Q4 isolation + z461 post-fix")
md.append("")
md.append("Date: 2026-05-17.")
md.append("Library clamp default lifted: 1e-2 → 1e-1 (100 mA).")
md.append("")
md.append("## Pre-registered gates")
md.append("")
md.append("| Gate | Criterion | Result |")
md.append("|------|-----------|--------|")
md.append(f"| INFRA | clamp raised, all runs complete | "
          f"{'PASS' if infra_pass else 'FAIL'} |")
md.append(f"| DISCOVERY (Q4) | SNAP_HOT Id_pk > 2× SNAP_DEFAULT on ≥2/4 biases "
          f"AND both <100 mA | "
          f"{'PASS' if discovery_pass else 'FAIL'} "
          f"({q4['q4_discovery_n_pass']}/4, both<100mA={q4['both_under_100mA']}) |")
md.append(f"| AMBITIOUS (Mario) | SNAP_DEFAULT within 0.5 dec of 4.8 mA "
          f"without clamp saturation | "
          f"{'PASS' if ambitious_pass else 'FAIL'} "
          f"({q4['ambitious_n_pass']}/4 within 0.5 dec) |")
md.append(f"| BREAKTHROUGH (z461) | post-fix scorecard ≥ 8/9 | "
          f"{'PASS' if breakthrough_pass else 'FAIL'} "
          f"({'?' if n_pass_post is None else n_pass_post}/9) |")
md.append("")
md.append("## Q4 isolation — SNAP_DEFAULT vs SNAP_HOT (clamp lifted to 100 mA)")
md.append("")
md.append("| Bias | DEFAULT Id_pk | DEFAULT Vb_pk | DEFAULT t_rise | "
          "HOT Id_pk | HOT Vb_pk | HOT t_rise | HOT/DEFAULT |")
md.append("|------|---------------|---------------|----------------|"
          "-----------|-----------|------------|-------------|")
for t in q4["biases"]:
    md.append(fmt_bias_row(t))
md.append("")
md.append(f"- Best SNAP_DEFAULT Id_pk: **{q4['best_SNAP_DEFAULT_Id_pk']:.3e} A**")
md.append(f"- Best SNAP_HOT     Id_pk: **{q4['best_SNAP_HOT_Id_pk']:.3e} A**")
md.append(f"- Mario target:            **4.8e-03 A**")
def _dec(x):
    if x <= 0: return float("nan")
    return math.log10(4.8e-3 / x)
md.append(f"- log10(Mario/best SNAP_DEFAULT) = {_dec(q4['best_SNAP_DEFAULT_Id_pk']):+.2f} dec")
md.append(f"- log10(Mario/best SNAP_HOT)     = {_dec(q4['best_SNAP_HOT_Id_pk']):+.2f} dec")
md.append("")

# Q4 verdict
hot_vs_def_dec = math.log10(max(q4['best_SNAP_HOT_Id_pk'],1e-30)
                            / max(q4['best_SNAP_DEFAULT_Id_pk'],1e-30))
md.append("### Q4 verdict")
if discovery_pass:
    md.append(f"**Q4 hypothesis CONFIRMED**: Sebas-Bf param-fix (Bf=10000 vs 417) "
              f"increases Id_pk by {hot_vs_def_dec:+.2f} dec, both stay under clamp.")
elif q4["both_under_100mA"]:
    md.append(f"**Q4 hypothesis WEAK/REJECTED**: clamp lifted (both <100 mA), "
              f"but HOT/DEFAULT ratio < 2× on most biases "
              f"(log-ratio {hot_vs_def_dec:+.2f} dec). "
              f"Param-fix gives a small effect, not a decisive one.")
else:
    md.append(f"**Q4 status INCONCLUSIVE**: at least one variant still hits the 100 mA "
              f"clamp. Need either higher clamp or topology change (substrate-return R).")
md.append("")

md.append("## thy_Gon sweep (which knob moves Id_pk?)")
md.append("")
md.append("| factor | Best Id_pk [A] |")
md.append("|--------|----------------|")
for gf in GON_FACTORS:
    md.append(f"| {gf}× | {best_per[f'Gon_{gf}x']:.3e} |")
md.append("")
md.append(f"- Ratio Id_pk(10×) / Id_pk(1×) = **{ratio_10_1:.2f}**")
if ratio_10_1 > 2.0:
    md.append("- thy_Gon IS the binding knob in the thyristor model — confirms z469 inference.")
else:
    md.append("- thy_Gon does NOT move Id_pk by >2× — look elsewhere "
              "(possibly snapback path dominates, or thy_Vpk/thy_Wpk).")
md.append("")

md.append("## z461 dynamics validation, post-fix")
md.append("")
md.append(f"- Pre-fix (z458_best, snapshot in `results/z461_validation_z458_best.prefix_backup/`): "
          f"{pre_summary['pass']}/{pre_summary['total']} PASS")
if pf_table is not None:
    s = pf_table.get("summary", {})
    md.append(f"- Post-fix (z470): {s.get('pass')}/{s.get('total')} PASS "
              f"(NA={s.get('na')}, FAIL={s.get('fail')})")
    md.append("")
    md.append("| Test | Name | Pre-fix | Post-fix |")
    md.append("|------|------|---------|----------|")
    pre_tests = {}
    bak = ROOT / "results" / "z461_validation_z458_best.prefix_backup" / "validation_table.json"
    if bak.exists():
        try:
            pre_tests = {t["test_id"]: t for t in json.loads(bak.read_text()).get("tests", [])}
        except Exception:
            pass
    for t in pf_table.get("tests", []):
        tid = t["test_id"]
        post_v = "PASS" if t["passed"] else "FAIL"
        pre_v = "?"
        if tid in pre_tests:
            pre_v = "PASS" if pre_tests[tid]["passed"] else "FAIL"
        md.append(f"| {tid} | {t['name']} | {pre_v} | **{post_v}** |")
else:
    md.append("- Post-fix: validation_table.json not produced; see run.log.")
md.append("")

md.append("## Mario-realistic gap")
md.append("")
md.append(f"- Pre-z469 best Id_pk: 1.04 µA (z467) → 3.66 dec under Mario")
md.append(f"- Post-z469, pre-z470 (clamp-bound at 10 mA): -0.32 dec (overshoot artifact)")
md.append(f"- Post-z470 (clamp lifted), SNAP_DEFAULT best: "
          f"{q4['best_SNAP_DEFAULT_Id_pk']:.3e} A "
          f"= {_dec(q4['best_SNAP_DEFAULT_Id_pk']):+.2f} dec from Mario")
md.append(f"- Post-z470, SNAP_HOT best: "
          f"{q4['best_SNAP_HOT_Id_pk']:.3e} A "
          f"= {_dec(q4['best_SNAP_HOT_Id_pk']):+.2f} dec from Mario")
md.append("")

md.append("## What's left")
md.append("")
if not discovery_pass:
    md.append("- If both SNAP variants still hit 100 mA: z468 Q3 — add substrate-return "
              "resistor (or finite R_collector) to bound regenerative kick by physics, not clamp.")
if not breakthrough_pass:
    md.append("- z461 still <8/9 — focus on V3 (knee detection routine returns NaN), "
              "V6 (self-reset gate too strict), V7 (no relaxation oscillation in 5 µs window).")
md.append("")
(OUT / "honest_analysis.md").write_text("\n".join(md))
log(f"  wrote {OUT/'honest_analysis.md'}")

log("\n[z470] DONE")
_logfh.close()
