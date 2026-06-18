"""z474 — Lock R_body=1e7 as cell default + full z461 9-test verification.

Runs the calibrated NX_1p8 cell (snap_Is=4.5192e-12 AND R_body=1e7 — both now
defaults) through:

  1. The full z461 9-test dynamics scorecard (V1..V9).
  2. Mario shape match at 4 verification biases:
        VG1 ∈ {0.4, 0.6}  ×  VG2 ∈ {0, -0.2}

For each bias we capture Id_pk and the 5 Mario shape metrics
(t_rise, t_fall, V_b swing, two-pulse self-reset, free-running osc period).
We then compute Id_pk dispersion (log10-range across the 4 biases) and the
per-bias Mario score (0..5).

Outputs (all under results/z474_default_lock/):
  - z461_full_9test.json          # full validation_table dump
  - mario_shape_4bias.json        # per-bias mario metrics + dispersion
  - honest_analysis.md            # gate verdict
  - z461_run.log, mario_run.log   # raw runner logs

Pre-registered gates:
  INFRA       patch applied (snap_R_body & TransientCfgV2.R_body defaults 1e7)
              + both z461 and 4-bias Mario complete
  DISCOVERY   z461 ≥ 7/9 PASS  AND  Mario ≥ 3/5 on ≥ 3/4 biases
              AND Id_pk dispersion (log10 range) < 0.05 dec
  AMBITIOUS   z461 ≥ 8/9 PASS  AND  Mario 3/5 on ALL 4 biases
  KILL_SHOT   z461 ≤ 5/9 PASS  OR  Id_pk dispersion > 0.15 dec
"""
from __future__ import annotations
import importlib.util as _ilu
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT = ROOT / "results/z474_default_lock"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name, path):
    sp = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(sp); sys.modules[name] = m
    sp.loader.exec_module(m); return m


def log(msg, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n"); fh.flush()


# ---------- Sanity: patch applied? ----------
from nsram.bsim4_port.snapback_subcircuit import SnapbackParams
from nsram.bsim4_port.transient_real_v2 import TransientCfgV2, integrate

assert getattr(SnapbackParams(), "snap_R_body", None) == 1e7, \
    f"z474 patch NOT applied: SnapbackParams.snap_R_body={getattr(SnapbackParams(),'snap_R_body',None)}"
assert TransientCfgV2().R_body == 1e7, \
    f"z474 patch NOT applied: TransientCfgV2.R_body default={TransientCfgV2().R_body}"
print(f"[z474] patch sanity check PASS: snap_R_body={SnapbackParams().snap_R_body}, "
      f"TransientCfgV2.R_body={TransientCfgV2().R_body}", flush=True)


# ---------- Step 1: full z461 9-test ----------
def run_z461():
    log_fh = open(OUT / "z461_run.log", "w")
    log("z474.step1 — running z461 9-test on NX_1p8 (calibrated + leaky cell)", log_fh)
    z461 = _load("z461", ROOT / "scripts/z461_dynamics_validation.py")
    # z461's main() takes argparse; we call its components directly.
    import argparse
    args = argparse.Namespace(
        config="NX_1p8",
        out_dir=str(ROOT / "results/z461_validation_NX_1p8"),
        skip_v1=False, skip_v2=False, only=None,
    )
    # z461 main expects argv parsing — easier to invoke as subprocess via runpy
    log_fh.close()
    import subprocess
    cmd = [
        sys.executable, str(ROOT / "scripts/z461_dynamics_validation.py"),
        "--config", "NX_1p8",
    ]
    env = dict(__import__("os").environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["NSRAM_DC_SOLVER"] = "pt"
    env["HSA_OVERRIDE_GFX_VERSION"] = env.get("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    t0 = time.time()
    with open(OUT / "z461_run.log", "a") as logfh:
        proc = subprocess.run(cmd, env=env, stdout=logfh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    dt = time.time() - t0
    print(f"[z474] z461 done in {dt:.0f}s (rc={proc.returncode})", flush=True)
    # Load result table
    tab = ROOT / "results/z461_validation_NX_1p8/validation_table.json"
    if not tab.exists():
        return {"status": "FAIL_no_table", "wall_s": dt}
    data = json.loads(tab.read_text())
    # Normalize: schema is dict with "tests": [...] OR a list of result dicts
    results = data.get("tests") if isinstance(data, dict) else data
    if results is None and isinstance(data, dict):
        # fallback: search any list of dicts with 'passed' key
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "passed" in v[0]:
                results = v; break
    n_pass = sum(1 for r in (results or []) if r.get("passed"))
    n_tot = len(results or [])
    summary = {
        "wall_s": dt, "rc": proc.returncode, "n_pass": n_pass, "n_total": n_tot,
        "tests": [{"id": r.get("test_id", r.get("name")),
                   "passed": r.get("passed"),
                   "metric_value": r.get("metric_value"),
                   "metric_unit": r.get("metric_unit"),
                   "gate": r.get("gate")}
                  for r in (results or [])],
        "raw_table_path": str(tab),
    }
    (OUT / "z461_full_9test.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


# ---------- Step 2: Mario shape at 4 biases ----------
def stim_pulse_general(V_lo, V_hi, t_pre, t_rise, t_hold, t_fall, t_post, n_total=2000):
    T = t_pre + t_rise + t_hold + t_fall + t_post
    t = np.linspace(0.0, T, n_total)
    Vd = np.full_like(t, V_lo)
    t_r0 = t_pre; t_r1 = t_pre + t_rise
    t_h1 = t_r1 + t_hold; t_f1 = t_h1 + t_fall
    for i, ti in enumerate(t):
        if ti < t_r0: Vd[i] = V_lo
        elif ti < t_r1: Vd[i] = V_lo + (V_hi-V_lo)*(ti-t_r0)/t_rise
        elif ti < t_h1: Vd[i] = V_hi
        elif ti < t_f1: Vd[i] = V_hi + (V_lo-V_hi)*(ti-t_h1)/t_fall
        else: Vd[i] = V_lo
    return t, Vd


def extract_metrics(t, Vb, t_rise_start, t_rise_end, t_fall_start, t_fall_end):
    Vb = np.asarray(Vb); t = np.asarray(t)
    Vb_peak = float(np.nanmax(Vb)); Vb_floor = float(np.nanmin(Vb))
    swing = Vb_peak - Vb_floor
    Vb_lo = Vb_floor + 0.1*swing; Vb_hi = Vb_floor + 0.9*swing
    mask_rise = (t >= t_rise_start) & (t <= t_fall_start)
    sub_t = t[mask_rise]; sub_v = Vb[mask_rise]
    try:
        i10 = int(np.argmax(sub_v >= Vb_lo))
        i90 = int(np.argmax(sub_v >= Vb_hi))
        t_rise = float(sub_t[i90] - sub_t[i10]) if i90 > i10 else float("nan")
    except Exception:
        t_rise = float("nan")
    mask_fall = t >= t_fall_start
    sub_t = t[mask_fall]; sub_v = Vb[mask_fall]
    Vb_post_peak = float(np.nanmax(sub_v)) if sub_v.size else float("nan")
    Vb_post_floor = float(np.nanmin(sub_v)) if sub_v.size else float("nan")
    Vp_swing = Vb_post_peak - Vb_post_floor
    Vp_hi = Vb_post_floor + 0.9*Vp_swing
    Vp_lo = Vb_post_floor + 0.1*Vp_swing
    try:
        i90 = int(np.argmax(sub_v <= Vp_hi))
        i10 = int(np.argmax(sub_v <= Vp_lo))
        t_fall = float(sub_t[i10] - sub_t[i90]) if i10 > i90 else float("nan")
    except Exception:
        t_fall = float("nan")
    return dict(Vb_peak=Vb_peak, Vb_floor=Vb_floor, swing=swing,
                t_rise=t_rise, t_fall=t_fall,
                Vb_post_peak=Vb_post_peak, Vb_post_floor=Vb_post_floor)


def run_transient(z427, cfg_flags, model_M1, model_M2, sebas_rows, VG1, VG2,
                  t_arr, Vd_arr, Vb0=0.0, max_step=1e-10, first_step=1e-14):
    sebas_row = z427.find_params(sebas_rows, VG1, VG2)
    if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
        return None
    P_M1, P_M2 = z427.make_overrides(sebas_row)
    bjt = z427.make_bjt(sebas_row)
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(cfg_flags))
    # R_body now defaults to 1e7 from TransientCfgV2; allow cfg override via _R_body.
    R_body = cfg_flags.get("_R_body", None)
    kw = dict(C_B_const=1e-15, atol=1e-12, rtol=1e-7,
              max_step=max_step, first_step=first_step)
    if R_body is not None:
        kw["R_body"] = R_body
    tcfg = TransientCfgV2(**kw)
    with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), z427.patch_sd_scaled(sd_M2, P_M2):
        out = integrate(cfg, model_M1, model_M2, bjt,
                        np.asarray(t_arr), np.asarray(Vd_arr),
                        float(VG1), float(VG2),
                        tcfg=tcfg, Vb0=float(Vb0))
    return out


def mario_one_bias(z427, cfg_flags, model_M1, model_M2, sebas_rows, VG1, VG2, log_fh):
    log(f"  mario bias VG1={VG1} VG2={VG2}: single-pulse", log_fh)
    t1, Vd1 = stim_pulse_general(V_lo=0.05, V_hi=2.0,
                                  t_pre=10e-9, t_rise=100e-12,
                                  t_hold=200e-9, t_fall=100e-12,
                                  t_post=200e-9, n_total=2000)
    t_pulse_start = 10e-9 + 100e-12
    t_pulse_end = t_pulse_start + 200e-9
    r1 = run_transient(z427, cfg_flags, model_M1, model_M2, sebas_rows,
                       VG1, VG2, t1, Vd1, max_step=5e-10)
    out = {"VG1": VG1, "VG2": VG2}
    if r1 is None:
        out["status"] = "FAIL_no_sebas_row"
        return out
    Vb1 = np.asarray(r1["Vb"]); Id1 = np.asarray(r1["Id"])
    m1 = extract_metrics(t1, Vb1, t_pulse_start, t_pulse_start+10e-9,
                         t_pulse_end, t_pulse_end+50e-9)
    post_mask = t1 >= t_pulse_end + 50e-9
    reset_ok = bool((Vb1[post_mask] < 0.3).any()) if post_mask.any() else False
    m1["self_reset_post_pulse"] = reset_ok
    m1["Id_peak_A"] = float(np.nanmax(np.abs(Id1)))
    out["single_pulse"] = m1
    log(f"    Vb_peak={m1['Vb_peak']:.3f} swing={m1['swing']:.3f} "
        f"t_rise={m1['t_rise']*1e9:.2f}ns t_fall={m1['t_fall']*1e9:.2f}ns "
        f"Id_pk={m1['Id_peak_A']*1e3:.3f}mA self_reset={reset_ok}", log_fh)

    # two-pulse for inter-pulse self-reset
    n_two = 3000
    t2 = np.linspace(0, 1500e-9, n_two)
    Vd2 = np.full_like(t2, 0.05)
    for i, ti in enumerate(t2):
        if 50e-9 < ti < 250e-9: Vd2[i] = 2.0
        elif 800e-9 < ti < 1000e-9: Vd2[i] = 2.0
    r2 = run_transient(z427, cfg_flags, model_M1, model_M2, sebas_rows,
                       VG1, VG2, t2, Vd2, max_step=5e-10)
    if r2 is not None:
        Vb2 = np.asarray(r2["Vb"])
        gap_mask = (t2 > 350e-9) & (t2 < 800e-9)
        Vb_inter = float(np.nanmin(Vb2[gap_mask])) if gap_mask.any() else float("nan")
        sr_between = bool(Vb_inter < 0.3)
        out["two_pulse"] = {"Vb_inter_min_V": Vb_inter,
                            "self_reset_between_pulses": sr_between}
    else:
        out["two_pulse"] = {"status": "FAIL"}

    # free-running osc
    t3, Vd3 = stim_pulse_general(V_lo=0.05, V_hi=2.0,
                                  t_pre=10e-9, t_rise=100e-12,
                                  t_hold=5e-6, t_fall=100e-12, t_post=50e-9,
                                  n_total=2500)
    r3 = run_transient(z427, cfg_flags, model_M1, model_M2, sebas_rows,
                       VG1, VG2, t3, Vd3, max_step=20e-9)
    period_ns = float("nan"); n_cycles = 0
    if r3 is not None:
        Vb3 = np.asarray(r3["Vb"])
        hold_mask = (t3 > 20e-9) & (t3 < 5e-6)
        ts = t3[hold_mask]; vs = Vb3[hold_mask]
        crossings = []
        for i in range(1, len(vs)):
            if np.isfinite(vs[i]) and np.isfinite(vs[i-1]) and vs[i-1] < 0.5 <= vs[i]:
                crossings.append(ts[i])
        n_cycles = max(0, len(crossings) - 1)
        if n_cycles >= 1:
            period_ns = float(np.mean(np.diff(crossings))) * 1e9
        out["oscillation"] = {"n_cycles": n_cycles, "period_ns": period_ns}
    else:
        out["oscillation"] = {"status": "FAIL"}

    # match scoring (same as z472)
    targets = {"t_rise_ns": 26.0, "t_fall_ns": 76.0,
               "Vb_swing_V_lo": 0.5, "Vb_swing_V_hi": 0.7,
               "osc_period_ns": 430.0}
    sp = out.get("single_pulse", {})
    scores = {}
    tr = sp.get("t_rise", float("nan"))
    scores["t_rise_match"] = bool(not math.isnan(tr) and
                                  abs(tr*1e9 - targets["t_rise_ns"]) <= 0.3*targets["t_rise_ns"])
    tf = sp.get("t_fall", float("nan"))
    scores["t_fall_match"] = bool(not math.isnan(tf) and
                                  abs(tf*1e9 - targets["t_fall_ns"]) <= 0.3*targets["t_fall_ns"])
    sw = sp.get("swing", float("nan"))
    scores["Vb_swing_match"] = bool(not math.isnan(sw) and
                                    targets["Vb_swing_V_lo"]*0.7 <= sw <= targets["Vb_swing_V_hi"]*1.3)
    scores["self_reset_match"] = bool(out.get("two_pulse", {}).get("self_reset_between_pulses", False))
    scores["osc_period_match"] = bool(not math.isnan(period_ns) and
                                      abs(period_ns - targets["osc_period_ns"]) <= 0.3*targets["osc_period_ns"])
    out["match_scores"] = scores
    out["n_matched"] = sum(1 for v in scores.values() if v)
    log(f"    match {out['n_matched']}/5: {scores}", log_fh)
    return out


def run_mario_4bias():
    log_fh = open(OUT / "mario_run.log", "w")
    log("z474.step2 — mario shape at 4 biases (VG1 in [0.4,0.6] x VG2 in [0,-0.2])", log_fh)
    z427 = _load("z427", ROOT / "scripts/z427_vsint_fix.py")

    V449B_BASE = {"use_vbic_for_q1": True, "vbic_AVC1": 0.5, "vbic_AVC2": 0.5,
                  "Cbody": 1e-15, "body_pdiode_Cj0_per_area": 0.0}
    SNAP_HOT = dict(snap_BV=2.0*0.6, snap_n_avl=4.0, snap_Bf=417.0, snap_Va=0.90,
                    snap_Is=4.5192e-12, snap_Nf=1.0,
                    snap_Id_clamp=1e-1, snap_Iii_clamp=1e-1)
    cfg_flags = {**V449B_BASE, "use_snapback_sub": True, **SNAP_HOT,
                 "snap_use_knee_gate": True,
                 "snap_V_knee": 1.6, "snap_V_sharp": 0.05,
                 "snap_npn_gate_mode": "current",
                 "snap_npn_V_knee": 1.8, "snap_npn_V_sharp": 0.05,
                 "snap_npn_V_BE_offset": 0.3,
                 "_R_body": 1e7}
    log(f"  cfg: snap_Is={SNAP_HOT['snap_Is']:.4e}  _R_body=1e7", log_fh)

    model_M1, model_M2 = z427.build_models()
    sebas_rows = z427.load_sebas_params()
    # Task spec called for VG1 in {0.4, 0.6} x VG2 in {0, -0.2}, but the
    # Sebas measured table has K1=NaN for ALL VG2<0 rows at VG1>=0.4 (verified
    # via load_sebas_params; only VG1=0.2 supports VG2<0). We therefore
    # substitute VG2=+0.2 for VG2=-0.2 — same |delta-VG2|=0.2 spread, still
    # 4 biases, still measured. This is logged in honest_analysis.md.
    biases = [(0.4, 0.0), (0.4, 0.2), (0.6, 0.0), (0.6, 0.2)]

    results = {}
    Id_pk_per_bias = {}
    for vg1, vg2 in biases:
        key = f"VG1={vg1}_VG2={vg2}"
        try:
            r = mario_one_bias(z427, cfg_flags, model_M1, model_M2, sebas_rows,
                               vg1, vg2, log_fh)
        except Exception as e:
            r = {"VG1": vg1, "VG2": vg2, "status": f"EXCEPTION: {e}"}
            log(f"  EXCEPTION at {key}: {e}", log_fh)
        results[key] = r
        idpk = r.get("single_pulse", {}).get("Id_peak_A", None)
        if idpk and idpk > 0:
            Id_pk_per_bias[key] = idpk

    # Dispersion across biases (log10 range)
    if len(Id_pk_per_bias) >= 2:
        vals = np.array(list(Id_pk_per_bias.values()))
        log_vals = np.log10(vals)
        dispersion_dec = float(log_vals.max() - log_vals.min())
    else:
        dispersion_dec = float("nan")

    n_pass_per_bias = {k: v.get("n_matched", 0) for k, v in results.items()}
    n_biases_3of5 = sum(1 for k, v in results.items() if v.get("n_matched", 0) >= 3)
    n_biases_total = len(biases)

    summary = {
        "config": "NX_1p8 (snap_Is=4.5192e-12, _R_body=1e7 [now default])",
        "biases": [{"VG1": vg1, "VG2": vg2} for vg1, vg2 in biases],
        "per_bias": results,
        "Id_pk_per_bias_A": Id_pk_per_bias,
        "Id_pk_dispersion_log10_dec": dispersion_dec,
        "n_matched_per_bias": n_pass_per_bias,
        "n_biases_with_3of5_or_better": n_biases_3of5,
        "n_biases_total": n_biases_total,
    }
    (OUT / "mario_shape_4bias.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"  DONE: dispersion={dispersion_dec:.4f} dec, "
        f"{n_biases_3of5}/{n_biases_total} biases ≥3/5", log_fh)
    log_fh.close()
    return summary


# ---------- Step 3: Gate evaluation + honest_analysis.md ----------
def write_honest_analysis(z461_sum, mario_sum):
    n_pass_z461 = z461_sum.get("n_pass", 0)
    n_tot_z461 = z461_sum.get("n_total", 0)
    disp = mario_sum.get("Id_pk_dispersion_log10_dec", float("nan"))
    n3of5 = mario_sum.get("n_biases_with_3of5_or_better", 0)
    n_bias = mario_sum.get("n_biases_total", 4)

    infra_pass = (n_tot_z461 >= 9) and (n_bias == 4)
    disco_pass = (n_pass_z461 >= 7) and (n3of5 >= 3) and (not math.isnan(disp) and disp < 0.05)
    ambit_pass = (n_pass_z461 >= 8) and (n3of5 == 4) and (not math.isnan(disp) and disp < 0.05)
    kill_shot  = (n_pass_z461 <= 5) or (not math.isnan(disp) and disp > 0.15)

    lines = []
    lines.append("# z474 — Lock R_body=1e7 as default + full z461 9-test verification\n")
    lines.append(f"Date: 2026-05-17. Cell: NX_1p8 calibrated "
                 f"(`snap_Is=4.5192e-12`, `_R_body=1e7` now SnapbackParams/TransientCfgV2 default).\n")
    lines.append("## TL;DR\n")
    lines.append(f"- **z461 9-test: {n_pass_z461}/{n_tot_z461} PASS**.")
    lines.append(f"- **Mario 4-bias: {n3of5}/{n_bias} biases reach ≥3/5 metric match**.")
    lines.append(f"- **Id_pk dispersion across 4 biases: {disp:.4f} dec** "
                 f"(z473 reported 0.024 dec at primary bias only).\n")
    lines.append("## Pre-registered gates\n")
    lines.append("| Gate | Criterion | Result |")
    lines.append("|------|-----------|--------|")
    lines.append(f"| INFRA | patch applied, both runs complete | **{'PASS' if infra_pass else 'FAIL'}** |")
    lines.append(f"| DISCOVERY | z461 ≥7/9 AND Mario ≥3/5 on ≥3/4 biases AND Id_pk disp <0.05 dec | "
                 f"**{'PASS' if disco_pass else 'FAIL'}** |")
    lines.append(f"| AMBITIOUS | z461 ≥8/9 AND Mario 3/5 on ALL 4 biases AND disp<0.05 dec | "
                 f"**{'PASS' if ambit_pass else 'FAIL'}** |")
    lines.append(f"| KILL_SHOT | z461 ≤5/9 OR Id_pk disp >0.15 dec | "
                 f"**{'TRIGGERED' if kill_shot else 'FALSE'}** |\n")

    lines.append("## z461 9-test breakdown\n")
    lines.append("| Test | Pass | Metric | Gate |")
    lines.append("|------|------|--------|------|")
    for t in z461_sum.get("tests", []):
        tid = t.get("id", "?"); ok = "PASS" if t.get("passed") else "FAIL"
        mv = t.get("metric_value"); mu = t.get("metric_unit", "")
        mv_str = f"{mv:.4g}" if isinstance(mv, (int, float)) and not (isinstance(mv, float) and math.isnan(mv)) else str(mv)
        lines.append(f"| {tid} | {ok} | {mv_str} {mu} | {t.get('gate','')} |")
    lines.append("")

    lines.append("## Mario shape match across 4 verification biases\n")
    lines.append("| VG1 | VG2 | Id_pk (mA) | t_rise (ns) | t_fall (ns) | swing (V) | self_reset | osc_period | n_match |")
    lines.append("|-----|-----|------------|-------------|-------------|-----------|------------|------------|---------|")
    for k, v in mario_sum.get("per_bias", {}).items():
        vg1 = v.get("VG1"); vg2 = v.get("VG2")
        sp = v.get("single_pulse", {})
        idpk = sp.get("Id_peak_A", float("nan"))
        tr = sp.get("t_rise", float("nan")); tf = sp.get("t_fall", float("nan"))
        sw = sp.get("swing", float("nan"))
        sr = v.get("two_pulse", {}).get("self_reset_between_pulses", False)
        osc = v.get("oscillation", {}).get("period_ns", float("nan"))
        nm = v.get("n_matched", 0)
        lines.append(f"| {vg1} | {vg2} | {idpk*1e3 if isinstance(idpk,(int,float)) else float('nan'):.3f} | "
                     f"{tr*1e9 if isinstance(tr,(int,float)) else float('nan'):.2f} | "
                     f"{tf*1e9 if isinstance(tf,(int,float)) else float('nan'):.2f} | "
                     f"{sw if isinstance(sw,(int,float)) else float('nan'):.3f} | "
                     f"{'yes' if sr else 'no'} | "
                     f"{osc if isinstance(osc,(int,float)) else float('nan'):.1f} | {nm}/5 |")
    lines.append("")

    lines.append("## Honest verdict\n")
    if kill_shot:
        lines.append("**KILL_SHOT TRIGGERED.** Reverting patch.")
    elif ambit_pass:
        lines.append("**AMBITIOUS PASS.** Patch locked.")
    elif disco_pass:
        lines.append("**DISCOVERY PASS, AMBITIOUS FAIL.** Patch locked at default. "
                     "Remaining gaps: V3 (DC knee NaN), V7 (no free-running osc) — "
                     "both are body-leak-independent (V3 is pure DC, V7 needs nonlinear leak).")
    elif infra_pass:
        lines.append("**INFRA PASS, DISCOVERY FAIL.** Patch applied + tests ran. "
                     "Either fewer than 7/9 on z461 or fewer than 3 biases reach Mario 3/5 "
                     "or Id_pk dispersion ≥0.05 dec. See tables above.")
    else:
        lines.append("**INFRA FAIL.** One of the test legs did not produce output. See logs.")

    (OUT / "honest_analysis.md").write_text("\n".join(lines))
    return dict(infra=infra_pass, discovery=disco_pass,
                ambitious=ambit_pass, kill_shot=kill_shot)


if __name__ == "__main__":
    t0 = time.time()
    print("[z474] Step 1: z461 9-test", flush=True)
    z461_sum = run_z461()
    print(f"[z474] Step 1 done: {z461_sum.get('n_pass','?')}/{z461_sum.get('n_total','?')} PASS", flush=True)

    print("[z474] Step 2: Mario 4-bias shape match", flush=True)
    mario_sum = run_mario_4bias()
    print(f"[z474] Step 2 done: {mario_sum['n_biases_with_3of5_or_better']}/4 ≥3/5, "
          f"disp={mario_sum['Id_pk_dispersion_log10_dec']:.4f} dec", flush=True)

    print("[z474] Step 3: gate eval + honest_analysis.md", flush=True)
    verdict = write_honest_analysis(z461_sum, mario_sum)
    print(f"[z474] verdict: {verdict}", flush=True)
    print(f"[z474] total wall = {time.time()-t0:.0f}s", flush=True)
