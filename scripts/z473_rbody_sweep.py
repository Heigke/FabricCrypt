"""z473 — R_body sweep to enable reset path; flip V3/V6/V7 triplet.

Re-uses z472 transient/DC harness on calibrated NX_1p8 (snap_Is=4.5192e-12)
and parametrically lowers R_body to provide body-to-ground leak path.

Outputs (all to results/z473_rbody_sweep/):
    rbody_sweep.json
    v3v6v7_post_sweep.json
    mario_shape_v2.json
    transient_overlay_with_reset.png
    honest_analysis.md
    patch.diff (deliberate, manual)
"""
from __future__ import annotations
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

import importlib.util as _ilu


def _load(name, path):
    sp = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(sp); sys.modules[name] = m
    sp.loader.exec_module(m); return m


z427 = _load("z427", ROOT / "scripts/z427_vsint_fix.py")
z429 = _load("z429", ROOT / "scripts/z429_multisolver_debug.py")
z449 = _load("z449", ROOT / "scripts/z449_vbic_bdf_combo.py")

from nsram.bsim4_port import transient_real_v2 as trv2
from nsram.bsim4_port.transient_real_v2 import integrate, TransientCfgV2

# ----------- Config (NX_1p8 from z461) -----------
V449B_BASE = {
    "use_vbic_for_q1": True, "vbic_AVC1": 0.5, "vbic_AVC2": 0.5,
    "Cbody": 1e-15, "body_pdiode_Cj0_per_area": 0.0,
}
SNAP_HOT = dict(snap_BV=2.0*0.6, snap_n_avl=4.0, snap_Bf=417.0, snap_Va=0.90,
                snap_Is=4.5192e-12, snap_Nf=1.0,
                snap_Id_clamp=1e-1, snap_Iii_clamp=1e-1)


def make_NX_1p8():
    return {**V449B_BASE, "use_snapback_sub": True, **SNAP_HOT,
            "snap_use_knee_gate": True,
            "snap_V_knee": 1.6, "snap_V_sharp": 0.05,
            "snap_npn_gate_mode": "current",
            "snap_npn_V_knee": 1.8, "snap_npn_V_sharp": 0.05,
            "snap_npn_V_BE_offset": 0.3}


def stim_pulse(V_lo, V_hi, t_pre, t_rise, t_hold, t_fall, t_post, n_total=2000):
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


def run_transient(cfg_flags, model_M1, model_M2, sebas_rows, VG1, VG2,
                  t_arr, Vd_arr, R_body=None, Vb0=0.0,
                  max_step=1e-10, first_step=1e-14):
    sebas_row = z427.find_params(sebas_rows, VG1, VG2)
    if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
        return None
    P_M1, P_M2 = z427.make_overrides(sebas_row)
    bjt = z427.make_bjt(sebas_row)
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(cfg_flags))
    cfg.Cbody = 1e-15
    tcfg = TransientCfgV2(C_B_const=1e-15,
                          atol=1e-12, rtol=1e-7,
                          max_step=max_step, first_step=first_step,
                          R_body=R_body)
    z449._VBIC_CTX["cfg"] = cfg
    z449._VBIC_CTX["bjt"] = bjt
    try:
        with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), \
             z427.patch_sd_scaled(sd_M2, P_M2):
            out = integrate(cfg, model_M1, model_M2, bjt,
                            np.asarray(t_arr), np.asarray(Vd_arr),
                            float(VG1), float(VG2),
                            tcfg=tcfg, Vb0=float(Vb0))
    finally:
        z449._VBIC_CTX["cfg"] = None
        z449._VBIC_CTX["bjt"] = None
    return out


def measure_decay_tau(t, Vb, t_release):
    """Fit exponential decay tau on V_b after t_release."""
    t = np.asarray(t); Vb = np.asarray(Vb)
    post = t >= t_release
    if not post.any(): return float("nan"), float("nan")
    tp = t[post] - t_release
    vp = Vb[post]
    # baseline floor: tail mean
    Vb_floor = float(np.nanmin(vp))
    Vb_start = float(vp[0])
    if Vb_start - Vb_floor < 1e-3:
        return float("inf"), Vb_floor   # already reset / no decay
    # time to fall to 1/e of (Vb_start - Vb_floor)
    target = Vb_floor + (Vb_start - Vb_floor) / math.e
    below = vp <= target
    if not below.any():
        return float("inf"), Vb_floor
    idx = int(np.argmax(below))
    return float(tp[idx]), Vb_floor


def primary_pulse(n_total=2000):
    """Single primary pulse @ VG1=0.6, VG2=0, Vd=2V, 200ns hold."""
    return stim_pulse(V_lo=0.05, V_hi=2.0,
                      t_pre=10e-9, t_rise=100e-12,
                      t_hold=200e-9, t_fall=100e-12,
                      t_post=300e-9, n_total=n_total)


# ---------- Step 1: sweep ----------
def step1_sweep(cfg_flags, model_M1, model_M2, sebas_rows, log):
    """5-point R_body sweep at primary bias."""
    log("Step 1: R_body sweep at VG1=0.6, VG2=0, Vd=2V pulse")
    R_values = [None, 1e9, 1e8, 1e7, 1e6]   # None == default (infinity, no leak)
    rows = []
    t_arr, Vd_arr = primary_pulse(n_total=1500)
    t_release = 10e-9 + 100e-12 + 200e-9 + 100e-12   # end of fall
    for R in R_values:
        tag = "inf (default)" if R is None else f"{R:.0e}"
        t0 = time.time()
        r = run_transient(cfg_flags, model_M1, model_M2, sebas_rows,
                          0.6, 0.0, t_arr, Vd_arr, R_body=R, max_step=5e-10)
        dt = time.time() - t0
        if r is None:
            rows.append({"R_body": R, "tag": tag, "status": "fail_no_sebas"})
            log(f"  R={tag}: FAIL no sebas row")
            continue
        Vb = np.asarray(r["Vb"]); Id = np.asarray(r["Id"])
        Id_pk_mA = float(np.nanmax(np.abs(Id)) * 1e3)
        Vb_pk = float(np.nanmax(Vb))
        tau_ns, Vb_floor = measure_decay_tau(t_arr, Vb, t_release)
        # Reset within 200 ns post-fall?
        post_200 = (t_arr >= t_release + 200e-9)
        Vb_at_200 = float(np.nanmean(Vb[post_200][:5])) if post_200.any() else float("nan")
        reset_200 = bool(Vb_at_200 < 0.4)
        row = {
            "R_body": R, "tag": tag,
            "Id_pk_mA": Id_pk_mA,
            "Vb_pk_V": Vb_pk,
            "tau_decay_ns": (None if math.isinf(tau_ns) else tau_ns * 1e9),
            "Vb_at_200ns_post": Vb_at_200,
            "reset_lt_0p4_within_200ns": reset_200,
            "wall_s": dt,
        }
        rows.append(row)
        log(f"  R={tag}: Id_pk={Id_pk_mA:.2f}mA Vb_pk={Vb_pk:.3f}V "
            f"tau={row['tau_decay_ns']}ns Vb@200ns={Vb_at_200:.3f}V reset={reset_200} "
            f"({dt:.1f}s)")
    return rows


def pick_R_body(rows, log, target_Id_pk=4.23, max_drift_dec=0.15):
    """Choose lowest R_body that holds Id_pk inside drift gate and resets."""
    log("Step 3: pick R_body sweet spot")
    log(f"  target Id_pk={target_Id_pk}mA, drift gate < {max_drift_dec} dec")
    candidate = None
    for r in rows:
        if r.get("status") == "fail_no_sebas":
            continue
        Id = r["Id_pk_mA"]
        if Id <= 0:
            continue
        drift = abs(math.log10(Id / target_Id_pk))
        r["Id_pk_drift_dec"] = drift
        in_band = (drift < max_drift_dec)
        resets = r["reset_lt_0p4_within_200ns"]
        log(f"    R={r['tag']}: Id_pk_drift={drift:.3f}dec in_band={in_band} resets={resets}")
        if in_band and resets and candidate is None:
            candidate = r
    if candidate is None:
        # fall back: lowest R that still in_band (even if no reset)
        for r in rows:
            if r.get("status") == "fail_no_sebas":
                continue
            if r.get("Id_pk_drift_dec", 99) < max_drift_dec:
                if candidate is None or (r["R_body"] is not None and
                                          (candidate["R_body"] is None or
                                           r["R_body"] < candidate["R_body"])):
                    candidate = r
    return candidate


# ---------- Step 4: V3 / V6 / V7 on chosen R_body ----------
def run_V3(cfg_flags, model_M1, model_M2, sebas_rows, log):
    log("V3 — DC knee (R_body has NO effect on DC; reported for completeness)")
    sebas_row = z427.find_params(sebas_rows, 0.6, 0.0)
    if sebas_row is None:
        return {"test": "V3", "V_knee": None, "passed": False,
                "notes": "no sebas row"}
    P_M1, P_M2 = z427.make_overrides(sebas_row)
    bjt = z427.make_bjt(sebas_row)
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(cfg_flags))
    Vd_seq_fwd = np.linspace(0.05, 2.0, 60)
    Id_fwd = np.zeros_like(Vd_seq_fwd)
    Vd_seq_bwd = Vd_seq_fwd[::-1]
    Id_bwd = np.zeros_like(Vd_seq_bwd)
    with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), \
         z427.patch_sd_scaled(sd_M2, P_M2):
        Vb_warm = 0.0
        for i, Vd_f in enumerate(Vd_seq_fwd):
            r = z429.run_vsint_pinned(cfg, model_M1, model_M2, bjt,
                                       float(Vd_f), 0.6, 0.0,
                                       Vsint_pin=0.0, Vb_init=Vb_warm)
            Id_fwd[i] = abs(r["Id"]) if r.get("Id") is not None else 0.0
            Vb_warm = r["Vb"] if r["converged"] else 0.0
        # Backward sweep continues from last Vb (latched)
        for i, Vd_b in enumerate(Vd_seq_bwd):
            r = z429.run_vsint_pinned(cfg, model_M1, model_M2, bjt,
                                       float(Vd_b), 0.6, 0.0,
                                       Vsint_pin=0.0, Vb_init=Vb_warm)
            Id_bwd[i] = abs(r["Id"]) if r.get("Id") is not None else 0.0
            Vb_warm = r["Vb"] if r["converged"] else Vb_warm
    target = 10e-6
    above_fwd = np.where(Id_fwd >= target)[0]
    V_knee_fwd = float(Vd_seq_fwd[above_fwd[0]]) if len(above_fwd) else float("nan")
    above_bwd = np.where(Id_bwd >= target)[0]
    V_knee_bwd = float(Vd_seq_bwd[above_bwd[0]]) if len(above_bwd) else float("nan")
    V_knee = min(v for v in (V_knee_fwd, V_knee_bwd) if not math.isnan(v)) \
             if (not math.isnan(V_knee_fwd) or not math.isnan(V_knee_bwd)) \
             else float("nan")
    passed = (not math.isnan(V_knee)) and abs(V_knee - 1.5) <= 0.3
    log(f"  V_knee_fwd={V_knee_fwd}V V_knee_bwd={V_knee_bwd}V passed={passed}")
    return {"test": "V3", "V_knee_fwd": V_knee_fwd, "V_knee_bwd": V_knee_bwd,
            "V_knee_used": V_knee, "passed": passed, "gate": "|V_knee-1.5|<=0.3"}


def run_V6(cfg_flags, model_M1, model_M2, sebas_rows, R_body, log):
    log(f"V6 — Self-reset (1µs hold then release) R_body={R_body}")
    t_arr, Vd_arr = stim_pulse(V_lo=0.05, V_hi=2.0,
                                t_pre=10e-9, t_rise=100e-12,
                                t_hold=1e-6, t_fall=100e-12,
                                t_post=100e-9, n_total=1500)
    r = run_transient(cfg_flags, model_M1, model_M2, sebas_rows,
                      0.6, 0.0, t_arr, Vd_arr, R_body=R_body, max_step=5e-9)
    if r is None:
        return {"test": "V6", "passed": False, "notes": "no transient"}
    Vb = np.asarray(r["Vb"])
    t_release = 10e-9 + 100e-12 + 1e-6 + 100e-12
    t_ns = np.asarray(t_arr) * 1e9
    post = t_ns >= t_release * 1e9
    Vb_post_mean = float(np.nanmean(Vb[post])) if post.any() else float("nan")
    if post.any():
        idx_post = np.where(post)[0]
        below = Vb[idx_post] < 0.3
        t_reset_ns = float(t_ns[idx_post[np.argmax(below)]] - t_release*1e9) \
                     if below.any() else float("inf")
    else:
        t_reset_ns = float("inf")
    passed = (t_reset_ns < 1e5) and (Vb_post_mean < 0.3)
    log(f"  V_B_post_mean={Vb_post_mean:.3f}V t_reset={t_reset_ns}ns passed={passed}")
    return {"test": "V6", "Vb_post_mean": Vb_post_mean,
            "t_reset_ns": t_reset_ns, "passed": passed,
            "gate": "t_reset<100µs AND Vb_post<0.3V"}


def run_V7(cfg_flags, model_M1, model_M2, sebas_rows, R_body, log):
    log(f"V7 — Oscillation (5µs const drive) R_body={R_body}")
    t_arr, Vd_arr = stim_pulse(V_lo=0.05, V_hi=2.0,
                                t_pre=10e-9, t_rise=100e-12,
                                t_hold=5e-6, t_fall=100e-12,
                                t_post=100e-9, n_total=2500)
    r = run_transient(cfg_flags, model_M1, model_M2, sebas_rows,
                      0.6, 0.0, t_arr, Vd_arr, R_body=R_body, max_step=20e-9)
    if r is None:
        return {"test": "V7", "passed": False, "notes": "no transient"}, None, None
    Vb = np.asarray(r["Vb"])
    t_ns = np.asarray(t_arr) * 1e9
    crossings = []
    for i in range(1, len(Vb)):
        if np.isfinite(Vb[i]) and np.isfinite(Vb[i-1]) and Vb[i-1] < 0.5 <= Vb[i]:
            crossings.append(t_ns[i])
    n_cycles = max(0, len(crossings) - 1)
    period_ns = float(np.mean(np.diff(crossings))) if len(crossings) >= 2 else float("nan")
    passed = (n_cycles >= 3) and (100 <= period_ns <= 1000)
    log(f"  n_cycles={n_cycles} period={period_ns}ns passed={passed}")
    return ({"test": "V7", "n_cycles": n_cycles, "period_ns": period_ns,
             "passed": passed, "gate": ">=3 cycles AND period in [100,1000]ns"},
            t_arr, Vb)


# ---------- Step 6: Mario shape v2 ----------
def extract_metrics(t, Vb, t_pulse_start, t_pulse_end):
    Vb = np.asarray(Vb); t = np.asarray(t)
    Vb_peak = float(np.nanmax(Vb))
    Vb_floor = float(np.nanmin(Vb))
    swing = Vb_peak - Vb_floor
    Vb_lo = Vb_floor + 0.1*swing
    Vb_hi = Vb_floor + 0.9*swing
    mask_rise = (t >= t_pulse_start) & (t <= t_pulse_end)
    sub_t = t[mask_rise]; sub_v = Vb[mask_rise]
    try:
        i10 = int(np.argmax(sub_v >= Vb_lo))
        i90 = int(np.argmax(sub_v >= Vb_hi))
        t_rise = float(sub_t[i90] - sub_t[i10]) if i90 > i10 else float("nan")
    except Exception:
        t_rise = float("nan")
    mask_fall = t >= t_pulse_end
    sub_t = t[mask_fall]; sub_v = Vb[mask_fall]
    Vp_peak = float(np.nanmax(sub_v)) if sub_v.size else float("nan")
    Vp_floor = float(np.nanmin(sub_v)) if sub_v.size else float("nan")
    Vp_swing = Vp_peak - Vp_floor
    Vp_hi = Vp_floor + 0.9 * Vp_swing
    Vp_lo = Vp_floor + 0.1 * Vp_swing
    try:
        i90 = int(np.argmax(sub_v <= Vp_hi))
        i10 = int(np.argmax(sub_v <= Vp_lo))
        t_fall = float(sub_t[i10] - sub_t[i90]) if i10 > i90 else float("nan")
    except Exception:
        t_fall = float("nan")
    return dict(Vb_peak=Vb_peak, Vb_floor=Vb_floor, swing=swing,
                t_rise=t_rise, t_fall=t_fall,
                Vb_post_peak=Vp_peak, Vb_post_floor=Vp_floor)


def mario_shape(cfg_flags, model_M1, model_M2, sebas_rows, R_body, period_ns, log):
    log(f"Mario shape v2 with R_body={R_body}")
    targets = {"t_rise_ns": 26.0, "t_fall_ns": 76.0,
               "Vb_swing_V_lo": 0.5, "Vb_swing_V_hi": 0.7,
               "osc_period_ns": 430.0}
    # 1) single 200 ns pulse
    t1, Vd1 = stim_pulse(V_lo=0.05, V_hi=2.0,
                          t_pre=10e-9, t_rise=100e-12,
                          t_hold=200e-9, t_fall=100e-12,
                          t_post=300e-9, n_total=2000)
    t_pulse_start = 10e-9 + 100e-12
    t_pulse_end = t_pulse_start + 200e-9
    r1 = run_transient(cfg_flags, model_M1, model_M2, sebas_rows,
                       0.6, 0.0, t1, Vd1, R_body=R_body, max_step=5e-10)
    if r1 is None:
        return {"status": "fail_no_transient"}, None
    Vb1 = np.asarray(r1["Vb"]); Id1 = np.asarray(r1["Id"])
    m1 = extract_metrics(t1, Vb1, t_pulse_start, t_pulse_end)
    post_mask = t1 >= t_pulse_end + 50e-9
    reset_ok = bool((Vb1[post_mask] < 0.3).any()) if post_mask.any() else False
    m1["self_reset_post_pulse"] = reset_ok
    m1["Id_peak_A"] = float(np.nanmax(np.abs(Id1)))

    # 2) Two-pulse (50-250 ns, 800-1000 ns) on 1500-ns axis
    n_two = 3000
    t2 = np.linspace(0, 1500e-9, n_two)
    Vd2 = np.full_like(t2, 0.05)
    for i, ti in enumerate(t2):
        if 50e-9 < ti < 250e-9: Vd2[i] = 2.0
        elif 800e-9 < ti < 1000e-9: Vd2[i] = 2.0
    r2 = run_transient(cfg_flags, model_M1, model_M2, sebas_rows,
                       0.6, 0.0, t2, Vd2, R_body=R_body, max_step=5e-10)
    Vb_inter = float("nan"); sr_between = False
    Vb2 = None
    if r2 is not None:
        Vb2 = np.asarray(r2["Vb"])
        gap_mask = (t2 > 350e-9) & (t2 < 800e-9)
        Vb_inter = float(np.nanmin(Vb2[gap_mask])) if gap_mask.any() else float("nan")
        sr_between = bool(Vb_inter < 0.3)

    # 3) Reuse V7 oscillation period (passed in)
    scores = {}
    scores["t_rise_match"] = bool(abs(m1["t_rise"]*1e9 - targets["t_rise_ns"])
                                  <= 0.3*targets["t_rise_ns"]) if not math.isnan(m1["t_rise"]) else False
    scores["t_fall_match"] = bool(abs(m1["t_fall"]*1e9 - targets["t_fall_ns"])
                                  <= 0.3*targets["t_fall_ns"]) if not math.isnan(m1["t_fall"]) else False
    sw = m1["swing"]
    scores["Vb_swing_match"] = bool(targets["Vb_swing_V_lo"]*0.7 <= sw
                                    <= targets["Vb_swing_V_hi"]*1.3) if not math.isnan(sw) else False
    scores["self_reset_match"] = sr_between
    scores["osc_period_match"] = bool(abs(period_ns - targets["osc_period_ns"])
                                       <= 0.3*targets["osc_period_ns"]) \
                                  if (period_ns is not None and not math.isnan(period_ns)) else False
    n_match = sum(1 for v in scores.values() if v)
    out = {"target": targets, "single_pulse": m1,
            "two_pulse": {"Vb_inter_min_V": Vb_inter,
                          "self_reset_between_pulses": sr_between},
            "oscillation": {"period_ns": period_ns},
            "match_scores": scores, "n_metrics_matched": n_match,
            "R_body": R_body}
    log(f"  Mario v2: {n_match}/5  {scores}")
    return out, (t1, Vb1, Vd1, t2, Vb2)


# ---------------- main ----------------
def main():
    out_dir = ROOT / "results/z473_rbody_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    log_file = open(log_path, "w")

    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True); log_file.write(line + "\n"); log_file.flush()

    t0 = time.time()
    log("z473 R_body sweep + V3/V6/V7 + Mario shape v2")
    cfg_flags = make_NX_1p8()
    log("loading models + sebas params")
    model_M1, model_M2 = z427.build_models()
    sebas_rows = z427.load_sebas_params()

    # Step 1: sweep
    rows = step1_sweep(cfg_flags, model_M1, model_M2, sebas_rows, log)
    (out_dir / "rbody_sweep.json").write_text(json.dumps(rows, indent=2, default=float))

    # Step 3: pick
    chosen = pick_R_body(rows, log)
    if chosen is None:
        log("  NO R_body satisfies drift+reset. Falling back to lowest R that didn't break Id_pk.")
        for r in rows:
            if "Id_pk_drift_dec" in r and r["Id_pk_drift_dec"] < 0.3:
                if chosen is None or (r["R_body"] is not None and
                                      (chosen["R_body"] is None or r["R_body"] < chosen["R_body"])):
                    chosen = r
    log(f"CHOSEN R_body = {chosen['tag'] if chosen else 'NONE'}")
    R_chosen = chosen["R_body"] if chosen else None
    Id_drift_chosen = chosen.get("Id_pk_drift_dec", float("nan")) if chosen else float("nan")

    # KILL_SHOT check: if Id_pk drift > 0.3 dec, abort
    kill_shot = False
    if chosen and chosen.get("Id_pk_drift_dec", 0) > 0.3:
        kill_shot = True
        log(f"KILL_SHOT: Id_pk drift {chosen['Id_pk_drift_dec']:.3f} > 0.3 dec on chosen R_body")

    # Step 4: V3/V6/V7
    v3_res = run_V3(cfg_flags, model_M1, model_M2, sebas_rows, log)
    v6_res = run_V6(cfg_flags, model_M1, model_M2, sebas_rows, R_chosen, log)
    v7_res, t_v7, Vb_v7 = run_V7(cfg_flags, model_M1, model_M2, sebas_rows, R_chosen, log)
    triplet = {"R_body_chosen": R_chosen, "Id_pk_drift_dec": Id_drift_chosen,
                "V3": v3_res, "V6": v6_res, "V7": v7_res,
                "kill_shot_triggered": kill_shot}
    triplet_pass = sum(1 for x in (v3_res, v6_res, v7_res) if x.get("passed"))
    triplet["n_pass"] = triplet_pass
    (out_dir / "v3v6v7_post_sweep.json").write_text(json.dumps(triplet, indent=2, default=float))
    log(f"Triplet V3/V6/V7: {triplet_pass}/3 PASS")

    # Step 5+6: Mario shape v2 (reuse V7 period)
    period_ns = v7_res.get("period_ns") if isinstance(v7_res, dict) else None
    mario, traces = mario_shape(cfg_flags, model_M1, model_M2, sebas_rows,
                                 R_chosen, period_ns, log)
    (out_dir / "mario_shape_v2.json").write_text(json.dumps(mario, indent=2, default=float))

    # Overlay plot: single pulse + V7 oscillation + two-pulse
    if traces is not None:
        t1s, Vb1s, Vd1s, t2s, Vb2s = traces
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        ax = axes[0]
        ax.plot(t1s*1e9, Vb1s, "b-", lw=1.3, label="V_B")
        ax.plot(t1s*1e9, Vd1s, "k-", lw=0.5, alpha=0.6, label="V_D")
        ax.axhline(0.3, color="red", ls=":", label="0.3V (reset)")
        ax.set_xlabel("t [ns]"); ax.set_ylabel("V [V]")
        ax.set_title(f"Single 200ns pulse @ R_body={chosen['tag'] if chosen else 'None'} — "
                     f"t_rise={mario['single_pulse']['t_rise']*1e9:.1f}ns (tgt 26) "
                     f"t_fall={mario['single_pulse']['t_fall']*1e9:.1f}ns (tgt 76) "
                     f"swing={mario['single_pulse']['swing']:.3f}V")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[1]
        if Vb2s is not None:
            ax.plot(t2s*1e9, Vb2s, "b-", lw=1.0, label="V_B (two-pulse)")
            ax.plot(t2s*1e9, np.full_like(t2s, 0.3), "r:", label="0.3V reset")
        ax.set_xlabel("t [ns]"); ax.set_ylabel("V_B [V]")
        sr = mario["two_pulse"]["self_reset_between_pulses"]
        ax.set_title(f"Two-pulse train — self_reset_between={sr}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[2]
        if t_v7 is not None and Vb_v7 is not None:
            ax.plot(np.asarray(t_v7)*1e9, Vb_v7, "b-", lw=0.8)
            ax.axhline(0.5, color="red", ls=":", label="0.5V threshold")
        n_c = v7_res.get("n_cycles", 0)
        per = v7_res.get("period_ns", float("nan"))
        ax.set_xlabel("t [ns]"); ax.set_ylabel("V_B [V]")
        ax.set_title(f"V7 free-running osc — n_cycles={n_c} period={per}ns (tgt 430)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "transient_overlay_with_reset.png", dpi=120)
        plt.close(fig)
        log("wrote transient_overlay_with_reset.png")

    log(f"DONE in {time.time()-t0:.1f}s")
    log_file.close()


if __name__ == "__main__":
    main()
