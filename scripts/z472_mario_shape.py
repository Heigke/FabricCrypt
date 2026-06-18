"""z472 — Mario shape-match transient extraction.

Re-uses the z461 transient harness to produce a single V_b(t) trace at
the primary calibration bias (VG1=0.6, VG2=0, Vd=2V pulse), then extracts:

  - t_rise (10%→90%, target 26 ns)
  - t_fall (90%→10% post-pulse, target 76 ns)
  - V_b swing amplitude (peak−floor, Mario 0.5-0.7 V)
  - self-reset between pulses (yes/no)
  - oscillation period (free-running, target 430 ns)

Also produces a transient_overlay.png that compares our V_b(t) shape to a
schematic Mario target waveform.
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
z454 = _load("z454", ROOT / "scripts/z454_snapback_integration.py")

from nsram.bsim4_port import transient_real_v2 as trv2
from nsram.bsim4_port.transient_real_v2 import integrate, TransientCfgV2

# z461 config builder (inlined to avoid dataclass scope bug)
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


def run_transient(cfg_flags, model_M1, model_M2, sebas_rows, VG1, VG2,
                  t_arr, Vd_arr, Vb0=0.0, max_step=1e-10, first_step=1e-14):
    sebas_row = z427.find_params(sebas_rows, VG1, VG2)
    if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
        return None
    P_M1, P_M2 = z427.make_overrides(sebas_row)
    bjt = z427.make_bjt(sebas_row)
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(cfg_flags))
    tcfg = TransientCfgV2(C_B_const=1e-15,
                          atol=1e-12, rtol=1e-7,
                          max_step=max_step, first_step=first_step)
    with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), z427.patch_sd_scaled(sd_M2, P_M2):
        out = integrate(cfg, model_M1, model_M2, bjt,
                        np.asarray(t_arr), np.asarray(Vd_arr),
                        float(VG1), float(VG2),
                        tcfg=tcfg, Vb0=float(Vb0))
    return out


def extract_metrics(t, Vb, t_rise_start, t_rise_end, t_fall_start, t_fall_end):
    """t in seconds, Vb in volts."""
    Vb = np.asarray(Vb); t = np.asarray(t)
    Vb_peak = float(np.nanmax(Vb))
    Vb_floor = float(np.nanmin(Vb))
    swing = Vb_peak - Vb_floor

    # t_rise: 10%→90% during the pulse rise+hold region.
    Vb_lo = Vb_floor + 0.1*swing
    Vb_hi = Vb_floor + 0.9*swing
    mask_rise = (t >= t_rise_start) & (t <= t_fall_start)
    sub_t = t[mask_rise]; sub_v = Vb[mask_rise]
    try:
        i10 = int(np.argmax(sub_v >= Vb_lo))
        i90 = int(np.argmax(sub_v >= Vb_hi))
        t_rise = float(sub_t[i90] - sub_t[i10]) if i90 > i10 else float("nan")
    except Exception:
        t_rise = float("nan")

    # t_fall: 90%→10% post-pulse (decay)
    mask_fall = t >= t_fall_start
    sub_t = t[mask_fall]; sub_v = Vb[mask_fall]
    Vb_post_peak = float(np.nanmax(sub_v)) if sub_v.size else float("nan")
    Vb_post_floor = float(np.nanmin(sub_v)) if sub_v.size else float("nan")
    Vp_swing = Vb_post_peak - Vb_post_floor
    Vp_hi = Vb_post_floor + 0.9 * Vp_swing
    Vp_lo = Vb_post_floor + 0.1 * Vp_swing
    try:
        i90 = int(np.argmax(sub_v <= Vp_hi))
        i10 = int(np.argmax(sub_v <= Vp_lo))
        t_fall = float(sub_t[i10] - sub_t[i90]) if i10 > i90 else float("nan")
    except Exception:
        t_fall = float("nan")

    return dict(Vb_peak=Vb_peak, Vb_floor=Vb_floor, swing=swing,
                t_rise=t_rise, t_fall=t_fall,
                Vb_post_peak=Vb_post_peak, Vb_post_floor=Vb_post_floor)


def main():
    out_dir = ROOT / "results/z472_v1_fix"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_flags = make_NX_1p8()
    print(f"[mario] config NX_1p8 with calibrated snap_Is={SNAP_HOT['snap_Is']:.3e}", flush=True)
    print("[mario] building models + loading sebas params...", flush=True)
    model_M1, model_M2 = z427.build_models()
    sebas_rows = z427.load_sebas_params()

    metrics = {"target_mario": {
        "t_rise_ns": 26.0, "t_fall_ns": 76.0,
        "Vb_swing_V_lo": 0.5, "Vb_swing_V_hi": 0.7,
        "osc_period_ns": 430.0
    }}

    # ----- Single short pulse for t_rise / t_fall / swing -----
    print("[mario] running single 100ns hold pulse @ VG1=0.6 VG2=0 Vd=2V...", flush=True)
    t1, Vd1 = stim_pulse_general(V_lo=0.05, V_hi=2.0,
                                  t_pre=10e-9, t_rise=100e-12,
                                  t_hold=200e-9, t_fall=100e-12,
                                  t_post=200e-9, n_total=2000)
    t_pulse_start = 10e-9 + 100e-12
    t_pulse_end = t_pulse_start + 200e-9
    r1 = run_transient(cfg_flags, model_M1, model_M2, sebas_rows,
                       0.6, 0.0, t1, Vd1, max_step=5e-10)
    if r1 is None:
        print("[mario] transient FAILED — no sebas row", flush=True)
        metrics["single_pulse"] = {"status": "FAIL_no_sebas_row"}
    else:
        Vb1 = np.asarray(r1["Vb"]); Id1 = np.asarray(r1["Id"])
        m1 = extract_metrics(t1, Vb1, t_pulse_start, t_pulse_start+10e-9,
                             t_pulse_end, t_pulse_end+50e-9)
        # Single-pulse self-reset: V_b returns to <0.3V within 500ns post-fall
        post_mask = t1 >= t_pulse_end + 50e-9
        reset_ok = bool((Vb1[post_mask] < 0.3).any()) if post_mask.any() else False
        m1["self_reset_post_pulse"] = reset_ok
        m1["Id_peak_A"] = float(np.nanmax(np.abs(Id1)))
        metrics["single_pulse"] = m1
        print(f"[mario] single pulse: Vb_peak={m1['Vb_peak']:.3f}V swing={m1['swing']:.3f}V "
              f"t_rise={m1['t_rise']*1e9:.2f}ns t_fall={m1['t_fall']*1e9:.2f}ns "
              f"self_reset={reset_ok} Id_pk={m1['Id_peak_A']*1e3:.2f}mA", flush=True)
        # Save trace
        metrics["single_pulse_trace"] = {
            "t_ns": (t1 * 1e9).tolist()[::4],
            "Vb_V": Vb1.tolist()[::4],
            "Vd_V": Vd1.tolist()[::4],
        }

    # ----- Two-pulse for inter-pulse self-reset -----
    print("[mario] running two-pulse train for self-reset check...", flush=True)
    n_two = 3000
    t2 = np.linspace(0, 1500e-9, n_two)
    Vd2 = np.full_like(t2, 0.05)
    # pulse 1: 50-250 ns, pulse 2: 800-1000 ns
    for i, ti in enumerate(t2):
        if 50e-9 < ti < 250e-9: Vd2[i] = 2.0
        elif 800e-9 < ti < 1000e-9: Vd2[i] = 2.0
    r2 = run_transient(cfg_flags, model_M1, model_M2, sebas_rows,
                       0.6, 0.0, t2, Vd2, max_step=5e-10)
    if r2 is not None:
        Vb2 = np.asarray(r2["Vb"])
        gap_mask = (t2 > 350e-9) & (t2 < 800e-9)
        Vb_inter = float(np.nanmin(Vb2[gap_mask])) if gap_mask.any() else float("nan")
        # self-reset between pulses: Vb returns below 0.3V
        sr_between = bool(Vb_inter < 0.3)
        metrics["two_pulse"] = {
            "Vb_inter_min_V": Vb_inter,
            "self_reset_between_pulses": sr_between,
        }
        print(f"[mario] two-pulse Vb_inter_min={Vb_inter:.3f}V self_reset_between={sr_between}", flush=True)
    else:
        metrics["two_pulse"] = {"status": "FAIL"}

    # ----- Free-running for oscillation period -----
    print("[mario] running 5us constant drive for oscillation...", flush=True)
    t3, Vd3 = stim_pulse_general(V_lo=0.05, V_hi=2.0,
                                  t_pre=10e-9, t_rise=100e-12,
                                  t_hold=5e-6, t_fall=100e-12, t_post=50e-9,
                                  n_total=2500)
    r3 = run_transient(cfg_flags, model_M1, model_M2, sebas_rows,
                       0.6, 0.0, t3, Vd3, max_step=20e-9)
    period_ns = float("nan"); n_cycles = 0
    if r3 is not None:
        Vb3 = np.asarray(r3["Vb"])
        # Upward 0.5V crossings during hold
        hold_mask = (t3 > 20e-9) & (t3 < 5e-6)
        ts = t3[hold_mask]; vs = Vb3[hold_mask]
        crossings = []
        for i in range(1, len(vs)):
            if np.isfinite(vs[i]) and np.isfinite(vs[i-1]) and vs[i-1] < 0.5 <= vs[i]:
                crossings.append(ts[i])
        n_cycles = max(0, len(crossings) - 1)
        if n_cycles >= 1:
            period_ns = float(np.mean(np.diff(crossings))) * 1e9
        metrics["oscillation"] = {
            "n_cycles": n_cycles,
            "period_ns": period_ns,
        }
        print(f"[mario] oscillation: n_cycles={n_cycles} period_ns={period_ns:.1f}", flush=True)
    else:
        metrics["oscillation"] = {"status": "FAIL"}

    # ----- Match scoring -----
    targets = metrics["target_mario"]
    sp = metrics.get("single_pulse", {})
    scores = {}
    if sp.get("t_rise") and not math.isnan(sp.get("t_rise", float("nan"))):
        scores["t_rise_match"] = bool(abs(sp["t_rise"]*1e9 - targets["t_rise_ns"]) <= 0.3*targets["t_rise_ns"])
    else:
        scores["t_rise_match"] = False
    if sp.get("t_fall") and not math.isnan(sp.get("t_fall", float("nan"))):
        scores["t_fall_match"] = bool(abs(sp["t_fall"]*1e9 - targets["t_fall_ns"]) <= 0.3*targets["t_fall_ns"])
    else:
        scores["t_fall_match"] = False
    sw = sp.get("swing", float("nan"))
    scores["Vb_swing_match"] = bool(targets["Vb_swing_V_lo"]*0.7 <= sw <= targets["Vb_swing_V_hi"]*1.3) if not math.isnan(sw) else False
    scores["self_reset_match"] = bool(metrics.get("two_pulse", {}).get("self_reset_between_pulses", False))
    if not math.isnan(period_ns):
        scores["osc_period_match"] = bool(abs(period_ns - targets["osc_period_ns"]) <= 0.3*targets["osc_period_ns"])
    else:
        scores["osc_period_match"] = False
    n_match = sum(1 for v in scores.values() if v)
    metrics["match_scores"] = scores
    metrics["n_metrics_matched"] = n_match
    print(f"[mario] match {n_match}/5: {scores}", flush=True)

    # ----- Plot overlay -----
    if "single_pulse_trace" in metrics:
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=False)
        t_ns = np.array(metrics["single_pulse_trace"]["t_ns"])
        Vb_arr = np.array(metrics["single_pulse_trace"]["Vb_V"])
        Vd_arr_p = np.array(metrics["single_pulse_trace"]["Vd_V"])
        ax = axes[0]
        ax.plot(t_ns, Vb_arr, "b-", lw=1.5, label="V_B (our cell)")
        ax.plot(t_ns, Vd_arr_p, "k-", lw=0.6, alpha=0.6, label="V_D")
        ax.axhline(0.5, color="red", ls=":", alpha=0.5, label="0.5 V")
        ax.set_xlabel("t [ns]"); ax.set_ylabel("V [V]")
        ax.set_title(f"z472 — V_b(t) on calibrated NS-RAM cell vs Mario slide-08\n"
                     f"t_rise={sp.get('t_rise', float('nan'))*1e9:.1f} ns (tgt 26), "
                     f"t_fall={sp.get('t_fall', float('nan'))*1e9:.1f} ns (tgt 76), "
                     f"swing={sp.get('swing', float('nan')):.3f} V (tgt 0.5-0.7)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Schematic Mario target (trapezoidal idealization for visual reference)
        ax = axes[1]
        t_m = np.linspace(0, 500, 5000)
        V_m = np.where(t_m < 50, 0.05,
              np.where(t_m < 76, 0.05 + (0.6-0.05)*(t_m-50)/26.0,
              np.where(t_m < 350, 0.6,
              np.where(t_m < 426, 0.6 - (0.6-0.05)*(t_m-350)/76.0, 0.05))))
        ax.plot(t_m, V_m, "g-", lw=1.5, label="Mario target (schematic)")
        ax.set_xlabel("t [ns]"); ax.set_ylabel("V_B [V]")
        ax.set_title("Mario slide-08 schematic target")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "transient_overlay.png", dpi=120)
        plt.close(fig)
        print(f"[mario] wrote transient_overlay.png", flush=True)

    # Save metrics
    (out_dir / "mario_shape_match.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"[mario] wrote mario_shape_match.json", flush=True)


if __name__ == "__main__":
    main()
