"""z476 — Sweep `snap_npn_V_knee` to re-open regenerative loop for V7 oscillation.

z475 KILL_SHOT showed body-leak cannot manufacture Hopf bifurcation: V_B is a
globally attracting equilibrium at ~0.62V. z475's first recommendation was to
WEAKEN the sigma-knee gate on the parasitic NPN (snap_npn_V_knee 1.8 -> 1.4-1.5)
to push the BJT super-critical and re-open the regenerative loop.

This script:
  1. Single-bias V7 free-osc transient (VG1=0.6, VG2=0, Vd=2V hold 5us)
  2. Sweep snap_npn_V_knee in {1.8 (baseline), 1.6, 1.5, 1.4, 1.3, 1.2} V
  3. Record V_b(t), Vb_max, Vb_min, oscillation cycles, period
  4. If DISCOVERY: 4-bias Id_pk verify (z471-style) + V6 self-reset verify (z461)
  5. If KILL_SHOT: log + revert, document tradeoff

Outputs to results/z476_npn_knee/:
  v_knee_sweep.json
  transient_osc_overlay.png
  z471_4bias_post.json   (if applicable)
  z461_V6_post.json      (if applicable)
  honest_analysis.md
  patch.diff             (if default updated)
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
    m = _ilu.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


z427 = _load("z427", ROOT / "scripts/z427_vsint_fix.py")
z449 = _load("z449", ROOT / "scripts/z449_vbic_bdf_combo.py")
z473 = _load("z473", ROOT / "scripts/z473_rbody_sweep.py")

from nsram.bsim4_port import transient_real_v2 as trv2
from nsram.bsim4_port.transient_real_v2 import integrate, TransientCfgV2

OUT = ROOT / "results/z476_npn_knee"
OUT.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT / "run.log"
LOG_FH = open(LOG_PATH, "w")


def log(msg):
    print(msg, flush=True)
    LOG_FH.write(msg + "\n")
    LOG_FH.flush()


def make_cfg_flags(V_knee_npn: float):
    """Same as z473.make_NX_1p8() but with overridable snap_npn_V_knee."""
    f = z473.make_NX_1p8()
    f["snap_npn_V_knee"] = float(V_knee_npn)
    return f


def run_transient(cfg_flags, model_M1, model_M2, sebas_rows, VG1, VG2,
                  t_arr, Vd_arr, *, R_body=1e7, Vb0=0.0,
                  max_step=1e-10, first_step=1e-14):
    sebas_row = z427.find_params(sebas_rows, VG1, VG2)
    if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
        return None
    P_M1, P_M2 = z427.make_overrides(sebas_row)
    bjt = z427.make_bjt(sebas_row)
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(cfg_flags))
    cfg.Cbody = 1e-15
    tcfg = TransientCfgV2(
        C_B_const=1e-15, atol=1e-12, rtol=1e-7,
        max_step=max_step, first_step=first_step,
        R_body=R_body,
    )
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


def v7_stim():
    return z473.stim_pulse(V_lo=0.05, V_hi=2.0,
                           t_pre=10e-9, t_rise=100e-12,
                           t_hold=5e-6, t_fall=100e-12,
                           t_post=100e-9, n_total=2500)


def v6_stim():
    return z473.stim_pulse(V_lo=0.05, V_hi=2.0,
                           t_pre=10e-9, t_rise=100e-12,
                           t_hold=1e-6, t_fall=100e-12,
                           t_post=100e-9, n_total=1500)


def measure_oscillation(t_arr, Vb, level=None):
    """Count crossings of mid-level; returns (n_cycles, period_ns, crossings)."""
    t_ns = np.asarray(t_arr) * 1e9
    Vb = np.asarray(Vb)
    if level is None:
        vmax = float(np.nanmax(Vb))
        vmin = float(np.nanmin(Vb))
        level = 0.5 * (vmax + vmin)
        if (vmax - vmin) < 5e-3:
            return 0, float("nan"), [], level
    crossings = []
    for i in range(1, len(Vb)):
        if (np.isfinite(Vb[i]) and np.isfinite(Vb[i - 1])
                and Vb[i - 1] < level <= Vb[i]):
            crossings.append(float(t_ns[i]))
    n_cycles = max(0, len(crossings) - 1)
    if len(crossings) >= 2:
        period_ns = float(np.mean(np.diff(crossings)))
    else:
        period_ns = float("nan")
    return n_cycles, period_ns, crossings, level


def sweep_v_knee(model_M1, model_M2, sebas_rows):
    Vk_values = [1.8, 1.6, 1.5, 1.4, 1.3, 1.2]
    t_arr, Vd_arr = v7_stim()
    rows = []
    traces = {}
    for Vk in Vk_values:
        cfg_flags = make_cfg_flags(Vk)
        t0 = time.time()
        try:
            r = run_transient(cfg_flags, model_M1, model_M2, sebas_rows,
                              0.6, 0.0, t_arr, Vd_arr,
                              R_body=1e7, max_step=20e-9)
        except Exception as exc:
            log(f"  Vk={Vk}: EXCEPTION {exc}")
            rows.append({"V_knee": Vk, "status": "exception",
                         "msg": str(exc)})
            continue
        dt = time.time() - t0
        if r is None:
            rows.append({"V_knee": Vk, "status": "no_sebas"})
            continue
        Vb = np.asarray(r["Vb"])
        Id = np.asarray(r["Id"])
        n_cyc, T_ns, crossings, level = measure_oscillation(t_arr, Vb)
        # also count using 0.5V absolute (z475 metric)
        n_cyc_0p5, T_ns_0p5, crossings_0p5, _ = measure_oscillation(
            t_arr, Vb, level=0.5)
        Vb_max = float(np.nanmax(Vb))
        Vb_min = float(np.nanmin(Vb))
        Id_pk_mA = float(np.nanmax(np.abs(Id)) * 1e3)
        sustained = (n_cyc >= 3)
        in_range = (math.isfinite(T_ns) and 300.0 <= T_ns <= 600.0)
        row = {
            "V_knee": Vk,
            "n_cycles_mid": int(n_cyc),
            "n_cycles_0p5": int(n_cyc_0p5),
            "period_ns_mid": (None if not math.isfinite(T_ns) else T_ns),
            "period_ns_0p5": (None if not math.isfinite(T_ns_0p5) else T_ns_0p5),
            "level_mid_V": float(level),
            "Vb_max_V": Vb_max,
            "Vb_min_V": Vb_min,
            "Vb_swing_V": Vb_max - Vb_min,
            "Id_pk_mA": Id_pk_mA,
            "sustained_ge3": bool(sustained),
            "in_300_600_ns": bool(in_range),
            "wall_s": dt,
            "status": "ok",
        }
        rows.append(row)
        traces[Vk] = (np.asarray(t_arr).copy(), Vb.copy())
        log(f"  Vk={Vk}: cyc_mid={n_cyc} T_mid={T_ns} ns "
            f"cyc_0.5={n_cyc_0p5} T_0.5={T_ns_0p5} ns "
            f"Vb_min={Vb_min:.3f} Vb_max={Vb_max:.3f} "
            f"Id_pk={Id_pk_mA:.3f}mA dt={dt:.1f}s")
    return rows, traces


def make_overlay_plot(traces, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.95, len(traces)))
    for (Vk, (t, Vb)), c in zip(sorted(traces.items()), colors):
        ax.plot(np.asarray(t) * 1e9, Vb, lw=0.9, color=c,
                label=f"V_knee={Vk}V")
    ax.axhline(0.5, color="red", ls=":", lw=0.7, alpha=0.6,
               label="0.5V ref")
    ax.set_xlabel("time [ns]")
    ax.set_ylabel("V_B [V]")
    ax.set_title("z476: V_B(t) free-osc under V7 DC hold, sweep snap_npn_V_knee")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def verify_4bias_id_pk(Vk_chosen, model_M1, model_M2, sebas_rows,
                       baseline_table):
    """z471-style 4-bias Id_pk verify under primary 200ns pulse."""
    biases = [(0.4, 0.0), (0.4, 0.2), (0.6, 0.0), (0.6, 0.2)]
    cfg_flags = make_cfg_flags(Vk_chosen)
    t_arr, Vd_arr = z473.primary_pulse(n_total=1500)
    rows = []
    max_drift = 0.0
    for vg1, vg2 in biases:
        t0 = time.time()
        try:
            r = run_transient(cfg_flags, model_M1, model_M2, sebas_rows,
                              vg1, vg2, t_arr, Vd_arr,
                              R_body=1e7, max_step=5e-10)
        except Exception as exc:
            rows.append({"VG1": vg1, "VG2": vg2,
                         "status": "exception", "msg": str(exc)})
            continue
        if r is None:
            rows.append({"VG1": vg1, "VG2": vg2, "status": "no_sebas"})
            continue
        Id = np.asarray(r["Id"])
        Id_pk_mA = float(np.nanmax(np.abs(Id)) * 1e3)
        base = baseline_table.get((vg1, vg2))
        drift = (abs(math.log10(max(Id_pk_mA, 1e-12))
                     - math.log10(max(base, 1e-12)))
                 if base is not None else None)
        if drift is not None and drift > max_drift:
            max_drift = drift
        rows.append({"VG1": vg1, "VG2": vg2,
                     "Id_pk_mA": Id_pk_mA,
                     "baseline_mA": base,
                     "drift_dec": drift,
                     "wall_s": time.time() - t0,
                     "status": "ok"})
        log(f"    4-bias (VG1={vg1}, VG2={vg2}): "
            f"Id_pk={Id_pk_mA:.3f}mA base={base} drift={drift}")
    return {"V_knee": Vk_chosen, "rows": rows, "max_drift_dec": max_drift}


def baseline_4bias(model_M1, model_M2, sebas_rows):
    """Run 4-bias Id_pk with V_knee=1.8 (current default) to get baselines."""
    log("Building 4-bias Id_pk baseline at V_knee=1.8 (current default)...")
    biases = [(0.4, 0.0), (0.4, 0.2), (0.6, 0.0), (0.6, 0.2)]
    cfg_flags = make_cfg_flags(1.8)
    t_arr, Vd_arr = z473.primary_pulse(n_total=1500)
    table = {}
    for vg1, vg2 in biases:
        r = run_transient(cfg_flags, model_M1, model_M2, sebas_rows,
                          vg1, vg2, t_arr, Vd_arr,
                          R_body=1e7, max_step=5e-10)
        if r is None:
            table[(vg1, vg2)] = None
            continue
        Id = np.asarray(r["Id"])
        Id_pk_mA = float(np.nanmax(np.abs(Id)) * 1e3)
        table[(vg1, vg2)] = Id_pk_mA
        log(f"    baseline (VG1={vg1}, VG2={vg2}): Id_pk={Id_pk_mA:.3f}mA")
    return table


def verify_v6_self_reset(Vk_chosen, model_M1, model_M2, sebas_rows):
    """z461 V6 — V_B drains below 0.3V within 1us after V_d=2V release."""
    cfg_flags = make_cfg_flags(Vk_chosen)
    t_arr, Vd_arr = v6_stim()
    r = run_transient(cfg_flags, model_M1, model_M2, sebas_rows,
                      0.6, 0.0, t_arr, Vd_arr,
                      R_body=1e7, max_step=5e-9)
    if r is None:
        return {"V_knee": Vk_chosen, "passed": False,
                "notes": "no transient"}
    Vb = np.asarray(r["Vb"])
    t_release = 10e-9 + 100e-12 + 1e-6 + 100e-12
    t_ns = np.asarray(t_arr) * 1e9
    post = t_ns >= t_release * 1e9
    Vb_post_mean = (float(np.nanmean(Vb[post]))
                    if post.any() else float("nan"))
    if post.any():
        idx_post = np.where(post)[0]
        below = Vb[idx_post] < 0.3
        if below.any():
            t_reset_ns = float(t_ns[idx_post[np.argmax(below)]]
                               - t_release * 1e9)
        else:
            t_reset_ns = float("inf")
    else:
        t_reset_ns = float("inf")
    passed = (t_reset_ns < 1e5) and (Vb_post_mean < 0.3)
    out = {"V_knee": Vk_chosen,
           "passed": bool(passed),
           "t_reset_ns": t_reset_ns,
           "Vb_post_mean": Vb_post_mean}
    log(f"    V6 self-reset @ Vk={Vk_chosen}: t_reset={t_reset_ns}ns "
        f"Vb_post={Vb_post_mean:.3f}V passed={passed}")
    return out


def main():
    log("z476 — snap_npn_V_knee sweep for V7 free oscillation")
    log("Loading models / sebas...")
    model_M1, model_M2 = z427.build_models()
    sebas_rows = z427.load_sebas_params()

    log("=== STEP 1: V_knee sweep on V7 stimulus ===")
    rows, traces = sweep_v_knee(model_M1, model_M2, sebas_rows)
    summary = {"sweep_rows": rows}

    # Identify DISCOVERY candidates: sustained >=3 cycles AND period in [300,600]
    discoveries = [r for r in rows
                   if r.get("status") == "ok"
                   and r.get("sustained_ge3")
                   and r.get("in_300_600_ns")]
    # Relaxed: any sustained osc with finite period (in case period is out of band)
    any_sustained = [r for r in rows
                     if r.get("status") == "ok"
                     and r.get("sustained_ge3")]

    log("")
    log(f"DISCOVERY candidates (>=3 cyc, T in [300,600]ns): "
        f"{len(discoveries)}")
    log(f"Any sustained (>=3 cyc, any T): {len(any_sustained)}")

    # Overlay plot regardless
    overlay_path = OUT / "transient_osc_overlay.png"
    make_overlay_plot(traces, overlay_path)
    log(f"wrote {overlay_path}")

    chosen = None
    if discoveries:
        # Prefer V_knee closest to baseline (1.8) to minimize calibration impact
        chosen = max(discoveries, key=lambda r: r["V_knee"])
        log(f"CHOSEN DISCOVERY candidate: V_knee={chosen['V_knee']} V "
            f"T={chosen.get('period_ns_mid')}ns "
            f"cyc={chosen['n_cycles_mid']}")
        summary["chosen_V_knee"] = chosen["V_knee"]
        summary["chosen_rationale"] = "discovery (>=3 cyc in [300,600]ns)"
    elif any_sustained:
        chosen = max(any_sustained, key=lambda r: r["V_knee"])
        log(f"Partial: V_knee={chosen['V_knee']} V sustained but "
            f"T={chosen.get('period_ns_mid')}ns out of band")

    # Calibration verify
    cal_verify = None
    v6_verify = None
    kill_shot = False
    if chosen is not None and chosen["V_knee"] != 1.8:
        log("")
        log("=== STEP 2: 4-bias Id_pk verify (baseline at V_knee=1.8) ===")
        baseline = baseline_4bias(model_M1, model_M2, sebas_rows)
        log("")
        log(f"Running 4-bias verify at V_knee={chosen['V_knee']}...")
        cal_verify = verify_4bias_id_pk(chosen["V_knee"],
                                        model_M1, model_M2, sebas_rows,
                                        baseline)
        (OUT / "z471_4bias_post.json").write_text(
            json.dumps({**cal_verify,
                        "baseline_at_V_knee_1p8":
                            {f"{k[0]},{k[1]}": v
                             for k, v in baseline.items()}},
                       indent=2, default=float))
        log(f"    4-bias max Id_pk drift: {cal_verify['max_drift_dec']:.3f} dec")

        log("")
        log("=== STEP 3: V6 self-reset verify at chosen V_knee ===")
        v6_verify = verify_v6_self_reset(chosen["V_knee"],
                                         model_M1, model_M2, sebas_rows)
        (OUT / "z461_V6_post.json").write_text(
            json.dumps(v6_verify, indent=2, default=float))

        # KILL_SHOT check: drift > 0.3 dec
        if cal_verify["max_drift_dec"] > 0.3:
            kill_shot = True
            log("KILL_SHOT: 4-bias Id_pk drift > 0.3 dec — calibration destroyed")

    summary["calibration_verify"] = cal_verify
    summary["v6_verify"] = v6_verify
    summary["kill_shot_triggered"] = kill_shot

    (OUT / "v_knee_sweep.json").write_text(
        json.dumps(summary, indent=2, default=float))
    log(f"wrote v_knee_sweep.json")

    # Honest analysis
    write_honest_analysis(summary, rows, chosen, cal_verify,
                          v6_verify, kill_shot)
    log("DONE.")


def write_honest_analysis(summary, rows, chosen, cal_verify,
                          v6_verify, kill_shot):
    lines = ["# z476 — Honest Analysis: snap_npn_V_knee sweep for V7 osc",
             "",
             "## Premise",
             ("z475 KILL_SHOT showed body-leak alone cannot manufacture Hopf"
              " bifurcation — V_B converges to a stable fixed point ~0.62V"
              " under V7 stimulus. z475's first recommendation was to weaken"
              " the sigma-knee gate on the parasitic NPN (snap_npn_V_knee"
              " 1.8 -> 1.4-1.5) so the BJT goes super-critical at V_d=2V"
              " and pushes V_B past its fixed point."),
             "",
             "## Sweep results", "",
             "| V_knee | cyc(mid) | T(mid)[ns] | cyc(0.5V) | T(0.5V)[ns] | "
             "Vb_min | Vb_max | swing | Id_pk[mA] | status |",
             "|--------|----------|------------|-----------|-------------|"
             "--------|--------|-------|-----------|--------|"]
    for r in rows:
        if r.get("status") != "ok":
            lines.append(f"| {r.get('V_knee')} | - | - | - | - | - | - | - | - "
                         f"| {r.get('status')} |")
            continue
        lines.append("| {Vk} | {cm} | {Tm} | {c5} | {T5} | {vmin:.3f} | "
                     "{vmax:.3f} | {sw:.3f} | {Id:.3f} | ok |".format(
                         Vk=r["V_knee"],
                         cm=r["n_cycles_mid"],
                         Tm=("-" if r["period_ns_mid"] is None
                             else f"{r['period_ns_mid']:.1f}"),
                         c5=r["n_cycles_0p5"],
                         T5=("-" if r["period_ns_0p5"] is None
                             else f"{r['period_ns_0p5']:.1f}"),
                         vmin=r["Vb_min_V"],
                         vmax=r["Vb_max_V"],
                         sw=r["Vb_swing_V"],
                         Id=r["Id_pk_mA"]))
    lines.append("")

    # Discovery / kill-shot scoring
    any_disco = any(r.get("status") == "ok" and r.get("sustained_ge3")
                    and r.get("in_300_600_ns") for r in rows)
    any_sust = any(r.get("status") == "ok" and r.get("sustained_ge3")
                   for r in rows)

    lines.append("## Pre-reg scoring")
    lines.append("")
    lines.append(f"- **INFRA**: DONE — 6-point V_knee sweep + overlay plot")
    if any_disco:
        lines.append("- **DISCOVERY**: HIT — at least one V_knee gives "
                     "sustained (>=3 cyc) osc with period in [300,600]ns")
    elif any_sust:
        lines.append("- **DISCOVERY**: PARTIAL — sustained osc found but "
                     "period out of [300,600]ns band")
    else:
        lines.append("- **DISCOVERY**: MISS — no V_knee gives sustained osc")
    if chosen and cal_verify:
        drift = cal_verify["max_drift_dec"]
        v6p = v6_verify.get("passed", False) if v6_verify else False
        if any_disco and drift <= 0.1 and v6p:
            lines.append("- **AMBITIOUS**: HIT — DISCOVERY AND Id_pk drift "
                         f"({drift:.3f} dec) <= 0.1 AND V6 still PASS")
        else:
            lines.append(f"- **AMBITIOUS**: MISS — drift={drift:.3f} dec, "
                         f"V6_pass={v6p}")
    else:
        lines.append("- **AMBITIOUS**: N/A (no DISCOVERY)")
    if kill_shot:
        lines.append("- **KILL_SHOT**: TRIGGERED — weakening V_knee destroyed "
                     f"calibration (Id_pk drift > 0.3 dec)")
    else:
        lines.append("- **KILL_SHOT**: not triggered")

    lines.append("")
    lines.append("## NO-CHEAT statement")
    lines.append("")
    lines.append(
        "Oscillation cycle counts are reported using TWO crossing levels: "
        "(a) midpoint = 0.5*(Vb_max+Vb_min), and (b) 0.5V absolute (z475's "
        "metric). Small-amplitude wobble near a fixed point can produce "
        "midpoint crossings but no 0.5V crossings; both are reported "
        "honestly. If swing < 5 mV, midpoint metric returns 0 by "
        "construction.")
    lines.append("")
    if chosen and cal_verify:
        lines.append("## Calibration tradeoff")
        lines.append("")
        for r in cal_verify["rows"]:
            if r.get("status") != "ok":
                continue
            lines.append(f"- bias VG1={r['VG1']}, VG2={r['VG2']}: "
                         f"Id_pk={r['Id_pk_mA']:.3f}mA "
                         f"(baseline {r['baseline_mA']}mA, "
                         f"drift={r['drift_dec']} dec)")
        lines.append("")
    if not any_sust:
        lines.append("## Honest conclusion")
        lines.append("")
        lines.append(
            "Weakening snap_npn_V_knee from 1.8 down through 1.2 V did NOT "
            "produce sustained free oscillation in any sweep point. The body "
            "node still settles to a stable equilibrium under DC V_d=2V hold. "
            "z475's diagnosis remains structurally correct: relaxation "
            "oscillation requires either an unstable equilibrium (Hopf) or a "
            "true hysteretic latch with a slow recovery state. Adjusting "
            "only the BJT gate cannot manufacture a Hopf bifurcation that "
            "the underlying ODE does not possess.")
        lines.append("")
    elif kill_shot:
        lines.append("## Honest conclusion — tradeoff")
        lines.append("")
        lines.append(
            "Free oscillation appears at lower V_knee values but at the cost "
            "of calibration drift > 0.3 dec on the primary bias. Brief v4.5 "
            "must pick one: (a) keep the calibrated DC fit and abandon V7, "
            "or (b) accept the calibration drift to enable V7. Default "
            "snap_npn_V_knee=1.8 is NOT being changed.")
        lines.append("")
    elif any_disco and chosen and cal_verify and cal_verify["max_drift_dec"] <= 0.1:
        v6p = v6_verify.get("passed", False) if v6_verify else False
        if v6p:
            lines.append("## Honest conclusion — DISCOVERY locked")
            lines.append("")
            lines.append(
                f"snap_npn_V_knee={chosen['V_knee']} V produces sustained "
                f"oscillation (period in band) AND preserves Id_pk on the "
                f"4-bias verify (max drift "
                f"{cal_verify['max_drift_dec']:.3f} dec) AND V6 self-reset "
                f"still passes. Recommend locking new default.")
    (OUT / "honest_analysis.md").write_text("\n".join(lines) + "\n")
    log(f"wrote honest_analysis.md")


if __name__ == "__main__":
    try:
        main()
    finally:
        LOG_FH.close()
