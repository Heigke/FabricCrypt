"""z475 — threshold-gated nonlinear body leak for V7 free oscillation.

Linear R_body (z473) gives V6 self-reset but cannot break the BJT positive-
feedback latch during a DC V_d hold — V7 free oscillation fails. This script
implements I_leak = relu(V_b - V_th_leak) * G_leak in transient_real_v2 and
sweeps V_th_leak × G_leak to find a combo producing sustained oscillation
with period in [300, 600] ns (Mario slide-08 target ~430 ns).

Outputs (all to results/z475_nonlinear_leak/):
    sweep_grid.json          - full (V_th, G) sweep results
    v7_pass_config.json      - chosen sweet spot (if any)
    z461_post_z475.json      - V6/V7/Mario re-check at sweet spot
    mario_shape_v3.json      - mario metrics at sweet spot
    transient_osc.png        - V_B(t) trace showing oscillation
    honest_analysis.md       - findings, kill-shots, no-cheat assessment
    impl.diff                - git diff of transient_real_v2.py
"""
from __future__ import annotations
import json
import math
import os
import sys
import time
import subprocess
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
z449 = _load("z449", ROOT / "scripts/z449_vbic_bdf_combo.py")
z473 = _load("z473", ROOT / "scripts/z473_rbody_sweep.py")

from nsram.bsim4_port import transient_real_v2 as trv2
from nsram.bsim4_port.transient_real_v2 import integrate, TransientCfgV2

OUT = ROOT / "results/z475_nonlinear_leak"
OUT.mkdir(parents=True, exist_ok=True)


def run_transient_nl(cfg_flags, model_M1, model_M2, sebas_rows, VG1, VG2,
                     t_arr, Vd_arr, *, body_leak_kind="threshold",
                     V_th_leak=0.4, G_leak=1e-3, V_th_sharp=0.0,
                     R_body=None, Vb0=0.0,
                     max_step=1e-10, first_step=1e-14):
    """Mirror of z473.run_transient with nonlinear-leak hooks."""
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
        body_leak_kind=body_leak_kind,
        V_th_leak=V_th_leak, G_leak=G_leak, V_th_sharp=V_th_sharp,
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


def measure_oscillation(t_arr, Vb):
    """Return (n_cycles, period_ns, crossings) using up-crossings of 0.5 V."""
    t_ns = np.asarray(t_arr) * 1e9
    Vb = np.asarray(Vb)
    crossings = []
    for i in range(1, len(Vb)):
        if (np.isfinite(Vb[i]) and np.isfinite(Vb[i-1])
                and Vb[i-1] < 0.5 <= Vb[i]):
            crossings.append(float(t_ns[i]))
    n_cycles = max(0, len(crossings) - 1)
    if len(crossings) >= 2:
        period_ns = float(np.mean(np.diff(crossings)))
    else:
        period_ns = float("nan")
    return n_cycles, period_ns, crossings


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


def primary_pulse_stim():
    return z473.primary_pulse(n_total=1500)


def sweep(cfg_flags, model_M1, model_M2, sebas_rows, log):
    """Sweep V_th_leak × G_leak under V7 stimulus, score oscillation."""
    V_th_vals = [0.3, 0.4, 0.5, 0.6]
    G_vals = [1e-4, 1e-3, 1e-2]
    rows = []
    t_arr, Vd_arr = v7_stim()
    best = None
    for Vth in V_th_vals:
        for G in G_vals:
            t0 = time.time()
            try:
                r = run_transient_nl(cfg_flags, model_M1, model_M2, sebas_rows,
                                     0.6, 0.0, t_arr, Vd_arr,
                                     V_th_leak=Vth, G_leak=G,
                                     max_step=20e-9)
            except Exception as exc:
                log(f"  V_th={Vth} G={G:.0e}: EXCEPTION {exc}")
                rows.append({"V_th_leak": Vth, "G_leak": G,
                             "status": "exception", "msg": str(exc)})
                continue
            dt = time.time() - t0
            if r is None:
                rows.append({"V_th_leak": Vth, "G_leak": G,
                             "status": "no_sebas"})
                continue
            Vb = np.asarray(r["Vb"]); Id = np.asarray(r["Id"])
            n_cyc, T_ns, crossings = measure_oscillation(t_arr, Vb)
            Id_pk_mA = float(np.nanmax(np.abs(Id)) * 1e3)
            Vb_pk = float(np.nanmax(Vb))
            in_range = (math.isfinite(T_ns)
                        and 300.0 <= T_ns <= 600.0)
            sustained = (n_cyc >= 3)
            row = {"V_th_leak": Vth, "G_leak": G,
                   "n_cycles": int(n_cyc),
                   "period_ns": (None if not math.isfinite(T_ns) else T_ns),
                   "Vb_peak_V": Vb_pk,
                   "Id_pk_mA": Id_pk_mA,
                   "n_crossings": len(crossings),
                   "in_300_600_ns": bool(in_range),
                   "sustained_ge3": bool(sustained),
                   "dt_s": dt,
                   "status": "ok"}
            rows.append(row)
            log(f"  V_th={Vth} G={G:.0e}: cyc={n_cyc} T={T_ns} ns "
                f"Vb_pk={Vb_pk:.3f} Id_pk={Id_pk_mA:.3f} mA dt={dt:.1f}s")
            # Best = sustained AND period in 300-600. Among those prefer
            # closest to 430 ns; fallback: maximize n_cycles.
            score_target = abs(T_ns - 430.0) if math.isfinite(T_ns) else 1e9
            if sustained and in_range:
                if (best is None) or (score_target < best["_dist430"]):
                    best = {**row, "_dist430": score_target,
                            "_trace": (list(t_arr), list(Vb))}
    return rows, best


def make_v7_plot(t_arr, Vb, period_ns, n_cyc, out_path):
    t_ns = np.asarray(t_arr) * 1e9
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    ax.plot(t_ns, Vb, "b-", lw=0.9, label="V_B")
    ax.axhline(0.5, color="red", ls=":", lw=0.8, label="0.5 V")
    ax.set_xlabel("time [ns]"); ax.set_ylabel("V_B [V]")
    title = f"z475 V7 free osc: n_cycles={n_cyc}, period={period_ns:.1f} ns"
    ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120); plt.close(fig)


def run_post_check(cfg_flags, model_M1, model_M2, sebas_rows, sweet, log):
    """Run V6, V7 (already known), plus mario shape at the sweet spot."""
    Vth = sweet["V_th_leak"]; G = sweet["G_leak"]
    # V6 — should still pass: nonlinear leak strictly dominates linear above
    # threshold, so V_B drains AT LEAST as fast as the linear leak at V_B>Vth.
    log(f"  V6 self-reset @ V_th={Vth} G={G:.0e}")
    t_arr, Vd_arr = v6_stim()
    r6 = run_transient_nl(cfg_flags, model_M1, model_M2, sebas_rows,
                          0.6, 0.0, t_arr, Vd_arr,
                          V_th_leak=Vth, G_leak=G, max_step=5e-9)
    if r6 is None:
        v6_out = {"passed": False, "notes": "no transient"}
    else:
        Vb = np.asarray(r6["Vb"])
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
        v6_out = {"passed": bool(passed),
                  "t_reset_ns": t_reset_ns,
                  "Vb_post_mean": Vb_post_mean}
        log(f"    V6 t_reset={t_reset_ns} ns Vb_post={Vb_post_mean:.3f} V "
            f"passed={passed}")

    # V7 already verified in sweep but re-run here for clean record.
    t_arr, Vd_arr = v7_stim()
    r7 = run_transient_nl(cfg_flags, model_M1, model_M2, sebas_rows,
                          0.6, 0.0, t_arr, Vd_arr,
                          V_th_leak=Vth, G_leak=G, max_step=20e-9)
    Vb7 = np.asarray(r7["Vb"]) if r7 is not None else None
    if Vb7 is None:
        v7_out = {"passed": False, "notes": "no transient"}
    else:
        n_cyc, T_ns, _ = measure_oscillation(t_arr, Vb7)
        passed = (n_cyc >= 3) and (100 <= T_ns <= 1000)
        v7_out = {"passed": bool(passed),
                  "n_cycles": int(n_cyc),
                  "period_ns": (None if not math.isfinite(T_ns) else T_ns)}
        log(f"    V7 cyc={n_cyc} T={T_ns} ns passed={passed}")

    # Id_pk preserved check — primary 200ns pulse, must hold within 0.1 dec of
    # z473 baseline (Id_pk≈4.298 mA at R_body=1e7).
    t_arr, Vd_arr = primary_pulse_stim()
    r_pk = run_transient_nl(cfg_flags, model_M1, model_M2, sebas_rows,
                            0.6, 0.0, t_arr, Vd_arr,
                            V_th_leak=Vth, G_leak=G, max_step=5e-10)
    if r_pk is None:
        idpk_out = {"passed": False, "notes": "no transient"}
    else:
        Id = np.asarray(r_pk["Id"])
        Id_pk_mA = float(np.nanmax(np.abs(Id)) * 1e3)
        # baseline from retry_lower.json @ R_body=1e7: 4.298 mA
        baseline_mA = 4.298
        drift_dec = abs(math.log10(max(Id_pk_mA, 1e-12)) -
                        math.log10(baseline_mA))
        passed = drift_dec <= 0.1
        idpk_out = {"passed": bool(passed),
                    "Id_pk_mA": Id_pk_mA,
                    "baseline_mA": baseline_mA,
                    "drift_dec": drift_dec}
        log(f"    Id_pk={Id_pk_mA:.3f} mA drift={drift_dec:.3f} dec passed={passed}")

    return {"V6": v6_out, "V7": v7_out, "Id_pk": idpk_out,
            "V_th_leak": Vth, "G_leak": G}


def mario_shape_v3(cfg_flags, model_M1, model_M2, sebas_rows, sweet, log):
    """Mario metrics under threshold leak."""
    Vth = sweet["V_th_leak"]; G = sweet["G_leak"]
    targets = {"t_rise_ns": 26.0, "t_fall_ns": 76.0,
               "Vb_swing_V_lo": 0.5, "Vb_swing_V_hi": 0.7,
               "osc_period_ns": 430.0}
    # Single 200ns pulse
    t1, Vd1 = z473.stim_pulse(V_lo=0.05, V_hi=2.0,
                               t_pre=10e-9, t_rise=100e-12,
                               t_hold=200e-9, t_fall=100e-12,
                               t_post=300e-9, n_total=2000)
    t_pulse_start = 10e-9 + 100e-12
    t_pulse_end = t_pulse_start + 200e-9
    r1 = run_transient_nl(cfg_flags, model_M1, model_M2, sebas_rows,
                          0.6, 0.0, t1, Vd1,
                          V_th_leak=Vth, G_leak=G, max_step=5e-10)
    if r1 is None:
        return {"status": "fail_no_transient"}
    Vb1 = np.asarray(r1["Vb"]); Id1 = np.asarray(r1["Id"])
    m1 = z473.extract_metrics(t1, Vb1, t_pulse_start, t_pulse_end)
    post_mask = t1 >= t_pulse_end + 50e-9
    reset_ok = bool((Vb1[post_mask] < 0.3).any()) if post_mask.any() else False
    m1["self_reset_post_pulse"] = reset_ok
    m1["Id_peak_A"] = float(np.nanmax(np.abs(Id1)))

    # period from sweep best
    period_ns = sweet.get("period_ns", float("nan"))

    scores = {
        "t_rise_match": bool(1e-9 < m1["t_rise"] < 50e-9),
        "t_fall_match": bool(40e-9 < m1["t_fall"] < 200e-9),
        "Vb_swing_match": bool(0.4 < m1["swing"] < 0.9),
        "self_reset_match": bool(reset_ok),
        "osc_period_match": bool(period_ns is not None
                                  and math.isfinite(period_ns)
                                  and 300 <= period_ns <= 600),
    }
    n_match = sum(1 for v in scores.values() if v)
    return {"target": targets, "single_pulse": m1,
            "oscillation": {"period_ns": period_ns},
            "match_scores": scores, "n_metrics_matched": n_match,
            "V_th_leak": Vth, "G_leak": G}


def main():
    log = lambda m: print(m, flush=True)
    log("z475 — threshold nonlinear leak: smoke + sweep")
    cfg_flags = z473.make_NX_1p8()
    log("loading models / sebas")
    model_M1, model_M2 = z427.build_models()
    sebas_rows = z427.load_sebas_params()

    log("=== STEP 1: V_th × G sweep on V7 stimulus ===")
    rows, best = sweep(cfg_flags, model_M1, model_M2, sebas_rows, log)
    sweep_path = OUT / "sweep_grid.json"
    sweep_path.write_text(json.dumps(
        {"rows": rows,
         "best_summary": (None if best is None
                          else {k: v for k, v in best.items()
                                if not k.startswith("_")})},
        indent=2, default=float))
    log(f"wrote {sweep_path.name}")

    if best is None:
        log("NO sweet spot found — no (V_th,G) gave sustained osc in [300,600] ns")
        # write fallback honest analysis & return
        (OUT / "v7_pass_config.json").write_text(
            json.dumps({"found": False, "rows": rows}, indent=2, default=float))
        (OUT / "honest_analysis.md").write_text(
            "# z475 — Honest Analysis\n\n"
            "## Result: NO sweet spot found\n\n"
            "Sweep V_th ∈ {0.3,0.4,0.5,0.6} V × G ∈ {1e-4, 1e-3, 1e-2} S\n"
            "produced no parameter combination with ≥3 oscillation cycles\n"
            "and period in [300, 600] ns under DC V_d=2 V hold for 5 µs.\n\n"
            "Likely cause: the parasitic-NPN regenerative latch held at the\n"
            "Slotboom multiplier knee; threshold-relu leak alone cannot break\n"
            "the latch without unrealistically large G or low V_th.\n\n"
            "## NO-CHEAT: not flipping V7 by inflating params.\n"
        )
        log("DONE (no V7 pass).")
        return

    log(f"=== BEST: V_th={best['V_th_leak']} G={best['G_leak']:.0e} "
        f"T={best['period_ns']} ns cyc={best['n_cycles']} ===")
    # Save v7_pass_config
    pass_cfg = {k: v for k, v in best.items() if not k.startswith("_")}
    pass_cfg["found"] = True
    (OUT / "v7_pass_config.json").write_text(
        json.dumps(pass_cfg, indent=2, default=float))

    # Plot transient
    t_arr_t, Vb_t = best["_trace"]
    make_v7_plot(t_arr_t, Vb_t,
                 period_ns=best["period_ns"], n_cyc=best["n_cycles"],
                 out_path=OUT / "transient_osc.png")
    log(f"wrote transient_osc.png")

    log("=== STEP 2: post-check V6/V7/Id_pk at sweet spot ===")
    post = run_post_check(cfg_flags, model_M1, model_M2, sebas_rows, best, log)
    (OUT / "z461_post_z475.json").write_text(
        json.dumps(post, indent=2, default=float))

    log("=== STEP 3: mario shape v3 at sweet spot ===")
    mario = mario_shape_v3(cfg_flags, model_M1, model_M2, sebas_rows, best, log)
    (OUT / "mario_shape_v3.json").write_text(
        json.dumps(mario, indent=2, default=float))

    # honest analysis
    v6_ok = post["V6"].get("passed", False)
    v7_ok = post["V7"].get("passed", False)
    idpk_ok = post["Id_pk"].get("passed", False)
    drift = post["Id_pk"].get("drift_dec", float("nan"))
    nmatch = mario.get("n_metrics_matched", 0)
    nocheat_warn = ""
    if best["G_leak"] > 1.0:
        nocheat_warn += ("- KILL_SHOT WARN: G_leak > 1 S is "
                         "non-physical for body-leak.\n")
    if best["V_th_leak"] > 1.0:
        nocheat_warn += ("- KILL_SHOT WARN: V_th_leak > 1 V is "
                         "non-physical (above BJT Vbe on).\n")
    if drift > 0.3:
        nocheat_warn += (f"- KILL_SHOT: Id_pk drift {drift:.3f} dec > "
                         "0.3 dec calibration loss.\n")
    md = (
        "# z475 — Honest Analysis (threshold-gated nonlinear body leak)\n\n"
        f"## Sweet spot\nV_th_leak = {best['V_th_leak']} V, "
        f"G_leak = {best['G_leak']:.0e} S\n"
        f"period = {best['period_ns']:.1f} ns, n_cycles = {best['n_cycles']}\n\n"
        "## Post-check\n"
        f"- V6 self-reset: {'PASS' if v6_ok else 'FAIL'} "
        f"(t_reset={post['V6'].get('t_reset_ns')} ns, "
        f"Vb_post={post['V6'].get('Vb_post_mean')})\n"
        f"- V7 oscillation: {'PASS' if v7_ok else 'FAIL'} "
        f"(cyc={post['V7'].get('n_cycles')}, "
        f"T={post['V7'].get('period_ns')} ns)\n"
        f"- Id_pk preserved: {'PASS' if idpk_ok else 'FAIL'} "
        f"(Id_pk={post['Id_pk'].get('Id_pk_mA'):.3f} mA, "
        f"drift={drift:.3f} dec)\n\n"
        f"## Mario shape (n_matched = {nmatch}/5)\n"
        f"```\n{json.dumps(mario.get('match_scores', {}), indent=2)}\n```\n\n"
        "## NO-CHEAT\n"
        f"{nocheat_warn or '- params within physical range (V_th<1V, G<1S, drift<0.3dec).'}\n"
    )
    (OUT / "honest_analysis.md").write_text(md)

    # impl.diff (git diff vs HEAD on transient_real_v2.py)
    try:
        diff = subprocess.check_output(
            ["git", "diff", "HEAD", "--",
             "nsram/nsram/bsim4_port/transient_real_v2.py"],
            cwd=str(ROOT), text=True)
        (OUT / "impl.diff").write_text(diff)
    except Exception as exc:
        (OUT / "impl.diff").write_text(f"# git diff failed: {exc}\n")

    log("DONE.")


if __name__ == "__main__":
    main()
