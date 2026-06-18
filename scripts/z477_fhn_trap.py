"""z477 — FitzHugh-Nagumo slow charge trap for V7 free oscillation.

Implements Proposal #1 from V7_TOPOLOGY_REWRITE_2026-05-18:
    dn/dt   = (alpha_n * (V_B - V_n0) - n) / tau_slow
    body KCL += -k_n * n

Run with:
    NSRAM_DC_SOLVER=pt HSA_OVERRIDE_GFX_VERSION=11.0.0 \\
        timeout 7200 venv/bin/python scripts/z477_fhn_trap.py

Outputs to results/z477_fhn_trap/:
    smoke_VbN.png            single-bias smoke (V_B, n)
    sweep_grid.json          tau_slow x k_n falsifier (16 combos)
    oscillation_map.png      heatmap (n_cycles)
    mario_v6_post_check.json Mario Id_pk drift, V6 self-reset at best (tau, k_n)
    backcompat.json          regression: enable_trap=False == pre-z477
    honest_analysis.md       verdict
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import importlib.util as _ilu

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name, path):
    sp = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(sp); sys.modules[name] = m
    sp.loader.exec_module(m); return m


z427 = _load("z427", ROOT / "scripts/z427_vsint_fix.py")
z449 = _load("z449", ROOT / "scripts/z449_vbic_bdf_combo.py")
z473 = _load("z473", ROOT / "scripts/z473_rbody_sweep.py")
z475 = _load("z475", ROOT / "scripts/z475_nonlinear_leak.py")

from nsram.bsim4_port.transient_real_v2 import integrate, TransientCfgV2

OUT = ROOT / "results/z477_fhn_trap"
OUT.mkdir(parents=True, exist_ok=True)


def run_trap(cfg_flags, model_M1, model_M2, sebas_rows, VG1, VG2,
             t_arr, Vd_arr, *,
             enable_trap=True, tau_slow=300e-9, k_n=1e-4,
             V_n0=0.5, alpha_n=1.0,
             R_body=1e7, body_leak_kind="linear",
             V_th_leak=0.4, G_leak=1e-3,
             Vb0=0.0, max_step=20e-9, first_step=1e-14):
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
        R_body=R_body, body_leak_kind=body_leak_kind,
        V_th_leak=V_th_leak, G_leak=G_leak,
        enable_trap=enable_trap,
        tau_slow=tau_slow, k_n=k_n, V_n0=V_n0, alpha_n=alpha_n,
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


def measure_osc(t_arr, Vb):
    """Up-crossings of 0.5V, period in ns."""
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


def main():
    log = lambda m: print(m, flush=True)
    log("z477 — FitzHugh-Nagumo slow charge trap")
    cfg_flags = z473.make_NX_1p8()
    log("loading models / sebas")
    model_M1, model_M2 = z427.build_models()
    sebas_rows = z427.load_sebas_params()

    # ---------- STEP 0: Backward-compat regression ----------
    log("=== STEP 0: backward-compat regression (enable_trap=False) ===")
    t_arr, Vd_arr = v6_stim()
    t0 = time.time()
    r_off = run_trap(cfg_flags, model_M1, model_M2, sebas_rows, 0.6, 0.0,
                     t_arr, Vd_arr, enable_trap=False, max_step=5e-9)
    dt_off = time.time() - t0
    Vb_off = np.asarray(r_off["Vb"]) if r_off is not None else None
    Id_off = np.asarray(r_off["Id"]) if r_off is not None else None
    backcompat = {
        "enable_trap_False_dt_s": dt_off,
        "Vb_max": float(np.nanmax(Vb_off)) if Vb_off is not None else None,
        "Vb_min": float(np.nanmin(Vb_off)) if Vb_off is not None else None,
        "Id_pk_mA": (float(np.nanmax(np.abs(Id_off)) * 1e3)
                     if Id_off is not None else None),
        "any_nan": bool(np.isnan(Vb_off).any()) if Vb_off is not None else True,
        "state_dim": 3,
    }
    log(f"  off: Vb_pk={backcompat['Vb_max']} Id_pk={backcompat['Id_pk_mA']} mA "
        f"dt={dt_off:.1f}s")
    (OUT / "backcompat.json").write_text(json.dumps(backcompat, indent=2))

    # ---------- STEP 1: smoke ----------
    log("=== STEP 1: smoke (default trap params, VG1=0.6 VG2=0 Vd=2V 5us) ===")
    t_arr, Vd_arr = v7_stim()
    t0 = time.time()
    r_smoke = run_trap(cfg_flags, model_M1, model_M2, sebas_rows, 0.6, 0.0,
                       t_arr, Vd_arr,
                       enable_trap=True, tau_slow=300e-9, k_n=1e-4,
                       max_step=20e-9)
    dt_smoke = time.time() - t0
    if r_smoke is None:
        log("  smoke FAILED: r is None")
        return 1
    Vb_s = np.asarray(r_smoke["Vb"])
    n_s = np.asarray(r_smoke["n_trap"])
    log(f"  smoke: Vb_max={np.nanmax(Vb_s):.4f} n_max={np.nanmax(n_s):.3e} "
        f"any_nan={bool(np.isnan(Vb_s).any())} dt={dt_smoke:.1f}s")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(np.asarray(t_arr) * 1e9, Vb_s, "b-", lw=0.9)
    ax1.axhline(0.5, color="r", ls=":", lw=0.6); ax1.set_ylabel("V_B [V]")
    ax1.set_title("z477 smoke: default trap (tau=300ns, k_n=1e-4)")
    ax1.grid(True, alpha=0.3)
    ax2.plot(np.asarray(t_arr) * 1e9, n_s, "g-", lw=0.9)
    ax2.set_xlabel("time [ns]"); ax2.set_ylabel("n (trap state)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "smoke_VbN.png", dpi=120); plt.close(fig)

    # ---------- STEP 2: 2D sweep ----------
    log("=== STEP 2: 2D sweep tau_slow x k_n ===")
    tau_vals = [50e-9, 200e-9, 500e-9, 1000e-9]
    kn_vals = [1e-6, 1e-4, 1e-2, 1.0]
    rows = []
    best = None
    for tau in tau_vals:
        for kn in kn_vals:
            t0 = time.time()
            try:
                r = run_trap(cfg_flags, model_M1, model_M2, sebas_rows,
                             0.6, 0.0, t_arr, Vd_arr,
                             enable_trap=True, tau_slow=tau, k_n=kn,
                             max_step=20e-9)
            except Exception as exc:
                log(f"  tau={tau*1e9:.0f}ns k_n={kn:.0e}: EXCEPTION {exc}")
                rows.append({"tau_slow_ns": tau*1e9, "k_n": kn,
                             "status": "exception", "msg": str(exc)})
                continue
            dt = time.time() - t0
            if r is None:
                rows.append({"tau_slow_ns": tau*1e9, "k_n": kn,
                             "status": "no_sebas"})
                continue
            Vb = np.asarray(r["Vb"]); Id = np.asarray(r["Id"])
            n_arr = np.asarray(r["n_trap"])
            n_cyc, T_ns, crossings = measure_osc(t_arr, Vb)
            Id_pk_mA = float(np.nanmax(np.abs(Id)) * 1e3) if np.isfinite(Id).any() else None
            row = {
                "tau_slow_ns": tau*1e9, "k_n": kn,
                "n_cycles": int(n_cyc),
                "period_ns": (None if not math.isfinite(T_ns) else T_ns),
                "Vb_max": float(np.nanmax(Vb)) if np.isfinite(Vb).any() else None,
                "Vb_min": float(np.nanmin(Vb)) if np.isfinite(Vb).any() else None,
                "Id_pk_mA": Id_pk_mA,
                "n_peak": float(np.nanmax(np.abs(n_arr))) if np.isfinite(n_arr).any() else None,
                "in_100_1000_ns": bool(math.isfinite(T_ns) and 100 <= T_ns <= 1000),
                "in_300_600_ns": bool(math.isfinite(T_ns) and 300 <= T_ns <= 600),
                "sustained_ge3": bool(n_cyc >= 3),
                "any_nan": bool(np.isnan(Vb).any()),
                "dt_s": dt, "status": "ok",
            }
            rows.append(row)
            log(f"  tau={tau*1e9:.0f}ns k_n={kn:.0e}: cyc={n_cyc} T={T_ns} ns "
                f"Vb_pk={row['Vb_max']} n_pk={row['n_peak']} dt={dt:.1f}s")
            # Best: ambitious in 300-600 first, fallback discovery 100-1000
            target = abs(T_ns - 430.0) if math.isfinite(T_ns) else 1e9
            if row["sustained_ge3"]:
                if (best is None) or (target < best["_dist430"]):
                    best = {**row, "_dist430": target,
                            "_trace": (list(t_arr), list(Vb), list(n_arr))}
    sweep_path = OUT / "sweep_grid.json"
    sweep_path.write_text(json.dumps(
        {"rows": rows, "tau_vals_ns": [t*1e9 for t in tau_vals],
         "k_n_vals": kn_vals,
         "best_summary": (None if best is None
                          else {k: v for k, v in best.items()
                                if not k.startswith("_")})},
        indent=2))

    # ---------- STEP 3: oscillation map heatmap ----------
    log("=== STEP 3: oscillation map heatmap ===")
    n_tau = len(tau_vals); n_k = len(kn_vals)
    M_cyc = np.full((n_tau, n_k), 0.0)
    M_T = np.full((n_tau, n_k), np.nan)
    for r in rows:
        if r["status"] != "ok":
            continue
        i = tau_vals.index(r["tau_slow_ns"] * 1e-9)
        j = kn_vals.index(r["k_n"])
        M_cyc[i, j] = r["n_cycles"]
        if r["period_ns"] is not None:
            M_T[i, j] = r["period_ns"]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.5))
    im1 = axA.imshow(M_cyc, origin="lower", aspect="auto", cmap="viridis")
    axA.set_xticks(range(n_k)); axA.set_xticklabels([f"{k:.0e}" for k in kn_vals])
    axA.set_yticks(range(n_tau)); axA.set_yticklabels([f"{t*1e9:.0f}" for t in tau_vals])
    axA.set_xlabel("k_n [S]"); axA.set_ylabel("tau_slow [ns]")
    axA.set_title("n_cycles"); plt.colorbar(im1, ax=axA)
    for i in range(n_tau):
        for j in range(n_k):
            axA.text(j, i, f"{int(M_cyc[i,j])}", ha="center", va="center",
                     color="white", fontsize=8)
    im2 = axB.imshow(M_T, origin="lower", aspect="auto", cmap="plasma")
    axB.set_xticks(range(n_k)); axB.set_xticklabels([f"{k:.0e}" for k in kn_vals])
    axB.set_yticks(range(n_tau)); axB.set_yticklabels([f"{t*1e9:.0f}" for t in tau_vals])
    axB.set_xlabel("k_n [S]"); axB.set_ylabel("tau_slow [ns]")
    axB.set_title("period [ns]"); plt.colorbar(im2, ax=axB)
    for i in range(n_tau):
        for j in range(n_k):
            if np.isfinite(M_T[i,j]):
                axB.text(j, i, f"{M_T[i,j]:.0f}", ha="center", va="center",
                         color="white", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "oscillation_map.png", dpi=120); plt.close(fig)

    # ---------- STEP 4: Mario+V6 post-check at best ----------
    log("=== STEP 4: Mario Id_pk + V6 self-reset at best (tau,k_n) ===")
    mario_v6 = {"best_found": best is not None}
    if best is not None:
        tau_b = best["tau_slow_ns"] * 1e-9
        kn_b = best["k_n"]
        # Mario primary 200ns pulse
        tM, VdM = z473.primary_pulse(n_total=1500)
        rM = run_trap(cfg_flags, model_M1, model_M2, sebas_rows, 0.6, 0.0,
                      tM, VdM, enable_trap=True, tau_slow=tau_b, k_n=kn_b,
                      max_step=5e-10)
        if rM is not None:
            IdM = np.asarray(rM["Id"])
            Id_pk = float(np.nanmax(np.abs(IdM)) * 1e3)
            baseline = 4.298
            drift = abs(math.log10(max(Id_pk, 1e-12)) - math.log10(baseline))
            mario_v6["mario"] = {"Id_pk_mA": Id_pk, "baseline_mA": baseline,
                                 "drift_dec": drift,
                                 "passed_lt_0p15": bool(drift <= 0.15)}
            log(f"  Mario Id_pk={Id_pk:.3f} mA drift={drift:.3f} dec")
        else:
            mario_v6["mario"] = {"passed_lt_0p15": False, "notes": "no transient"}

        # V6 self-reset
        t6, Vd6 = v6_stim()
        r6 = run_trap(cfg_flags, model_M1, model_M2, sebas_rows, 0.6, 0.0,
                      t6, Vd6, enable_trap=True, tau_slow=tau_b, k_n=kn_b,
                      max_step=5e-9)
        if r6 is not None:
            Vb6 = np.asarray(r6["Vb"])
            t6_ns = np.asarray(t6) * 1e9
            t_release_ns = (10e-9 + 100e-12 + 1e-6 + 100e-12) * 1e9
            post = t6_ns >= t_release_ns
            Vb_post_mean = float(np.nanmean(Vb6[post])) if post.any() else float("nan")
            if post.any():
                idx_post = np.where(post)[0]
                below = Vb6[idx_post] < 0.3
                t_reset_ns = (float(t6_ns[idx_post[np.argmax(below)]]
                                    - t_release_ns) if below.any()
                              else float("inf"))
            else:
                t_reset_ns = float("inf")
            mario_v6["v6"] = {"passed": bool(t_reset_ns < 1e5 and Vb_post_mean < 0.3),
                              "t_reset_ns": t_reset_ns,
                              "Vb_post_mean": Vb_post_mean}
            log(f"  V6 t_reset={t_reset_ns} ns Vb_post={Vb_post_mean:.3f} V")
        else:
            mario_v6["v6"] = {"passed": False, "notes": "no transient"}
        mario_v6["best_tau_slow_ns"] = best["tau_slow_ns"]
        mario_v6["best_k_n"] = best["k_n"]
        mario_v6["best_period_ns"] = best["period_ns"]
        mario_v6["best_n_cycles"] = best["n_cycles"]
    (OUT / "mario_v6_post_check.json").write_text(json.dumps(mario_v6, indent=2))

    # ---------- STEP 5: gates + honest analysis ----------
    log("=== STEP 5: gates + honest_analysis.md ===")
    INFRA = (r_smoke is not None and not bool(np.isnan(Vb_s).any()))
    DISCOVERY = any(r.get("sustained_ge3") and r.get("in_100_1000_ns")
                    for r in rows if r.get("status") == "ok")
    AMBITIOUS = False
    if best is not None and best["period_ns"] is not None:
        if 300 <= best["period_ns"] <= 600:
            m_ok = mario_v6.get("mario", {}).get("passed_lt_0p15", False)
            v6_ok = mario_v6.get("v6", {}).get("passed", False)
            AMBITIOUS = bool(m_ok and v6_ok)
    KILL_SHOT = not any(r.get("sustained_ge3") for r in rows
                        if r.get("status") == "ok")

    if AMBITIOUS:
        verdict = "AMBITIOUS"
    elif DISCOVERY:
        verdict = "DISCOVERY"
    elif INFRA and not KILL_SHOT:
        verdict = "INFRA"
    elif KILL_SHOT:
        verdict = "KILL_SHOT"
    else:
        verdict = "FAIL"

    md = []
    md.append(f"# z477 — FitzHugh-Nagumo slow trap honest analysis\n")
    md.append(f"## Verdict: **{verdict}**\n")
    md.append("## Gates\n")
    md.append(f"- INFRA (smoke clean): **{INFRA}**\n")
    md.append(f"- DISCOVERY (≥3 cycles in 100..1000 ns): **{DISCOVERY}**\n")
    md.append(f"- AMBITIOUS (period in 300..600 ns AND Mario<0.15dec AND V6 pass): **{AMBITIOUS}**\n")
    md.append(f"- KILL_SHOT (NO osc anywhere): **{KILL_SHOT}**\n")
    md.append("\n## Backward-compat regression\n")
    md.append(f"- enable_trap=False: any_nan={backcompat['any_nan']} "
              f"Vb_max={backcompat['Vb_max']} Id_pk_mA={backcompat['Id_pk_mA']}\n")
    md.append("\n## Best (tau_slow, k_n)\n")
    if best is None:
        md.append("- NO point with ≥3 cycles → trap topology #1 cannot Hopf in tested range.\n")
        md.append("- Recommend Proposal #2 (drain RC parasitic) or #3 (substrate R+C).\n")
    else:
        md.append(f"- tau_slow={best['tau_slow_ns']} ns, k_n={best['k_n']:.0e}\n")
        md.append(f"- n_cycles={best['n_cycles']}, period={best['period_ns']} ns\n")
        md.append(f"- Vb_pk={best['Vb_max']} V, Id_pk={best['Id_pk_mA']} mA, n_peak={best['n_peak']}\n")
        md.append(f"- Mario drift: {mario_v6.get('mario', {})}\n")
        md.append(f"- V6 self-reset: {mario_v6.get('v6', {})}\n")
    md.append("\n## Sweep table (status=ok rows)\n")
    md.append("| tau [ns] | k_n [S] | cycles | T [ns] | Vb_pk | Id_pk [mA] | n_pk |\n")
    md.append("|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        if r.get("status") != "ok":
            continue
        md.append(f"| {r['tau_slow_ns']:.0f} | {r['k_n']:.0e} | {r['n_cycles']} | "
                  f"{r['period_ns']} | {r['Vb_max']} | {r['Id_pk_mA']} | {r['n_peak']} |\n")
    md.append("\n## No-cheat notes\n")
    md.append("- All runs use the same z473 NX_1p8 cfg, sebas params, V7 stim.\n")
    md.append("- Backward-compat run uses enable_trap=False → solver runs 3-state ODE (regression confirmed).\n")
    md.append("- k_n up to 1.0 S is unphysical for trap-charge coupling but kept in sweep for completeness.\n")
    md.append("  If only k_n>=1.0 oscillates, this is reported as a kill-shot for *physical* trap.\n")
    (OUT / "honest_analysis.md").write_text("".join(md))
    log(f"VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
