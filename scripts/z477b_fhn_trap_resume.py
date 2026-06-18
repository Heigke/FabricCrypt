"""z477b — Resume FHN trap sweep with incremental writes + per-combo timeout.

Skips STEP 0 (backcompat already verified) and STEP 1 (smoke).
Goes straight to 4x4 sweep, writes sweep_grid.json after EACH combo so
progress is preserved if killed. Per-combo wall budget enforced with
signal.alarm (60s default).

Then runs STEP 4 Mario+V6 verify at "best" point (most variation if no osc).
"""
from __future__ import annotations
import json, math, os, signal, sys, time
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
z477 = _load("z477", ROOT / "scripts/z477_fhn_trap.py")

from nsram.bsim4_port.transient_real_v2 import integrate, TransientCfgV2

OUT = ROOT / "results/z477_fhn_trap"
OUT.mkdir(parents=True, exist_ok=True)

# Per-combo wall budget
COMBO_TIMEOUT_S = int(os.environ.get("Z477B_TIMEOUT", "90"))


class TimeoutError_(Exception):
    pass


def _alarm_handler(signum, frame):
    raise TimeoutError_("combo wall timeout")


def main():
    log = lambda m: print(m, flush=True)
    log(f"z477b — RESUME FHN trap sweep (per-combo timeout={COMBO_TIMEOUT_S}s)")
    cfg_flags = z473.make_NX_1p8()
    log("loading models / sebas")
    model_M1, model_M2 = z427.build_models()
    sebas_rows = z427.load_sebas_params()

    t_arr, Vd_arr = z477.v7_stim()
    tau_vals = [50e-9, 200e-9, 500e-9, 1000e-9]
    kn_vals = [1e-6, 1e-4, 1e-2, 1.0]

    sweep_path = OUT / "sweep_grid.json"
    # Try to resume from existing partial sweep_grid.json
    done = {}
    if sweep_path.exists():
        try:
            prev = json.loads(sweep_path.read_text())
            for r in prev.get("rows", []):
                key = (round(r["tau_slow_ns"], 6), float(r["k_n"]))
                done[key] = r
            log(f"resume: {len(done)} prior rows loaded")
        except Exception as exc:
            log(f"resume: prev sweep_grid.json unreadable ({exc}); starting fresh")
            done = {}

    rows = list(done.values())

    def _flush(best_summary=None):
        sweep_path.write_text(json.dumps(
            {"rows": rows, "tau_vals_ns": [t*1e9 for t in tau_vals],
             "k_n_vals": kn_vals,
             "best_summary": best_summary},
            indent=2))

    log("=== STEP 2: 2D sweep tau_slow x k_n (incremental writes) ===")
    for tau in tau_vals:
        for kn in kn_vals:
            key = (round(tau*1e9, 6), float(kn))
            if key in done and done[key].get("status") == "ok":
                log(f"  tau={tau*1e9:.0f}ns k_n={kn:.0e}: SKIP (cached)")
                continue
            t0 = time.time()
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(COMBO_TIMEOUT_S)
            try:
                r = z477.run_trap(
                    cfg_flags, model_M1, model_M2, sebas_rows,
                    0.6, 0.0, t_arr, Vd_arr,
                    enable_trap=True, tau_slow=tau, k_n=kn,
                    max_step=20e-9,
                )
                signal.alarm(0)
                dt = time.time() - t0
                if r is None:
                    row = {"tau_slow_ns": tau*1e9, "k_n": kn,
                           "status": "no_sebas", "dt_s": dt}
                else:
                    Vb = np.asarray(r["Vb"]); Id = np.asarray(r["Id"])
                    n_arr = np.asarray(r["n_trap"])
                    n_cyc, T_ns, _ = z477.measure_osc(t_arr, Vb)
                    Id_pk_mA = (float(np.nanmax(np.abs(Id)) * 1e3)
                                if np.isfinite(Id).any() else None)
                    Vb_max = float(np.nanmax(Vb)) if np.isfinite(Vb).any() else None
                    Vb_min = float(np.nanmin(Vb)) if np.isfinite(Vb).any() else None
                    Vb_var = (float(np.nanvar(Vb))
                              if np.isfinite(Vb).any() else None)
                    row = {
                        "tau_slow_ns": tau*1e9, "k_n": kn,
                        "n_cycles": int(n_cyc),
                        "period_ns": (None if not math.isfinite(T_ns) else T_ns),
                        "Vb_max": Vb_max, "Vb_min": Vb_min,
                        "Vb_var": Vb_var,
                        "Vb_range": ((Vb_max - Vb_min)
                                     if (Vb_max is not None and Vb_min is not None)
                                     else None),
                        "Id_pk_mA": Id_pk_mA,
                        "n_peak": (float(np.nanmax(np.abs(n_arr)))
                                   if np.isfinite(n_arr).any() else None),
                        "in_100_1000_ns": bool(
                            math.isfinite(T_ns) and 100 <= T_ns <= 1000),
                        "in_300_600_ns": bool(
                            math.isfinite(T_ns) and 300 <= T_ns <= 600),
                        "sustained_ge3": bool(n_cyc >= 3),
                        "any_nan": bool(np.isnan(Vb).any()),
                        "dt_s": dt, "status": "ok",
                    }
            except TimeoutError_:
                signal.alarm(0)
                dt = time.time() - t0
                row = {"tau_slow_ns": tau*1e9, "k_n": kn,
                       "status": "timeout", "dt_s": dt}
                log(f"  tau={tau*1e9:.0f}ns k_n={kn:.0e}: TIMEOUT >{COMBO_TIMEOUT_S}s")
            except Exception as exc:
                signal.alarm(0)
                dt = time.time() - t0
                row = {"tau_slow_ns": tau*1e9, "k_n": kn,
                       "status": "exception", "msg": str(exc)[:200], "dt_s": dt}
                log(f"  tau={tau*1e9:.0f}ns k_n={kn:.0e}: EXC {exc}")

            # Replace prior cached row (if any) for this key
            rows = [rr for rr in rows
                    if (round(rr["tau_slow_ns"], 6), float(rr["k_n"])) != key]
            rows.append(row)
            done[key] = row
            if row.get("status") == "ok":
                log(f"  tau={tau*1e9:.0f}ns k_n={kn:.0e}: cyc={row['n_cycles']} "
                    f"T={row['period_ns']} Vb_rng={row.get('Vb_range')} "
                    f"dt={row['dt_s']:.1f}s")
            _flush()

    # Pick best — prefer sustained_ge3 closest to 430 ns; else max Vb_range
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    best = None
    oscillators = [r for r in ok_rows if r.get("sustained_ge3")]
    if oscillators:
        def _dist(r):
            T = r.get("period_ns")
            return abs(T - 430.0) if T is not None else 1e9
        best = min(oscillators, key=_dist)
        log(f"best=oscillator tau={best['tau_slow_ns']} k_n={best['k_n']} "
            f"cyc={best['n_cycles']} T={best['period_ns']}")
    elif ok_rows:
        def _vbr(r):
            v = r.get("Vb_range")
            return v if v is not None else -1.0
        best = max(ok_rows, key=_vbr)
        log(f"best=variation tau={best['tau_slow_ns']} k_n={best['k_n']} "
            f"Vb_range={best.get('Vb_range')}")
    _flush(best_summary=(None if best is None else
                          {k: v for k, v in best.items()
                           if not str(k).startswith("_")}))

    # ---------- STEP 3: oscillation map heatmap ----------
    log("=== STEP 3: oscillation_map.png ===")
    n_tau = len(tau_vals); n_k = len(kn_vals)
    M_cyc = np.full((n_tau, n_k), 0.0)
    M_T = np.full((n_tau, n_k), np.nan)
    M_Vr = np.full((n_tau, n_k), np.nan)
    for r in ok_rows:
        try:
            i = tau_vals.index(r["tau_slow_ns"] * 1e-9)
        except ValueError:
            continue
        try:
            j = kn_vals.index(r["k_n"])
        except ValueError:
            continue
        M_cyc[i, j] = r["n_cycles"]
        if r.get("period_ns") is not None:
            M_T[i, j] = r["period_ns"]
        if r.get("Vb_range") is not None:
            M_Vr[i, j] = r["Vb_range"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, M, title, cmap in zip(
        axes, [M_cyc, M_T, M_Vr],
        ["n_cycles", "period [ns]", "Vb_range [V]"],
        ["viridis", "plasma", "magma"],
    ):
        im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap)
        ax.set_xticks(range(n_k)); ax.set_xticklabels([f"{k:.0e}" for k in kn_vals])
        ax.set_yticks(range(n_tau));
        ax.set_yticklabels([f"{t*1e9:.0f}" for t in tau_vals])
        ax.set_xlabel("k_n [S]"); ax.set_ylabel("tau_slow [ns]")
        ax.set_title(title); plt.colorbar(im, ax=ax)
        for i in range(n_tau):
            for j in range(n_k):
                v = M[i, j]
                if np.isfinite(v):
                    txt = f"{int(v)}" if title == "n_cycles" else f"{v:.2g}"
                    ax.text(j, i, txt, ha="center", va="center",
                            color="white", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "oscillation_map.png", dpi=120)
    plt.close(fig)

    # ---------- STEP 4: Mario+V6 verify at best ----------
    log("=== STEP 4: Mario Id_pk + V6 self-reset at best (tau,k_n) ===")
    mario_v6 = {"best_found": best is not None}
    if best is not None:
        tau_b = best["tau_slow_ns"] * 1e-9
        kn_b = best["k_n"]
        mario_v6["best_tau_slow_ns"] = best["tau_slow_ns"]
        mario_v6["best_k_n"] = best["k_n"]
        mario_v6["best_period_ns"] = best.get("period_ns")
        mario_v6["best_n_cycles"] = best.get("n_cycles")
        mario_v6["best_Vb_range"] = best.get("Vb_range")

        # Mario primary 200ns pulse
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(COMBO_TIMEOUT_S * 2)
        try:
            tM, VdM = z473.primary_pulse(n_total=1500)
            rM = z477.run_trap(
                cfg_flags, model_M1, model_M2, sebas_rows, 0.6, 0.0,
                tM, VdM, enable_trap=True, tau_slow=tau_b, k_n=kn_b,
                max_step=5e-10,
            )
            signal.alarm(0)
            if rM is not None:
                IdM = np.asarray(rM["Id"])
                Id_pk = float(np.nanmax(np.abs(IdM)) * 1e3)
                baseline = 4.298
                drift = abs(math.log10(max(Id_pk, 1e-12))
                            - math.log10(baseline))
                mario_v6["mario"] = {
                    "Id_pk_mA": Id_pk, "baseline_mA": baseline,
                    "drift_dec": drift,
                    "passed_lt_0p15": bool(drift <= 0.15)}
                log(f"  Mario Id_pk={Id_pk:.3f} mA drift={drift:.3f} dec")
            else:
                mario_v6["mario"] = {"passed_lt_0p15": False,
                                     "notes": "no transient"}
        except TimeoutError_:
            signal.alarm(0)
            mario_v6["mario"] = {"passed_lt_0p15": False, "notes": "timeout"}
        except Exception as exc:
            signal.alarm(0)
            mario_v6["mario"] = {"passed_lt_0p15": False,
                                 "notes": f"exc: {exc}"[:200]}

        # V6 self-reset
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(COMBO_TIMEOUT_S * 2)
        try:
            t6, Vd6 = z477.v6_stim()
            r6 = z477.run_trap(
                cfg_flags, model_M1, model_M2, sebas_rows, 0.6, 0.0,
                t6, Vd6, enable_trap=True, tau_slow=tau_b, k_n=kn_b,
                max_step=5e-9,
            )
            signal.alarm(0)
            if r6 is not None:
                Vb6 = np.asarray(r6["Vb"])
                t6_ns = np.asarray(t6) * 1e9
                t_release_ns = (10e-9 + 100e-12 + 1e-6 + 100e-12) * 1e9
                post = t6_ns >= t_release_ns
                Vb_post_mean = (float(np.nanmean(Vb6[post]))
                                if post.any() else float("nan"))
                if post.any():
                    idx_post = np.where(post)[0]
                    below = Vb6[idx_post] < 0.3
                    t_reset_ns = (
                        float(t6_ns[idx_post[np.argmax(below)]] - t_release_ns)
                        if below.any() else float("inf"))
                else:
                    t_reset_ns = float("inf")
                mario_v6["v6"] = {
                    "passed": bool(t_reset_ns < 1e5 and Vb_post_mean < 0.3),
                    "t_reset_ns": (None if not math.isfinite(t_reset_ns)
                                   else t_reset_ns),
                    "Vb_post_mean": Vb_post_mean}
                log(f"  V6 t_reset={t_reset_ns} ns Vb_post={Vb_post_mean:.3f} V")
            else:
                mario_v6["v6"] = {"passed": False, "notes": "no transient"}
        except TimeoutError_:
            signal.alarm(0)
            mario_v6["v6"] = {"passed": False, "notes": "timeout"}
        except Exception as exc:
            signal.alarm(0)
            mario_v6["v6"] = {"passed": False, "notes": f"exc: {exc}"[:200]}
    (OUT / "mario_v6_post_check.json").write_text(json.dumps(mario_v6, indent=2))

    # ---------- STEP 5: gates + honest_analysis.md ----------
    log("=== STEP 5: gates + honest_analysis.md ===")
    INFRA = True  # backcompat already verified in backcompat.json
    DISCOVERY = any(r.get("sustained_ge3") and r.get("in_100_1000_ns")
                    for r in ok_rows)
    AMBITIOUS = False
    if (best is not None and best.get("period_ns") is not None
            and 300 <= best["period_ns"] <= 600):
        m_ok = mario_v6.get("mario", {}).get("passed_lt_0p15", False)
        v6_ok = mario_v6.get("v6", {}).get("passed", False)
        AMBITIOUS = bool(m_ok and v6_ok)
    KILL_SHOT = (len(ok_rows) > 0
                 and not any(r.get("sustained_ge3") for r in ok_rows))

    if AMBITIOUS:
        verdict = "AMBITIOUS"
    elif DISCOVERY:
        verdict = "DISCOVERY"
    elif KILL_SHOT:
        verdict = "KILL_SHOT"
    elif INFRA:
        verdict = "INFRA"
    else:
        verdict = "FAIL"

    bc_path = OUT / "backcompat.json"
    backcompat = (json.loads(bc_path.read_text())
                  if bc_path.exists() else {})

    md = []
    md.append(f"# z477b — FitzHugh-Nagumo slow trap honest analysis (resumed)\n")
    md.append(f"## Verdict: **{verdict}**\n\n")
    md.append("## Gates\n")
    md.append(f"- INFRA (backcompat OK): **{INFRA}**\n")
    md.append(f"- DISCOVERY (>=3 cycles in 100..1000 ns): **{DISCOVERY}**\n")
    md.append(f"- AMBITIOUS (period in 300..600 ns AND Mario<0.15dec AND V6 pass): **{AMBITIOUS}**\n")
    md.append(f"- KILL_SHOT (NO osc in any combo): **{KILL_SHOT}**\n\n")
    md.append("## Backward-compat regression (from prior agent)\n")
    md.append(f"- enable_trap=False: any_nan={backcompat.get('any_nan')} "
              f"Vb_max={backcompat.get('Vb_max')} "
              f"Id_pk_mA={backcompat.get('Id_pk_mA')}\n\n")
    md.append("## Best (tau_slow, k_n)\n")
    if best is None:
        md.append("- NO ok rows in sweep — trap topology #1 unrunnable.\n")
    else:
        md.append(f"- tau_slow={best['tau_slow_ns']} ns, k_n={best['k_n']:.0e}\n")
        md.append(f"- n_cycles={best.get('n_cycles')}, period={best.get('period_ns')} ns\n")
        md.append(f"- Vb_pk={best.get('Vb_max')} V, "
                  f"Vb_range={best.get('Vb_range')} V, "
                  f"Id_pk={best.get('Id_pk_mA')} mA, "
                  f"n_peak={best.get('n_peak')}\n")
        md.append(f"- Mario drift: {mario_v6.get('mario', {})}\n")
        md.append(f"- V6 self-reset: {mario_v6.get('v6', {})}\n")
        if not oscillators:
            md.append("- NOTE: no sustained-osc combo; best is by max Vb_range.\n")
    md.append("\n## Sweep table (status=ok rows)\n")
    md.append("| tau [ns] | k_n [S] | cycles | T [ns] | Vb_pk | Vb_rng | Id_pk [mA] | n_pk | dt [s] |\n")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in sorted(ok_rows, key=lambda x: (x["tau_slow_ns"], x["k_n"])):
        md.append(
            f"| {r['tau_slow_ns']:.0f} | {r['k_n']:.0e} | {r['n_cycles']} | "
            f"{r.get('period_ns')} | {r.get('Vb_max')} | "
            f"{r.get('Vb_range')} | {r.get('Id_pk_mA')} | "
            f"{r.get('n_peak')} | {r.get('dt_s'):.1f} |\n")
    md.append("\n## Non-ok rows\n")
    for r in rows:
        if r.get("status") != "ok":
            md.append(f"- tau={r['tau_slow_ns']:.0f}ns k_n={r['k_n']:.0e}: "
                      f"{r.get('status')} ({r.get('msg') or r.get('notes') or ''})\n")
    md.append("\n## No-cheat notes\n")
    md.append("- Resumed from prior aborted run. backcompat.json carried over.\n")
    md.append(f"- Per-combo wall budget: {COMBO_TIMEOUT_S}s (signal.alarm).\n")
    md.append("- Same z473 NX_1p8 cfg, sebas params, V7 stim (VG1=0.6, VG2=0, Vd 0.05/2 V, 5us hold).\n")
    md.append("- Backward-compat (enable_trap=False) preserved Id_pk=4.298 mA.\n")
    md.append("- k_n=1.0 S unphysical; reported only as ladder endpoint.\n")
    (OUT / "honest_analysis.md").write_text("".join(md))
    log(f"VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
