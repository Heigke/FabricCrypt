"""z477c — Finsweep around z477b's Hopf point; physical V_b clamp test.

Sweep: tau_slow in {800, 1000, 1200} ns x k_n in {3e-5, 1e-4, 3e-4} (9 combos).
Stim: V7 (VG1=0.6, VG2=0, Vd 0.05->2 V, 5us hold).

Two passes per combo:
    A) UNCLAMPED (Vb_min=-50, Vb_max=+50)  - matches z477b context
    B) PHYSICAL CLAMP (hard) Vb in [-0.5, +1.2]  - hard saturating clamp on dVb

Outputs to results/z477c_finsweep/:
    finsweep_grid.json
    physical_clamp_compare.json
    oscillation_map_phys.png
    honest_verdict.md
"""
from __future__ import annotations
import json, math, sys, time, signal
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
z477 = _load("z477", ROOT / "scripts/z477_fhn_trap.py")

from nsram.bsim4_port import transient_real_v2 as trv2
from nsram.bsim4_port.transient_real_v2 import integrate, TransientCfgV2

OUT = ROOT / "results/z477c_finsweep"
OUT.mkdir(parents=True, exist_ok=True)


# ---- hard-clamp wrapper for dVb ----
_HARD_CLAMP = {"on": False, "lo": -0.5, "hi": 1.2}
_orig_build_rhs = trv2._build_rhs

def _build_rhs_hardclamp(cfg, model_M1, model_M2, bjt, VG1_f, VG2_f,
                         Vd_of_t, tcfg, T_K, Vsint_state):
    inner = _orig_build_rhs(cfg, model_M1, model_M2, bjt, VG1_f, VG2_f,
                            Vd_of_t, tcfg, T_K, Vsint_state)
    if not _HARD_CLAMP["on"]:
        return inner
    lo = _HARD_CLAMP["lo"]; hi = _HARD_CLAMP["hi"]
    def rhs(t, x):
        dx = inner(t, x)
        Vb = float(x[0])
        dVb = float(dx[0])
        # Hard saturating clamp: when at/past boundary AND pushing further out,
        # zero dVb. Otherwise pass through.
        if Vb <= lo and dVb < 0.0:
            dx[0] = 0.0
        elif Vb >= hi and dVb > 0.0:
            dx[0] = 0.0
        return dx
    return rhs


def measure_osc(t_arr, Vb):
    t_ns = np.asarray(t_arr) * 1e9
    Vb = np.asarray(Vb)
    crossings = []
    for i in range(1, len(Vb)):
        if (np.isfinite(Vb[i]) and np.isfinite(Vb[i-1])
                and Vb[i-1] < 0.5 <= Vb[i]):
            crossings.append(float(t_ns[i]))
    n_cycles = max(0, len(crossings) - 1)
    period_ns = float(np.mean(np.diff(crossings))) if len(crossings) >= 2 else float("nan")
    return n_cycles, period_ns, crossings


def run_one(cfg_flags, model_M1, model_M2, sebas_rows, t_arr, Vd_arr,
            tau, kn, hard_clamp=False, wall_budget_s=120):
    _HARD_CLAMP["on"] = hard_clamp
    _HARD_CLAMP["lo"] = -0.5
    _HARD_CLAMP["hi"] = +1.2
    # Patch
    trv2._build_rhs = _build_rhs_hardclamp
    # Loose soft Vb clamp on tcfg so it doesn't interfere when hard clamp off;
    # when hard clamp on, set soft clamp to same bounds to avoid weird currents.
    if hard_clamp:
        vmin, vmax = -0.5, +1.2
    else:
        vmin, vmax = -50.0, +50.0
    # monkey-patch run_trap to inject Vb_min/Vb_max in TransientCfgV2
    # We replicate z477.run_trap inline to control TransientCfgV2 fields.
    sebas_row = z427.find_params(sebas_rows, 0.6, 0.0)
    if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
        return None
    P_M1, P_M2 = z427.make_overrides(sebas_row)
    bjt = z427.make_bjt(sebas_row)
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(cfg_flags))
    cfg.Cbody = 1e-15
    tcfg = TransientCfgV2(
        C_B_const=1e-15, atol=1e-12, rtol=1e-7,
        max_step=20e-9, first_step=1e-14,
        R_body=1e7, body_leak_kind="linear",
        V_th_leak=0.4, G_leak=1e-3,
        enable_trap=True,
        tau_slow=tau, k_n=kn, V_n0=0.5, alpha_n=1.0,
        Vb_min=vmin, Vb_max=vmax,
    )
    z449._VBIC_CTX["cfg"] = cfg
    z449._VBIC_CTX["bjt"] = bjt

    class _TO(Exception): pass
    def _h(signum, frame): raise _TO()
    old_h = signal.signal(signal.SIGALRM, _h)
    signal.alarm(int(wall_budget_s))
    try:
        with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), \
             z427.patch_sd_scaled(sd_M2, P_M2):
            out = integrate(cfg, model_M1, model_M2, bjt,
                            np.asarray(t_arr), np.asarray(Vd_arr),
                            0.6, 0.0, tcfg=tcfg, Vb0=0.0)
        status = "ok"
    except _TO:
        out = None; status = "timeout"
    except Exception as e:
        out = None; status = f"exception:{type(e).__name__}:{e}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_h)
        z449._VBIC_CTX["cfg"] = None
        z449._VBIC_CTX["bjt"] = None
        trv2._build_rhs = _orig_build_rhs  # restore
        _HARD_CLAMP["on"] = False
    return out, status


def summarize(out, t_arr):
    if out is None:
        return {"status": "fail"}
    Vb = np.asarray(out["Vb"]); Id = np.asarray(out["Id"])
    n_arr = np.asarray(out["n_trap"]) if out.get("n_trap") is not None else np.array([np.nan])
    n_cyc, T_ns, _ = measure_osc(t_arr, Vb)
    return {
        "n_cycles": int(n_cyc),
        "period_ns": (None if not math.isfinite(T_ns) else float(T_ns)),
        "Vb_max": float(np.nanmax(Vb)) if np.isfinite(Vb).any() else None,
        "Vb_min": float(np.nanmin(Vb)) if np.isfinite(Vb).any() else None,
        "Vb_range": (float(np.nanmax(Vb) - np.nanmin(Vb))
                     if np.isfinite(Vb).any() else None),
        "Id_pk_mA": (float(np.nanmax(np.abs(Id)) * 1e3)
                     if np.isfinite(Id).any() else None),
        "n_peak": (float(np.nanmax(np.abs(n_arr)))
                   if np.isfinite(n_arr).any() else None),
        "any_nan": bool(np.isnan(Vb).any()),
    }


def classify(period_ns, Vb_min, Vb_max):
    phys = (Vb_min is not None and Vb_max is not None
            and Vb_min >= -0.5 - 1e-9 and Vb_max <= +1.2 + 1e-9)
    in_band = (period_ns is not None and 300.0 <= period_ns <= 600.0)
    in_100_1000 = (period_ns is not None and 100.0 <= period_ns <= 1000.0)
    return {"physical_Vb": bool(phys), "period_300_600": bool(in_band),
            "period_100_1000": bool(in_100_1000)}


def main():
    log = lambda m: print(m, flush=True)
    log("z477c — finsweep around Hopf point")
    cfg_flags = z473.make_NX_1p8()
    log("loading models / sebas")
    model_M1, model_M2 = z427.build_models()
    sebas_rows = z427.load_sebas_params()

    tau_vals = [800e-9, 1000e-9, 1200e-9]
    kn_vals = [3e-5, 1e-4, 3e-4]

    t_arr, Vd_arr = z477.v7_stim()
    log(f"V7 stim: N={len(t_arr)}, t_max={t_arr[-1]*1e9:.1f} ns")

    rows = []
    written = 0
    for tau in tau_vals:
        for kn in kn_vals:
            t0 = time.time()
            log(f"-- UNCLAMP tau={tau*1e9:.0f}ns k_n={kn:.0e}")
            out, st = run_one(cfg_flags, model_M1, model_M2, sebas_rows,
                              t_arr, Vd_arr, tau, kn,
                              hard_clamp=False, wall_budget_s=120)
            dt_a = time.time() - t0
            sumA = summarize(out, t_arr) if st == "ok" else {"status": st}
            sumA["dt_s"] = dt_a; sumA["status"] = st
            classA = classify(sumA.get("period_ns"), sumA.get("Vb_min"), sumA.get("Vb_max"))
            log(f"   A: status={st} cyc={sumA.get('n_cycles')} "
                f"T={sumA.get('period_ns')} Vb_rng=[{sumA.get('Vb_min')},{sumA.get('Vb_max')}] "
                f"Id_pk={sumA.get('Id_pk_mA')} dt={dt_a:.1f}s")

            t0 = time.time()
            log(f"-- CLAMP[-0.5,+1.2] tau={tau*1e9:.0f}ns k_n={kn:.0e}")
            out_c, st_c = run_one(cfg_flags, model_M1, model_M2, sebas_rows,
                                  t_arr, Vd_arr, tau, kn,
                                  hard_clamp=True, wall_budget_s=120)
            dt_b = time.time() - t0
            sumB = summarize(out_c, t_arr) if st_c == "ok" else {"status": st_c}
            sumB["dt_s"] = dt_b; sumB["status"] = st_c
            classB = classify(sumB.get("period_ns"), sumB.get("Vb_min"), sumB.get("Vb_max"))
            log(f"   B: status={st_c} cyc={sumB.get('n_cycles')} "
                f"T={sumB.get('period_ns')} Vb_rng=[{sumB.get('Vb_min')},{sumB.get('Vb_max')}] "
                f"Id_pk={sumB.get('Id_pk_mA')} dt={dt_b:.1f}s")

            rows.append({
                "tau_slow_ns": tau * 1e9, "k_n": kn,
                "unclamped": {**sumA, **classA},
                "clamped":   {**sumB, **classB},
            })
            written += 1
            # incremental save every 3 combos
            if written % 3 == 0:
                (OUT / "finsweep_grid.json").write_text(json.dumps(
                    {"rows": rows, "complete": False}, indent=2))
                log(f"   [incremental save: {written}/9]")

    # final write
    (OUT / "finsweep_grid.json").write_text(json.dumps(
        {"rows": rows, "complete": True,
         "tau_vals_ns": [t*1e9 for t in tau_vals],
         "k_n_vals": kn_vals,
         "clamp_bounds_V": [-0.5, 1.2]}, indent=2))

    # ---- physical_clamp_compare.json ----
    cmp_rows = []
    for r in rows:
        cmp_rows.append({
            "tau_ns": r["tau_slow_ns"], "k_n": r["k_n"],
            "unclamped_cycles": r["unclamped"].get("n_cycles"),
            "unclamped_period_ns": r["unclamped"].get("period_ns"),
            "unclamped_Vb_range": r["unclamped"].get("Vb_range"),
            "clamped_cycles": r["clamped"].get("n_cycles"),
            "clamped_period_ns": r["clamped"].get("period_ns"),
            "clamped_Vb_range": r["clamped"].get("Vb_range"),
            "osc_survives_clamp": bool(
                (r["clamped"].get("n_cycles") or 0) >= 3),
        })
    (OUT / "physical_clamp_compare.json").write_text(json.dumps(
        cmp_rows, indent=2))

    # ---- oscillation_map_phys.png ----
    n_tau = len(tau_vals); n_k = len(kn_vals)
    M_cyc_unc = np.zeros((n_tau, n_k))
    M_T_unc = np.full((n_tau, n_k), np.nan)
    M_cyc_cl = np.zeros((n_tau, n_k))
    M_T_cl = np.full((n_tau, n_k), np.nan)
    for r in rows:
        i = tau_vals.index(r["tau_slow_ns"] * 1e-9)
        j = kn_vals.index(r["k_n"])
        M_cyc_unc[i, j] = r["unclamped"].get("n_cycles") or 0
        if r["unclamped"].get("period_ns") is not None:
            M_T_unc[i, j] = r["unclamped"]["period_ns"]
        M_cyc_cl[i, j] = r["clamped"].get("n_cycles") or 0
        if r["clamped"].get("period_ns") is not None:
            M_T_cl[i, j] = r["clamped"]["period_ns"]
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    for ax, M, title, cmap in [
        (axs[0, 0], M_cyc_unc, "n_cycles (UNCLAMPED)", "viridis"),
        (axs[0, 1], M_T_unc,   "period [ns] (UNCLAMPED)", "plasma"),
        (axs[1, 0], M_cyc_cl,  "n_cycles (CLAMP [-0.5,+1.2] V)", "viridis"),
        (axs[1, 1], M_T_cl,    "period [ns] (CLAMP)", "plasma"),
    ]:
        im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap)
        ax.set_xticks(range(n_k)); ax.set_xticklabels([f"{k:.0e}" for k in kn_vals])
        ax.set_yticks(range(n_tau)); ax.set_yticklabels([f"{t*1e9:.0f}" for t in tau_vals])
        ax.set_xlabel("k_n [S]"); ax.set_ylabel("tau_slow [ns]")
        ax.set_title(title); plt.colorbar(im, ax=ax)
        for i in range(n_tau):
            for j in range(n_k):
                v = M[i, j]
                if np.isfinite(v):
                    txt = f"{int(v)}" if "cycles" in title else f"{v:.0f}"
                    ax.text(j, i, txt, ha="center", va="center",
                            color="white", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "oscillation_map_phys.png", dpi=120)
    plt.close(fig)

    # ---- gate check ----
    discovery_plus = []   # 1 combo physical + period 300-600
    ambitious = []        # >=3 combos physical + period 300-600
    any_phys_kill_pass = []  # >=1 combo physical Vb + period 100-1000
    for r in rows:
        c = r["clamped"]
        if c.get("physical_Vb") and c.get("period_300_600"):
            discovery_plus.append((r["tau_slow_ns"], r["k_n"], c.get("period_ns")))
        if c.get("physical_Vb") and c.get("period_100_1000"):
            any_phys_kill_pass.append((r["tau_slow_ns"], r["k_n"], c.get("period_ns")))
    ambitious = list(discovery_plus)

    verdict = "KILL"
    if len(any_phys_kill_pass) >= 1 and len(discovery_plus) == 0:
        verdict = "PHYSICAL_OSC_BUT_OUT_OF_BAND"
    if len(discovery_plus) >= 1:
        verdict = "DISCOVERY+"
    if len(ambitious) >= 3:
        verdict = "AMBITIOUS"
    if len(any_phys_kill_pass) == 0 and len(discovery_plus) == 0:
        verdict = "KILL"

    md = []
    md.append(f"# z477c — Finsweep honest verdict")
    md.append(f"## Verdict: **{verdict}**\n")
    md.append("## Gate evaluations")
    md.append(f"- DISCOVERY+ (>=1 combo, physical Vb in [-0.5,+1.2] AND period in 300-600 ns): "
              f"**{len(discovery_plus) >= 1}** (n={len(discovery_plus)})")
    md.append(f"- AMBITIOUS (>=3 combos physical + 300-600 ns): "
              f"**{len(ambitious) >= 3}** (n={len(ambitious)})")
    md.append(f"- KILL (no combo physical AND period 100-1000 ns): "
              f"**{len(any_phys_kill_pass) == 0}**\n")
    md.append("## Combos satisfying physical Vb + period in 300-600 ns")
    if discovery_plus:
        for tau, kn, T in discovery_plus:
            md.append(f"- tau={tau:.0f} ns, k_n={kn:.0e}, period={T:.1f} ns")
    else:
        md.append("- (none)")
    md.append("\n## Combos satisfying physical Vb + period in 100-1000 ns")
    if any_phys_kill_pass:
        for tau, kn, T in any_phys_kill_pass:
            md.append(f"- tau={tau:.0f} ns, k_n={kn:.0e}, period={T:.1f} ns")
    else:
        md.append("- (none)")

    md.append("\n## Full grid (CLAMPED [-0.5,+1.2] V)")
    md.append("| tau [ns] | k_n | cyc | T [ns] | Vb_rng | Vb_min | Vb_max | Id_pk [mA] | status |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        c = r["clamped"]
        md.append(f"| {r['tau_slow_ns']:.0f} | {r['k_n']:.0e} | "
                  f"{c.get('n_cycles')} | {c.get('period_ns')} | "
                  f"{c.get('Vb_range')} | {c.get('Vb_min')} | {c.get('Vb_max')} | "
                  f"{c.get('Id_pk_mA')} | {c.get('status')} |")
    md.append("\n## Full grid (UNCLAMPED, wide Vb)")
    md.append("| tau [ns] | k_n | cyc | T [ns] | Vb_rng | Vb_min | Vb_max | Id_pk [mA] | status |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        a = r["unclamped"]
        md.append(f"| {r['tau_slow_ns']:.0f} | {r['k_n']:.0e} | "
                  f"{a.get('n_cycles')} | {a.get('period_ns')} | "
                  f"{a.get('Vb_range')} | {a.get('Vb_min')} | {a.get('Vb_max')} | "
                  f"{a.get('Id_pk_mA')} | {a.get('status')} |")

    md.append("\n## No-cheat notes")
    md.append("- Hard clamp implemented as: when state Vb at/past boundary AND dVb pushing outward, zero dVb (saturating).")
    md.append("- TransientCfgV2.Vb_min/Vb_max also tightened to clamp bounds so currents are evaluated at physical Vb.")
    md.append("- If oscillation persists under clamp -> mechanism is real within Vdd range.")
    md.append("- If clamp kills oscillation entirely -> original 7-cycle z477b result was BSIM4 extrapolation artifact (Vb ranged -2 V to +36 V).")
    md.append("- Per-combo wall budget: 120 s (signal.alarm).")
    md.append("- V7 stim: VG1=0.6, VG2=0.0, Vd 0.05->2.0 V, hold 5 us.")
    (OUT / "honest_verdict.md").write_text("\n".join(md))
    log(f"\nVERDICT: {verdict}")
    log(f"  discovery+ combos: {len(discovery_plus)}")
    log(f"  ambitious combos: {len(ambitious)}")
    log(f"  any-phys-osc combos: {len(any_phys_kill_pass)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
