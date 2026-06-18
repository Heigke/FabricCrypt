#!/usr/bin/env python3
"""Track ALPHA — ALPHA0 10× fix ablation on canonical baseline.

Reuses build_pyport_base() + run_grid() + summarize() from
scripts/pillar_I_C3_jts_tat.py. Only ALPHA0 override is changed
(BASE = CSV 7.842e-5, FIX = card 7.83756e-4, plus sweep). All other
params at v5.3 baseline (Bf=100, η≤1, JTS default OFF).

Outputs:
  results/track_ALPHA_alpha0_fix/ablation.json
  results/track_ALPHA_alpha0_fix/verdict.md
"""
from __future__ import annotations
import os, sys, json, math, time
from pathlib import Path
import numpy as np

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

import importlib.util
sp = importlib.util.spec_from_file_location("pillar_I", ROOT / "scripts/pillar_I_C3_jts_tat.py")
pillar = importlib.util.module_from_spec(sp); sp.loader.exec_module(pillar)

OUT = ROOT / "results/track_ALPHA_alpha0_fix"
OUT.mkdir(parents=True, exist_ok=True)

CSV_VALUE   = 7.842e-5     # current default (Sebas CSV)
CARD_VALUE  = 7.83756e-4   # Mario LALPHA0_FIX card value (10× higher)
SWEEP = [1e-5, 7.842e-5, 2.5e-4, 7.83756e-4, 2.5e-3, 1e-3]
# sweep includes CSV (CSV_VALUE), CARD (CARD_VALUE), and decade-spaced explorations

# Per-VG1 worst-bias summary helper
def worst_subset_metrics(rows, vg1_target=0.6):
    sub = [r for r in rows if abs(r["VG1"] - vg1_target) < 1e-6 and np.isfinite(r["med_dec"])]
    if not sub:
        return {"n": 0, "median_dec": float("nan"), "max_dec": float("nan"),
                "Imeas_over_Ipred_med": float("nan")}
    decs = np.array([r["med_dec"] for r in sub])
    ratios = []
    for r in sub:
        if r["Ipred_peak"] > 0 and np.isfinite(r["Ipred_peak"]):
            ratios.append(r["Imeas_peak"] / max(r["Ipred_peak"], 1e-30))
    ratios = np.array(ratios) if ratios else np.array([np.nan])
    return {
        "n": len(sub),
        "median_dec": float(np.median(decs)),
        "max_dec": float(np.max(decs)),
        "Imeas_over_Ipred_med": float(np.nanmedian(ratios)),
        "Imeas_over_Ipred_max": float(np.nanmax(ratios)),
    }


def run_one(label: str, alpha0_value: float, curves, sebas_rows, time_budget_per=120):
    """Run grid with ALPHA0 forced to value. Returns summary dict."""
    print(f"[track_ALPHA] === {label}  ALPHA0={alpha0_value:.4e} ===", flush=True)
    cfg, M1, M2, bjt = pillar.build_pyport_base()

    # Patch make_overrides so it forces ALPHA0 to alpha0_value for every bias.
    orig_make = pillar.make_overrides
    def patched_make(sebas_row):
        P_M1, P_M2 = orig_make(sebas_row)
        if P_M1 is None: P_M1 = {}
        P_M1["alpha0"] = float(alpha0_value)
        if P_M2 is None: P_M2 = {}
        P_M2["alpha0"] = float(alpha0_value)
        return P_M1, P_M2
    pillar.make_overrides = patched_make
    try:
        t0 = time.time()
        rows, nan_count = pillar.run_grid(cfg, M1, M2, bjt, curves, sebas_rows, label, do_bwd=True)
        dt = time.time() - t0
    finally:
        pillar.make_overrides = orig_make

    summ = pillar.summarize(rows, label)
    summ["alpha0"] = float(alpha0_value)
    summ["nan_count"] = int(nan_count)
    summ["runtime_s"] = float(dt)
    finite = sum(1 for r in rows if np.isfinite(r["med_dec"]) and r["med_dec"] > 0)
    summ["n_rows"] = len(rows)
    summ["n_finite"] = finite
    summ["convergence_rate"] = finite / max(len(rows), 1)
    summ["worst_VG1=0.6"] = worst_subset_metrics(rows, 0.6)
    summ["worst_VG1=0.4"] = worst_subset_metrics(rows, 0.4)
    summ["worst_VG1=0.2"] = worst_subset_metrics(rows, 0.2)
    # Carry the per-curve rows for later diagnostics
    summ["_rows"] = [
        {k: (v if isinstance(v, (str, int, float, bool, type(None))) else float(v))
         for k, v in r.items()}
        for r in rows
    ]
    return summ


def main():
    sebas_rows = pillar.load_sebas_params()
    curves = pillar.load_curves()
    print(f"[track_ALPHA] loaded {len(curves)} curves, {len(sebas_rows)} sebas rows", flush=True)

    results = {}
    # Run baseline (CSV value) explicitly for our own metric so it matches the rest
    for v in SWEEP:
        tag = f"ALPHA0={v:.4e}"
        try:
            results[tag] = run_one(tag, v, curves, sebas_rows)
        except Exception as e:
            import traceback
            print(f"[track_ALPHA] FAIL {tag}: {e}", flush=True)
            traceback.print_exc()
            results[tag] = {"label": tag, "alpha0": v, "error": str(e)}
        # Persist incrementally
        with open(OUT / "ablation.json", "w") as f:
            # strip _rows from persisted (too large) but keep slim summary
            slim = {}
            for k, summ in results.items():
                slim[k] = {kk: vv for kk, vv in summ.items() if kk != "_rows"}
            json.dump(slim, f, indent=2, default=str)

    # Build verdict.md
    lines = []
    lines.append("# Track ALPHA — ALPHA0 10× Fix Ablation (canonical baseline, 33-bias fwd+bwd)\n")
    lines.append(f"Canonical baseline (CSV ALPHA0=7.842e-5): median_dec target reference = 1.163 dec (n=66)\n")
    lines.append(f"Mario LALPHA0_FIX card value: ALPHA0=7.83756e-4 (10× larger)\n")
    lines.append("\n## Sweep table\n")
    lines.append("| ALPHA0 | median_dec (all) | CI95 | conv | VG1=0.2 | VG1=0.4 | VG1=0.6 | VG1=0.6 Imeas/Ipred (med) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, s in results.items():
        if "error" in s:
            lines.append(f"| {s['alpha0']:.4e} | ERROR: {s['error'][:40]} |")
            continue
        m = s["median_dec_all"]
        lines.append(
            f"| {s['alpha0']:.4e} | {m['median']:.3f} | [{m['ci95_lo']:.3f}, {m['ci95_hi']:.3f}] | "
            f"{s['convergence_rate']:.2f} | "
            f"{s['median_dec_VG1=0.2']['median']:.3f} | "
            f"{s['median_dec_VG1=0.4']['median']:.3f} | "
            f"{s['median_dec_VG1=0.6']['median']:.3f} | "
            f"{s['worst_VG1=0.6']['Imeas_over_Ipred_med']:.2e} |"
        )

    # Compare CSV (baseline) vs CARD (fix)
    base = None; fix = None
    for k, s in results.items():
        if "error" in s: continue
        if abs(s["alpha0"] - CSV_VALUE) / CSV_VALUE < 0.01: base = s
        if abs(s["alpha0"] - CARD_VALUE) / CARD_VALUE < 0.01: fix = s

    lines.append("\n## CSV (baseline) → CARD (fix) delta\n")
    if base and fix:
        d_all = fix["median_dec_all"]["median"] - base["median_dec_all"]["median"]
        d_06  = fix["median_dec_VG1=0.6"]["median"] - base["median_dec_VG1=0.6"]["median"]
        r_base = base["worst_VG1=0.6"]["Imeas_over_Ipred_med"]
        r_fix  = fix["worst_VG1=0.6"]["Imeas_over_Ipred_med"]
        lines.append(f"- Δmedian_dec (all 33 biases, fwd+bwd) = {d_all:+.3f} dec")
        lines.append(f"- Δmedian_dec (VG1=0.6 subset)         = {d_06:+.3f} dec")
        lines.append(f"- Imeas/Ipred at VG1=0.6 (median): baseline {r_base:.2e} → fix {r_fix:.2e}")
        # Verdict band
        if d_all <= -0.5:
            verdict = "PASS — Δ ≤ −0.5 dec, very close to 0.5 dec target"
        elif d_all <= -0.2:
            verdict = "STRONG — Δ ∈ (−0.5, −0.2], combine with other fixes"
        elif d_all <= 0:
            verdict = "WEAK — Δ ∈ (−0.2, 0], one of multiple needed fixes"
        else:
            verdict = "KILLSHOT/REGRESSION — Δ > 0, ALPHA0 fix alone makes it worse"
        lines.append(f"\n**Verdict: {verdict}**\n")

        # The smoking gun question
        gap_closed = r_base / max(r_fix, 1e-30)
        lines.append(f"\n## Did the 46× shortfall at VG1=0.6 close?")
        lines.append(f"- Baseline (CSV) Imeas/Ipred median at VG1=0.6 = **{r_base:.2e}×**")
        lines.append(f"- Fix (10× ALPHA0)  Imeas/Ipred median at VG1=0.6 = **{r_fix:.2e}×**")
        lines.append(f"- Gap closure factor = {gap_closed:.2f}× (>>1 = closed)")
        if r_fix > 5:
            lines.append("- **KILLSHOT NOTE**: Even at the 10× ALPHA0 fix the VG1=0.6 saturation regime still under-predicts by orders of magnitude. ALPHA0 alone CANNOT close the 46× shortfall. Consistent with A1m_alpha0_scale_test.md verdict (case (ii): missing body-charging path). Need additional body-injection mechanism.")
    else:
        lines.append("(baseline or fix run missing — see errors above)")

    # Best ALPHA0 in sweep
    best_k, best_med = None, float("inf")
    for k, s in results.items():
        if "error" in s: continue
        m = s["median_dec_all"]["median"]
        if np.isfinite(m) and m < best_med:
            best_med = m; best_k = k
    if best_k:
        lines.append(f"\n## Best ALPHA0 in sweep\n- **{best_k}** → median_dec = {best_med:.3f}")

    # PWL note
    lines.append("\n## PWL(V_G) impact-ionization — documented as future work")
    lines.append("- Sebas's 2Tcell_BSIM_param_DC.csv has ALPHA0 = 7.842e-5 CONSTANT across all 33 (VG1, VG2) rows (verified S5C2_zoom_deep_findings_2026-05-15.md, line 140).")
    lines.append("- Mario slide 21 is about transient oscillation, NOT impact-ionization PWL.")
    lines.append("- BSIM4 §6.1 standard ALPHA0 is a scalar; PWL(V_G) is NOT supported by the foundry card data.")
    lines.append("- Therefore PWL implementation is out of scope for this track; deferred.")

    lines.append("\n## Provenance")
    lines.append("- Baseline builder: `scripts/pillar_I_C3_jts_tat.py::build_pyport_base()` (Bf=100, η≤1, JTS default)")
    lines.append("- Source of CSV value: `data/sebas_2026_04_22/2Tcell_BSIM_param_DC.csv` (33 biases, ALPHA0 constant 7.842e-5)")
    lines.append("- Source of CARD value: `data/sebas_2026_04_22/M1_130DNWFB_LALPHA0_FIX.txt` line: `alpha0 = 7.83756e-4`")
    lines.append("- Prior art: `research_plan/artifacts/A1m_alpha0_scale_test.md` (single-bias 4-decade sweep, falsifying at WORST bias)")

    (OUT / "verdict.md").write_text("\n".join(lines) + "\n")
    print(f"[track_ALPHA] wrote {OUT / 'verdict.md'}", flush=True)
    print(f"[track_ALPHA] wrote {OUT / 'ablation.json'}", flush=True)


if __name__ == "__main__":
    main()
