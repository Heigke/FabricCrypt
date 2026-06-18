#!/usr/bin/env python3
"""Track COMBO — K1@VG1=0.6 + ALPHA0 combined ablation on canonical baseline.

Combines two prior single-variable fixes:
  - K1@VG1=0.6 ∈ {0.41825 (hand-tuned baseline), 0.53825 (BSIM card), 0.6459 (×1.2)}
  - ALPHA0   ∈ {7.842e-5 (CSV baseline), 7.83756e-4 (Mario LALPHA0_FIX card, 10×)}

Both override pillar.BRANCH_FLAT[0.6]["K1"] (used when CSV row K1 is NaN — but
also Sebas CSV's K1 at VG1=0.6 IS 0.41825 already, so we also force K1 in
make_overrides like ALPHA0). Sweep = 3 × 2 = 6 conditions, full 33 biases fwd+bwd.

Outputs:
  results/track_combo_k1_alpha0/ablation.json
  results/track_combo_k1_alpha0/verdict.md
"""
from __future__ import annotations
import os, sys, json, time, traceback
from pathlib import Path
import numpy as np

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

import importlib.util
sp = importlib.util.spec_from_file_location("pillar_I", ROOT / "scripts/pillar_I_C3_jts_tat.py")
pillar = importlib.util.module_from_spec(sp); sp.loader.exec_module(pillar)

OUT = ROOT / "results/track_combo_k1_alpha0"
OUT.mkdir(parents=True, exist_ok=True)

# Baselines (canonical)
K1_BASELINE   = 0.41825   # current hand-tuned at VG1=0.6
ALPHA0_CSV    = 7.842e-5  # CSV baseline
# Card / fix values
K1_CARD       = 0.53825   # BSIM card (matches VG1=0.4)
K1_PUSH       = 0.6459    # 1.2× card
ALPHA0_CARD   = 7.83756e-4  # Mario LALPHA0_FIX card (10× CSV)

K1_GRID     = [K1_BASELINE, K1_CARD, K1_PUSH]
ALPHA0_GRID = [ALPHA0_CSV, ALPHA0_CARD]

BASELINE_MEDIAN_DEC = 1.163  # canonical reference: build_pyport_base() (n=66)


def worst_subset_metrics(rows, vg1_target):
    sub = [r for r in rows if abs(r["VG1"] - vg1_target) < 1e-6 and np.isfinite(r["med_dec"])]
    if not sub:
        return {"n": 0, "median_dec": float("nan"), "max_dec": float("nan"),
                "Imeas_over_Ipred_med": float("nan"), "Imeas_over_Ipred_max": float("nan")}
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


def run_one(label, k1_val, alpha0_val, curves, sebas_rows):
    print(f"[combo] === {label}  K1@0.6={k1_val:.5f}  ALPHA0={alpha0_val:.4e} ===", flush=True)
    cfg, M1, M2, bjt = pillar.build_pyport_base()

    # Patch 1: BRANCH_FLAT[0.6]["K1"] (covers NaN-imputed rows)
    saved_branch_k1 = pillar.BRANCH_FLAT[0.6]["K1"]
    pillar.BRANCH_FLAT[0.6]["K1"] = float(k1_val)

    # Patch 2: make_overrides — force k1 at VG1≈0.6 AND alpha0 everywhere
    orig_make = pillar.make_overrides
    def patched_make(sebas_row):
        P_M1, P_M2 = orig_make(sebas_row)
        if P_M1 is None: P_M1 = {}
        if P_M2 is None: P_M2 = {}
        P_M1["alpha0"] = float(alpha0_val)
        P_M2["alpha0"] = float(alpha0_val)
        if sebas_row is not None and abs(sebas_row.get("VG1", float("nan")) - 0.6) < 1e-6:
            P_M1["k1"] = float(k1_val)
        return P_M1, P_M2
    pillar.make_overrides = patched_make

    try:
        t0 = time.time()
        rows, nan_count = pillar.run_grid(cfg, M1, M2, bjt, curves, sebas_rows, label, do_bwd=True)
        dt = time.time() - t0
    finally:
        pillar.make_overrides = orig_make
        pillar.BRANCH_FLAT[0.6]["K1"] = saved_branch_k1

    summ = pillar.summarize(rows, label)
    summ["k1_vg1_0p6"] = float(k1_val)
    summ["alpha0"] = float(alpha0_val)
    summ["nan_count"] = int(nan_count)
    summ["runtime_s"] = float(dt)
    finite = sum(1 for r in rows if np.isfinite(r["med_dec"]) and r["med_dec"] > 0)
    summ["n_rows"] = len(rows)
    summ["n_finite"] = finite
    summ["convergence_rate"] = finite / max(len(rows), 1)
    summ["worst_VG1=0.2"] = worst_subset_metrics(rows, 0.2)
    summ["worst_VG1=0.4"] = worst_subset_metrics(rows, 0.4)
    summ["worst_VG1=0.6"] = worst_subset_metrics(rows, 0.6)
    return summ


def main():
    sebas_rows = pillar.load_sebas_params()
    curves = pillar.load_curves()
    print(f"[combo] loaded {len(curves)} curves, {len(sebas_rows)} sebas rows", flush=True)

    results = {}
    for k1 in K1_GRID:
        for a0 in ALPHA0_GRID:
            tag = f"K1={k1:.5f}__ALPHA0={a0:.4e}"
            try:
                results[tag] = run_one(tag, k1, a0, curves, sebas_rows)
            except Exception as e:
                print(f"[combo] FAIL {tag}: {e}", flush=True)
                traceback.print_exc()
                results[tag] = {"label": tag, "k1_vg1_0p6": k1, "alpha0": a0, "error": str(e)}
            # Persist incrementally
            with open(OUT / "ablation.json", "w") as f:
                json.dump(results, f, indent=2, default=str)

    # Build verdict.md
    lines = []
    lines.append("# Track COMBO — K1@VG1=0.6 × ALPHA0 Ablation (canonical baseline, 33-bias fwd+bwd)\n")
    lines.append(f"Canonical baseline: median_dec = {BASELINE_MEDIAN_DEC} dec (n=66)\n")
    lines.append(f"- K1 grid (@VG1=0.6): {K1_GRID}  (baseline=0.41825, card=0.53825, push=0.6459)")
    lines.append(f"- ALPHA0 grid: {ALPHA0_GRID}  (CSV baseline=7.842e-5, card=7.83756e-4)\n")

    lines.append("## Sweep table\n")
    lines.append("| K1@0.6 | ALPHA0 | median_dec (all) | CI95 | conv | VG1=0.2 | VG1=0.4 | VG1=0.6 | VG1=0.6 Imeas/Ipred (med) | Δ vs 1.163 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tag, s in results.items():
        if "error" in s:
            lines.append(f"| {s['k1_vg1_0p6']:.5f} | {s['alpha0']:.4e} | ERROR: {s['error'][:40]} | | | | | | | |")
            continue
        m = s["median_dec_all"]
        d = m["median"] - BASELINE_MEDIAN_DEC
        lines.append(
            f"| {s['k1_vg1_0p6']:.5f} | {s['alpha0']:.4e} | {m['median']:.3f} | "
            f"[{m['ci95_lo']:.3f}, {m['ci95_hi']:.3f}] | {s['convergence_rate']:.2f} | "
            f"{s['median_dec_VG1=0.2']['median']:.3f} | "
            f"{s['median_dec_VG1=0.4']['median']:.3f} | "
            f"{s['median_dec_VG1=0.6']['median']:.3f} | "
            f"{s['worst_VG1=0.6']['Imeas_over_Ipred_med']:.2e} | "
            f"{d:+.3f} |"
        )

    # Find best, compute deltas
    best_tag, best_med = None, float("inf")
    for tag, s in results.items():
        if "error" in s: continue
        m = s["median_dec_all"]["median"]
        if np.isfinite(m) and m < best_med:
            best_med = m; best_tag = tag

    # Locate canonical baseline cell (K1=0.41825, ALPHA0=7.842e-5)
    base_cell = None
    for tag, s in results.items():
        if "error" in s: continue
        if abs(s["k1_vg1_0p6"] - K1_BASELINE) < 1e-6 and abs(s["alpha0"] - ALPHA0_CSV)/ALPHA0_CSV < 0.01:
            base_cell = s

    # Single-fix anchors
    k1card_a0csv = None    # K1 card only
    k1base_a0card = None   # ALPHA0 card only
    k1card_a0card = None   # both card
    k1push_a0card = None   # K1 push + ALPHA0 card
    for tag, s in results.items():
        if "error" in s: continue
        if abs(s["k1_vg1_0p6"] - K1_CARD) < 1e-6 and abs(s["alpha0"] - ALPHA0_CSV)/ALPHA0_CSV < 0.01:
            k1card_a0csv = s
        if abs(s["k1_vg1_0p6"] - K1_BASELINE) < 1e-6 and abs(s["alpha0"] - ALPHA0_CARD)/ALPHA0_CARD < 0.01:
            k1base_a0card = s
        if abs(s["k1_vg1_0p6"] - K1_CARD) < 1e-6 and abs(s["alpha0"] - ALPHA0_CARD)/ALPHA0_CARD < 0.01:
            k1card_a0card = s
        if abs(s["k1_vg1_0p6"] - K1_PUSH) < 1e-6 and abs(s["alpha0"] - ALPHA0_CARD)/ALPHA0_CARD < 0.01:
            k1push_a0card = s

    lines.append("\n## Compounding check\n")
    def fmt(s):
        if s is None: return "MISSING"
        return f"{s['median_dec_all']['median']:.3f} (Δ={s['median_dec_all']['median']-BASELINE_MEDIAN_DEC:+.3f})"
    lines.append(f"- Baseline (K1=0.41825, ALPHA0=7.842e-5): {fmt(base_cell)}")
    lines.append(f"- K1=card only  (0.53825, ALPHA0=CSV)    : {fmt(k1card_a0csv)}")
    lines.append(f"- ALPHA0=card only (K1=baseline, 7.838e-4): {fmt(k1base_a0card)}")
    lines.append(f"- COMBO card+card (K1=0.53825 + 7.838e-4) : {fmt(k1card_a0card)}")
    lines.append(f"- COMBO push+card (K1=0.6459 + 7.838e-4)  : {fmt(k1push_a0card)}")

    # Compoundedness diagnostic
    if all(x is not None for x in (base_cell, k1card_a0csv, k1base_a0card, k1card_a0card)):
        d_k1   = k1card_a0csv ["median_dec_all"]["median"] - base_cell["median_dec_all"]["median"]
        d_a0   = k1base_a0card["median_dec_all"]["median"] - base_cell["median_dec_all"]["median"]
        d_both = k1card_a0card["median_dec_all"]["median"] - base_cell["median_dec_all"]["median"]
        expected_additive = d_k1 + d_a0
        synergy = d_both - expected_additive
        lines.append(f"\n- Δ(K1 only)    = {d_k1:+.3f}")
        lines.append(f"- Δ(ALPHA0 only)= {d_a0:+.3f}")
        lines.append(f"- Δ(both card)  = {d_both:+.3f}  (expected additive: {expected_additive:+.3f}, synergy: {synergy:+.3f})")
        if d_both < min(d_k1, d_a0):
            lines.append(f"- **Compound: YES** — combo beats each individual fix.")
        else:
            lines.append(f"- **Compound: NO** — combo did not beat best individual fix.")

    # Target check
    lines.append(f"\n## Best in grid\n- **{best_tag}** → median_dec = {best_med:.3f}  (Δ = {best_med - BASELINE_MEDIAN_DEC:+.3f})")
    if np.isfinite(best_med) and best_med <= 0.5:
        lines.append("- **TARGET REACHED**: median_dec ≤ 0.5 dec ✔")
    else:
        lines.append(f"- Target median_dec ≤ 0.5 NOT yet reached (gap = {best_med - 0.5:+.3f} dec).")

    # Conclusion paragraph
    lines.append("\n## Conclusion\n")
    if all(x is not None for x in (base_cell, k1card_a0csv, k1base_a0card, k1card_a0card)):
        d_k1   = k1card_a0csv ["median_dec_all"]["median"] - base_cell["median_dec_all"]["median"]
        d_a0   = k1base_a0card["median_dec_all"]["median"] - base_cell["median_dec_all"]["median"]
        d_both = k1card_a0card["median_dec_all"]["median"] - base_cell["median_dec_all"]["median"]
        compound_word = "compounded" if d_both < min(d_k1, d_a0) else "did NOT compound"
        target_word   = "reached" if best_med <= 0.5 else "not yet reached"
        lines.append(
            f"K1@0.6 card-revert and ALPHA0 10× card fixes {compound_word} on the canonical 33-bias DC fit "
            f"(Δ_K1={d_k1:+.3f}, Δ_ALPHA0={d_a0:+.3f}, Δ_both={d_both:+.3f} dec; "
            f"baseline {BASELINE_MEDIAN_DEC} → combo {k1card_a0card['median_dec_all']['median']:.3f}). "
            f"Target median_dec ≤ 0.5 dec is {target_word} (best in grid = {best_med:.3f} at {best_tag}). "
            f"VG1=0.6 Imeas/Ipred ratio at combo card+card = {k1card_a0card['worst_VG1=0.6']['Imeas_over_Ipred_med']:.2e}× "
            f"(baseline {base_cell['worst_VG1=0.6']['Imeas_over_Ipred_med']:.2e}×)."
        )
    else:
        lines.append("Conclusion: some required cells failed or were missing — see sweep table.")

    lines.append("\n## Provenance")
    lines.append("- Baseline builder: `scripts/pillar_I_C3_jts_tat.py::build_pyport_base()` (Bf=100, η≤1)")
    lines.append("- K1 fix locus: `BRANCH_FLAT[0.6]['K1']` and per-bias `make_overrides`")
    lines.append("- ALPHA0 fix locus: per-bias `make_overrides` (M1+M2)")
    lines.append("- Reference scripts: `track_ALPHA_alpha0_fix.py`, `track_triode_vg1_60_sweep.py`")

    (OUT / "verdict.md").write_text("\n".join(lines) + "\n")
    print(f"[combo] wrote {OUT / 'verdict.md'}", flush=True)
    print(f"[combo] wrote {OUT / 'ablation.json'}", flush=True)


if __name__ == "__main__":
    main()
