#!/usr/bin/env python3
"""03_compare_thermal.py — Phase 1b thermal-controlled identity verdict.

Reads signature_thermal.json from both ikaros and daedalus, for each regime
(cold/idle/warm) computes:
    - intra-HD per device (from raw_<regime>.npz bit votes)
    - inter-HD (ikaros vs daedalus stable signature bits) for that regime
    - KL(knee_slope hist), KL(RTN-rate hist)
    - spatial-corr MSE
    - PERF KL (noise control)

Tabulates per-regime divergence and emits a verdict markdown to
research_plan/IDENTITY_BENCHMARK_2026-05-30_PHASE1B.md.

Verdict logic (using KL(knee) as the strongest Phase-1 signal, value 6.54
in the unmatched-temp Phase 1 baseline):

  SILICON-CONFIRMED  : KL(knee) at matched temp >= 0.5 * baseline AND RTN
                        asymmetry sign persists at all 3 regimes
  THERMAL-DOMINATED  : KL(knee) at matched temp < 0.5 * baseline AND/OR
                        RTN sign flips across regimes
  MIXED              : one channel (knee OR rtn OR spatial_corr) survives,
                        others don't

Usage: python 03_compare_thermal.py
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

# reuse helpers from 02
import importlib.util
spec = importlib.util.spec_from_file_location("cmp02", HERE / "02_compare_signatures.py")
cmp02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp02)


# Phase-1 baseline numbers (unmatched temp), for reference
PHASE1_BASELINE = {
    "kl_knee": 6.5442,
    "kl_rtn": 25.1053,
    "spatial_corr_mse": 0.0923,
    "kl_perf": 0.1096,
    "rtn_a_mean": 0.0000,
    "rtn_b_mean": 0.1149,
    "spatial_corr_a": 0.0563,
    "spatial_corr_b": 0.3601,
    "knee_a_mean": 0.2018,
    "knee_b_mean": 0.0883,
    "inter_hd": 0.2953,
    "intra_hd_avg": 0.2698,
}


def kl_div_hist(p_counts, q_counts, eps=1e-9):
    p = np.asarray(p_counts, dtype=np.float64) + eps
    q = np.asarray(q_counts, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    n = min(len(p), len(q))
    return float(np.sum(p[:n] * np.log(p[:n] / q[:n])))


def hamming(a, b):
    a = np.asarray(a).astype(np.uint8).ravel()
    b = np.asarray(b).astype(np.uint8).ravel()
    n = min(a.size, b.size)
    if n == 0:
        return float("nan")
    return float(np.mean(a[:n] != b[:n]))


def load_thermal_sig(results_root: Path, name: str):
    sig_path = results_root / name / "signature_thermal.json"
    if not sig_path.exists():
        raise FileNotFoundError(sig_path)
    sig = json.load(open(sig_path))
    raws = {}
    for regime, reg_sig in sig["regimes"].items():
        if "raw_npz" not in reg_sig:
            continue
        npz = Path(reg_sig["raw_npz"])
        if not npz.exists():
            npz = results_root / name / npz.name
        if npz.exists():
            raws[regime] = np.load(npz)
    return sig, raws


def compare_regime(sig_a, raw_a, sig_b, raw_b, regime):
    reg_a = sig_a["regimes"][regime]
    reg_b = sig_b["regimes"][regime]
    out = {"regime": regime}
    # achieved temps
    out["temp_a_C"] = reg_a.get("temp_during_run_mean",
                                reg_a.get("clamp", {}).get("achieved_temp_C"))
    out["temp_b_C"] = reg_b.get("temp_during_run_mean",
                                reg_b.get("clamp", {}).get("achieved_temp_C"))
    out["temp_delta_C"] = (out["temp_a_C"] or 0) - (out["temp_b_C"] or 0)
    out["in_band_a"] = reg_a.get("clamp", {}).get("in_band", False)
    out["in_band_b"] = reg_b.get("clamp", {}).get("in_band", False)

    # intra-HD per device
    bv_a, S_a = cmp02.per_rep_bit_votes_from_raw(raw_a)
    bv_b, S_b = cmp02.per_rep_bit_votes_from_raw(raw_b)
    out["intra_hd_a"], _, _ = cmp02.intra_hd(bv_a, S_a)
    out["intra_hd_b"], _, _ = cmp02.intra_hd(bv_b, S_b)
    out["intra_hd_avg"] = 0.5 * (out["intra_hd_a"] + out["intra_hd_b"])

    # inter-HD on stable bits
    out["inter_hd"] = hamming(S_a, S_b)

    # process-stat channels
    knees_a = np.array(reg_a["process_stat"]["knee_slopes"], dtype=float)
    knees_b = np.array(reg_b["process_stat"]["knee_slopes"], dtype=float)
    knees_a = knees_a[np.isfinite(knees_a)]
    knees_b = knees_b[np.isfinite(knees_b)]
    rtn_a = np.array(reg_a["process_stat"]["rtn_transition_rate_per_cu"], dtype=float)
    rtn_b = np.array(reg_b["process_stat"]["rtn_transition_rate_per_cu"], dtype=float)
    sc_a = reg_a["process_stat"]["spatial_corr_mean_abs"]
    sc_b = reg_b["process_stat"]["spatial_corr_mean_abs"]

    if knees_a.size and knees_b.size:
        lo = min(knees_a.min(), knees_b.min())
        hi = max(knees_a.max(), knees_b.max())
        bins = np.linspace(lo, hi, 16)
        ha, _ = np.histogram(knees_a, bins=bins)
        hb, _ = np.histogram(knees_b, bins=bins)
        out["kl_knee"] = kl_div_hist(ha, hb)
    else:
        out["kl_knee"] = float("nan")
    out["knee_a_mean"] = float(knees_a.mean()) if knees_a.size else float("nan")
    out["knee_b_mean"] = float(knees_b.mean()) if knees_b.size else float("nan")

    bins = np.linspace(0, max(rtn_a.max(), rtn_b.max(), 1e-3), 16)
    ra, _ = np.histogram(rtn_a, bins=bins)
    rb, _ = np.histogram(rtn_b, bins=bins)
    out["kl_rtn"] = kl_div_hist(ra, rb)
    out["rtn_a_mean"] = float(rtn_a.mean())
    out["rtn_b_mean"] = float(rtn_b.mean())

    out["spatial_corr_a"] = float(sc_a)
    out["spatial_corr_b"] = float(sc_b)
    out["spatial_corr_mse"] = float((sc_a - sc_b) ** 2)

    # perf control
    perf_a = np.array(reg_a["noise_control"]["perf_hist"], dtype=float)
    perf_b = np.array(reg_b["noise_control"]["perf_hist"], dtype=float)
    out["kl_perf"] = kl_div_hist(perf_a, perf_b)

    return out


def verdict(regimes):
    """Return (verdict_label, justification_list)."""
    base = PHASE1_BASELINE
    just = []
    # KL(knee) preservation
    knee_ratios = []
    for r in regimes:
        if np.isfinite(r["kl_knee"]):
            knee_ratios.append(r["kl_knee"] / base["kl_knee"])
            just.append(f"KL(knee)[{r['regime']}]={r['kl_knee']:.3f} "
                        f"({r['kl_knee']/base['kl_knee']*100:.0f}% of Phase 1 "
                        f"baseline {base['kl_knee']:.3f})")
    knee_survives = all(kr >= 0.5 for kr in knee_ratios) if knee_ratios else False

    # RTN sign asymmetry: does sign(rtn_a - rtn_b) persist?
    rtn_signs = []
    for r in regimes:
        rtn_signs.append(np.sign(r["rtn_a_mean"] - r["rtn_b_mean"]))
        just.append(f"RTN[{r['regime']}] a={r['rtn_a_mean']:.4f} "
                    f"b={r['rtn_b_mean']:.4f} sign={int(rtn_signs[-1])}")
    rtn_persists = len(set(s for s in rtn_signs if s != 0)) == 1

    # spatial-corr difference
    sc_persists = all(r["spatial_corr_mse"] >= 0.3 * base["spatial_corr_mse"]
                      for r in regimes)
    for r in regimes:
        just.append(f"spatial-MSE[{r['regime']}]={r['spatial_corr_mse']:.4f}")

    survived = sum([knee_survives, rtn_persists, sc_persists])
    if knee_survives and rtn_persists:
        label = "SILICON-CONFIRMED"
    elif survived == 0:
        label = "THERMAL-DOMINATED"
    else:
        label = "MIXED"
    just.append(f"Channel survival: knee={knee_survives} rtn={rtn_persists} "
                f"spatial={sc_persists} -> {survived}/3")
    return label, just


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=str(
        REPO_ROOT / "results" / "IDENTITY_BENCHMARK_2026-05-30"))
    ap.add_argument("--report-out", default=str(
        REPO_ROOT / "research_plan" / "IDENTITY_BENCHMARK_2026-05-30_PHASE1B.md"))
    args = ap.parse_args()

    rroot = Path(args.results_root)
    sig_a, raws_a = load_thermal_sig(rroot, "ikaros")
    sig_b, raws_b = load_thermal_sig(rroot, "daedalus")
    regimes_present = sorted(set(raws_a.keys()) & set(raws_b.keys()))
    print(f"Regimes both devices have raw for: {regimes_present}")

    results = []
    for r in regimes_present:
        try:
            res = compare_regime(sig_a, raws_a[r], sig_b, raws_b[r], r)
            results.append(res)
            print(f"  {r}: temps {res['temp_a_C']:.1f}/{res['temp_b_C']:.1f}C "
                  f"KL(knee)={res['kl_knee']:.3f} inter_hd={res['inter_hd']:.3f}")
        except Exception as e:
            print(f"  {r} FAILED: {e}")

    label, just = verdict(results)
    print(f"\nVERDICT: {label}")
    for j in just:
        print(f"  - {j}")

    # write markdown
    lines = []
    lines.append("# Identity Benchmark — Phase 1B Verdict (Thermal-Controlled)")
    lines.append("")
    lines.append(f"Date: 2026-05-30 · Devices: ikaros vs daedalus")
    lines.append(f"Phase 1B run: ikaros {sig_a['timestamp']} · daedalus {sig_b['timestamp']}")
    lines.append("")
    lines.append(f"## Verdict: **{label}**")
    lines.append("")
    lines.append("## Achieved-temperature matrix")
    lines.append("")
    lines.append("| regime | ikaros (°C) | daedalus (°C) | Δ | ikaros in_band | daedalus in_band |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['regime']} | {r['temp_a_C']:.1f} | {r['temp_b_C']:.1f} | "
            f"{r['temp_delta_C']:+.1f} | {r['in_band_a']} | {r['in_band_b']} |")
    lines.append("")
    lines.append("## Per-regime divergence")
    lines.append("")
    lines.append("| regime | intra_a | intra_b | inter | KL(knee) | KL(RTN) | RTN_a | RTN_b | spatial_MSE | KL(perf) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['regime']} | {r['intra_hd_a']:.3f} | {r['intra_hd_b']:.3f} | "
            f"{r['inter_hd']:.3f} | {r['kl_knee']:.3f} | {r['kl_rtn']:.3f} | "
            f"{r['rtn_a_mean']:.4f} | {r['rtn_b_mean']:.4f} | "
            f"{r['spatial_corr_mse']:.4f} | {r['kl_perf']:.3f} |")
    lines.append("")
    lines.append("## Phase 1 baseline (unmatched temp, for reference)")
    lines.append("")
    lines.append(f"- KL(knee) = {PHASE1_BASELINE['kl_knee']:.3f}")
    lines.append(f"- KL(RTN) = {PHASE1_BASELINE['kl_rtn']:.3f}")
    lines.append(f"- spatial-corr MSE = {PHASE1_BASELINE['spatial_corr_mse']:.4f}")
    lines.append(f"- inter-HD = {PHASE1_BASELINE['inter_hd']:.3f} vs intra-HD = {PHASE1_BASELINE['intra_hd_avg']:.3f}")
    lines.append(f"- RTN ikaros={PHASE1_BASELINE['rtn_a_mean']:.4f} daedalus={PHASE1_BASELINE['rtn_b_mean']:.4f}")
    lines.append(f"- spatial-corr ikaros={PHASE1_BASELINE['spatial_corr_a']:.4f} daedalus={PHASE1_BASELINE['spatial_corr_b']:.4f}")
    lines.append("")
    lines.append("## Justification")
    lines.append("")
    for j in just:
        lines.append(f"- {j}")
    lines.append("")
    lines.append("## Decomposition")
    lines.append("")
    if label == "SILICON-CONFIRMED":
        lines.append("Signal magnitude is preserved at matched temperatures and ")
        lines.append("RTN-asymmetry sign persists across all measured regimes. The ")
        lines.append("Phase 1 divergence reflects silicon process variation, not ")
        lines.append("ambient temperature. **Phase 2 transplant matrix: GREENLIT.**")
    elif label == "THERMAL-DOMINATED":
        lines.append("Signal shrinks substantially at matched temperatures: the ")
        lines.append("Phase 1 divergence was driven by the ~15°C ambient difference, ")
        lines.append("not by silicon identity. **Phase 2 transplant matrix: NOT ")
        lines.append("greenlit as-is — re-design with explicit thermal controls ")
        lines.append("baked into every cross-device comparison; consider abandoning ")
        lines.append("the process-stat channel and switching to a temperature-")
        lines.append("invariant signature (e.g. cycle-rank stable bits).**")
    else:
        lines.append("Partial survival: some channels show silicon-driven divergence ")
        lines.append("at matched temperatures, others collapse. **Phase 2 transplant ")
        lines.append("matrix: CONDITIONAL — proceed using only the surviving ")
        lines.append("channels listed above; document the thermal-sensitive channels ")
        lines.append("as confounded.**")
    lines.append("")
    lines.append("## Raw data paths")
    lines.append("")
    for dev in ("ikaros", "daedalus"):
        for r in regimes_present:
            lines.append(f"- {dev}/{r}: `{rroot/dev/f'raw_{r}.npz'}`")
        lines.append(f"- {dev}/signature_thermal.json: `{rroot/dev/'signature_thermal.json'}`")

    Path(args.report_out).write_text("\n".join(lines))
    print(f"\nReport: {args.report_out}")


if __name__ == "__main__":
    main()
