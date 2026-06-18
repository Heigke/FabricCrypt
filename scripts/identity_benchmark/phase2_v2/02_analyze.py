"""Bootstrap CI + verdict for phase2_v2 transplant matrix."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "IDENTITY_BENCHMARK_2026-05-30" / "phase2_v2"
REPORT = ROOT / "research_plan" / "IDENTITY_BENCHMARK_2026-05-30_PHASE2_V2.md"


def boot_ci(x, n_boot=4000, ci=0.95):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(0)
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    return float(x.mean()), lo, hi


def analyze_task(rows, metric, lower_is_better=True):
    """Aggregate by (variant, train==eval). Return summary dict."""
    out = {}
    for variant in sorted(set(r["variant"] for r in rows)):
        sub = [r for r in rows if r["variant"] == variant]
        if variant == "SHUFFLE":
            vals = np.asarray([r[metric] for r in sub])
            m, lo, hi = boot_ci(vals)
            out[variant] = {"shuffle_mean": m, "ci": [lo, hi], "n": len(vals)}
            continue
        diag = np.asarray([r[metric] for r in sub if r["train"] == r["eval"]])
        off  = np.asarray([r[metric] for r in sub if r["train"] != r["eval"]])
        delta = off - diag[:len(off)] if len(diag) == len(off) else (off.mean() - diag.mean())
        # Paired delta across seeds: pair by index since rows alternate
        # Simpler: compute per-seed (off-mean - diag-mean) bootstrap on row-level
        m_d, lo_d, hi_d = boot_ci(diag)
        m_o, lo_o, hi_o = boot_ci(off)
        # Per-seed paired delta
        # group by seed
        per_seed = {}
        for r in sub:
            per_seed.setdefault(r["seed"], {"diag": [], "off": []})
            (per_seed[r["seed"]]["diag" if r["train"] == r["eval"] else "off"]
                .append(r[metric]))
        seed_deltas = []
        for s, d in per_seed.items():
            if d["diag"] and d["off"]:
                seed_deltas.append(np.mean(d["off"]) - np.mean(d["diag"]))
        seed_deltas = np.asarray(seed_deltas)
        m_dl, lo_dl, hi_dl = boot_ci(seed_deltas)
        out[variant] = {
            "diag_mean": m_d, "diag_ci": [lo_d, hi_d],
            "off_mean": m_o, "off_ci": [lo_o, hi_o],
            "delta_mean": m_dl, "delta_ci": [lo_dl, hi_dl],
            "delta_std": float(seed_deltas.std()),
            "n": len(sub), "n_seeds": len(seed_deltas),
        }
    return out


def main():
    data = json.loads((OUT / "matrix.json").read_text())
    n10 = analyze_task(data["narma10"], "nrmse_test", lower_is_better=True)
    pmn = analyze_task(data["pmnist"], "acc", lower_is_better=False)

    summary = {"narma10": n10, "pmnist": pmn, "meta": data["meta"]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # --- Verdict logic ---
    hw = n10.get("HW", {})
    sw = n10.get("SW_MATCHED", {})
    shuf = n10.get("SHUFFLE", {})
    nosub = n10.get("NO_SUB", {})

    hw_delta = hw.get("delta_mean", 0)
    hw_sigma = hw.get("delta_std", 1e-9) or 1e-9
    sw_delta = sw.get("delta_mean", 0)
    sw_sigma = sw.get("delta_std", 1e-9) or 1e-9

    # Z-score of hw_delta relative to SW_MATCHED distribution
    z_hw_vs_sw = (hw_delta - sw_delta) / max(sw_sigma, 1e-9)
    shuf_flat = abs(shuf.get("shuffle_mean", 0) - hw.get("diag_mean", 0)) < 0.05 * abs(hw.get("diag_mean", 1))

    gate_pass = (z_hw_vs_sw > 2.0) and shuf_flat

    pm_hw = pmn.get("HW", {})
    pm_sw = pmn.get("SW_MATCHED", {})
    cross_task = pm_hw.get("delta_mean", 0) < pm_sw.get("delta_mean", 0) - pm_sw.get("delta_std", 0)

    verdict = "DISCOVERY" if gate_pass else ("AMBITIOUS" if cross_task and gate_pass else "NULL")
    if hw_delta < sw_delta:
        verdict = "NULL"

    md = []
    md.append("# Phase 2 v2 — Envelope Substrate Transplant Matrix\n")
    md.append(f"**Date**: 2026-05-30   **N seeds**: {data['meta']['n_seeds']}   "
              f"**Substrate features**: {data['meta']['n_features']}\n")
    md.append(f"||z_ikaros - z_daedalus||_2 = **{data['meta']['L2_dist_z']:.3f}** "
              "(distance of the two device signatures in shared z-space)\n")
    md.append("## NARMA-10 transplant\n")
    md.append("| variant | diag NRMSE | off-diag NRMSE | Δ (off−diag) | n_seeds |")
    md.append("|---|---|---|---|---|")
    for v in ("HW", "SW_MATCHED", "NO_SUB"):
        d = n10.get(v, {})
        md.append(f"| {v} | {d.get('diag_mean',0):.4f} [{d.get('diag_ci',[0,0])[0]:.4f}, "
                  f"{d.get('diag_ci',[0,0])[1]:.4f}] | "
                  f"{d.get('off_mean',0):.4f} [{d.get('off_ci',[0,0])[0]:.4f}, "
                  f"{d.get('off_ci',[0,0])[1]:.4f}] | "
                  f"{d.get('delta_mean',0):+.4f} [{d.get('delta_ci',[0,0])[0]:+.4f}, "
                  f"{d.get('delta_ci',[0,0])[1]:+.4f}] | {d.get('n_seeds',0)} |")
    md.append(f"\nSHUFFLE: mean NRMSE = {shuf.get('shuffle_mean',0):.4f} "
              f"CI {shuf.get('ci',[0,0])}\n")
    md.append(f"\n**HW Δ vs SW_MATCHED z-score = {z_hw_vs_sw:+.2f}σ**   "
              f"SHUFFLE flat? **{shuf_flat}**\n")

    md.append("\n## Permuted-MNIST lite (5 tasks, K=4 classes)\n")
    md.append("| variant | diag acc | off-diag acc | Δ (off−diag) |")
    md.append("|---|---|---|---|")
    for v in ("HW", "SW_MATCHED"):
        d = pmn.get(v, {})
        md.append(f"| {v} | {d.get('diag_mean',0):.4f} | {d.get('off_mean',0):.4f} | "
                  f"{d.get('delta_mean',0):+.4f} |")
    md.append(f"\nCross-task transplant degradation: **{cross_task}**\n")

    md.append("\n## Verdict\n")
    md.append(f"- HW Δ NRMSE: {hw_delta:+.4f}  σ={hw_sigma:.4f}")
    md.append(f"- SW_MATCHED Δ NRMSE: {sw_delta:+.4f}  σ={sw_sigma:.4f}")
    md.append(f"- z(HW vs SW_MATCHED) = {z_hw_vs_sw:+.2f}σ   (gate: >2σ)")
    md.append(f"- SHUFFLE flat: {shuf_flat}")
    md.append(f"- Cross-task pMNIST corroboration: {cross_task}")
    md.append(f"\n### **PHASE 2 v2 VERDICT: {verdict}**\n")
    md.append("\nInterpretation:")
    if verdict == "NULL":
        md.append("- The 23-feature envelope substrate is also FUNGIBLE on NARMA-10. "
                  "Off-diagonal transplant does not degrade more than software-matched "
                  "Gaussian envelope of the same mean/std. The HW silicon-bound channels "
                  "(power/thermal/per-core latency) discriminate the *devices* with Cohen "
                  "d≥3, but they do NOT propagate into a learned reservoir readout in a "
                  "device-specific way. Identity remains *recognisable* but not *constitutive*.")
    elif verdict == "DISCOVERY":
        md.append("- Off-diagonal transplant degrades NRMSE significantly more than the "
                  "matched-Gaussian control, with SHUFFLE flat — the readout has learned "
                  "to exploit device-specific envelope structure that survives randomisation "
                  "of magnitude but not of *which feature is which*.")

    REPORT.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
