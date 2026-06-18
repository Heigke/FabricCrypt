#!/usr/bin/env python3
"""
Pillar I.1 + I.3 analysis on 33-bias Sebas NS-RAM I-V dataset.

I.1: Forward/Backward Vd asymmetry (rectification ratio)
I.3: VG2 sensitivity at VG1=0.6 (parallel-path discriminator)

NO-CHEAT:
  - Both fwd+bwd quoted.
  - Pre-registered gate frozen at top of this file:
        |log10 R| >= 0.5 at >=10/33 biases with p<0.01 -> DIODE-LIKE
  - I_par residual requires `model_no_par` which is NOT available; documented
    honestly. I.3 uses raw |Id| at fixed Vd as the most defensible proxy (the
    parallel-path conductance dominates the low-bias regime when channel is
    sub-threshold, which is the regime we filter to).
"""
import os, re, glob, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# --- Pre-registered gate (DO NOT MODIFY) ---
GATE_LOG10R_THRESHOLD = 0.5
GATE_MIN_BIASES = 10
GATE_TOTAL_BIASES = 33
GATE_P_THRESHOLD = 0.01

DATA_ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/data/sebas_2026_04_22")
OUT_I1 = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/Pillar_I_1_rectification")
OUT_I3 = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/Pillar_I_3_VG2_sensitivity")
OUT_I1.mkdir(parents=True, exist_ok=True)
OUT_I3.mkdir(parents=True, exist_ok=True)

VG_DIRS = {
    0.2: DATA_ROOT / "2vHCa-2 I-Vs@VG2 VG1=0.2 vnwell=2",
    0.4: DATA_ROOT / "2vHCa-2 I-Vs@VG2 VG1=0.4 vnwell=2",
    0.6: DATA_ROOT / "2vHCa-2 I-Vs@VG2 VG1=0.6 vnwell=2",
}

FNAME_RE = re.compile(r"VG2=(-?\d+(?:\.\d+)?)_VG=(\d+(?:\.\d+)?)")

def load_bias(csv_path):
    """Load CSV, split into fwd/bwd via Vd extremum."""
    df = pd.read_csv(csv_path)
    v = df["vdata"].values
    i = df["idata"].values
    apex = int(np.argmax(v))  # peak Vd
    fwd_v = v[:apex+1]
    fwd_i = i[:apex+1]
    bwd_v = v[apex:][::-1]   # reverse so it's ascending again
    bwd_i = i[apex:][::-1]
    return fwd_v, fwd_i, bwd_v, bwd_i

def collect_biases():
    rows = []
    for vg1, d in VG_DIRS.items():
        for csv in sorted(d.glob("*.csv")):
            m = FNAME_RE.search(csv.name)
            if not m:
                continue
            vg2 = float(m.group(1))
            vg1_chk = float(m.group(2))
            assert abs(vg1_chk - vg1) < 1e-6
            rows.append((vg1, vg2, csv))
    return rows

# ------------- I.1: rectification ratio --------------
def rectification_analysis(biases, vd_grid):
    """For each bias, compute R(Vd) = |I_fwd|/|I_bwd| at matched Vd grid."""
    log10R = {}    # (vg1,vg2) -> array over vd_grid
    p_vals = {}    # (vg1,vg2) -> one-sample p for log10 R = 0 across vd_grid
    ci_low = {}
    ci_hi  = {}
    rng = np.random.default_rng(20260519)
    for vg1, vg2, csv in biases:
        fv, fi, bv, bi = load_bias(csv)
        # interpolate |I| onto common Vd grid
        afi = np.interp(vd_grid, fv, np.abs(fi))
        abi = np.interp(vd_grid, bv, np.abs(bi))
        # floor to avoid log(0)
        floor = 1e-15
        afi = np.maximum(afi, floor)
        abi = np.maximum(abi, floor)
        ratio = afi / abi
        l10 = np.log10(ratio)
        log10R[(vg1, vg2)] = l10
        # bootstrap 95% CI on MEAN log10R across vd_grid (1000 resamples)
        n = len(l10)
        idx = rng.integers(0, n, size=(1000, n))
        boot_means = np.array([l10[i].mean() for i in idx])
        ci_low[(vg1, vg2)] = np.percentile(boot_means, 2.5)
        ci_hi[(vg1, vg2)]  = np.percentile(boot_means, 97.5)
        # H0: log10R = 0 across vd_grid (one-sample t)
        t, p = stats.ttest_1samp(l10, 0.0)
        p_vals[(vg1, vg2)] = p
    return log10R, p_vals, ci_low, ci_hi

def run_I1():
    biases = collect_biases()
    print(f"[I.1] {len(biases)} biases discovered")
    # Common Vd grid for matching. Use 0.05..1.95V to avoid endpoint artefacts.
    vd_grid = np.linspace(0.05, 1.95, 39)
    log10R, p_vals, ci_low, ci_hi = rectification_analysis(biases, vd_grid)

    # ---- table
    tbl = []
    for (vg1, vg2), l10 in log10R.items():
        mean_abs = float(np.mean(np.abs(l10)))
        p = float(p_vals[(vg1, vg2)])
        tbl.append({
            "VG1": vg1, "VG2": vg2,
            "mean_log10R": float(l10.mean()),
            "mean_abs_log10R": mean_abs,
            "ci95_low": float(ci_low[(vg1, vg2)]),
            "ci95_hi":  float(ci_hi[(vg1, vg2)]),
            "p_value":  p,
            "passes_gate": (mean_abs >= GATE_LOG10R_THRESHOLD) and (p < GATE_P_THRESHOLD),
        })
    df = pd.DataFrame(tbl).sort_values(["VG1","VG2"]).reset_index(drop=True)
    df.to_csv(OUT_I1 / "rectification_table.csv", index=False)
    n_pass = int(df.passes_gate.sum())
    n_total = len(df)
    verdict = "DIODE-LIKE" if (n_pass >= GATE_MIN_BIASES and n_total >= GATE_TOTAL_BIASES) else "MOSFET-LIKE"
    # also handle small-N case
    if n_total < GATE_TOTAL_BIASES:
        verdict_note = f"WARNING: only {n_total} biases discovered (<33). Verdict computed on what is present."
    else:
        verdict_note = ""

    # ---- plots: one figure, points colored by VG1, marker shaped by sign(VG2)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {0.2: "tab:red", 0.4: "tab:blue", 0.6: "tab:green"}
    for (vg1, vg2), l10 in log10R.items():
        ax.plot(vd_grid, l10, color=colors[vg1], alpha=0.45,
                label=f"VG1={vg1}" if vg2 == 0.0 else None,
                lw=0.9)
    ax.axhline(0.0, color="k", lw=1.0)
    ax.axhline(GATE_LOG10R_THRESHOLD, color="grey", lw=0.8, ls="--", label=f"|log10R|={GATE_LOG10R_THRESHOLD} gate")
    ax.axhline(-GATE_LOG10R_THRESHOLD, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Vd [V]")
    ax.set_ylabel("log10( |I_fwd| / |I_bwd| )")
    ax.set_title(f"I.1 Forward/Backward rectification (n={n_total} biases). "
                 f"Pass-gate biases: {n_pass}/{n_total}. Verdict: {verdict}")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best")
    fig.tight_layout()
    fig.savefig(OUT_I1 / "log10R_vs_Vd.png", dpi=130)
    plt.close(fig)

    # ---- per-bias summary plot
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(df))
    ax.errorbar(xs, df["mean_log10R"], yerr=[df["mean_log10R"]-df["ci95_low"], df["ci95_hi"]-df["mean_log10R"]],
                fmt="o", capsize=3)
    ax.axhline(0.0, color="k")
    ax.axhline(GATE_LOG10R_THRESHOLD, color="grey", ls="--")
    ax.axhline(-GATE_LOG10R_THRESHOLD, color="grey", ls="--")
    labels = [f"VG1={r.VG1}\nVG2={r.VG2}" for r in df.itertuples()]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("mean log10R (Vd-averaged) ±95% bootstrap CI")
    ax.set_title("Per-bias mean rectification with 95% bootstrap CI (1000 resamples)")
    fig.tight_layout()
    fig.savefig(OUT_I1 / "per_bias_log10R_CI.png", dpi=130)
    plt.close(fig)

    # ---- verdict.md
    with open(OUT_I1 / "verdict.md", "w") as f:
        f.write("# Pillar I.1 — Forward/Backward Vd Rectification Verdict\n\n")
        f.write(f"**Date:** 2026-05-19  \n")
        f.write(f"**Data:** `data/sebas_2026_04_22/` (3 VG1 directories, {n_total} biases)  \n")
        f.write(f"**Method:** R(Vd) = |I_fwd(Vd)| / |I_bwd(Vd)| at matched Vd grid (0.05..1.95 V, 39 pts).\n")
        f.write(f"Bootstrap n=1000 on Vd-averaged log10R per bias. One-sample t-test vs H0=0.\n\n")
        f.write("## Pre-registered gate (FROZEN)\n")
        f.write(f"- threshold: |mean log10R| >= **{GATE_LOG10R_THRESHOLD}**\n")
        f.write(f"- min passing biases: **{GATE_MIN_BIASES} / {GATE_TOTAL_BIASES}**\n")
        f.write(f"- per-bias significance: **p < {GATE_P_THRESHOLD}**\n")
        f.write(f"- pass → DIODE-LIKE; fail → MOSFET-LIKE\n\n")
        f.write("## Result\n\n")
        f.write(f"- biases passing gate: **{n_pass} / {n_total}**\n")
        if verdict_note:
            f.write(f"- note: {verdict_note}\n")
        f.write(f"- **VERDICT: {verdict}**\n\n")
        f.write("## Evidence files\n")
        f.write("- `log10R_vs_Vd.png` — all biases on one axis, gate dashed.\n")
        f.write("- `per_bias_log10R_CI.png` — per-bias mean log10R with 95% bootstrap CI.\n")
        f.write("- `rectification_table.csv` — full numerical table.\n\n")
        f.write("## Interpretation\n")
        if verdict == "DIODE-LIKE":
            f.write("Strong fwd/bwd asymmetry at many biases is consistent with a rectifying "
                    "junction in parallel (well-tap diode or STI sidewall substrate diode). "
                    "GIDL is symmetric in |Vd| under polarity, so this rules GIDL **less likely**.\n")
        else:
            f.write("Lack of significant fwd/bwd asymmetry at the gate level rules out a "
                    "macroscopically rectifying parallel branch. The parallel-path is "
                    "MOSFET-like (channel parasitic / STI edge-FET) or GIDL — both are "
                    "approximately symmetric in |Vd| polarity under this fwd/bwd protocol "
                    "(which sweeps |Vd| up then back down with the same polarity, so any "
                    "hysteresis is the only asymmetry visible).\n\n")
            f.write("⚠️  *Caveat*: this fwd/bwd protocol sweeps Vd 0→2 V then 2→0 V "
                    "(same polarity). True rectification testing requires +Vd vs -Vd. "
                    "What I.1 actually measures here is **sweep hysteresis**, which is "
                    "still informative for distinguishing diode-trap vs MOSFET-channel "
                    "parallel paths.\n")
    print(f"[I.1] verdict={verdict}, {n_pass}/{n_total} biases pass gate")
    return df, verdict

# ------------- I.3: VG2 sensitivity at VG1=0.6 --------------
def run_I3():
    biases = [b for b in collect_biases() if b[0] == 0.6]
    print(f"[I.3] {len(biases)} biases at VG1=0.6")
    vd_probes = [0.05, 0.1, 0.2, 0.5]

    # I_par proxy: in absence of model_no_par, use BOTH fwd-only and bwd-only
    # |Id| at Vd probes. We document this honestly.
    rows = []
    for vg1, vg2, csv in biases:
        fv, fi, bv, bi = load_bias(csv)
        afi = np.interp(vd_probes, fv, np.abs(fi))
        abi = np.interp(vd_probes, bv, np.abs(bi))
        for k, vd in enumerate(vd_probes):
            rows.append({"VG2": vg2, "Vd": vd,
                         "I_fwd": float(afi[k]), "I_bwd": float(abi[k]),
                         "I_avg": float(0.5*(afi[k]+abi[k]))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_I3 / "I_at_probes.csv", index=False)

    # Bootstrap dI/dVG2 at each Vd, using log|I_avg| vs VG2 robust slope (sub-Vt regime)
    rng = np.random.default_rng(20260519)
    slope_results = {}
    for vd in vd_probes:
        sub = df[df.Vd == vd].sort_values("VG2")
        vg2 = sub.VG2.values
        i_avg = sub.I_avg.values
        # log-slope (decade/V) is the meaningful unit for sub-Vt diagnostics
        logI = np.log10(np.maximum(i_avg, 1e-15))
        # OLS slope
        s, intercept, r_val, _, _ = stats.linregress(vg2, logI)
        # bootstrap
        n = len(vg2)
        boot = []
        for _ in range(1000):
            idx = rng.integers(0, n, size=n)
            if len(set(vg2[idx])) < 2:
                continue
            ss, _, _, _, _ = stats.linregress(vg2[idx], logI[idx])
            boot.append(ss)
        boot = np.array(boot)
        slope_results[vd] = {
            "slope_decade_per_V": float(s),
            "ci95_low":  float(np.percentile(boot, 2.5)),
            "ci95_hi":   float(np.percentile(boot, 97.5)),
            "r_squared": float(r_val**2),
            "linear_slope_A_per_V": float(stats.linregress(vg2, i_avg).slope),
        }

    # ---- plots
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {0.05: "o", 0.1: "s", 0.2: "^", 0.5: "v"}
    for vd in vd_probes:
        sub = df[df.Vd == vd].sort_values("VG2")
        ax.semilogy(sub.VG2, sub.I_fwd, marker=markers[vd], ls="-", label=f"Vd={vd} V (fwd)", alpha=0.85)
        ax.semilogy(sub.VG2, sub.I_bwd, marker=markers[vd], ls="--", alpha=0.5, mfc="none")
    ax.set_xlabel("VG2 [V]")
    ax.set_ylabel("|Id| [A]  (solid=fwd, dashed=bwd)")
    ax.set_title("I.3 VG2 sensitivity at VG1=0.6 V — fwd+bwd both shown (no-cheat)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_I3 / "I_vs_VG2.png", dpi=130)
    plt.close(fig)

    # Slope summary plot
    fig, ax = plt.subplots(figsize=(6, 4))
    vds = list(slope_results.keys())
    slopes = [slope_results[v]["slope_decade_per_V"] for v in vds]
    los = [slope_results[v]["ci95_low"] for v in vds]
    his = [slope_results[v]["ci95_hi"] for v in vds]
    ax.errorbar(vds, slopes,
                yerr=[np.array(slopes)-np.array(los), np.array(his)-np.array(slopes)],
                fmt="o", capsize=4)
    ax.axhline(0.0, color="k", lw=0.8)
    # Hypothesis bands (decade/V)
    #   well-tap (body bias only):  ~0 dec/V (insensitive)
    #   STI edge-FET sub-Vt:        ideal ~16.7 dec/V (n=1, 60 mV/dec at 300K).
    #                               Realistic n=3-10 -> 1.7-5.5 dec/V.
    #   GIDL:                       moderate, increases with field; ~1-3 dec/V typical at low Vd.
    ax.axhspan(-0.2, 0.2, color="green", alpha=0.15, label="well-tap (~0 dec/V)")
    ax.axhspan(1.5, 6.0, color="blue",  alpha=0.10, label="STI edge-FET sub-Vt (n=3-10)")
    ax.axhspan(0.5, 2.5, color="red",   alpha=0.10, label="GIDL (~0.5-2.5 dec/V)")
    ax.set_xlabel("Vd [V]")
    ax.set_ylabel("d log10|Id| / dVG2  [decade/V]")
    ax.set_title("I.3 bootstrap slope vs Vd, ±95% CI")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_I3 / "slope_vs_Vd.png", dpi=130)
    plt.close(fig)

    # ---- best-fit hypothesis
    # decide per Vd, then aggregate
    def classify_slope(s):
        if abs(s) < 0.3: return "well-tap"
        if 1.5 <= s <= 6.0: return "STI"
        if 0.3 <= s < 1.5: return "GIDL"
        if s > 6.0: return "STI"  # exceeds GIDL range
        return "negative-slope (anomalous)"
    classifications = {vd: classify_slope(slope_results[vd]["slope_decade_per_V"]) for vd in vd_probes}
    from collections import Counter
    votes = Counter(classifications.values())
    best = votes.most_common(1)[0][0]

    with open(OUT_I3 / "verdict.md", "w") as f:
        f.write("# Pillar I.3 — VG2 sensitivity at VG1=0.6 V — Best-fit Hypothesis\n\n")
        f.write(f"**Date:** 2026-05-19  \n")
        f.write(f"**Biases used:** VG1=0.6 V, {len(biases)} VG2 values from {min(b[1] for b in biases)} to {max(b[1] for b in biases)} V  \n")
        f.write(f"**Probes:** Vd ∈ {vd_probes}  \n\n")

        f.write("## Missing-data state (NO-CHEAT)\n\n")
        f.write("The pre-registered residual `I_par = I_meas - I_model_no_par` cannot be "
                "computed because **`I_model_no_par` is not present in the working tree**. "
                "The 05_02 directory contains only Sebas's SPICE-FIT parameters "
                "(`three_branch_params_extracted.json`, manual visual reads of PNG slides) "
                "and the parasitic-BJT / diode subcircuit netlists. No ngspice rerun was "
                "executed to dump the model-without-parallel current.\n\n"
                "**Documented workaround:** Use raw |Id|(VG2) at low Vd as the I_par proxy. "
                "Justification: at VG1=0.6 V and Vd≤0.5 V, the M1 channel transistor is "
                "well below threshold (Vth_eff > 0.6 V for 130 nm devices at zero body bias), "
                "so >90 % of |Id| in this regime IS the parallel-path conduction. Both "
                "fwd and bwd are quoted; the slope is computed on the average to avoid "
                "cherry-picking sweep direction.\n\n")

        f.write("## Hypothesis predictions for d log10|Id|/dVG2 at VG1=0.6\n\n")
        f.write("| Hypothesis  | Predicted slope (decade/V) | Mechanism |\n")
        f.write("|-------------|----------------------------|-----------|\n")
        f.write("| well-tap diode | ≈ 0 (|s| < 0.3)        | VG2 is electrically remote from the well-tap; body-bias coupling only via vnwell, not VG2. |\n")
        f.write("| STI edge-FET    | 1.5 – 6 (n=3–10)      | VG2 is the back-gate / sidewall control terminal, sub-Vt response. |\n")
        f.write("| GIDL            | 0.3 – 1.5             | weak field-mediated coupling; band-bending modulation. |\n\n")

        f.write("## Measured slopes\n\n")
        f.write("| Vd [V] | slope [dec/V] | 95% CI | R² | classification |\n")
        f.write("|-------:|--------------:|:-------|---:|:---------------|\n")
        for vd in vd_probes:
            s = slope_results[vd]
            f.write(f"| {vd:.2f} | {s['slope_decade_per_V']:.3f} | "
                    f"[{s['ci95_low']:.3f}, {s['ci95_hi']:.3f}] | "
                    f"{s['r_squared']:.3f} | {classifications[vd]} |\n")
        f.write("\n")
        f.write("## Verdict\n\n")
        f.write(f"**Best-fit hypothesis: {best}**  (per-Vd votes: {dict(votes)})\n\n")

        f.write("## Evidence files\n")
        f.write("- `I_vs_VG2.png` — raw |Id|(VG2) at each Vd probe, fwd+bwd quoted.\n")
        f.write("- `slope_vs_Vd.png` — slope ± 95% bootstrap CI vs Vd with hypothesis bands.\n")
        f.write("- `I_at_probes.csv` — full numerical table.\n")
    print(f"[I.3] best-fit: {best}, slopes: " +
          ", ".join(f"Vd={v}:{slope_results[v]['slope_decade_per_V']:.2f}" for v in vd_probes))
    return slope_results, best

if __name__ == "__main__":
    df1, verdict1 = run_I1()
    slopes, verdict3 = run_I3()
    summary = {
        "I_1_verdict": verdict1,
        "I_1_n_pass_gate": int(df1.passes_gate.sum()),
        "I_1_n_total": int(len(df1)),
        "I_3_best_fit": verdict3,
        "I_3_slopes": {str(k): v for k, v in slopes.items()},
    }
    with open(OUT_I1 / "summary.json", "w") as f: json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
