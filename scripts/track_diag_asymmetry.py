#!/usr/bin/env python3
"""Track Diag: Is the 1.163 dec model gap on build_pyport_base() symmetric or
asymmetric in Vd sweep direction? And is the MEASUREMENT itself hysteretic?

Reuses cached per_bias_LEGACY rows from results/Pillar_I_C3_jts_tat/summary.json
(LEGACY = enable_jts_dsd=False = canonical build_pyport_base() baseline; same
code path as JTS_OFF/JTS_ON, only the JTS flag differs).

Adds measurement-hysteresis analysis by re-reading the raw CSVs and comparing
fwd vs bwd branches at matched Vd points.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
DATA = ROOT / "data/sebas_2026_04_22"
CACHE = ROOT / "results/Pillar_I_C3_jts_tat/summary.json"
OUT = ROOT / "results/track_diag_asymmetry"
OUT.mkdir(parents=True, exist_ok=True)

DEC_FLOOR_MEAS = 1e-12

# ── Load model residuals from cached LEGACY run ───────────────────
d = json.load(open(CACHE))
rows = d["per_bias_LEGACY"]
assert len(rows) == 66, f"expected 66 rows (33 biases × fwd+bwd), got {len(rows)}"

# Pair fwd/bwd per (VG1, VG2)
by_bias = {}
for r in rows:
    key = (round(r["VG1"], 3), round(r["VG2"], 3))
    by_bias.setdefault(key, {})[r["branch"]] = r

assert len(by_bias) == 33, f"expected 33 unique biases, got {len(by_bias)}"

# ── Re-read raw CSVs to compute MEASUREMENT hysteresis ─────────────
def load_csv_pair(fname):
    """Load a CSV and split into fwd/bwd at Vd apex."""
    for sub in DATA.iterdir():
        if not sub.is_dir(): continue
        p = sub / fname
        if p.exists():
            d = np.loadtxt(p, delimiter=",", skiprows=1)
            Vd = d[:, 0].astype(np.float64); Id = np.abs(d[:, 1]).astype(np.float64)
            apex = int(np.argmax(Vd))
            return (Vd[:apex+1], Id[:apex+1], Vd[apex:][::-1].copy(), Id[apex:][::-1].copy())
    return None

per_bias_table = []
for (vg1, vg2), branches in sorted(by_bias.items()):
    fwd_row = branches.get("fwd"); bwd_row = branches.get("bwd")
    if fwd_row is None or bwd_row is None: continue
    rmse_fwd = fwd_row["med_dec"]; rmse_bwd = bwd_row["med_dec"]
    asym_model = abs(rmse_fwd - rmse_bwd)

    # Measurement hysteresis: interpolate bwd I onto fwd Vd grid, compare
    pair = load_csv_pair(fwd_row["file"])
    meas_hys_med = float("nan"); meas_hys_max = float("nan")
    if pair is not None:
        fVd, fId, bVd, bId = pair
        # restrict to overlap and Vd>0.3 (matches model residual window)
        mask = (fVd > 0.3) & (fId > DEC_FLOOR_MEAS)
        if mask.sum() >= 3:
            bId_interp = np.interp(fVd[mask], bVd, bId)
            bId_interp = np.clip(bId_interp, DEC_FLOOR_MEAS, None)
            fId_clip = np.clip(fId[mask], DEC_FLOOR_MEAS, None)
            dlog = np.abs(np.log10(fId_clip) - np.log10(bId_interp))
            meas_hys_med = float(np.median(dlog))
            meas_hys_max = float(np.max(dlog))

    per_bias_table.append({
        "VG1": vg1, "VG2": vg2,
        "log10_rmse_fwd": rmse_fwd,
        "log10_rmse_bwd": rmse_bwd,
        "asymmetry_model": asym_model,
        "measurement_hysteresis_med_dec": meas_hys_med,
        "measurement_hysteresis_max_dec": meas_hys_max,
        "Imeas_peak_fwd": fwd_row["Imeas_peak"],
        "Ipred_peak_fwd": fwd_row["Ipred_peak"],
    })

# ── Summary stats ─────────────────────────────────────────────────
fwd_arr = np.array([r["log10_rmse_fwd"] for r in per_bias_table if np.isfinite(r["log10_rmse_fwd"])])
bwd_arr = np.array([r["log10_rmse_bwd"] for r in per_bias_table if np.isfinite(r["log10_rmse_bwd"])])
asym_arr = np.array([r["asymmetry_model"] for r in per_bias_table if np.isfinite(r["asymmetry_model"])])
mhys_med = np.array([r["measurement_hysteresis_med_dec"] for r in per_bias_table
                     if np.isfinite(r["measurement_hysteresis_med_dec"])])
mhys_max = np.array([r["measurement_hysteresis_max_dec"] for r in per_bias_table
                     if np.isfinite(r["measurement_hysteresis_max_dec"])])

sp_rho, sp_p = spearmanr(fwd_arr, bwd_arr)

summary = {
    "n_biases": len(per_bias_table),
    "median_rmse_fwd_dec": float(np.median(fwd_arr)),
    "median_rmse_bwd_dec": float(np.median(bwd_arr)),
    "mean_rmse_fwd_dec":   float(np.mean(fwd_arr)),
    "mean_rmse_bwd_dec":   float(np.mean(bwd_arr)),
    "median_model_asymmetry_dec": float(np.median(asym_arr)),
    "max_model_asymmetry_dec":    float(np.max(asym_arr)),
    "spearman_fwd_vs_bwd_rho":  float(sp_rho),
    "spearman_fwd_vs_bwd_p":    float(sp_p),
    "median_measurement_hysteresis_dec": float(np.median(mhys_med)) if mhys_med.size else None,
    "max_measurement_hysteresis_dec":    float(np.max(mhys_max)) if mhys_max.size else None,
    "n_biases_with_meas_hys": int(mhys_med.size),
}

# ── Top-5 outliers (by mean of fwd+bwd, drives the 1.163 median) ──
per_bias_table_sorted = sorted(
    per_bias_table,
    key=lambda r: -0.5 * (r["log10_rmse_fwd"] + r["log10_rmse_bwd"])
        if np.isfinite(r["log10_rmse_fwd"]) and np.isfinite(r["log10_rmse_bwd"]) else 0
)
top5 = per_bias_table_sorted[:5]

# ── Save table ────────────────────────────────────────────────────
json.dump({
    "summary": summary,
    "per_bias": per_bias_table,
    "top5_outliers": top5,
    "note": "LEGACY = enable_jts_dsd=False = build_pyport_base() baseline. n=66 (33 biases × fwd+bwd).",
    "median_dec_all_from_cache": d["summary_LEGACY"]["median_dec_all"]["median"],
}, open(OUT / "per_bias_residuals.json", "w"), indent=2)

# ── Scatter plot ──────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(6.5, 6))
mfwd = np.array([r["log10_rmse_fwd"] for r in per_bias_table])
mbwd = np.array([r["log10_rmse_bwd"] for r in per_bias_table])
vg1s = np.array([r["VG1"] for r in per_bias_table])
colors = {0.2: "tab:blue", 0.4: "tab:orange", 0.6: "tab:red"}
for vg1, col in colors.items():
    m = np.abs(vg1s - vg1) < 1e-6
    ax.scatter(mfwd[m], mbwd[m], c=col, label=f"VG1={vg1}", s=70, alpha=0.75, edgecolors="black")
lo = 0; hi = max(mfwd.max(), mbwd.max()) * 1.05
ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="fwd = bwd (symmetric)")
ax.set_xlabel("log10 RMSE fwd (dec)")
ax.set_ylabel("log10 RMSE bwd (dec)")
ax.set_title(f"build_pyport_base(): 33 biases\nSpearman ρ(fwd,bwd) = {sp_rho:.3f} (p={sp_p:.1e})\n"
             f"median asymmetry |fwd−bwd| = {summary['median_model_asymmetry_dec']:.3f} dec")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
plt.tight_layout()
plt.savefig(OUT / "scatter_fwd_bwd.png", dpi=140)
plt.close()

# ── Histogram of measurement hysteresis ───────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
if mhys_med.size:
    ax.hist(mhys_med, bins=20, color="tab:purple", alpha=0.7, edgecolor="black", label="med per bias")
    ax.axvline(np.median(mhys_med), color="black", ls="--",
               label=f"median = {np.median(mhys_med):.3f} dec")
ax.set_xlabel("Measurement hysteresis |log10(I_fwd) − log10(I_bwd)| (dec)")
ax.set_ylabel("# biases")
ax.set_title("Hysteresis IN THE DATA (33 biases, Vd>0.3V)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "hist_measurement_hysteresis.png", dpi=140)
plt.close()

# ── Verdict ───────────────────────────────────────────────────────
# Classify
fwd_med = summary["median_rmse_fwd_dec"]
bwd_med = summary["median_rmse_bwd_dec"]
asym_med = summary["median_model_asymmetry_dec"]
mhys_m = summary["median_measurement_hysteresis_dec"]
top5_share = sum(0.5*(r["log10_rmse_fwd"]+r["log10_rmse_bwd"]) for r in top5) / \
             sum(0.5*(r["log10_rmse_fwd"]+r["log10_rmse_bwd"]) for r in per_bias_table)

verdict_lines = []
verdict_lines.append(f"# Track Diag — Asymmetry diagnosis of build_pyport_base() 1.163 dec gap\n")
verdict_lines.append(f"**Baseline**: LEGACY (enable_jts_dsd=False) per-bias rows from cached "
                     f"`results/Pillar_I_C3_jts_tat/summary.json` (same code path as JTS_OFF/JTS_ON; "
                     f"identical to canonical `build_pyport_base()`).\n")
verdict_lines.append(f"## TLDR numbers (n=33 biases, fwd+bwd each)\n")
verdict_lines.append(f"| metric | value |")
verdict_lines.append(f"|---|---|")
verdict_lines.append(f"| median(rmse_fwd) | **{fwd_med:.4f} dec** |")
verdict_lines.append(f"| median(rmse_bwd) | **{bwd_med:.4f} dec** |")
verdict_lines.append(f"| |fwd−bwd| of medians | **{abs(fwd_med-bwd_med):.4f} dec** |")
verdict_lines.append(f"| median per-bias |fwd−bwd| | **{asym_med:.4f} dec** |")
verdict_lines.append(f"| max per-bias |fwd−bwd|    | {summary['max_model_asymmetry_dec']:.4f} dec |")
verdict_lines.append(f"| Spearman ρ(fwd, bwd) over 33 biases | **{sp_rho:.3f}** (p={sp_p:.2e}) |")
verdict_lines.append(f"| median measurement hysteresis | **{mhys_m:.4f} dec** (n={summary['n_biases_with_meas_hys']}) |")
verdict_lines.append(f"| max measurement hysteresis    | {summary['max_measurement_hysteresis_dec']:.4f} dec |")
verdict_lines.append(f"| top-5 outliers share of total | **{top5_share*100:.1f}%** |")
verdict_lines.append(f"| (cached) median_dec_all | {d['summary_LEGACY']['median_dec_all']['median']:.4f} dec |\n")

# Decision tree
classification = []
if asym_med < 0.10 and abs(fwd_med - bwd_med) < 0.05 and sp_rho > 0.7:
    classification.append("(a) **SYMMETRIC** — fwd≈bwd, ρ high. The 1.163 dec gap is "
                          "*conventional static-physics shortfall* (missing parallel path or "
                          "wrong subthreshold/triode physics), NOT memory.")
elif asym_med >= 0.3 or abs(fwd_med - bwd_med) >= 0.2:
    classification.append("(b) **ASYMMETRIC** — fwd vs bwd differ. Memory / hysteresis dominant.")
else:
    classification.append("(a/c) Mostly symmetric with mild per-bias variance.")

if top5_share >= 0.5:
    classification.append(f"(c) **OUTLIER-DRIVEN** — top-5 biases carry {top5_share*100:.1f}% of the dec-weight.")

if mhys_m is not None and mhys_m >= 0.3:
    classification.append(f"(d) **MEASUREMENT-HYSTERETIC** — the *data itself* has "
                          f"{mhys_m:.3f} dec median fwd-vs-bwd separation. Static DC fit "
                          f"is structurally limited; device is in dynamic regime.")
elif mhys_m is not None and mhys_m >= 0.1:
    classification.append(f"(d?) Mild measurement hysteresis ({mhys_m:.3f} dec). "
                          f"Device is *mostly* static but not fully.")
else:
    classification.append(f"Measurement is *static* (median hysteresis "
                          f"{mhys_m:.3f} dec). The dataset is a fair static-DC target.")

verdict_lines.append(f"## Classification\n")
for c in classification: verdict_lines.append(f"- {c}")

verdict_lines.append(f"\n## Plain-words answer\n")
if asym_med < 0.10 and (mhys_m is None or mhys_m < 0.1):
    verdict_lines.append(
        f"The 1.163 dec gap is **symmetric** (fwd {fwd_med:.3f} vs bwd {bwd_med:.3f}; "
        f"per-bias |Δ| median {asym_med:.3f} dec; ρ={sp_rho:.2f}) AND the measurement "
        f"itself is **non-hysteretic** ({mhys_m:.3f} dec). The model is missing **static physics** "
        f"(parallel leakage / subthreshold / triode regime), not memory effects. "
        f"The Pazos cell may be a memory cell *in pulsed operation*, but the Sebas DC sweep "
        f"at issue here is static-clean — the 1.163 dec gap is NOT explained by hysteresis."
    )
elif asym_med >= 0.2 or (mhys_m is not None and mhys_m >= 0.2):
    verdict_lines.append(
        f"The gap is **asymmetric / memory-dominated**: model fwd-bwd Δ={asym_med:.3f} dec; "
        f"measurement hysteresis {mhys_m:.3f} dec. Static DC frame is wrong."
    )
else:
    verdict_lines.append(
        f"Mixed: model is mostly symmetric ({asym_med:.3f} dec) but {top5_share*100:.0f}% of "
        f"gap concentrated in top-5 biases."
    )

verdict_lines.append(f"\n## Top-5 outlier biases (dominate the 1.163 median)\n")
verdict_lines.append(f"| rank | VG1 | VG2 | rmse_fwd | rmse_bwd | |Δ| | Imeas_peak |")
verdict_lines.append(f"|---|---|---|---|---|---|---|")
for i, r in enumerate(top5, 1):
    verdict_lines.append(f"| {i} | {r['VG1']:.2f} | {r['VG2']:+.2f} | "
                         f"{r['log10_rmse_fwd']:.3f} | {r['log10_rmse_bwd']:.3f} | "
                         f"{r['asymmetry_model']:.3f} | {r['Imeas_peak_fwd']:.2e} |")

verdict_lines.append(f"\n## Files\n")
verdict_lines.append(f"- `per_bias_residuals.json` — full table with all 33 biases")
verdict_lines.append(f"- `scatter_fwd_bwd.png` — diagnostic plot (diagonal = symmetric)")
verdict_lines.append(f"- `hist_measurement_hysteresis.png` — fwd-vs-bwd hysteresis IN THE DATA")
verdict_lines.append(f"- `top5_outliers.md` — outlier breakdown\n")

(OUT / "verdict.md").write_text("\n".join(verdict_lines))

# top5_outliers.md
top5_lines = ["# Top-5 outlier biases driving the 1.163 dec median\n"]
top5_lines.append("Sorted by 0.5*(rmse_fwd + rmse_bwd), descending.\n")
top5_lines.append(f"They contribute {top5_share*100:.1f}% of the total dec-weight across 33 biases.\n")
top5_lines.append(f"| rank | VG1 | VG2 | rmse_fwd | rmse_bwd | |Δ_model| | meas_hys_med | Imeas_peak | Ipred_peak | ratio (meas/pred) |")
top5_lines.append(f"|---|---|---|---|---|---|---|---|---|---|")
for i, r in enumerate(top5, 1):
    ratio = r["Imeas_peak_fwd"] / r["Ipred_peak_fwd"] if r["Ipred_peak_fwd"] else float("inf")
    top5_lines.append(f"| {i} | {r['VG1']:.2f} | {r['VG2']:+.2f} | "
                      f"{r['log10_rmse_fwd']:.3f} | {r['log10_rmse_bwd']:.3f} | "
                      f"{r['asymmetry_model']:.3f} | {r['measurement_hysteresis_med_dec']:.3f} | "
                      f"{r['Imeas_peak_fwd']:.2e} | {r['Ipred_peak_fwd']:.2e} | {ratio:.1e} |")
top5_lines.append(f"\n## Pattern\n")
vg1_top = [r["VG1"] for r in top5]
vg1_count = {v: vg1_top.count(v) for v in sorted(set(vg1_top))}
top5_lines.append(f"VG1 distribution in top-5: {vg1_count}")
top5_lines.append(f"(Reminder: full set has VG1∈{{0.2(7), 0.4(11), 0.6(15)}} biases.)")
(OUT / "top5_outliers.md").write_text("\n".join(top5_lines))

print(json.dumps(summary, indent=2))
print()
print(f"Wrote {OUT}/verdict.md, per_bias_residuals.json, scatter_fwd_bwd.png, hist_measurement_hysteresis.png, top5_outliers.md")
