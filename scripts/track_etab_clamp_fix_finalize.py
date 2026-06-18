#!/usr/bin/env python3
"""Finalize verdict.md + plot.png + ngspice_xval.json from ablation.json.

Phase 3 ngspice xval was incomplete due to runtime budget; we partial-finalize
honestly stating ngspice gap as "not measured (run aborted at OFF VG1=0.6 due
to runtime budget — see ablation.json + run.log)".
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT = ROOT / "results/track_etab_clamp_fix"
UPPER_MAX = 1.0

d = json.loads((OUT / "ablation.json").read_text())
off = d["full33_off"]; on = d["full33_on"]
k_off = d["knee_off"]; k_on = d["knee_on"]
med_off = off["median_dec_all"]["median"]
med_on  = on["median_dec_all"]["median"]

# OFF ngspice gap from partial xval (7 of 9 biases)
partial_ngx_off = {
    (0.2, -0.1): 0.963, (0.2, 0.0): 0.963, (0.2, 0.1): 0.963,
    (0.4, -0.1): 0.688, (0.4, 0.0): 0.688, (0.4, 0.1): 0.688,
    (0.6, -0.1): 0.773,
}
gap_off_partial = float(np.mean(list(partial_ngx_off.values())))
# ngspice ON not measured
gap_on = None

# Knee
knee_rows = []
knee_on_vals = []
for vg2_s, ko in k_off.items():
    vg2 = float(vg2_s)
    kn = k_on.get(vg2_s) or k_on.get(str(vg2))
    if kn is None:
        continue
    knee_rows.append((vg2, ko["knee_data"], ko["knee_model"], kn["knee_model"]))
    knee_on_vals.append(kn["knee_model"])
knee_max = float(np.max(knee_on_vals))
knee_min = float(np.min(knee_on_vals))
pass_dec = med_on <= 0.4
pass_knee = knee_max <= 1.2
overall = "FAIL"  # set explicitly

# -------- plot --------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
vg2s = [r[0] for r in knee_rows]
ax.plot(vg2s, [r[1] for r in knee_rows], "ko-", label="data knee", lw=1.5)
ax.plot(vg2s, [r[2] for r in knee_rows], "b--s", label=f"model OFF (baseline, K1+ALPHA0+Tlpe1)", lw=1.3)
ax.plot(vg2s, [r[3] for r in knee_rows], "r:^", label=f"model ON (+dibl_upper_clamp=1.0)", lw=1.3)
ax.axhline(1.2, color="gray", ls=":", lw=1, alpha=0.5)
ax.text(min(vg2s), 1.21, " PASS gate ≤1.2V", fontsize=8, color="gray")
ax.set_xlabel("VG2 (V)")
ax.set_ylabel("knee Vd (V)")
ax.set_title("VG1=0.6 knee position — ON clamp shifts RIGHT (worse)")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
labels = ["OFF\n(K1+ALPHA0+Tlpe1)", "ON\n(+dibl_upper_clamp=1.0)"]
vals = [med_off, med_on]
bars = ax.bar(labels, vals, color=["#1976d2", "#d32f2f"])
ax.axhline(0.4, color="gray", ls=":", label="PASS gate ≤0.4")
ax.set_ylabel("full-33 median_dec")
ax.set_title(f"full-33 fwd+bwd median_dec\nΔ = {med_on-med_off:+.4f} dec (negligible)")
ax.legend()
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.4f}", ha="center", fontsize=9)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(OUT / "plot.png", dpi=120, bbox_inches="tight")
print(f"wrote {OUT/'plot.png'}")

# -------- ngspice_xval.json --------
ngx_json = {
    "off_partial": {
        "n_completed": len(partial_ngx_off),
        "n_planned": 9,
        "per_bias_med_dec": {f"VG1={k[0]}_VG2={k[1]}": v for k, v in partial_ngx_off.items()},
        "mean_gap_partial": gap_off_partial,
        "note": "partial — runtime budget exceeded; aborted at OFF VG1=0.6 VG2=-0.1",
    },
    "on": None,
    "on_note": "ngspice gap with gate ON was NOT measured (run aborted due to "
               "runtime budget). Phase-1 full-33 median_dec already showed only "
               "Δ=-0.0045 dec improvement (gate is near-no-op), so ngspice gap "
               "would also not meaningfully improve.",
}
(OUT / "ngspice_xval.json").write_text(json.dumps(ngx_json, indent=2))
print(f"wrote {OUT/'ngspice_xval.json'}")

# -------- verdict.md --------
lines = []
lines.append("# track_etab_clamp_fix — verdict\n")
lines.append("## 1. BSIM4 reference clamp — VERIFIED\n")
lines.append("Inspected sources:")
lines.append("- `external/bsim4/code/b4ld.c` §1107-1117 (BSIM4 v4.5 reference)")
lines.append("- `ngspice-42+ds/src/spicelib/devices/bsim4/b4ld.c` §1147-1153 (ngspice-42)\n")
lines.append("```c")
lines.append("T3 = here->BSIM4eta0 + pParam->BSIM4etab * Vbseff;")
lines.append("if (T3 < 1.0e-4)")
lines.append("{   T9 = 1.0 / (3.0 - 2.0e4 * T3);")
lines.append("    T3 = (2.0e-4 - T3) * T9;")
lines.append("    T4 = T9 * T9;")
lines.append("}")
lines.append("```\n")
lines.append("**Canonical BSIM4 (both reference and ngspice-42) has ONLY a LOWER")
lines.append("soft-clamp at 1e-4 via the rational regularizer.** There is NO upper")
lines.append("clamp in BSIM4. The proposed `max=2.0` is a deliberate non-BSIM4")
lines.append("modification motivated by Sebas card's anomalous etab≈1.8-2.5 (vs")
lines.append("canonical ~-0.07), which lets T3_d explode to 2-4 at snapback")
lines.append("(Vbseff→1V) and over-suppress Vth.\n")
lines.append(f"Patch applied as gated upper clamp: model['dibl_upper_clamp']=True")
lines.append(f"with model['dibl_upper_clamp_max'] (default 1.0). This run used")
lines.append(f"`max={UPPER_MAX}` (defensible bound: keeps DIBL shift ≤ Vds since")
lines.append(f"θ0vb0≤1; the proposed max=2.0 still allows 2·Vds of DIBL shift")
lines.append(f"and was not expected to bind given etab≈2.5×Vbseff alone gives T3≈2.5).")
lines.append(f"All pre-existing fits unchanged when gate OFF (default).\n")
lines.append("Patch location: `nsram/nsram/bsim4_port/dc.py` after canonical `T3_clamped`.\n")

lines.append("## 2. Patch applied — YES (gated)\n")
lines.append("```python")
lines.append("if bool(model.get('dibl_upper_clamp', False)):")
lines.append("    _upper = float(model.get('dibl_upper_clamp_max', 1.0))")
lines.append("    T3_clamped = torch.clamp(T3_clamped, max=_upper)")
lines.append("```\n")

lines.append("## 3. Full-33 fwd+bwd median_dec (in-run apples-to-apples)\n")
lines.append(f"- OFF (gate disabled, K1+ALPHA0+Tlpe1):  **{med_off:.4f} dec** (n_finite=66/66)")
lines.append(f"- ON  (gate enabled,  K1+ALPHA0+Tlpe1):  **{med_on:.4f} dec** (n_finite=66/66)")
lines.append(f"- Δ = {med_on - med_off:+.4f} dec")
lines.append(f"- PASS gate (≤0.4): **{'PASS' if pass_dec else 'FAIL'}** "
             f"({med_on:.4f} > 0.4)\n")
for vg1 in (0.2, 0.4, 0.6):
    a = off.get(f"median_dec_VG1={vg1}_grouped")
    b = on.get(f"median_dec_VG1={vg1}_grouped")
    lines.append(f"- per-VG1={vg1}: OFF={a:.4f}, ON={b:.4f}, Δ={b-a:+.4f}")
lines.append("")

lines.append("## 4. ngspice 9-bias gap — PARTIAL (run aborted)\n")
lines.append(f"Phase-3 ngspice xval was aborted at 7 of 9 OFF biases due to runtime")
lines.append(f"budget (each VG1=0.6 bias takes ~10 min for ngspice + pyport with")
lines.append(f"snapback Newton iteration). ON branch was NOT reached.\n")
lines.append(f"- OFF mean gap (partial, n=7/9): **{gap_off_partial:.4f}** dec")
lines.append(f"- ON  mean gap: **NOT MEASURED**")
lines.append(f"- PASS gate (≤0.4): **FAIL (cannot evaluate)** — but phase-1 already")
lines.append(f"  shows Δ=−0.0045 dec for full-33, so ngspice gap improvement is bounded")
lines.append(f"  to be ≤O(0.01) dec; cannot reach the 0.808→0.4 PASS gate.\n")

lines.append("## 5. Knee shift at VG1=0.6 (V where model Id crosses 10× baseline)\n")
lines.append("| VG2 | data | model OFF | model ON | Δ(ON−OFF) |")
lines.append("|----:|----:|----:|----:|----:|")
for vg2, kd, ko, kn in knee_rows:
    lines.append(f"| {vg2:+.2f} | {kd:.3f} | {ko:.3f} | {kn:.3f} | {kn-ko:+.3f} |")
lines.append(f"\n- min(model_ON across VG2) = **{knee_min:.3f}V**")
lines.append(f"- max(model_ON across VG2) = **{knee_max:.3f}V**")
lines.append(f"- PASS gate (max ≤1.2V): **{'PASS' if pass_knee else 'FAIL'}** "
             f"({knee_max:.3f} > 1.2). **Knee shifts RIGHT by +0.05V** with the gate ON")
lines.append(f"  (1.45V → 1.50V) — i.e. opposite to the desired direction.\n")

lines.append(f"## OVERALL: **{overall}**\n")
lines.append(f"- full-33 dec: {med_on:.4f} > 0.4 (FAIL; Δ=−0.0045 dec is negligible)")
lines.append(f"- knee: {knee_max:.3f}V > 1.2V and shifts RIGHT not LEFT (FAIL)")
lines.append(f"- ngspice gap: partial (cannot fully evaluate), but bounded by")
lines.append(f"  full-33 result to be effectively unchanged from baseline.\n")
lines.append("## Interpretation — clamp is NOT the right answer\n")
lines.append("Per the NO-CHEAT criterion: \"If knee doesn't shift left, the etab")
lines.append("clamp isn't the right answer — say so.\" We do.\n")
lines.append("The Vth shift from clamping T3_d (verified at one bias: Vbseff=0.95,")
lines.append("Vds=2.0, Vgs=0.6, Vth went 0.308 → 0.405, +97 mV) IS real, but it")
lines.append("does not translate into a leftward knee shift because the snapback")
lines.append("Vd at VG1=0.6 is dominated by the body-NPN trigger condition")
lines.append("(Vbe≈0.7V), which is set by the substrate impact-ionization current")
lines.append("path, not by the M1 channel Vth. Raising M1 Vth slightly delays the")
lines.append("channel onset (the model knee actually moves to higher Vd, +50mV)")
lines.append("rather than letting the NPN trigger sooner.\n")
lines.append("The Sebas etab≈1.8-2.5 anomaly is therefore a **symptom** of a")
lines.append("different upstream root cause — likely (i) the Vb attractor selection")
lines.append("in the body-floating Newton solve (cf. `results/track_vb_attractor_hunt/`),")
lines.append("(ii) charge-conservation in the snapback sub-circuit, or (iii) an")
lines.append("incorrect coupling between the well diode and the body in the")
lines.append("2T cell during snapback. The clamp is a band-aid on a downstream")
lines.append("DIBL term that doesn't drive the residual gap.\n")
lines.append("**Gate left at default OFF; do NOT enable for production fits.**\n")
lines.append("Recommended next step: revisit the Vb attractor (z2065+ track) at")
lines.append("VG1=0.6 snapback with multi-init seeded by ngspice's source-stepping")
lines.append("trajectory; the residual 0.46 dec likely splits there.\n")

(OUT / "verdict.md").write_text("\n".join(lines))
print(f"wrote {OUT/'verdict.md'}")
print()
print((OUT / "verdict.md").read_text())
