"""Summarise Phase 11B results into a deliverable-ready table."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

REPO = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
RES = REPO / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment11b/abcd_neuralode.json"


def fmt_ci(ci):
    return f"[{ci[0]*100:+5.1f}%, {ci[1]*100:+5.1f}%]"


def main():
    data = json.loads(RES.read_text())
    summary = data.get("summary", {})
    meta = data.get("meta", {})

    lines = []
    lines.append(f"# Phase 11B — A/B/C/D ablation across architectures")
    lines.append(f"")
    lines.append(f"Seeds: {meta.get('n_seeds')}, T={meta.get('T')}, epochs={meta.get('epochs')}")
    lines.append(f"Task: {meta.get('task')}; substrate ch: {meta.get('substrate_channels')}")
    lines.append(f"Elapsed: {meta.get('elapsed_min', 0):.1f} min")
    lines.append("")
    header = f"{'arch':12s} | {'n':3s} | {'NRMSE A':>9s} {'NRMSE B':>9s} {'NRMSE C':>9s} {'NRMSE D':>9s} | {'A-B mean':>9s} {'A-B CI':>20s} | {'A-C mean':>9s} {'A-C CI':>20s} | gates"
    lines.append(header)
    lines.append("-" * len(header))
    for arch, s in summary.items():
        row = (
            f"{arch:12s} | {s['n']:3d} | "
            f"{s['nrmse_A_mean']:>9.4f} {s['nrmse_B_mean']:>9.4f} {s['nrmse_C_mean']:>9.4f} {s['nrmse_D_mean']:>9.4f} | "
            f"{s['rel_AminusB_mean']*100:>+8.2f}% {fmt_ci(s['rel_AminusB_CI95']):>20s} | "
            f"{s['rel_AminusC_mean']*100:>+8.2f}% {fmt_ci(s['rel_AminusC_CI95']):>20s} | "
            f"AB15={'PASS' if s['gate_AB_15pct'] else 'FAIL'} "
            f"AC10={'PASS' if s['gate_AC_10pct'] else 'FAIL'}"
        )
        lines.append(row)

    lines.append("")
    lines.append("## Architecture-was-bottleneck verdict")
    nod = summary.get("neural_ode", {})
    ridge = summary.get("ridge_esn", {})
    if nod and ridge:
        nod_ab = nod.get("rel_AminusB_mean", 0) * 100
        ridge_ab = ridge.get("rel_AminusB_mean", 0) * 100
        gate = nod.get("gate_AB_15pct", False)
        lines.append(f"Neural ODE A-B = {nod_ab:+.2f}%   Ridge ESN A-B = {ridge_ab:+.2f}%")
        if gate and ridge_ab < 1.0:
            lines.append("Verdict: TRUE — Neural ODE crosses 15% gate while Ridge does not.")
        elif gate:
            lines.append("Verdict: PARTIAL — Neural ODE crosses 15% gate; Ridge also shows substrate effect.")
        else:
            lines.append("Verdict: FALSE — Neural ODE does not cross the 15% gate.")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
