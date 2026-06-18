"""z276 — refined sweep grid v2, focused on the productive regime
benchmark revealed (C_b=8 fF, V_G2=0.2, dt=1e-7, g_in=0.2 → 67.4% acc).

Grid 5×4×4×4 = 320 cells × 4 seeds = 1280 runs.
Wall: ~1.8s/seed × 4 seeds × 320 cells = ~38 min single node.
"""
import json, itertools
from pathlib import Path

C_b_vals    = [5.0, 8.0, 10.0, 14.0, 20.0]              # Sebas-confirmed range
V_G2_vals   = [0.05, 0.15, 0.25, 0.35]                  # thin-ox sweep
dt_vals     = [1e-7, 5e-7, 1e-6, 5e-6]                  # productive timescale band
g_in_vals   = [0.10, 0.20, 0.40, 0.80]                  # input drive
seeds       = [0, 1, 2, 3]

cells = []
for i, (cb, vg2, dt, gi) in enumerate(itertools.product(
        C_b_vals, V_G2_vals, dt_vals, g_in_vals)):
    cells.append({
        "cell_id": f"d{i:03d}",
        "C_b_fF": cb, "V_G2_bias": vg2, "dt_s": dt, "g_in": gi,
    })

n_total = len(cells)
# Distribute by node capacity (ZGX has GB10 — assume 2× faster than 8060S)
assign = {}
for idx, c in enumerate(cells):
    # round-robin: 2x zgx, 1x daedalus, 1x ikaros
    bucket = idx % 4
    if bucket in {0, 1}:
        assign[c["cell_id"]] = "zgx"
    elif bucket == 2:
        assign[c["cell_id"]] = "daedalus"
    else:
        assign[c["cell_id"]] = "ikaros"

counts = {"zgx": 0, "daedalus": 0, "ikaros": 0}
for v in assign.values():
    counts[v] += 1
print(f"Total cells: {n_total}; seeds per cell: {len(seeds)}")
print(f"Assignment: {counts}")

out_root = Path("research_plan/sweep_v2")
out_root.mkdir(parents=True, exist_ok=True)
(out_root / "grid.json").write_text(json.dumps({
    "cells": cells, "seeds": seeds, "assign": assign,
    "C_b_fF": C_b_vals, "V_G2_bias": V_G2_vals,
    "dt_s": dt_vals, "g_in": g_in_vals,
    "benchmark_anchor": {
        "C_b_fF": 8.0, "V_G2_bias": 0.20, "dt_s": 1e-7, "g_in": 0.20,
        "test_acc_4seed_mean": 0.6744, "vb_rail": 0.0,
    },
}, indent=2))

for node in ["ikaros", "daedalus", "zgx"]:
    lines = []
    for c in cells:
        if assign[c["cell_id"]] != node:
            continue
        lines.append(" ".join([
            "--cell_id", c["cell_id"],
            "--C_b_fF", str(c["C_b_fF"]),
            "--V_G2_bias", str(c["V_G2_bias"]),
            "--dt_s", str(c["dt_s"]),
            "--g_in", str(c["g_in"]),
            "--seeds", *map(str, seeds),
            "--subsample", "10000", "2000",
        ]))
    out_path = out_root / f"node_{node}.txt"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  {node}: {len(lines)} cells → {out_path}")
