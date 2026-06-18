"""Generate D2 corrective grid + per-node assignment."""
import json, itertools
from pathlib import Path

V_G2_vals = [0.30, 0.35, 0.40, 0.45, 0.50]
dt_vals = [1e-7, 5e-7, 1e-6]
g_in_vals = [0.10, 0.20, 0.30]
C_b_vals = [5.0, 8.0, 10.0, 14.0]
seeds = list(range(10))

cells = []
for i, (cb, vg2, dt, gi) in enumerate(itertools.product(
        C_b_vals, V_G2_vals, dt_vals, g_in_vals)):
    cells.append({
        "cell_id": f"e{i:03d}",
        "C_b_fF": cb, "V_G2_bias": vg2, "dt_s": dt, "g_in": gi,
    })

assign = {}
for idx, c in enumerate(cells):
    b = idx % 4
    if b in {0, 1}: assign[c["cell_id"]] = "zgx"
    elif b == 2: assign[c["cell_id"]] = "daedalus"
    else: assign[c["cell_id"]] = "ikaros"

n_total = len(cells)
counts = {k: 0 for k in ("ikaros", "daedalus", "zgx")}
for v in assign.values(): counts[v] += 1
print(f"D2: {n_total} cells × {len(seeds)} seeds = {n_total*len(seeds)} runs")
print(f"  {counts}")

out_root = Path("research_plan/sweep_d2")
out_root.mkdir(parents=True, exist_ok=True)
(out_root / "grid.json").write_text(json.dumps({
    "cells": cells, "seeds": seeds, "assign": assign,
    "axes": {"V_G2": V_G2_vals, "dt_s": dt_vals, "g_in": g_in_vals, "C_b_fF": C_b_vals},
    "surrogate": "results/z278_mep2_surrogate_v3/surrogate_4d_v3.npz",
    "harness": "scripts/z280_d2_corrective_cell.py",
}, indent=2))

for node in ("ikaros", "daedalus", "zgx"):
    lines = []
    for c in cells:
        if assign[c["cell_id"]] != node: continue
        lines.append(" ".join([
            "--cell_id", c["cell_id"],
            "--C_b_fF", str(c["C_b_fF"]),
            "--V_G2_bias", str(c["V_G2_bias"]),
            "--dt_s", str(c["dt_s"]),
            "--g_in", str(c["g_in"]),
            "--seeds", *map(str, seeds),
        ]))
    out = out_root / f"node_{node}.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"  {node}: {len(lines)} cells → {out}")
