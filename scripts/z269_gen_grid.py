"""z269 — generate the 4D sweep grid + assign cells to nodes."""
import json, itertools
from pathlib import Path

C_b_vals    = [2.0, 5.0, 10.0, 20.0, 50.0]      # fF
V_G2_vals   = [0.10, 0.20, 0.30]                 # V
dt_vals     = [1e-7, 1e-6, 1e-5]                 # s
g_in_vals   = [0.10, 0.20, 0.40]
seeds       = [0, 1, 2, 3]

cells = []
for i, (cb, vg2, dt, gi) in enumerate(itertools.product(
        C_b_vals, V_G2_vals, dt_vals, g_in_vals)):
    cells.append({
        "cell_id": f"c{i:03d}",
        "C_b_fF": cb, "V_G2_bias": vg2, "dt_s": dt, "g_in": gi,
    })

# Assign by node speed:
#   ZGX (GB10): 67%
#   daedalus  : 22%
#   ikaros    : 11%
n_total = len(cells)
n_zgx       = int(round(n_total * 0.67))
n_daedalus  = int(round(n_total * 0.22))
n_ikaros    = n_total - n_zgx - n_daedalus

# Interleave so each node gets a representative cross-section
assign = {}
for idx, c in enumerate(cells):
    if idx % 9 in {0, 1, 2, 3, 4, 5}:        # 6/9 ~ 67% → ZGX
        assign[c["cell_id"]] = "zgx"
    elif idx % 9 in {6, 7}:                  # 2/9 ~ 22% → daedalus
        assign[c["cell_id"]] = "daedalus"
    else:                                    # 1/9 ~ 11% → ikaros
        assign[c["cell_id"]] = "ikaros"

# Print summary
counts = {"zgx": 0, "daedalus": 0, "ikaros": 0}
for v in assign.values():
    counts[v] += 1
print(f"Total cells: {n_total}; seeds per cell: {len(seeds)}")
print(f"Assignment: zgx={counts['zgx']}, daedalus={counts['daedalus']}, "
      f"ikaros={counts['ikaros']}")

out_root = Path("research_plan/sweep_v1")
out_root.mkdir(parents=True, exist_ok=True)
(out_root / "grid.json").write_text(json.dumps({
    "cells": cells, "seeds": seeds, "assign": assign,
    "C_b_fF": C_b_vals, "V_G2_bias": V_G2_vals,
    "dt_s": dt_vals, "g_in": g_in_vals,
}, indent=2))

# Per-node cell lists
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
            "--subsample", "5000", "1000",
        ]))
    out_path = out_root / f"node_{node}.txt"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  {node}: {len(lines)} cells → {out_path}")
