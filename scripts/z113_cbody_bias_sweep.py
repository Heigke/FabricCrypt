"""z113 — extend z110 C_body characterization across bias regimes.

z110 ran 3 op-points with one perturbation size. Two issues there:
  - Subthreshold (VG1=0.2/VG2=0.1) and near-threshold (0.4/0.3) both
    gave τ ≈ 2.1 µs — same value, suspicious.
  - Above-threshold (0.6/0.5) gave τ=inf (Vb didn't relax monotonically).

This script bounds the brief's "doubly empirical" Limitations bullet 3
with concrete numbers across bias space:

  - VG1 ∈ {0.2, 0.4, 0.6}, VG2 ∈ {0.0, 0.1, 0.3, 0.5}, Vd = 1.0 V
    (12 op-points).
  - δVb perturbation = 5 mV (small-signal).
  - Returns τ_body, R_internal, and a flag for non-monotonic decay
    (where the simple exponential fit doesn't apply).

Goal: produce a 3×4 grid of τ_body values that directly feeds the
brief's Limitations bullet 3 narrative.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z113_cbody_bias_sweep"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("z110", ROOT / "scripts/z110_cbody_characterization.py")
z110 = importlib.util.module_from_spec(sp); sp.loader.exec_module(z110)


def main():
    t0 = time.time()
    print(f"[z113] C_body bias sweep — 3×4 (VG1, VG2) grid, Vd=1.0 V, δVb=5 mV")
    VG1s = [0.2, 0.4, 0.6]
    VG2s = [0.0, 0.1, 0.3, 0.5]
    grid_tau = np.full((len(VG1s), len(VG2s)), np.nan)
    grid_Vbeq = np.full((len(VG1s), len(VG2s)), np.nan)
    grid_status = np.full((len(VG1s), len(VG2s)), "?", dtype=object)
    detail = []
    for i, VG1 in enumerate(VG1s):
        for j, VG2 in enumerate(VG2s):
            ti = time.time()
            print(f"  VG1={VG1}, VG2={VG2} ...", end="", flush=True)
            try:
                r = z110.run_relaxation(VG1, VG2, 1.0, delta=0.005,
                                         dt=0.005e-9, T=400)
                if r is None:
                    grid_status[i, j] = "fit-fail"
                    print(" fit failed (too few decay points)")
                    continue
                tau = r["tau_body_s"]
                grid_Vbeq[i, j] = r["Vb_eq"]
                if not np.isfinite(tau):
                    grid_status[i, j] = "non-mono"
                    print(f" non-monotonic decay (Vb_eq={r['Vb_eq']:+.3f})")
                    detail.append({"VG1": VG1, "VG2": VG2, **r,
                                    "status": "non-mono"})
                else:
                    grid_tau[i, j] = tau
                    grid_status[i, j] = "ok"
                    print(f" τ={tau*1e9:8.2f} ns  Vb_eq={r['Vb_eq']:+.3f} V "
                           f"({time.time()-ti:.1f}s)")
                    detail.append({"VG1": VG1, "VG2": VG2, **r, "status": "ok"})
            except Exception as e:
                grid_status[i, j] = f"err:{type(e).__name__}"
                print(f" exception: {e}")

    print(f"\n[z113] === τ_body GRID (rows VG1, cols VG2) ===")
    header = "  VG1\\VG2  " + "  ".join(f"{v:>10s}" for v in [f"{x:.2f}" for x in VG2s])
    print(header)
    for i, VG1 in enumerate(VG1s):
        row = f"  {VG1:>7.2f}  "
        for j in range(len(VG2s)):
            tau = grid_tau[i, j]
            st = grid_status[i, j]
            if st == "ok":
                row += f"  {tau*1e9:>8.2f} ns"
            else:
                row += f"  {st:>10s}"
        print(row)

    print(f"\n[z113] === Vb_eq GRID ===")
    print(header)
    for i, VG1 in enumerate(VG1s):
        row = f"  {VG1:>7.2f}  "
        for j in range(len(VG2s)):
            v = grid_Vbeq[i, j]
            row += f"  {v:>+8.3f} V" if np.isfinite(v) else f"  {'  N/A':>10s}"
        print(row)

    # Summary stats over OK points
    ok_taus = grid_tau[~np.isnan(grid_tau)]
    if len(ok_taus) > 0:
        print(f"\n[z113] === SUMMARY (OK points only, n={len(ok_taus)}) ===")
        print(f"  τ_body min  = {ok_taus.min()*1e9:.2f} ns")
        print(f"  τ_body max  = {ok_taus.max()*1e9:.2f} ns")
        print(f"  τ_body median = {np.median(ok_taus)*1e9:.2f} ns")
        print(f"  Range vs Pazos's silicon 0.7 ns:")
        print(f"    pyport min/0.7  = {ok_taus.min()*1e9 / 0.7:.0f}× slower")
        print(f"    pyport max/0.7  = {ok_taus.max()*1e9 / 0.7:.0f}× slower")

    json.dump({"VG1s": VG1s, "VG2s": VG2s, "Vd": 1.0,
                "tau_grid": grid_tau.tolist(),
                "Vbeq_grid": grid_Vbeq.tolist(),
                "status_grid": grid_status.tolist(),
                "detail": detail,
                "wall_s": float(time.time() - t0)},
               (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z113] wall: {time.time()-t0:.1f}s")
    print(f"[z113] saved {OUT}/summary.json")


if __name__ == "__main__":
    main()
