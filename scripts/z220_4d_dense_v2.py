"""z220 — Path A v2: denser Vb axis + diagnostic.

Per z219 finding (MC=2.5, baseline 1.0): direction validated, but
need denser Vb sampling around parasitic-NPN turn-on (0.55-0.65 V).

Build improved 4D surrogate:
  Vb axis: 10 points (was 5) focused 0.4-0.7 V where Id changes dramatically
  Other axes unchanged (5×5×4)
  = 1000 op-points, ~20s build time

Then run two diagnostics:
  1. Vb-sensitivity: at fixed (VG1=0.4, VG2=0.2, Vd=1.0), how does
     log_Id vary with Vb? Need >1 dec spread for body to matter.
  2. MC re-test: same Cb/dt sweep as z219, see if denser axis lifts MC.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results/z220_4d_dense"; OUT.mkdir(parents=True, exist_ok=True)


def main():
    # Override the 4D-surrogate axis defaults via monkeypatch
    import scripts.nsram_surrogate_4d as m4d
    # Denser Vb axis focused on parasitic-NPN turn-on
    m4d.VB_AXIS = np.array([0.00, 0.20, 0.40, 0.50, 0.55, 0.58, 0.60,
                             0.62, 0.65, 0.70], dtype=np.float64)

    out_path = OUT / "surrogate_4d_dense.npz"
    if not out_path.exists():
        print(f"[z220] building dense 4D surrogate to {out_path}")
        print(f"       Vb axis = {m4d.VB_AXIS}")
        m4d.build_4d_grid(out_path, verbose=True)
    else:
        print(f"[z220] surrogate exists at {out_path}")

    surr = m4d.NSRAMSurrogate4D(out_path)

    # DIAGNOSTIC 1: Vb-sensitivity at fixed bias
    print("\n=== Vb-sensitivity at (VG1=0.4, VG2=0.2, Vd=1.0) ===")
    print(f"  {'Vb':>6}  {'log_Id':>10}  {'Iii':>10}  {'Ileak':>10}  {'I_net':>10}")
    for vb in np.linspace(0.0, 0.7, 15):
        log_Id, Iii, Ileak = surr.eval(0.4, 0.2, 1.0, vb)
        net = Iii - Ileak
        print(f"  {vb:>4.2f}    {log_Id:>10.4f}  {Iii:>10.2e}  {Ileak:>10.2e}  {net:>+10.2e}")

    # DIAGNOSTIC 2: same at (VG1=0.6, VG2=0.3, Vd=1.5) — strong-on snapback
    print("\n=== Vb-sensitivity at (VG1=0.6, VG2=0.3, Vd=1.5) [snapback regime] ===")
    print(f"  {'Vb':>6}  {'log_Id':>10}  {'Iii':>10}  {'Ileak':>10}  {'I_net':>10}")
    for vb in np.linspace(0.0, 0.7, 15):
        log_Id, Iii, Ileak = surr.eval(0.6, 0.3, 1.5, vb)
        net = Iii - Ileak
        print(f"  {vb:>4.2f}    {log_Id:>10.4f}  {Iii:>10.2e}  {Ileak:>10.2e}  {net:>+10.2e}")

    # MC re-test
    print("\n=== MC test on dense 4D surrogate ===")
    from scripts.z219_mc_4d import memory_capacity_4d
    sweep = [(1.0e-15, 1e-7), (5.0e-15, 1e-6), (5.0e-15, 1e-7),
              (10.0e-15, 1e-6), (20.0e-15, 1e-6)]
    print(f"  {'Cb (fF)':>9}  {'dt (s)':>9}  {'MC_total':>12}  {'early':>7}  {'mid':>7}  {'late':>7}")
    out = []
    for Cb, dt in sweep:
        mcs = []
        full = []
        for s in [0, 1, 2]:
            mc = memory_capacity_4d(200, s, surr, Cb, dt)
            mcs.append(mc.sum())
            full.append(mc)
        full = np.stack(full).mean(axis=0)
        mc_mean = np.mean(mcs); mc_std = np.std(mcs)
        early = full[:5].sum()
        mid = full[5:15].sum() if len(full) >= 15 else 0
        late = full[15:30].sum() if len(full) >= 30 else 0
        print(f"  {Cb*1e15:>7.1f}    {dt:>9.0e}    "
              f"{mc_mean:>5.2f}±{mc_std:.2f}  {early:>7.2f}  {mid:>7.2f}  {late:>7.2f}")
        out.append({"Cb": Cb, "dt": dt, "mc_mean": float(mc_mean),
                     "mc_std": float(mc_std),
                     "early": float(early), "mid": float(mid), "late": float(late)})

    print(f"\nz219 best (5-pt Vb): MC = 2.49")
    print(f"z220 (10-pt Vb): see above. Lift = quality of dense interp.")
    (OUT / "summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
