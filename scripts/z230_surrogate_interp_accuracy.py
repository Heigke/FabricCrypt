"""z230 — R-track step 2: surrogate INTERPOLATION accuracy.

z229 mismatched 6.3 dec because solve_2t_steady_state (joint Newton)
diverges at reservoir biases, while the surrogate uses _solve_at_fixed_vb
(Vsint-only Newton with Vb as parameter).

Correct apples-to-apples test: at 32 OFF-GRID biases, compare surrogate
interp value to direct _solve_at_fixed_vb call. This measures the
interpolation error of the surrogate, NOT the joint-solver gap.

If max log10|Id| error < 0.5 dec at off-grid biases, the surrogate is
a faithful interpolant of the underlying physics-Newton (which IS
convergent at these biases per the 96.7% grid converged map).

CPU-only. ~30s wall.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/z230_surr_interp"; OUT.mkdir(parents=True, exist_ok=True)
SURR_PATH = ROOT / "results/z220_4d_dense/surrogate_4d_dense.npz"


def main():
    sys.path.insert(0, str(ROOT / "nsram"))
    from scripts.nsram_surrogate_4d import (
        NSRAMSurrogate4D, _solve_at_fixed_vb, _build_pyport_models,
    )

    surr = NSRAMSurrogate4D(SURR_PATH)
    cfg, M1, M2, bjt = _build_pyport_models()

    # 32 OFF-grid reservoir-typical biases. Use the SAME ranges the
    # reservoir actually visits (from z221+).
    rng = np.random.default_rng(42)
    N = 32
    VG1 = rng.uniform(0.15, 0.65, N)   # away from grid edges
    VG2 = rng.uniform(0.05, 0.55, N)
    Vd  = np.full(N, 1.0)
    # Vb chosen at typical reservoir steady state (~0.15-0.30 V observed)
    Vb  = rng.uniform(0.10, 0.30, N)

    print(f"=== z230 R-track step 2: surrogate interp vs _solve_at_fixed_vb, N={N} ===")

    log_Id_surr, _, _ = surr.eval(VG1, VG2, Vd, Vb)
    Id_pyp = np.zeros(N)
    conv_pyp = np.zeros(N, dtype=bool)
    t0 = time.time()
    for i in range(N):
        out = _solve_at_fixed_vb(cfg, M1, M2, bjt,
                                   Vd[i], VG1[i], VG2[i], Vb[i])
        Id_pyp[i] = out["Id"]
        conv_pyp[i] = out["converged"]
    wall = time.time() - t0
    print(f"pyport (_solve_at_fixed_vb) wall: {wall:.2f}s; converged {conv_pyp.sum()}/{N}")

    log_Id_pyp = np.log10(np.maximum(np.abs(Id_pyp), 1e-15))
    delta_dec = np.abs(log_Id_pyp - log_Id_surr)
    rms = float(np.sqrt((delta_dec ** 2).mean()))
    p95 = float(np.quantile(delta_dec, 0.95))
    mx  = float(delta_dec.max())

    print(f"\nlog10|Id|  surrogate (interp) vs _solve_at_fixed_vb (direct):")
    print(f"  RMS    : {rms:.3f}")
    print(f"  P95    : {p95:.3f}")
    print(f"  MAX    : {mx:.3f}  (gate: <0.50 PASS)")
    print(f"  GATE   : {'✅ PASS' if mx < 0.50 else '❌ FAIL'}")

    # converged-only stats
    if conv_pyp.any():
        d_c = delta_dec[conv_pyp]
        print(f"\nConverged-only ({conv_pyp.sum()} pts):")
        print(f"  RMS={float(np.sqrt((d_c**2).mean())):.3f}  MAX={float(d_c.max()):.3f}")

    # Top 5 worst
    print(f"\nWorst 5 biases:")
    print(f"  i  VG1   VG2    Vb     log_pyp  log_surr  Δdec  conv")
    for k in np.argsort(-delta_dec)[:5]:
        print(f"  {int(k):2d}  {VG1[k]:.3f} {VG2[k]:.3f}  {Vb[k]:.3f}  "
              f"{log_Id_pyp[k]:>7.3f}  {log_Id_surr[k]:>7.3f}  "
              f"{delta_dec[k]:.3f}  {bool(conv_pyp[k])}")

    out = {
        "N": N, "rms_dec": rms, "p95_dec": p95, "max_dec": mx,
        "gate_pass": bool(mx < 0.50),
        "wall_s": wall,
        "converged": int(conv_pyp.sum()),
        "rms_dec_conv_only": (float(np.sqrt((delta_dec[conv_pyp]**2).mean()))
                              if conv_pyp.any() else None),
        "max_dec_conv_only": (float(delta_dec[conv_pyp].max())
                              if conv_pyp.any() else None),
    }
    (OUT / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
