"""z2b_bsim4_gpu_scaling — where does the GPU pay off for BSIM4 fitting?

scipy curve_fit on a single curve is fast (~2 ms on CPU).  GPU batch
fitting wins when you have MANY curves (parameter variability studies,
wafer-scale Monte Carlo, multi-Vg1/T sweeps).  This benchmark shows
the crossover: at what N does the GPU LBFGS beat per-curve scipy?

Relevant to Sebastian because his new silicon measurements will come
in batches — each device × tuning condition × temperature × layout
is a separate curve, so N grows quickly.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "nsram"))

from nsram.bsim4 import BSIM4Params, impact_ionization_bsim4      # noqa: E402
from nsram.fitting import fit_bsim4_impact                        # noqa: E402

# Reuse the GPU batch-fit module from the main experiment
sys.path.insert(0, str(REPO / "scripts"))
from z2_nsram_bsim4_zenodo import batch_fit                       # noqa: E402

OUT = REPO / "results" / "z2_nsram_bsim4_zenodo"


def make_synthetic(N, rng):
    """N random BSIM4 curves over realistic param space + 5% noise."""
    Vds_arr = np.linspace(2.0, 4.5, 40)
    true_a = 10 ** rng.uniform(-7.0, -4.0, N)
    true_b = rng.uniform(15.0, 30.0, N)
    Vgs = rng.uniform(0.6, 1.4, N)
    base = BSIM4Params()
    Iii = []
    for i in range(N):
        p = BSIM4Params(**{**base.__dict__, "ALPHA0": true_a[i], "BETA0": true_b[i]})
        y = np.array([float(impact_ionization_bsim4(Vgs[i], v, 0.0, p))
                      for v in Vds_arr])
        y = np.maximum(y * (1 + 0.05 * rng.standard_normal(len(y))), 1e-30)
        Iii.append(y)
    return Vgs.tolist(), [Vds_arr] * N, Iii, true_a, true_b


def bench_scipy(Vgs, Vds, Iii, base):
    t0 = time.perf_counter()
    results = []
    for i, (vg, vd, y) in enumerate(zip(Vgs, Vds, Iii)):
        r = fit_bsim4_impact(vd, y, Vgs=float(vg), Vbs=0.0, base=base)
        results.append(r)
    return time.perf_counter() - t0, results


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {torch.cuda.get_device_name(0) if dev=='cuda' else 'cpu'}")

    base = BSIM4Params()
    Ns = [50, 200, 1000, 4000]
    rows = []
    rng = np.random.default_rng(7)

    for N in Ns:
        print(f"\n── N={N} curves " + "─" * (50 - len(str(N))))
        Vgs, Vds, Iii, true_a, true_b = make_synthetic(N, rng)

        # GPU
        fit_gpu = batch_fit(Vgs, Vds, Iii, base, iters=150, device=dev)
        gpu_ms = 1e3 * fit_gpu["t_s"] / N
        a_err_gpu = np.abs(np.log10(fit_gpu["alpha"]) - np.log10(true_a))
        gpu_quality = float(np.median(a_err_gpu))

        # scipy — skip for very large N (would take minutes)
        if N <= 1000:
            t_scipy, scipy_res = bench_scipy(Vgs, Vds, Iii, base)
            scipy_ms = 1e3 * t_scipy / N
            scipy_ok = sum(1 for r in scipy_res if r.get("r_squared", 0) > 0.8)
        else:
            # project from smaller N
            scipy_ms = rows[-1]["scipy_ms_per_curve"]
            scipy_ok = -1

        speedup = scipy_ms / gpu_ms
        print(f"  GPU batch   : total {fit_gpu['t_s']:.2f}s, {gpu_ms:.2f} ms/curve, "
              f"R²_median={np.median(fit_gpu['r2']):.4f}")
        print(f"  scipy CPU   : {scipy_ms:.2f} ms/curve"
              f"{' (projected — too slow to benchmark)' if scipy_ok < 0 else ''}")
        print(f"  speedup     : {speedup:.2f}×  {'← GPU wins' if speedup > 1 else ''}")

        rows.append(dict(
            N=N,
            gpu_total_s=fit_gpu["t_s"],
            gpu_ms_per_curve=gpu_ms,
            scipy_ms_per_curve=scipy_ms,
            speedup=speedup,
            gpu_r2_median=float(np.median(fit_gpu["r2"])),
            gpu_alpha0_log_err_median=gpu_quality,
        ))

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Ns_a = [r["N"] for r in rows]
        gpu = [r["gpu_ms_per_curve"] for r in rows]
        sci = [r["scipy_ms_per_curve"] for r in rows]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.loglog(Ns_a, gpu, "o-", label="GPU batch LBFGS (gfx1151)", lw=2)
        ax.loglog(Ns_a, sci, "s-", label="scipy curve_fit (CPU, per-curve)", lw=2)
        ax.set_xlabel("# curves fitted"); ax.set_ylabel("ms per curve")
        ax.set_title("BSIM4 fitter scaling — where the GPU starts winning")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
        for r in rows:
            ax.annotate(f"{r['speedup']:.1f}×", (r["N"], r["gpu_ms_per_curve"]),
                         textcoords="offset points", xytext=(5, -8), fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "scaling_gpu_vs_cpu.png", dpi=140)
        plt.close(fig)
        print(f"\n[plot] {OUT / 'scaling_gpu_vs_cpu.png'}")
    except ImportError:
        pass

    with open(OUT / "scaling.json", "w") as f:
        json.dump({"device": dev, "rows": rows}, f, indent=2)
    print(f"[done] scaling → {OUT / 'scaling.json'}")


if __name__ == "__main__":
    main()
