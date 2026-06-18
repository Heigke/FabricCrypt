"""z253 — STEP E of NEXT_DIRECTION_PLAN: NS-RAM hyperparam sweep on NARMA-10.

Mirrors z252 ESN-fairness sweep but for NS-RAM. Tests whether NS-RAM's
NRMSE 0.612 baseline could improve with hyperparameter tuning.

Sweep on NARMA-10 at N=200, 5 seeds:
  g_VG2 ∈ {0.10, 0.20, 0.30}   (input gain on V_G2; z241 showed 0.30 ≈ best on MNIST)
  leak  ∈ {0.10, 0.30, 0.60}
  dt    ∈ {1e-7, 5e-7, 5e-6}    (covers shorter and longer than body-RC tau ~1ms)
= 27 configs × 5 seeds = 135 runs.

ESN reference (best from z252 sweep): NRMSE 0.461.
NS-RAM default reference: 0.612 (z223 30-seed CI).

Pre-registered gate: any NS-RAM config that beats the BEST-tuned ESN
(0.461) would be a real signal. More realistic: does NS-RAM tuning
improve on its default 0.612?
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results/z253_nsram_hyperparam_sweep"; OUT.mkdir(parents=True, exist_ok=True)
SURR_PATH = ROOT / "results/z220_4d_dense/surrogate_4d_dense.npz"

from scripts.nsram_surrogate_4d import NSRAMSurrogate4D
from z249_nsram_vs_esn_scaling import gen_narma10, run_nsram


def main():
    print(f"=== z253 NS-RAM hyperparam sweep on NARMA-10 (N=200) ===", flush=True)
    print(f"References: best-tuned ESN 0.461 (z252), NS-RAM default 0.612 (z223)",
          flush=True)
    surr = NSRAMSurrogate4D(SURR_PATH)
    gVG2_vals = [0.10, 0.20, 0.30]
    leak_vals = [0.10, 0.30, 0.60]
    dt_vals = [1e-7, 5e-7, 5e-6]
    seeds = [0, 1, 2, 3, 4]
    T = 1500
    NSRAM_DEFAULT = 0.612
    BEST_ESN = 0.461

    configs = []
    t0 = time.time()
    for g_VG2 in gVG2_vals:
        for leak in leak_vals:
            for dt in dt_vals:
                vals = []
                for s in seeds:
                    u, y = gen_narma10(T, s)
                    r = run_nsram(surr, u, y, N=200, seed=s,
                                    leak=leak, g_VG2=g_VG2, dt=dt)
                    vals.append(r)
                m = float(np.mean(vals)); sd = float(np.std(vals))
                vs_default = "BETTER than default" if m < NSRAM_DEFAULT else "worse than default"
                vs_esn = "BEATS best ESN" if m < BEST_ESN else "loses to best ESN"
                print(f"  g_VG2={g_VG2:.2f} leak={leak:.2f} dt={dt:.0e}  "
                      f"NS-RAM={m:.4f}±{sd:.3f}  ({vs_default}, {vs_esn})",
                      flush=True)
                configs.append({"g_VG2": g_VG2, "leak": leak, "dt": dt,
                                  "nsram_mean": m, "nsram_std": sd,
                                  "better_than_default": bool(m < NSRAM_DEFAULT),
                                  "beats_best_esn": bool(m < BEST_ESN)})
    wall = time.time() - t0
    best = min(configs, key=lambda c: c["nsram_mean"])
    n_better_default = sum(1 for c in configs if c["better_than_default"])
    n_beats_esn = sum(1 for c in configs if c["beats_best_esn"])
    summary = {
        "n_configs": len(configs),
        "default_ref": NSRAM_DEFAULT, "best_esn_ref": BEST_ESN,
        "best_nsram_config": best,
        "n_better_than_default": n_better_default,
        "n_beats_best_esn": n_beats_esn,
        "configs": configs,
        "wall_s": wall,
        "interpretation": (
            f"Best NS-RAM config: NRMSE {best['nsram_mean']:.4f} "
            f"(g_VG2={best['g_VG2']}, leak={best['leak']}, dt={best['dt']:.0e}). "
            f"{n_better_default}/{len(configs)} beat default 0.612. "
            f"{n_beats_esn}/{len(configs)} beat best-tuned ESN 0.461. "
            f"{'NS-RAM has tuning headroom but still cannot match best ESN.' if n_beats_esn == 0 else 'NS-RAM beats best ESN at some configs.'}"
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== Summary ===", flush=True)
    print(f"Best NS-RAM: NRMSE {best['nsram_mean']:.4f} "
          f"(g_VG2={best['g_VG2']}, leak={best['leak']}, dt={best['dt']:.0e})",
          flush=True)
    print(f"Better than default: {n_better_default}/{len(configs)}", flush=True)
    print(f"Beats best-tuned ESN: {n_beats_esn}/{len(configs)}", flush=True)
    print(summary["interpretation"], flush=True)


if __name__ == "__main__":
    main()
