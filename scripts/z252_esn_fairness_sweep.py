"""z252 — ESN fairness sweep (O40 openai+grok optional follow-up).

Question: was the ESN baseline (sparse 10%, ρ=0.9, leak=0.30,
input gain 1.0) "accidentally over-tuned" and so an unfair benchmark?

Sweep ESN hyperparameters on NARMA-10 at N=200, 5 seeds each:
  spectral radius ∈ {0.50, 0.70, 0.90, 1.10}
  leak             ∈ {0.10, 0.30, 0.60}
  input gain       ∈ {0.30, 1.00, 3.00}
= 4×3×3 = 36 configs × 5 seeds = 180 runs at ~0.5s each = ~90s.

NS-RAM frozen reference: NRMSE 0.612 ± 0.030 (z223 30-seed CI).

Reports: fraction of ESN configs that fall ABOVE NS-RAM (i.e. NS-RAM
wins against that detuned ESN), and the worst ESN config. If even
poorly-tuned ESNs beat NS-RAM, the negative pattern is robust to
ESN-tuning concerns.
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
OUT = ROOT / "results/z252_esn_fairness_sweep"; OUT.mkdir(parents=True, exist_ok=True)

from z249_nsram_vs_esn_scaling import gen_narma10


def run_esn(u, y, N=200, seed=0, leak=0.30, sr=0.9, gain=1.0,
              washout=300, T_train=1000):
    rng = np.random.default_rng(seed + 1000)
    W = (rng.random((N, N)) < 0.10) * rng.normal(0, 1, (N, N))
    np.fill_diagonal(W, 0)
    eig = np.abs(np.linalg.eigvals(W)).max()
    if eig > 1e-9: W *= sr / eig
    W_in = rng.normal(0, 1.0, N)
    T = len(u)
    s = np.zeros(N)
    state = np.zeros((T, N))
    for t in range(T):
        s = (1-leak)*s + leak*np.tanh(W @ s + gain * W_in * u[t])
        state[t] = s
    X = np.hstack([state, np.ones((state.shape[0], 1))])
    Xt = X[washout:T_train]; yt = y[washout:T_train]
    Xv = X[T_train:];        yv = y[T_train:]
    XtX = Xt.T @ Xt + 1e-4 * np.eye(X.shape[1])
    w = np.linalg.solve(XtX, Xt.T @ yt)
    return float(np.sqrt(((Xv @ w - yv)**2).mean()) / yv.std())


def main():
    print(f"=== z252 ESN fairness sweep on NARMA-10 (N=200) ===", flush=True)
    print(f"NS-RAM reference: NRMSE 0.612 ± 0.030 (z223 30-seed CI)", flush=True)
    sr_vals   = [0.50, 0.70, 0.90, 1.10]
    leak_vals = [0.10, 0.30, 0.60]
    gain_vals = [0.30, 1.00, 3.00]
    seeds = [0, 1, 2, 3, 4]
    T = 1500
    NSRAM_REF = 0.612
    NSRAM_CI_HI = 0.624

    configs = []
    t0 = time.time()
    for sr in sr_vals:
        for leak in leak_vals:
            for gain in gain_vals:
                vals = []
                for s in seeds:
                    u, y = gen_narma10(T, s)
                    r = run_esn(u, y, N=200, seed=s, leak=leak, sr=sr, gain=gain)
                    vals.append(r)
                m = float(np.mean(vals)); sd = float(np.std(vals))
                nsram_wins = bool(m > NSRAM_CI_HI)  # ESN strictly worse than NS-RAM
                tag = "  NS-RAM wins" if nsram_wins else ("** ESN beats NS-RAM" if m < NSRAM_REF else "  ESN ≈ NS-RAM")
                print(f"  sr={sr:.2f} leak={leak:.2f} gain={gain:.2f}  ESN={m:.4f}±{sd:.3f}  {tag}",
                      flush=True)
                configs.append({"sr": sr, "leak": leak, "gain": gain,
                                  "esn_mean": m, "esn_std": sd,
                                  "nsram_wins_against_this_esn": nsram_wins})
    wall = time.time() - t0

    n_nsram_wins = sum(1 for c in configs if c["nsram_wins_against_this_esn"])
    worst_esn = max(configs, key=lambda c: c["esn_mean"])
    best_esn = min(configs, key=lambda c: c["esn_mean"])
    n_esn_strictly_beats = sum(1 for c in configs if c["esn_mean"] < NSRAM_REF)

    summary = {
        "nsram_ref_NARMA10": NSRAM_REF, "nsram_ci_upper": NSRAM_CI_HI,
        "n_configs_total": len(configs),
        "n_nsram_wins_against_detuned_esn": n_nsram_wins,
        "n_esn_strictly_beats_nsram": n_esn_strictly_beats,
        "worst_esn": worst_esn,
        "best_esn": best_esn,
        "configs": configs,
        "wall_s": wall,
        "interpretation": (
            f"NS-RAM beats detuned ESN at {n_nsram_wins}/{len(configs)} configs. "
            f"Best ESN: {best_esn['esn_mean']:.4f} (sr={best_esn['sr']}, "
            f"leak={best_esn['leak']}, gain={best_esn['gain']}). "
            f"Worst ESN: {worst_esn['esn_mean']:.4f}. "
            f"Even at non-default ESN configs, ESN still beats NS-RAM at "
            f"{n_esn_strictly_beats}/{len(configs)} cells (NRMSE < 0.612)."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== Summary ===", flush=True)
    print(f"NS-RAM beats detuned ESN: {n_nsram_wins}/{len(configs)} configs", flush=True)
    print(f"ESN strictly beats NS-RAM: {n_esn_strictly_beats}/{len(configs)}", flush=True)
    print(f"Best ESN: NRMSE {best_esn['esn_mean']:.4f}", flush=True)
    print(f"Worst ESN: NRMSE {worst_esn['esn_mean']:.4f}", flush=True)
    print(summary["interpretation"], flush=True)


if __name__ == "__main__":
    main()
