"""z243 — ESN control on NARMA-10 (Option A from USER_DECISIONS_PENDING).

Critical attribution test: NS-RAM hit NRMSE 0.612 ± 0.030 (z223 30-seed
CI) on NARMA-10 at N=200. Does a textbook tanh ESN at the same N
beat us, match us, or lose? This is the only remaining concrete
reservoir-quality claim NS-RAM has.

Setup: same NARMA-10 generation as z223. Replace NS-RAM body-state
surrogate with tanh ESN at matched N=200 (CPU is fine, same as z223).
Linear ridge readout, 30 seeds for direct comparability.

Acceptance gates:
  ESN NRMSE ≥ 0.612 + 0.030 (worse, outside z223 CI) → NS-RAM wins
  ESN NRMSE ≤ 0.612 − 0.030 (better, outside z223 CI) → ESN wins
  Within z223 CI → tie

Either outcome is informative for Mario v2:
  Win/tie → NS-RAM has a defensible reservoir-quality claim on NARMA
  Loss → brief shrinks to pure energy + physics-credibility
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
OUT = ROOT / "results/z243_narma_esn"; OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "live.log"


def log_line(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def gen_narma10(T, seed):
    """Same NARMA-10 sequence as z223."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, T)
    y = np.zeros(T)
    for k in range(10, T-1):
        y[k+1] = 0.3*y[k] + 0.05*y[k]*y[k-9:k+1].sum() + 1.5*u[k-9]*u[k] + 0.1
    return u, y


def make_sparse_esn_W(N, density=0.10, spectral_radius=0.9, seed=0):
    """Sparse random W spectral-normalized."""
    rng = np.random.default_rng(seed)
    W = (rng.random((N, N)) < density) * rng.normal(0, 1, (N, N))
    W = W.astype(np.float64)
    np.fill_diagonal(W, 0)
    eig = np.abs(np.linalg.eigvals(W)).max()
    if eig > 1e-9:
        W *= spectral_radius / eig
    return W


def run_esn(u, N=200, seed=0, leak=0.30, g_in=1.0):
    """Standard tanh ESN with leaky integration."""
    rng = np.random.default_rng(seed + 1000)
    W = make_sparse_esn_W(N, density=0.10, spectral_radius=0.9, seed=seed)
    W_in = rng.normal(0, 1.0, N)
    state = np.zeros((len(u), N))
    s = np.zeros(N)
    for t in range(len(u)):
        new = np.tanh(W @ s + g_in * W_in * u[t])
        s = (1 - leak) * s + leak * new
        state[t] = s
    return state


def main():
    log_line(f"=== z243 ESN control on NARMA-10 (vs z223 NSRAM 0.612±0.030) ===")
    log_line(f"Setup: N=200 (matches z223), 30 seeds, ridge readout")

    N = 200
    leak = 0.30
    g_in = 1.0  # standard ESN scaling
    T_total, washout, T_train = 1500, 300, 1000
    SEEDS = list(range(30))

    results = []
    t_start = time.time()
    for s in SEEDS:
        if time.time() - t_start > 30 * 60:
            log_line(f"BUDGET REACHED at s{s}")
            break
        fp = OUT / f"seed{s}.json"
        if fp.exists():
            results.append(json.loads(fp.read_text()))
            continue
        u, y = gen_narma10(T_total, s)
        state = run_esn(u, N=N, seed=s, leak=leak, g_in=g_in)
        # Add bias column
        X = np.hstack([state, np.ones((state.shape[0], 1))])
        Xt = X[washout:T_train]; yt = y[washout:T_train]
        Xv = X[T_train:];        yv = y[T_train:]
        # Ridge
        XtX = Xt.T @ Xt + 1e-4 * np.eye(X.shape[1])
        Xty = Xt.T @ yt
        w = np.linalg.solve(XtX, Xty)
        pred_v = Xv @ w
        nrmse = float(np.sqrt(((pred_v - yv)**2).mean()) / yv.std())

        r = {"seed": s, "N": N, "leak": leak, "g_in": g_in, "nrmse": nrmse}
        fp.write_text(json.dumps(r, indent=2))
        results.append(r)
        if s < 5 or s % 5 == 0:
            log_line(f"  s{s}: ESN NRMSE = {nrmse:.4f}")

    if results:
        nrmses = np.array([r["nrmse"] for r in results])
        rng2 = np.random.default_rng(0)
        boots = np.array([nrmses[rng2.integers(0, len(nrmses), len(nrmses))].mean()
                            for _ in range(5000)])
        ci_lo, ci_hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))

        # Comparison vs z223 (NS-RAM): 0.612 ± 0.030 (CI [0.601, 0.624])
        nsram_mean = 0.6122
        nsram_ci_lo = 0.6010
        nsram_ci_hi = 0.6235
        esn_mean = float(nrmses.mean())

        if ci_hi < nsram_ci_lo:
            verdict = "ESN_WINS (better, outside NS-RAM CI)"
        elif ci_lo > nsram_ci_hi:
            verdict = "NS-RAM_WINS (NS-RAM better, ESN worse)"
        else:
            verdict = "TIE (CIs overlap)"

        diff = esn_mean - nsram_mean
        summary = {
            "n_seeds": len(results),
            "task": "NARMA-10 ESN N=200",
            "esn_mean": esn_mean,
            "esn_std": float(nrmses.std()),
            "ci95_mean": [ci_lo, ci_hi],
            "nsram_z223_mean": nsram_mean,
            "nsram_z223_ci": [nsram_ci_lo, nsram_ci_hi],
            "esn_minus_nsram": diff,
            "verdict": verdict,
        }
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
        log_line(f"\n=== n={len(results)} ===")
        log_line(f"ESN  NRMSE: {esn_mean:.4f} ± {nrmses.std():.4f}  CI [{ci_lo:.4f}, {ci_hi:.4f}]")
        log_line(f"NSRAM (z223): 0.6122 ± 0.030  CI [0.601, 0.624]")
        log_line(f"ESN − NSRAM: {diff:+.4f}")
        log_line(f"\nVERDICT: {verdict}")
        log_line(f"  Implications for Mario v2:")
        if "ESN_WINS" in verdict:
            log_line(f"  → NS-RAM ALSO loses on NARMA-10. Brief shrinks to ENERGY+PHYSICS only.")
        elif "NS-RAM_WINS" in verdict:
            log_line(f"  → NS-RAM HAS a defensible reservoir-quality claim on NARMA-10.")
        else:
            log_line(f"  → NS-RAM ties ESN on NARMA-10. Reservoir claim defensible at NARMA scope.")


if __name__ == "__main__":
    main()
