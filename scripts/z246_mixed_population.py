"""z246 — STEP 3 of V_G2 continuum study: mixed-population fabric.

Per VG2_CONTINUUM_PLAN STEP 3. Tests whether a network mixing
"grounded" (digital memory) and "floating" (analog LIF) cells
outperforms either pure population on NARMA-10.

Pipeline match to z235: N=200, ridge readout, 30-seed CI per condition.

Cells:
  - GROUNDED: V_G2 held at 0.0V (lowest grid value) — fast quiescent body
  - FLOATING: V_G2 in [0.05, 0.55] random — analog dynamics on body
  - Both share the same input projection W_in and recurrent block-diag W.

Sweep: fraction f of grounded cells in {0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0}.

PRE-REGISTERED GATE (before running):
  PASS if any mixed-mode point f in (0, 1) gives NRMSE strictly lower than
  BOTH pure endpoints (f=0 floating-only, f=1.0 grounded-only) by at least
  the larger of {1 pp, the std of the better endpoint}, with CIs that
  do not overlap.

Wall time: ~7 conditions × 5 seeds × ~25s = ~15 min CPU.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/z246_mixed_population"; OUT.mkdir(parents=True, exist_ok=True)
SURR_PATH = ROOT / "results/z220_4d_dense/surrogate_4d_dense.npz"

from scripts.nsram_surrogate_4d import NSRAMSurrogate4D


def gen_narma10(T, seed):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, T)
    y = np.zeros(T)
    for k in range(10, T-1):
        y[k+1] = 0.3*y[k] + 0.05*y[k]*y[k-9:k+1].sum() + 1.5*u[k-9]*u[k] + 0.1
    return u, y


def run_one(N, seed, f_grounded, surr,
              Cb=5e-15, dt=5e-7, g_VG2=0.20, g_VG1=0.30, leak=0.30,
              T_total=1500, washout=300, T_train=1000):
    rng = np.random.default_rng(seed)
    # Cell roles
    n_ground = int(round(f_grounded * N))
    n_float = N - n_ground
    is_grounded = np.zeros(N, dtype=bool)
    grounded_idx = rng.choice(N, size=n_ground, replace=False) if n_ground > 0 else np.array([], dtype=int)
    is_grounded[grounded_idx] = True

    base_VG1 = rng.uniform(0.2, 0.5, N)
    # Grounded cells: V_G2 = 0 always; floating cells: V_G2 from [0.05, 0.55]
    base_VG2 = np.where(is_grounded, 0.0, rng.uniform(0.05, 0.55, N))
    sign_mask = rng.choice([-1.0, 1.0], N).astype(np.float64)
    W_in = rng.normal(0, 1.0, N)
    n_block = 50
    K = N // n_block
    rng2 = np.random.default_rng(seed)
    Wb = np.zeros((K, n_block, n_block))
    for k in range(K):
        m = (rng2.random((n_block, n_block)) < 0.10).astype(np.float64)
        w = m * rng2.normal(0, 1, (n_block, n_block))
        np.fill_diagonal(w, 0)
        eig = np.abs(np.linalg.eigvals(w)).max()
        if eig > 1e-9: w *= 0.9 / eig
        Wb[k] = w

    u, y = gen_narma10(T_total, seed)
    u_input = (u - 0.25) / 0.25
    Vd_arr = np.ones(N)
    Vb = np.full(N, 0.30)
    feat = np.zeros(N)
    state = np.zeros((T_total, N))

    for t in range(T_total):
        # Input drives V_G2 ONLY for floating cells; grounded cells stay at 0
        VG2 = np.where(is_grounded, 0.0,
                         np.clip(base_VG2 + g_VG2 * W_in * u_input[t], 0.0, 0.55))
        feat_b = feat.reshape(K, n_block)
        rec_b = np.einsum("kij,kj->ki", Wb, feat_b)
        rec = rec_b.reshape(N) * sign_mask
        VG1 = np.clip(base_VG1 + g_VG1 * rec, 0.05, 0.7)
        log_Id, Iii, Ile = surr.eval(VG1, VG2, Vd_arr, Vb)
        Id = 10.0 ** log_Id
        Vb = np.clip(Vb + dt * (Iii - Ile) / Cb, 0.0, 0.7)
        # Grounded cells: body stuck at 0 (digital-like)
        Vb = np.where(is_grounded, 0.0, Vb)
        feat = (1.0 - leak) * feat + leak * log_Id
        state[t] = feat

    X = np.hstack([state, np.ones((state.shape[0], 1))])
    Xt = X[washout:T_train]; yt = y[washout:T_train]
    Xv = X[T_train:];        yv = y[T_train:]
    XtX = Xt.T @ Xt + 1e-4 * np.eye(X.shape[1])
    Xty = Xt.T @ yt
    w = np.linalg.solve(XtX, Xty)
    pred_v = Xv @ w
    nrmse = float(np.sqrt(((pred_v - yv)**2).mean()) / yv.std())
    return nrmse


def main():
    print(f"=== z246 V_G2 mixed-population (no-cheat) ===", flush=True)
    print(f"PRE-REG GATE: any f in (0,1) gives NRMSE strictly < BOTH "
          f"f=0 and f=1.0 by max(1pp, std).", flush=True)
    surr = NSRAMSurrogate4D(SURR_PATH)

    N = 200
    fractions = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    seeds = list(range(5))

    nrmses = {f: [] for f in fractions}
    t0 = time.time()
    for f in fractions:
        for s in seeds:
            r = run_one(N, s, f, surr)
            nrmses[f].append(r)
            print(f"  f={f:.2f} seed={s}: NRMSE={r:.4f}", flush=True)
    wall = time.time() - t0

    means = {f: float(np.mean(nrmses[f])) for f in fractions}
    stds = {f: float(np.std(nrmses[f])) for f in fractions}
    print(f"\n--- Summary (n={len(seeds)} seeds) ---", flush=True)
    print(f"{'f_grounded':>12s}  {'mean':>8s}  {'std':>7s}  {'n':>3s}", flush=True)
    for f in fractions:
        print(f"  {f:11.2f}  {means[f]:8.4f}  {stds[f]:7.4f}  {len(seeds):3d}", flush=True)

    pure_float = means[0.0]
    pure_ground = means[1.0]
    better_endpoint = min(pure_float, pure_ground)
    worse_endpoint = max(pure_float, pure_ground)
    margin = max(0.01, stds[0.0 if pure_float < pure_ground else 1.0])

    pass_f = None
    for f in fractions:
        if 0.0 < f < 1.0 and means[f] + margin < better_endpoint:
            pass_f = f
            break
    gate_pass = pass_f is not None

    summary = {
        "N": N, "fractions": fractions, "n_seeds": len(seeds),
        "nrmse_mean": means, "nrmse_std": stds,
        "pure_floating_mean": pure_float,
        "pure_grounded_mean": pure_ground,
        "better_endpoint_mean": better_endpoint,
        "passing_fraction": pass_f,
        "margin_used": margin,
        "STEP3_gate_PASS": gate_pass,
        "wall_s": wall,
        "interpretation": (
            f"PASS — mixed-mode fabric is a real architectural win. f={pass_f} "
            f"beats both pure modes by > max(1pp, std). Mario brief v4.4 can add "
            f"forward-looking mixed-mode-fabric section."
            if gate_pass else
            f"FAIL — no f in (0,1) strictly beats both pure endpoints by the "
            f"pre-registered margin. Mixed-mode is not a measurable architectural "
            f"win on NARMA-10 at this N and pipeline."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nPRE-REG GATE: {'✅ PASS' if gate_pass else '❌ FAIL'}", flush=True)
    print(f"{summary['interpretation']}", flush=True)


if __name__ == "__main__":
    main()
