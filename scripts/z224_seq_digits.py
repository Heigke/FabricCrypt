"""z224 — Sequential Digits classification (cross-task on 4D surrogate).

Cross-task generalization test gemini O32/O33 demanded.
Uses sklearn load_digits (1797 × 8x8) as compact MNIST proxy.

Each digit presented row-by-row over 8 timesteps. Reservoir state
(N-dim, log_id features only — no Vb in readout to reduce overfit
risk on small dataset) is fed to multinomial linear classifier
(softmax) trained on cross-entropy.

Frozen z222-best config (Cb=5fF, dt=500ns, leak=0.30, g_VG2=0.05).
Tests whether Path-A win generalizes from regression to classification.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, json, time
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/z224_seq_digits"; OUT.mkdir(parents=True, exist_ok=True)
SURR_PATH = ROOT / "results/z220_4d_dense/surrogate_4d_dense.npz"


def config_key(args):
    name, seed = args
    return f"{name}_s{seed}"


def run_one(args):
    import os as _os
    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        _os.environ[_k] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except Exception: pass
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    import numpy as _np
    from sklearn.datasets import load_digits
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from scripts.nsram_surrogate_4d import NSRAMSurrogate4D
    from scripts.z200_topo_rule_sweep import build_topo

    name, seed = args
    surr = NSRAMSurrogate4D(SURR_PATH)
    rng = _np.random.default_rng(seed)
    N = 200
    Cb = 5e-15
    dt = 5e-7   # z222-best
    leak = 0.30
    g_VG2 = 0.05
    g_VG1 = 0.3

    # Reservoir setup
    base_VG1 = rng.uniform(0.2, 0.5, N).astype(float)
    base_VG2 = rng.uniform(0.05, 0.55, N).astype(float)
    sign_mask = rng.choice([-1.0, 1.0], N).astype(float)
    # Per-cell input mask: random projection from 8-pixel row to N cells
    W_in = rng.normal(0, 1.0 / _np.sqrt(8), size=(N, 8))
    W_rec = build_topo("ER_SPARSE" if name == "baseline" else "ER_SPARSE", N, rng)

    # Data
    digits = load_digits()
    X_imgs = digits.images.astype(_np.float32) / 16.0  # normalize 0-1
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X_imgs, y, train_size=1000, test_size=400, random_state=seed)

    Vd_arr = _np.ones(N)

    def eval_image(img):
        Vb = _np.full(N, 0.30)
        feat = _np.zeros(N)
        for row in img:
            # row is 8-vector; project to N-dim cell input via W_in
            cell_input = W_in @ row   # (N,)
            VG2 = _np.clip(base_VG2 + g_VG2 * cell_input, 0.0, 0.6)
            rec = (W_rec @ feat) * sign_mask
            VG1 = _np.clip(base_VG1 + g_VG1 * rec, 0.05, 0.7)
            log_Id, Iii, Ileak = surr.eval(VG1, VG2, Vd_arr, Vb)
            net = Iii - Ileak
            Vb = _np.clip(Vb + dt * net / Cb, 0.0, 0.7)
            feat = (1.0 - leak) * feat + leak * log_Id
        return feat   # final reservoir state after all 8 rows

    t0 = time.time()
    states_train = _np.array([eval_image(img) for img in X_train])
    states_test  = _np.array([eval_image(img) for img in X_test])
    enc_wall = time.time() - t0

    # Train softmax classifier (multi-class linear ridge essentially)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(states_train, y_train)
    train_acc = float(clf.score(states_train, y_train))
    test_acc  = float(clf.score(states_test, y_test))

    # Random-projection baseline (no reservoir, just W_in @ rows summed)
    def proj_only(img):
        return _np.sum(W_in @ img.T, axis=1)
    Xp_train = _np.array([proj_only(img) for img in X_train])
    Xp_test = _np.array([proj_only(img) for img in X_test])
    clf2 = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf2.fit(Xp_train, y_train)
    proj_test_acc = float(clf2.score(Xp_test, y_test))

    return {
        "name": name, "seed": seed, "N": N,
        "train_acc": train_acc, "test_acc": test_acc,
        "proj_only_test_acc": proj_test_acc,
        "encode_wall_s": enc_wall,
    }


def main():
    from scripts.util_safe_sweep import safe_sweep
    grid = [("baseline", s) for s in range(5)]
    print(f"[z224] {len(grid)} configs (5 seeds, cross-task: 8×8 sequential digits)")

    results = safe_sweep(
        run_fn=run_one, configs=grid, out_dir=OUT,
        config_key=config_key, max_workers=2,
        thermal_pause_c=75.0, thermal_kill_c=88.0,
        per_config_wall_cap_s=180.0,
    )

    print(f"\n=== Sequential Digits (8x8, 1000 train, 400 test) ===")
    print(f"{'seed':>5}  {'train':>7}  {'test':>7}  {'proj-only':>9}  {'wall':>5}")
    for r in sorted(results, key=lambda x: x["seed"]):
        print(f"  {r['seed']:>3}    {r['train_acc']:>7.4f}  {r['test_acc']:>7.4f}  "
              f"{r['proj_only_test_acc']:>9.4f}  {r['encode_wall_s']:>4.0f}s")

    if results:
        ts = np.array([r["test_acc"] for r in results])
        ps = np.array([r["proj_only_test_acc"] for r in results])
        if len(ts) >= 2:
            ci_t = stats.t.interval(0.95, len(ts)-1, loc=ts.mean(), scale=stats.sem(ts))
            ci_p = stats.t.interval(0.95, len(ps)-1, loc=ps.mean(), scale=stats.sem(ps))
            d = ts - ps
            t, p = stats.ttest_rel(ts, ps)
            print(f"\nReservoir test acc:  {ts.mean():.4f} ± {ts.std():.4f}  CI [{ci_t[0]:.4f}, {ci_t[1]:.4f}]")
            print(f"Proj-only test acc:  {ps.mean():.4f} ± {ps.std():.4f}  CI [{ci_p[0]:.4f}, {ci_p[1]:.4f}]")
            print(f"Paired t-test:       t={t:+.2f}  p={p:.4g}  Δ={d.mean():+.4f}")
            print(f"Chance: 0.10 (10 classes)")

    summary = {"n_results": len(results),
                "test_acc_mean": float(ts.mean()) if len(ts) else None,
                "test_acc_std": float(ts.std()) if len(ts) else None}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
