"""z236 — FashionMNIST 28x28 at strong_input config (3rd-task generalization).

Tests whether the g_VG2=0.20 retune that worked on MNIST (z235:
Δ=+5.1pp) also works on a DIFFERENT image task with same input
dimensionality. FashionMNIST is harder than MNIST (typical baselines
65-70% on simple linear vs 90%+ on MNIST) so absolute numbers will
be lower. Question is: does reservoir > projection hold?

Single config: leak=0.30, g_VG2=0.20, N=1000 (matches z235).
10 seeds (budget-conscious; 30 would take 30+ min).

Acceptance: Δ ≥ +3pp with CI excluding 0 → principle generalizes
within image-task class.
Negative: would tighten Mario v2 framing to "MNIST-specific" rather
than "image-class generalizable."
"""
from __future__ import annotations
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, json, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results/z236_fashion_mnist"; OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "live.log"


def get_apu():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return -1.0


def log_line(msg):
    line = f"[{time.strftime('%H:%M:%S')}] APU={get_apu():.1f}°C  {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def cooldown_to(target, timeout_s=120):
    t0 = time.time()
    while True:
        a = get_apu()
        if a < target: return a
        if time.time() - t0 > timeout_s: return a
        time.sleep(15)


def main():
    from z233_seq_mnist28_frozen import (
        GPUSurrogate4D, make_block_dense, encode_images, project_only,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import fetch_openml

    log_line(f"=== z236 FashionMNIST 28x28 strong_input config ===")
    log_line(f"Device: {torch.cuda.get_device_name(0)}")
    log_line(f"Config: leak=0.30, g_VG2=0.20, N=1000")

    SURR_PATH = ROOT / "results/z220_4d_dense/surrogate_4d_dense.npz"
    surr = GPUSurrogate4D(SURR_PATH)

    log_line(f"loading FashionMNIST...")
    X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True,
                          as_frame=False, parser="auto")
    X = X.astype(np.float32).reshape(-1, 28, 28) / 255.0
    y = y.astype(int)
    rng0 = np.random.default_rng(0)
    idx = rng0.permutation(len(X))
    X = X[idx]; y = y[idx]
    X_train, y_train = X[:1000], y[:1000]
    X_test,  y_test  = X[1000:1200], y[1000:1200]

    N = 1000; n_block = 500
    Cb, dt, g_VG1 = 5e-15, 5e-7, 0.30
    leak, g_VG2 = 0.30, 0.20

    SEEDS = list(range(10))
    results = []
    t_start = time.time()
    for s in SEEDS:
        if time.time() - t_start > 28*60:
            log_line(f"BUDGET REACHED at seed {s}")
            break
        fp = OUT / f"seed{s}.json"
        if fp.exists():
            results.append(json.loads(fp.read_text()))
            continue
        cooldown_to(60.0, 60)
        try:
            rng = np.random.default_rng(s + 3000)
            base_VG1 = torch.tensor(rng.uniform(0.2, 0.5, N).astype(np.float32),
                                      device="cuda")
            base_VG2 = torch.tensor(rng.uniform(0.05, 0.55, N).astype(np.float32),
                                      device="cuda")
            sign_mask = torch.tensor(rng.choice([-1.0, 1.0], N).astype(np.float32),
                                       device="cuda")
            W_in_np = rng.normal(0, 1.0/np.sqrt(28), size=(N, 28)).astype(np.float32)
            W_in = torch.tensor(W_in_np, dtype=torch.float32, device="cuda")
            Wb, K, nb = make_block_dense(N, n_block, seed=s)

            t0 = time.time()
            St_train = encode_images(X_train, surr, base_VG1, base_VG2, sign_mask,
                                        W_in, Wb, K, nb, N,
                                        Cb=Cb, dt=dt, g_VG2=g_VG2,
                                        g_VG1=g_VG1, leak=leak)
            St_test = encode_images(X_test, surr, base_VG1, base_VG2, sign_mask,
                                        W_in, Wb, K, nb, N,
                                        Cb=Cb, dt=dt, g_VG2=g_VG2,
                                        g_VG1=g_VG1, leak=leak)
            enc_wall = time.time() - t0

            clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
            clf.fit(St_train, y_train)
            test_acc = float(clf.score(St_test, y_test))
            Pp_train = project_only(X_train, W_in_np)
            Pp_test = project_only(X_test, W_in_np)
            clfp = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
            clfp.fit(Pp_train, y_train)
            proj_acc = float(clfp.score(Pp_test, y_test))

            r = {"seed": s, "N": N, "leak": leak, "g_VG2": g_VG2,
                   "test_acc": test_acc, "proj_acc": proj_acc,
                   "delta_pp": (test_acc - proj_acc) * 100,
                   "enc_wall_s": enc_wall, "apu_peak": get_apu()}
            fp.write_text(json.dumps(r, indent=2))
            results.append(r)
            log_line(f"  s{s:>2}: res={test_acc:.3f} proj={proj_acc:.3f} "
                     f"Δ={r['delta_pp']:+5.2f}pp  apu={r['apu_peak']:.0f}°C")
            if r["apu_peak"] > 92:
                log_line(f"  THERMAL KILL at s{s}")
                break
        except Exception as e:
            log_line(f"  s{s} FAILED: {e}")

    if results:
        accs = np.array([r["test_acc"] for r in results])
        projs = np.array([r["proj_acc"] for r in results])
        deltas = (accs - projs) * 100
        rng2 = np.random.default_rng(0)
        boots = np.array([np.median(deltas[rng2.integers(0, len(deltas), len(deltas))])
                            for _ in range(5000)])
        ci_lo, ci_hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
        from scipy import stats as scs
        t, p = scs.ttest_rel(accs, projs)
        n_pos = int((deltas > 0).sum())

        gate_3pp = ci_lo >= 3.0 and deltas.mean() >= 3.0
        gate_pos = ci_lo > 0 and deltas.mean() > 0

        summary = {
            "n_seeds": len(results),
            "task": "FashionMNIST 28x28",
            "config": {"leak": leak, "g_VG2": g_VG2, "N": N},
            "reservoir_mean": float(accs.mean()), "reservoir_std": float(accs.std()),
            "proj_mean": float(projs.mean()),     "proj_std": float(projs.std()),
            "delta_mean_pp": float(deltas.mean()),
            "delta_median_pp": float(np.median(deltas)),
            "ci95_pp_median": [ci_lo, ci_hi],
            "n_positive": n_pos,
            "paired_t": float(t), "p_value": float(p),
            "gate_3pp_pass": bool(gate_3pp),
            "gate_pos_pass": bool(gate_pos),
        }
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
        log_line(f"\n=== n={len(results)} seeds ===")
        log_line(f"reservoir : {accs.mean():.4f} ± {accs.std():.4f}")
        log_line(f"projection: {projs.mean():.4f} ± {projs.std():.4f}")
        log_line(f"Δ mean    : {deltas.mean():+.2f}  median {np.median(deltas):+.2f}pp")
        log_line(f"CI95      : [{ci_lo:+.2f}, {ci_hi:+.2f}] pp")
        log_line(f"positive  : {n_pos}/{len(results)}, paired t={t:+.2f} p={p:.3g}")
        log_line(f"\nGates:")
        log_line(f"  Δ≥+3pp AND CI_lo≥+3: {'✅' if gate_3pp else '❌'}")
        log_line(f"  Δ>0 AND CI excl. 0:  {'✅' if gate_pos else '❌'}")


if __name__ == "__main__":
    main()
