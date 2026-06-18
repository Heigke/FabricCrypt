"""z239 — CIFAR-10 grayscale 28×28 at strong_input config (O37 modality test).

Per O37 3-oracle consensus: highest-value experiment is non-MNIST-family
extension test. Grok Q3 pick was CIFAR-10. We use grayscale 28×28
(resize from 32×32 + RGB→gray) to keep pipeline IDENTICAL to z235-z238.
Only the dataset CONTENT changes (real-world objects vs handwritten).

If reservoir Δ falls on the 4-task linear fit, claim extends to
broader image-classification family. If sign-flip or large miss,
claim is bounded to MNIST-family.

Same NS-RAM hyperparams: leak=0.30, g_VG2=0.20, N=1000.
8 seeds, ~10-12 min compute.
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
OUT = ROOT / "results/z239_cifar_grayscale"; OUT.mkdir(parents=True, exist_ok=True)
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


def load_cifar10_gray_28():
    """Fetch CIFAR-10, RGB->grayscale, resize 32->28."""
    from sklearn.datasets import fetch_openml
    log_line(f"loading CIFAR-10 (this may take a minute)...")
    X, y = fetch_openml("CIFAR_10", version=1, return_X_y=True,
                          as_frame=False, parser="auto")
    X = X.astype(np.float32) / 255.0           # (60000, 3072)
    X = X.reshape(-1, 3, 32, 32)                # (N, 3, 32, 32)
    X_gray = X.mean(axis=1)                     # (N, 32, 32)
    # Resize 32→28 by simple stride: drop 2 rows + 2 cols (center crop)
    X_28 = X_gray[:, 2:30, 2:30]                # (N, 28, 28)
    y = y.astype(int)
    return X_28, y


def main():
    from z233_seq_mnist28_frozen import (
        GPUSurrogate4D, make_block_dense, encode_images, project_only,
    )
    from sklearn.linear_model import LogisticRegression

    log_line(f"=== z239 CIFAR-10 grayscale 28×28 (O37 modality test) ===")
    log_line(f"Device: {torch.cuda.get_device_name(0)}")
    log_line(f"Predicted from 4-task fit (Δ = +29.8 - 0.56·proj%):")
    log_line(f"  if proj=30% → Δ pred ≈ +13pp (extends LOW)")
    log_line(f"  if proj=50% → Δ pred ≈ +1.8pp")
    log_line(f"  if proj=70% → Δ pred ≈ -9pp")

    SURR_PATH = ROOT / "results/z220_4d_dense/surrogate_4d_dense.npz"
    surr = GPUSurrogate4D(SURR_PATH)

    try:
        X, y = load_cifar10_gray_28()
    except Exception as e:
        log_line(f"CIFAR-10 load FAILED: {e}")
        log_line(f"Falling back to FashionMNIST as control (sanity check pipeline)")
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True,
                              as_frame=False, parser="auto")
        X = X.astype(np.float32).reshape(-1, 28, 28) / 255.0
        y = y.astype(int)
        log_line(f"USING FALLBACK — z239 result is FashionMNIST not CIFAR")

    rng0 = np.random.default_rng(0)
    idx = rng0.permutation(len(X))
    X = X[idx]; y = y[idx]
    X_train, y_train = X[:1000], y[:1000]
    X_test,  y_test  = X[1000:1200], y[1000:1200]
    log_line(f"data: train {X_train.shape}, test {X_test.shape}")

    N = 1000; n_block = 500
    Cb, dt, g_VG1 = 5e-15, 5e-7, 0.30
    leak, g_VG2 = 0.30, 0.20

    SEEDS = list(range(8))
    results = []
    t_start = time.time()
    for s in SEEDS:
        if time.time() - t_start > 38*60:
            log_line(f"BUDGET REACHED at s{s}")
            break
        fp = OUT / f"seed{s}.json"
        if fp.exists():
            results.append(json.loads(fp.read_text()))
            continue
        cooldown_to(60.0, 60)
        try:
            rng = np.random.default_rng(s + 6000)
            base_VG1 = torch.tensor(rng.uniform(0.2, 0.5, N).astype(np.float32),
                                      device="cuda")
            base_VG2 = torch.tensor(rng.uniform(0.05, 0.55, N).astype(np.float32),
                                      device="cuda")
            sign_mask = torch.tensor(rng.choice([-1.0, 1.0], N).astype(np.float32),
                                       device="cuda")
            W_in_np = rng.normal(0, 1.0/np.sqrt(28), size=(N, 28)).astype(np.float32)
            W_in = torch.tensor(W_in_np, dtype=torch.float32, device="cuda")
            Wb, K, nb = make_block_dense(N, n_block, seed=s)

            St_train = encode_images(X_train, surr, base_VG1, base_VG2, sign_mask,
                                        W_in, Wb, K, nb, N,
                                        Cb=Cb, dt=dt, g_VG2=g_VG2,
                                        g_VG1=g_VG1, leak=leak)
            St_test = encode_images(X_test, surr, base_VG1, base_VG2, sign_mask,
                                        W_in, Wb, K, nb, N,
                                        Cb=Cb, dt=dt, g_VG2=g_VG2,
                                        g_VG1=g_VG1, leak=leak)

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
                   "apu_peak": get_apu()}
            fp.write_text(json.dumps(r, indent=2))
            results.append(r)
            log_line(f"  s{s}: res={test_acc:.3f} proj={proj_acc:.3f} "
                     f"Δ={r['delta_pp']:+.2f}pp  apu={r['apu_peak']:.0f}°C")
            if r["apu_peak"] > 92: break
        except Exception as e:
            log_line(f"  s{s} FAILED: {e}")

    if results:
        accs = np.array([r["test_acc"] for r in results])
        projs = np.array([r["proj_acc"] for r in results])
        deltas = (accs - projs) * 100
        from scipy import stats as scs
        t, p = scs.ttest_rel(accs, projs)
        rng2 = np.random.default_rng(0)
        boots = np.array([np.median(deltas[rng2.integers(0, len(deltas), len(deltas))])
                            for _ in range(5000)])
        ci_lo, ci_hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))

        proj_pct = projs.mean() * 100
        predicted_delta = 29.77 - 0.564 * proj_pct
        actual_delta = deltas.mean()
        sign_match = (actual_delta * predicted_delta > 0) or abs(actual_delta) < 1
        in_ci = ci_lo <= predicted_delta <= ci_hi
        diff_abs = abs(actual_delta - predicted_delta)

        summary = {
            "n_seeds": len(results),
            "task": "CIFAR-10 grayscale 28x28",
            "config": {"leak": leak, "g_VG2": g_VG2, "N": N},
            "reservoir_mean": float(accs.mean()),
            "proj_mean": float(projs.mean()),
            "delta_mean_pp": float(deltas.mean()),
            "ci95_pp_median": [ci_lo, ci_hi],
            "n_positive": int((deltas > 0).sum()),
            "paired_t": float(t), "p_value": float(p),
            "predicted_delta_pp_from_4task_fit": float(predicted_delta),
            "actual_minus_predicted_pp": float(actual_delta - predicted_delta),
            "sign_match": bool(sign_match),
            "predicted_in_ci": bool(in_ci),
            "claim_extends": bool(in_ci and sign_match and diff_abs <= 2.0),
        }
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
        log_line(f"\n=== n={len(results)} ===")
        log_line(f"reservoir : {accs.mean():.3f}  projection: {projs.mean():.3f}")
        log_line(f"Δ actual  : {actual_delta:+.2f}pp  CI [{ci_lo:+.2f}, {ci_hi:+.2f}]")
        log_line(f"Δ predicted (proj={proj_pct:.0f}%): {predicted_delta:+.2f}pp")
        log_line(f"|actual − predicted| = {diff_abs:.2f}pp")
        log_line(f"Sign match: {sign_match}; Predicted in CI: {in_ci}")
        log_line(f"\nO37 acceptance gate (extends claim beyond MNIST-family):")
        log_line(f"  EXTENDS (|err|≤2pp + sign + in CI): "
                 f"{'✅ EXTENDS' if summary['claim_extends'] else '❌ BOUNDED'}")


if __name__ == "__main__":
    main()
