"""z240 — CIFAR-10 grayscale 28×28 modality test (uses local cache).

Replaces failed z239 (openml HTTP 504). Loads CIFAR-10 directly from
data/cifar-10-batches-py/ which is already cached. Otherwise identical
to z239 setup: pipeline IDENTICAL to z235-z238 (28×28 grayscale,
projection+linear-classifier baseline, leak=0.30, g_VG2=0.20, N=1000).

This is the modality test that O37 3/3 oracles requested to address
the task-modality confound WARNING.
"""
from __future__ import annotations
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, json, time, pickle
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results/z240_cifar_local"; OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "live.log"
CIFAR_DIR = ROOT / "data/cifar-10-batches-py"


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


def load_cifar10_local():
    """Load CIFAR-10 from local pickle batches; return RGB (60000,3,32,32)."""
    Xs, ys = [], []
    for batch in ["data_batch_1", "data_batch_2", "data_batch_3",
                    "data_batch_4", "data_batch_5", "test_batch"]:
        with open(CIFAR_DIR / batch, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        Xs.append(d[b"data"])
        ys.extend(d[b"labels"])
    X = np.concatenate(Xs, axis=0)         # (60000, 3072)
    X = X.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y = np.array(ys, dtype=np.int64)
    return X, y


def main():
    from z233_seq_mnist28_frozen import (
        GPUSurrogate4D, make_block_dense, encode_images, project_only,
    )
    from sklearn.linear_model import LogisticRegression

    log_line(f"=== z240 CIFAR-10 grayscale 28×28 (modality test, local cache) ===")
    log_line(f"Device: {torch.cuda.get_device_name(0)}")
    log_line(f"Predicted from 4-task fit Δ=+29.8-0.56·proj%:")
    log_line(f"  if proj≈30%: Δ_pred≈+13pp (extends LOW)")
    log_line(f"  if proj≈40%: Δ_pred≈+7pp")
    log_line(f"  if proj≈50%: Δ_pred≈+1.8pp (zero-cross)")

    SURR_PATH = ROOT / "results/z220_4d_dense/surrogate_4d_dense.npz"
    surr = GPUSurrogate4D(SURR_PATH)

    log_line(f"loading CIFAR-10 from local cache...")
    X_rgb, y = load_cifar10_local()
    log_line(f"loaded {X_rgb.shape}")
    # RGB→gray + 32→28 center crop
    X_gray = X_rgb.mean(axis=1)              # (60000, 32, 32)
    X = X_gray[:, 2:30, 2:30]                # (60000, 28, 28)

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
        if time.time() - t_start > 28*60:
            log_line(f"BUDGET REACHED at s{s}")
            break
        fp = OUT / f"seed{s}.json"
        if fp.exists():
            results.append(json.loads(fp.read_text()))
            continue
        cooldown_to(60.0, 60)
        try:
            rng = np.random.default_rng(s + 7000)
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
        diff_abs = abs(actual_delta - predicted_delta)
        sign_match = (actual_delta * predicted_delta > 0) or abs(actual_delta) < 1
        in_ci = ci_lo <= predicted_delta <= ci_hi
        claim_extends = bool(in_ci and sign_match and diff_abs <= 2.0)

        summary = {
            "n_seeds": len(results),
            "task": "CIFAR-10 grayscale 28x28 (local cache)",
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
            "claim_extends": claim_extends,
        }
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
        log_line(f"\n=== n={len(results)} ===")
        log_line(f"reservoir : {accs.mean():.3f}  projection: {projs.mean():.3f}")
        log_line(f"Δ actual  : {actual_delta:+.2f}pp  CI [{ci_lo:+.2f}, {ci_hi:+.2f}]")
        log_line(f"Δ predicted (proj={proj_pct:.0f}%): {predicted_delta:+.2f}pp")
        log_line(f"|actual-predicted| = {diff_abs:.2f}pp")
        log_line(f"\nO37 ACCEPTANCE GATE (extends claim BEYOND MNIST-family):")
        log_line(f"  EXTENDS (|err|≤2pp + sign + in CI): "
                 f"{'✅ EXTENDS — addresses task-modality WARNING' if claim_extends else '❌ BOUNDED to MNIST-family'}")


if __name__ == "__main__":
    main()
