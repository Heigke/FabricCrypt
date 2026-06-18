"""z242 — ESN control on MNIST (O38 consensus #1, pipeline-vs-NS-RAM test).

Replace NS-RAM surrogate with standard tanh ESN, keep EVERYTHING ELSE
identical (W_in, projection baseline, linear classifier, N=1000,
block-diag W with same sparsity pattern). Tests if the linear-within-band
Δ-vs-baseline relationship is NS-RAM-specific or pipeline-level.

ESN formula: state[t+1] = tanh(W @ state[t] + W_in @ x[t])
We use leaky integration: state[t+1] = (1-leak)*state[t] + leak*tanh(...)
to match NS-RAM's leaky update.

Key statistic: ΔΔ = Δ_NSRAM (z235: +5.10pp) − Δ_ESN
  - If |ΔΔ| ≤ 1pp → effect is pipeline-level, NS-RAM not specifically needed
  - If ΔΔ ≥ +3pp → NS-RAM-specific gain at this bias
  - If ΔΔ ≤ −3pp → ESN better; revisit attribution

8 seeds on MNIST at strong_input-equivalent ESN gain.
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
OUT = ROOT / "results/z242_esn_control"; OUT.mkdir(parents=True, exist_ok=True)
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


def encode_images_esn(images, base_VG1, base_VG2, sign_mask, W_in, Wb, K, nb,
                        N, leak=0.30, g_in=0.20):
    """Standard tanh ESN. Replaces NS-RAM surrogate.
    Each image (28, 28) is presented row-by-row. Leaky tanh update.
    Returns final state.
    """
    M = images.shape[0]
    states = torch.zeros((M, N), dtype=torch.float32, device="cuda")
    img_t = torch.tensor(images, dtype=torch.float32, device="cuda")

    for m in range(M):
        feat = torch.zeros(N, dtype=torch.float32, device="cuda")
        for t in range(28):
            row = img_t[m, t]
            cell_in = (W_in @ row.unsqueeze(-1)).squeeze(-1)  # (N,)
            # Standard ESN: tanh of W·feat + W_in·input + bias-like base
            feat_b = feat.view(K, nb)
            rec = torch.bmm(Wb, feat_b.unsqueeze(-1)).squeeze(-1).view(N) * sign_mask
            # g_in scales the input drive — analogous to g_VG2 in NS-RAM
            new_state = torch.tanh(0.9 * rec + g_in * cell_in + 0.05 * base_VG1)
            feat = (1.0 - leak) * feat + leak * new_state
        states[m] = feat
    torch.cuda.synchronize()
    return states.cpu().numpy()


def project_only(images, W_in_np):
    M = images.shape[0]
    out = np.zeros((M, W_in_np.shape[0]), dtype=np.float32)
    for m in range(M):
        out[m] = (W_in_np @ images[m].T).mean(axis=1)
    return out


def main():
    from z233_seq_mnist28_frozen import make_block_dense
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import fetch_openml

    log_line(f"=== z242 ESN control on MNIST (O38 attribution test) ===")
    log_line(f"NS-RAM Δ on MNIST (z235, n=25): +5.10pp")
    log_line(f"Hypothesis: if ESN gives similar Δ → effect is pipeline-level")

    log_line(f"loading MNIST...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True,
                          as_frame=False, parser="auto")
    X = X.astype(np.float32).reshape(-1, 28, 28) / 255.0
    y = y.astype(int)
    rng0 = np.random.default_rng(0)
    idx = rng0.permutation(len(X))
    X = X[idx]; y = y[idx]
    X_train, y_train = X[:1000], y[:1000]
    X_test,  y_test  = X[1000:1200], y[1000:1200]

    N = 1000; n_block = 500
    leak = 0.30
    g_in = 0.20  # match NS-RAM g_VG2 strength

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
            rng = np.random.default_rng(s + 9000)
            base_VG1 = torch.tensor(rng.uniform(0.2, 0.5, N).astype(np.float32),
                                      device="cuda")
            base_VG2 = torch.tensor(rng.uniform(0.05, 0.55, N).astype(np.float32),
                                      device="cuda")
            sign_mask = torch.tensor(rng.choice([-1.0, 1.0], N).astype(np.float32),
                                       device="cuda")
            W_in_np = rng.normal(0, 1.0/np.sqrt(28), size=(N, 28)).astype(np.float32)
            W_in = torch.tensor(W_in_np, dtype=torch.float32, device="cuda")
            Wb, K, nb = make_block_dense(N, n_block, seed=s)

            St_train = encode_images_esn(X_train, base_VG1, base_VG2, sign_mask,
                                            W_in, Wb, K, nb, N,
                                            leak=leak, g_in=g_in)
            St_test = encode_images_esn(X_test, base_VG1, base_VG2, sign_mask,
                                            W_in, Wb, K, nb, N,
                                            leak=leak, g_in=g_in)

            clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
            clf.fit(St_train, y_train)
            test_acc = float(clf.score(St_test, y_test))
            Pp_train = project_only(X_train, W_in_np)
            Pp_test = project_only(X_test, W_in_np)
            clfp = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
            clfp.fit(Pp_train, y_train)
            proj_acc = float(clfp.score(Pp_test, y_test))

            r = {"seed": s, "N": N, "leak": leak, "g_in": g_in,
                   "test_acc": test_acc, "proj_acc": proj_acc,
                   "delta_pp": (test_acc - proj_acc) * 100,
                   "apu_peak": get_apu()}
            fp.write_text(json.dumps(r, indent=2))
            results.append(r)
            log_line(f"  s{s}: ESN={test_acc:.3f} proj={proj_acc:.3f} "
                     f"Δ_ESN={r['delta_pp']:+.2f}pp apu={r['apu_peak']:.0f}°C")
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

        delta_nsram = 5.10  # z235 mean
        delta_esn = float(deltas.mean())
        delta_delta = delta_nsram - delta_esn

        if abs(delta_delta) <= 1.0:
            verdict = "PIPELINE_LEVEL (effect not NS-RAM-specific)"
        elif delta_delta >= 3.0:
            verdict = "NS-RAM-SPECIFIC (NS-RAM > ESN by ≥3pp)"
        elif delta_delta <= -3.0:
            verdict = "ESN_BETTER (revisit attribution)"
        else:
            verdict = "AMBIGUOUS (1pp < |ΔΔ| < 3pp)"

        summary = {
            "n_seeds": len(results),
            "task": "MNIST 28x28 ESN control",
            "config": {"leak": leak, "g_in": g_in, "N": N},
            "esn_mean_acc": float(accs.mean()),
            "proj_mean_acc": float(projs.mean()),
            "delta_esn_pp": delta_esn,
            "ci95_pp_median": [ci_lo, ci_hi],
            "n_positive": int((deltas > 0).sum()),
            "paired_t": float(t), "p_value": float(p),
            "delta_nsram_z235": delta_nsram,
            "delta_delta_pp": delta_delta,
            "verdict": verdict,
        }
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
        log_line(f"\n=== n={len(results)} ===")
        log_line(f"ESN reservoir : {accs.mean():.3f}  projection: {projs.mean():.3f}")
        log_line(f"Δ_ESN  : {delta_esn:+.2f}pp  CI [{ci_lo:+.2f}, {ci_hi:+.2f}]")
        log_line(f"Δ_NSRAM (z235): {delta_nsram:+.2f}pp")
        log_line(f"ΔΔ (NSRAM - ESN): {delta_delta:+.2f}pp")
        log_line(f"\nVERDICT: {verdict}")
        log_line(f"  |ΔΔ|≤1pp → pipeline-level")
        log_line(f"  ΔΔ≥+3pp → NS-RAM specific")
        log_line(f"  ΔΔ≤-3pp → ESN better")


if __name__ == "__main__":
    main()
