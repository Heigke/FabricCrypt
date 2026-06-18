"""z241 — g_VG2 sensitivity sweep on MNIST (O37 risk c test).

Tests whether the monotonic Δ-vs-baseline claim depends specifically
on g_VG2=0.20 (winner's curse) OR on stronger-input in general.

Existing data:
  g_VG2=0.05 (frozen z233, n=27):    Δ=-4.7pp on MNIST
  g_VG2=0.20 (strong z235, n=25):    Δ=+5.1pp on MNIST

Sweep: g_VG2 ∈ {0.10, 0.15, 0.30} × 5 seeds each, MNIST 28×28, N=1000.
Plus replicate g_VG2=0.20 with 5 seeds for cross-check.

Hypotheses:
  H1 (smooth): Δ varies smoothly with g_VG2 → claim robust to knob choice
  H2 (sharp peak): only g_VG2=0.20 helps → winner's curse confirmed
  H3 (saturating): Δ saturates above some g_VG2 threshold → ok
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
OUT = ROOT / "results/z241_gvg2_sensitivity"; OUT.mkdir(parents=True, exist_ok=True)
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

    log_line(f"=== z241 g_VG2 sensitivity sweep on MNIST ===")
    log_line(f"Existing: g=0.05→-4.7pp(z233 n=27), g=0.20→+5.1pp(z235 n=25)")
    log_line(f"Sweeping: g_VG2 ∈ {{0.10, 0.15, 0.30}}, 5 seeds each")

    SURR_PATH = ROOT / "results/z220_4d_dense/surrogate_4d_dense.npz"
    surr = GPUSurrogate4D(SURR_PATH)

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
    Cb, dt, g_VG1, leak = 5e-15, 5e-7, 0.30, 0.30
    g_VG2_values = [0.10, 0.15, 0.30]
    SEEDS = [0, 1, 2, 3, 4]

    results = []
    t_start = time.time()
    for g_VG2 in g_VG2_values:
        for s in SEEDS:
            if time.time() - t_start > 30*60:
                log_line(f"BUDGET REACHED at g={g_VG2} s{s}")
                break
            fp = OUT / f"g{g_VG2}_s{s}.json"
            if fp.exists():
                results.append(json.loads(fp.read_text()))
                continue
            cooldown_to(60.0, 60)
            try:
                rng = np.random.default_rng(s + int(g_VG2*1000) + 8000)
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

                r = {"seed": s, "g_VG2": g_VG2, "leak": leak, "N": N,
                       "test_acc": test_acc, "proj_acc": proj_acc,
                       "delta_pp": (test_acc - proj_acc) * 100,
                       "apu_peak": get_apu()}
                fp.write_text(json.dumps(r, indent=2))
                results.append(r)
                log_line(f"  g={g_VG2:.2f} s{s}: res={test_acc:.3f} proj={proj_acc:.3f} "
                         f"Δ={r['delta_pp']:+.2f}pp apu={r['apu_peak']:.0f}°C")
                if r["apu_peak"] > 92: break
            except Exception as e:
                log_line(f"  g={g_VG2} s{s} FAILED: {e}")
        else:
            continue
        break  # outer break if inner broke on budget

    if results:
        from collections import defaultdict
        from scipy import stats as scs
        by_g = defaultdict(list)
        for r in results:
            by_g[r["g_VG2"]].append(r)

        log_line(f"\n=== Summary by g_VG2 (sorted) ===")
        log_line(f"{'g_VG2':>6}  {'n':>3}  {'res mean':>9}  "
                 f"{'proj mean':>10}  {'Δ mean':>8}  {'Δ std':>7}  source")

        # Add z233 (g=0.05) and z235 (g=0.20) from existing summaries for context
        existing = [
            (0.05, "z233", -4.67, 27),
            (0.20, "z235", +5.10, 25),
        ]
        for g_e, src, dm, n_e in existing:
            log_line(f"  {g_e:5.2f}  {n_e:3d}     —          —      {dm:+7.2f}     —    {src}")

        all_data = []
        for g, rs in sorted(by_g.items()):
            res_m = np.mean([r["test_acc"] for r in rs])
            proj_m = np.mean([r["proj_acc"] for r in rs])
            d_m = np.mean([r["delta_pp"] for r in rs])
            d_std = np.std([r["delta_pp"] for r in rs])
            log_line(f"  {g:5.2f}  {len(rs):3d}  {res_m:9.4f}  {proj_m:10.4f}  "
                     f"{d_m:+7.2f}  {d_std:6.2f}    z241")
            all_data.append((g, d_m, d_std, len(rs)))

        # Combined picture
        combined = [(0.05, -4.67), (0.20, +5.10)]
        combined.extend([(g, dm) for g, dm, _, _ in all_data])
        combined.sort()
        log_line(f"\nCombined Δ vs g_VG2:")
        for g, dm in combined:
            bar_len = int(abs(dm))
            bar = "█" * bar_len if dm > 0 else "░" * bar_len
            sign = "+" if dm >= 0 else "-"
            log_line(f"  g_VG2={g:.2f}  Δ={dm:+6.2f}pp  {sign}{bar}")

        summary = {
            "z241_results": [{"g_VG2": g, "delta_mean_pp": float(dm),
                                "delta_std_pp": float(ds), "n": int(n)}
                                 for g, dm, ds, n in all_data],
            "context_existing": [{"g_VG2": g, "source": src,
                                     "delta_mean_pp": dm, "n": n}
                                    for g, src, dm, n in existing],
        }
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
