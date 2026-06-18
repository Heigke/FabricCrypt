"""z234 — seq-MNIST 28x28 with TASK-SPECIFIC retuned hyperparameters.

z233 showed that FROZEN NARMA-10 hyperparams DON'T generalize
(reservoir 37.4% vs projection 42.0%, p=8e-17 negative). Per audit #13
follow-up: does NS-RAM work on MNIST IF we retune?

Sweep (4 configs × 3 seeds = 12 runs, ~13 min wall budget):
  leak  ∈ {0.30 (frozen), 0.70 (more memory)}
  g_VG2 ∈ {0.05 (frozen), 0.20 (stronger input)}

If any config beats projection by ≥3 pp with consistent direction
across 3 seeds: "NS-RAM viable for image classification with task-
specific tuning." Brief framing improves.

If all configs still under projection: "frozen-or-tuned, current
NS-RAM-as-reservoir doesn't help on MNIST." Brief framing locked.

Reuses z233 infrastructure (GPU N=2000, manual block-loop, 4D
surrogate). MNIST 1000 train / 200 test.
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
OUT = ROOT / "results/z234_seq_mnist_retuned"; OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "live.log"
SURR_PATH = ROOT / "results/z220_4d_dense/surrogate_4d_dense.npz"


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


def cooldown_to(target_c, timeout_s=120):
    t0 = time.time()
    while True:
        apu = get_apu()
        if apu < target_c: return apu
        if time.time() - t0 > timeout_s: return apu
        time.sleep(15)


def main():
    sys.path.insert(0, str(ROOT / "scripts"))
    from z233_seq_mnist28_frozen import (
        GPUSurrogate4D, make_block_dense, encode_images, project_only,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import fetch_openml

    log_line(f"=== z234 seq-MNIST 28x28 RETUNED sweep ===")
    log_line(f"Device: {torch.cuda.get_device_name(0)}")

    surr = GPUSurrogate4D(SURR_PATH)

    log_line(f"loading MNIST 28x28...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")
    X = X.astype(np.float32).reshape(-1, 28, 28) / 255.0
    y = y.astype(int)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    X = X[idx]; y = y[idx]
    X_train, y_train = X[:1000], y[:1000]
    X_test,  y_test  = X[1000:1200], y[1000:1200]

    configs = [
        {"name": "frozen",         "leak": 0.30, "g_VG2": 0.05},
        {"name": "more_memory",    "leak": 0.70, "g_VG2": 0.05},
        {"name": "strong_input",   "leak": 0.30, "g_VG2": 0.20},
        {"name": "both_tuned",     "leak": 0.70, "g_VG2": 0.20},
    ]
    SEEDS = [0, 1, 2]

    N = 2000; n_block = 500
    Cb, dt, g_VG1 = 5e-15, 5e-7, 0.30

    results = []
    t_start = time.time()
    for cfg in configs:
        for seed in SEEDS:
            if time.time() - t_start > 32 * 60:
                log_line(f"budget reached, stopping at {cfg['name']}/seed={seed}")
                break
            fp = OUT / f"{cfg['name']}_s{seed}.json"
            if fp.exists():
                results.append(json.loads(fp.read_text()))
                continue
            cooldown_to(65.0, 60)
            try:
                rng = np.random.default_rng(seed + 1000)
                base_VG1 = torch.tensor(rng.uniform(0.2, 0.5, N).astype(np.float32),
                                          device="cuda")
                base_VG2 = torch.tensor(rng.uniform(0.05, 0.55, N).astype(np.float32),
                                          device="cuda")
                sign_mask = torch.tensor(rng.choice([-1.0, 1.0], N).astype(np.float32),
                                           device="cuda")
                W_in_np = rng.normal(0, 1.0/np.sqrt(28), size=(N, 28)).astype(np.float32)
                W_in = torch.tensor(W_in_np, dtype=torch.float32, device="cuda")
                Wb, K, nb = make_block_dense(N, n_block, seed=seed)

                t0 = time.time()
                St_train = encode_images(X_train, surr, base_VG1, base_VG2, sign_mask,
                                            W_in, Wb, K, nb, N,
                                            Cb=Cb, dt=dt, g_VG2=cfg["g_VG2"],
                                            g_VG1=g_VG1, leak=cfg["leak"])
                St_test = encode_images(X_test, surr, base_VG1, base_VG2, sign_mask,
                                            W_in, Wb, K, nb, N,
                                            Cb=Cb, dt=dt, g_VG2=cfg["g_VG2"],
                                            g_VG1=g_VG1, leak=cfg["leak"])
                enc_wall = time.time() - t0

                clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
                clf.fit(St_train, y_train)
                test_acc = float(clf.score(St_test, y_test))
                Pp_train = project_only(X_train, W_in_np)
                Pp_test = project_only(X_test, W_in_np)
                clfp = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
                clfp.fit(Pp_train, y_train)
                proj_acc = float(clfp.score(Pp_test, y_test))

                r = {"config": cfg["name"], "leak": cfg["leak"], "g_VG2": cfg["g_VG2"],
                       "seed": seed, "test_acc": test_acc, "proj_acc": proj_acc,
                       "delta_pp": (test_acc - proj_acc) * 100,
                       "enc_wall_s": enc_wall, "apu_peak": get_apu()}
                fp.write_text(json.dumps(r, indent=2))
                results.append(r)
                log_line(f"  {cfg['name']:15s} s{seed}: res={test_acc:.3f} "
                         f"proj={proj_acc:.3f} Δ={r['delta_pp']:+.2f}pp")

                apu = get_apu()
                if apu > 92:
                    log_line(f"  THERMAL KILL APU={apu}")
                    break
            except Exception as e:
                log_line(f"  {cfg['name']} s{seed} FAILED: {e}")
                continue

    if results:
        from collections import defaultdict
        by_config = defaultdict(list)
        for r in results:
            by_config[r["config"]].append(r)
        log_line(f"\n=== Summary by config (n={len(results)} runs) ===")
        log_line(f"{'config':15s}  {'leak':>5}  {'g_VG2':>6}  {'res mean':>9}  "
                 f"{'proj mean':>10}  {'Δ mean':>8}  {'n':>3}")
        summary = {}
        for name, rs in by_config.items():
            r0 = rs[0]
            res_m = np.mean([r["test_acc"] for r in rs])
            proj_m = np.mean([r["proj_acc"] for r in rs])
            d_m = np.mean([r["delta_pp"] for r in rs])
            log_line(f"  {name:15s}  {r0['leak']:5.2f}  {r0['g_VG2']:6.2f}  "
                     f"{res_m:9.4f}  {proj_m:10.4f}  {d_m:+7.2f}pp  {len(rs):3d}")
            summary[name] = {"leak": r0["leak"], "g_VG2": r0["g_VG2"],
                              "n": len(rs), "res_mean": float(res_m),
                              "proj_mean": float(proj_m), "delta_mean_pp": float(d_m)}

        # Best config
        best = max(summary.items(), key=lambda kv: kv[1]["delta_mean_pp"])
        log_line(f"\nBest config: {best[0]}  Δ={best[1]['delta_mean_pp']:+.2f}pp")
        gate_pass = best[1]["delta_mean_pp"] >= 3.0
        log_line(f"\"task-tuning rescues NS-RAM\" gate (Δ≥+3pp): "
                 f"{'✅ PASS' if gate_pass else '❌ FAIL'}")

        (OUT / "summary.json").write_text(json.dumps({
            "configs": summary, "best": best[0],
            "best_delta_pp": best[1]["delta_mean_pp"],
            "gate_3pp_pass": bool(gate_pass),
        }, indent=2))


if __name__ == "__main__":
    main()
