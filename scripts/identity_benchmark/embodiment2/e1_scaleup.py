"""Phase E1: scale-up. N=128 → 512 → 2048 reservoirs.

Per N: train ikaros, transplant daedalus, measure G2 factor.
Skip eigvals for N=2048 (O(N^3)) — use spectral_radius via power iteration.
"""
from __future__ import annotations
import json, sys, time, argparse
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _lib import (load_vec, derive_structure_v2, train_eval_task,
                  transplant_eval, OUT2, build_reservoir as _br,
                  run_reservoir, ridge_fit, ridge_predict, nrmse,
                  narma10, WASHOUT)

TASK = "narma10"


def build_fast(struct, N, seed=0, spectral_radius=0.95, input_scale=1.0):
    """For large N: avoid eigvals — use power iteration."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((N, N)) / np.sqrt(N)
    W = W * struct["mask"]
    W = W * struct["weight_scale"][None, :]
    # power iteration for spectral radius (handles negative/complex eigvals via |Wv|/|v|)
    v = rng.standard_normal(N); v /= np.linalg.norm(v) + 1e-12
    rho = 0.0
    for _ in range(30):
        Wv = W @ v
        nrm = float(np.linalg.norm(Wv))
        if nrm < 1e-12 or not np.isfinite(nrm):
            rho = 0.0; break
        rho = nrm  # ||Wv||/||v||=||Wv|| since v unit
        v = Wv / nrm
    if rho > 1e-9 and np.isfinite(rho):
        W = W * (spectral_radius / rho)
    else:
        W = W * 0.5  # fallback
    Win = rng.standard_normal((N, 1)) * input_scale
    return W, Win


def train_eval_N(struct, N, seed, T_tr=1500, T_te=400):
    u_tr, y_tr = narma10(T_tr, seed=seed*13+7)
    u_te, y_te = narma10(T_te, seed=seed*13+9991)
    W, Win = build_fast(struct, N, seed=seed)
    X_tr = run_reservoir(u_tr, W, Win, struct, N)
    Wout = ridge_fit(X_tr[WASHOUT:], y_tr[WASHOUT:])
    X_te = run_reservoir(u_te, W, Win, struct, N)
    y_hat = ridge_predict(X_te[WASHOUT:], Wout)
    return nrmse(y_te[WASHOUT:], y_hat), {"W": W, "Win": Win, "Wout": Wout}


def transplant_N(weights, struct_new, N, seed, T_te=400):
    u_te, y_te = narma10(T_te, seed=seed*13+9991)
    W = weights["W"] * struct_new["mask"]; W = W * struct_new["weight_scale"][None, :]
    X_te = run_reservoir(u_te, W, weights["Win"], struct_new, N)
    y_hat = ridge_predict(X_te[WASHOUT:], weights["Wout"])
    return nrmse(y_te[WASHOUT:], y_hat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--Ns", default="128,512,2048")
    args = ap.parse_args()
    out_dir = OUT2 / "phase_e"; out_dir.mkdir(parents=True, exist_ok=True)
    Ns = [int(x) for x in args.Ns.split(",")]
    vk = load_vec("ikaros"); vd = load_vec("daedalus")
    results = {"task": TASK, "seeds": args.seeds, "Ns": Ns, "per_N": {}}
    for N in Ns:
        # thermal pre-check
        try:
            T = float(open("/sys/class/thermal/thermal_zone0/temp").read().strip()) / 1000.0
        except Exception: T = 0.0
        if T > 70:
            print(f"[E1][N={N}] APU={T:.1f}C — cool first...", flush=True)
            while T > 50:
                time.sleep(5)
                try: T = float(open("/sys/class/thermal/thermal_zone0/temp").read().strip()) / 1000.0
                except Exception: break
        s_ik = derive_structure_v2(vk, N); s_da = derive_structure_v2(vd, N)
        g1, g2 = [], []
        for s in range(args.seeds):
            t0 = time.time()
            nr1, w = train_eval_N(s_ik, N, s)
            nr2 = transplant_N(w, s_da, N, s)
            g1.append(nr1); g2.append(nr2)
            print(f"[E1][N={N}] seed={s} G1={nr1:.4f} G2={nr2:.2f} (t={time.time()-t0:.1f}s)", flush=True)
        factor = float(np.median(g2) / max(1e-9, np.median(g1)))
        results["per_N"][str(N)] = {"G1_median": float(np.median(g1)),
                                     "G2_median": float(np.median(g2)),
                                     "factor": factor,
                                     "G1_per_seed": g1, "G2_per_seed": g2}
        print(f"[E1][N={N}] factor={factor:.1f}x", flush=True)
    (out_dir / "E1_result.json").write_text(json.dumps(results, indent=2))
    print(f"[E1] wrote E1_result.json", flush=True)


if __name__ == "__main__":
    main()
