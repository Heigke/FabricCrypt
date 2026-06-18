"""Phase D3: live envelope sampling — multiplicative gate per neuron from
ratio of current envelope to startup envelope.

We can't easily re-collect 23-feat envelope mid-inference (too slow). Use
fast surrogate: APU temp + per-CPU jiffies as a tiny "live envelope" (4 dims)
sampled every M=50 steps. Gate g_t = sigmoid(alpha * (live_t - live_0)).

Test:
  (a) train on ikaros with live gating
  (b) transplant to daedalus (mask/etc + a NEW live envelope on daedalus)
  (c) measure G1 vs G2 degradation factor — should be larger than D1
  (d) stability: re-run on ikaros after 60s — gate values change, so model
      should self-correct via gate computation
"""
from __future__ import annotations
import json, sys, time, argparse, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _lib import (load_vec, derive_structure_v2, baseline_structure_v2,
                  build_reservoir, apply_act, ridge_fit, ridge_predict,
                  nrmse, narma10, OUT2, WASHOUT)

N = 128
TASK = "narma10"


def live_sample(n_dims=4):
    """Quick 4-dim 'live envelope': APU temp, GPU temp, load1, cpu jiffies hash."""
    try:
        apu = float(open("/sys/class/thermal/thermal_zone0/temp").read().strip()) / 1000.0
    except Exception:
        apu = 50.0
    try:
        gpu_files = [f for f in os.listdir("/sys/class/hwmon")
                     if "amdgpu" in open(f"/sys/class/hwmon/{f}/name").read().strip().lower()] if os.path.isdir("/sys/class/hwmon") else []
        if gpu_files:
            gpu = float(open(f"/sys/class/hwmon/{gpu_files[0]}/temp1_input").read().strip()) / 1000.0
        else: gpu = 0.0
    except Exception:
        gpu = 0.0
    try:
        load1 = float(open("/proc/loadavg").read().split()[0])
    except Exception:
        load1 = 0.0
    try:
        with open("/proc/stat") as f:
            cpu0 = f.readline()
        jiffy = float(int(cpu0.split()[1]) % 1000) / 1000.0
    except Exception:
        jiffy = 0.0
    return np.array([apu, gpu, load1, jiffy])


def run_live_reservoir(u, W, Win, struct, N, live0, M=50, alpha=0.5):
    T = len(u); x = np.zeros(N); X = np.zeros((T, N))
    acts = struct["acts"]; perm = struct["perm"]; leak = struct["leak"]
    kinds = {}
    for i, a in enumerate(acts):
        kinds.setdefault(a, []).append(i)
    kind_idx = {k: np.asarray(v, dtype=np.int32) for k, v in kinds.items()}
    # Compute gate vector from live envelope DELTA. Project 4-dim delta to N via fixed hash-derived projection
    rng_proj = np.random.default_rng(0xC0DE)
    proj = rng_proj.standard_normal((4, N)) * 0.1  # fixed (so gate is reproducible given live & live0)
    live_cur = live0.copy()
    for t in range(T):
        if t % M == 0:
            live_cur = live_sample(4)
        delta = live_cur - live0
        gate = 1.0 / (1.0 + np.exp(-alpha * (delta @ proj)))  # N-dim in (0,1)
        gate = 0.5 + gate  # range (0.5, 1.5) — multiplicative around 1
        pre = (W @ x) * gate + Win[:, 0] * u[t]
        post = np.empty(N)
        for k, idx in kind_idx.items():
            post[idx] = apply_act(k, pre[idx])
        x_new = np.empty(N)
        x_new[perm] = (1 - leak[perm]) * x[perm] + leak[perm] * post
        x = x_new
        X[t] = x
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    out_dir = OUT2 / "phase_d"; out_dir.mkdir(parents=True, exist_ok=True)
    vk = load_vec("ikaros"); vd = load_vec("daedalus")
    s_ik = derive_structure_v2(vk, N); s_da = derive_structure_v2(vd, N)

    # capture live0 at startup
    live0 = live_sample()
    print(f"[D3] live0 = {live0}", flush=True)

    # Train ikaros with live gating
    g1_live, g1_static, weights_list = [], [], []
    for s in range(args.seeds):
        u_tr, y_tr = narma10(2000, seed=s*13+7)
        u_te, y_te = narma10(500, seed=s*13+9991)
        W, Win = build_reservoir(s_ik, N, seed=s)
        # train (use live gating during training too)
        X_tr = run_live_reservoir(u_tr, W, Win, s_ik, N, live0)
        Wout = ridge_fit(X_tr[WASHOUT:], y_tr[WASHOUT:])
        # eval ikaros
        X_te = run_live_reservoir(u_te, W, Win, s_ik, N, live0)
        y_hat = ridge_predict(X_te[WASHOUT:], Wout)
        g1_live.append(nrmse(y_te[WASHOUT:], y_hat))
        weights_list.append({"W": W, "Win": Win, "Wout": Wout})
        print(f"[D3] seed={s} G1_live={g1_live[-1]:.4f}", flush=True)

    # Transplant to daedalus (different live0 — we have only ikaros live but
    # simulate "daedalus live" via a different reference baseline)
    rng = np.random.default_rng(0xDEAD)
    live0_da = live0 + rng.standard_normal(4) * 5.0  # simulate large delta
    g2 = []
    for s, w in zip(range(args.seeds), weights_list):
        u_te, y_te = narma10(500, seed=s*13+9991)
        W = w["W"] * s_da["mask"]; W = W * s_da["weight_scale"][None, :]
        X_te = run_live_reservoir(u_te, W, w["Win"], s_da, N, live0_da)
        y_hat = ridge_predict(X_te[WASHOUT:], w["Wout"])
        g2.append(nrmse(y_te[WASHOUT:], y_hat))

    # stability re-test: same machine, same live0, after 30s wait
    print("[D3] sleeping 30s for stability re-test...", flush=True)
    time.sleep(30)
    g1_after = []
    for s, w in zip(range(args.seeds), weights_list):
        u_te, y_te = narma10(500, seed=s*13+9991)
        X_te = run_live_reservoir(u_te, w["W"], w["Win"], s_ik, N, live0)
        y_hat = ridge_predict(X_te[WASHOUT:], w["Wout"])
        g1_after.append(nrmse(y_te[WASHOUT:], y_hat))

    results = {"N": N, "task": TASK, "seeds": args.seeds, "live0": live0.tolist(),
               "live0_daedalus_simulated": live0_da.tolist(),
               "G1_live_median": float(np.median(g1_live)),
               "G2_transplant_median": float(np.median(g2)),
               "G2_factor": float(np.median(g2) / max(1e-9, np.median(g1_live))),
               "G1_after_30s_median": float(np.median(g1_after)),
               "stability_drift": float(np.median(g1_after) - np.median(g1_live)),
               "G1_per_seed": g1_live, "G2_per_seed": g2,
               "G1_after_per_seed": g1_after}
    results["verdict"] = "STABLE_TIGHTER" if (results["G2_factor"] > 5.0 and abs(results["stability_drift"]) < 0.1) else (
        "STABLE_WEAK_BINDING" if abs(results["stability_drift"]) < 0.1 else "UNSTABLE")
    print(f"[D3] G1={results['G1_live_median']:.4f} G2={results['G2_transplant_median']:.2f} factor={results['G2_factor']:.1f}x drift={results['stability_drift']:.4f}  verdict={results['verdict']}", flush=True)
    (out_dir / "D3_result.json").write_text(json.dumps(results, indent=2))
    print(f"[D3] wrote D3_result.json", flush=True)


if __name__ == "__main__":
    main()
