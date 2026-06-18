"""Phase F2 — oracle-guided advantage hunt: conditioning-driven scale/leak.

Based on O106 oracle synthesis:
  - Permutation is the dominant binder. Optimize cycle spectrum for NARMA-10.
  - Use envelope to choose per-neuron gains/leaks that minimize
    κ(XᵀX + λI) (condition number) on a calibration stream.
  - Compare envelope-keyed vs baseline vs random envelope.

Approach for "conditioning equalization":
  Given a reservoir produces X(T,N), we want each column to have similar
  variance and low cross-correlation → diagonal covariance is closer to
  identity → kappa lower → readout generalizes better.

  Envelope-derived per-neuron scale s_i and leak l_i are normalized so
  that calibration-run column variance is approximately uniform:
    s_i_eff = s_i_env / sqrt(var(X_i^cal))
  This is an envelope-MODULATED equalizer: the BASE scale comes from
  envelope (so it's chassi-bound), but the EQUALIZATION is from
  calibration. If random envelope gives DIFFERENT base scales →
  different post-equalization fingerprints → potentially worse.
"""
from __future__ import annotations
import json, sys, time, argparse
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _lib import (load_vec, derive_structure_v2, baseline_structure_v2,
                  build_reservoir, run_reservoir, ridge_fit, ridge_predict,
                  nrmse, narma10, mackey_glass, memory_capacity_task,
                  OUT2, WASHOUT)

N = 128
TASK_SET = ["narma10", "mackey17"]


def calibrate_eq(struct, N, seed=42, T_cal=600):
    """Run reservoir on calibration noise, return per-column std."""
    rng = np.random.default_rng(seed)
    u_cal = rng.uniform(-1, 1, size=T_cal)
    W, Win = build_reservoir(struct, N, seed=seed)
    X = run_reservoir(u_cal, W, Win, struct, N)
    col_std = np.std(X[WASHOUT:], axis=0)
    return col_std + 1e-6


def kappa(X, alpha=1e-6):
    """Condition number of X^T X + alpha I."""
    XtX = X.T @ X + alpha * np.eye(X.shape[1])
    try:
        s = np.linalg.svd(XtX, compute_uv=False)
        return float(s[0] / s[-1])
    except Exception:
        return float("inf")


def env_keyed_equalized(vec, N, seed=0, T_cal=600):
    """Build a structure where:
       - mask, perm, acts: from envelope (binding axes)
       - weight_scale: envelope-base * 1/sqrt(calibration variance)
       - leak: from envelope
    """
    s_env = derive_structure_v2(vec, N)
    col_std = calibrate_eq(s_env, N, seed=seed, T_cal=T_cal)
    # eq is per neuron — apply to weight_scale to flatten column variance
    # but we apply per-row scale change which means inputs to each neuron — not exact
    # simple alternative: scale Win per neuron post-hoc. Skip Win, apply to weight_scale.
    new_scale = s_env["weight_scale"] / np.sqrt(col_std / (np.median(col_std) + 1e-9))
    s_env["weight_scale"] = np.clip(new_scale, 0.05, 5.0)
    return s_env


def baseline_equalized(N, seed=0, T_cal=600):
    s_base = baseline_structure_v2(N, seed=seed)
    col_std = calibrate_eq(s_base, N, seed=seed, T_cal=T_cal)
    s_base["weight_scale"] = np.clip(1.0 / np.sqrt(col_std / (np.median(col_std) + 1e-9)),
                                      0.05, 5.0)
    return s_base


def eval_struct(struct, N, seed, task):
    if task == "narma10":
        u_tr, y_tr = narma10(1500, seed=seed*13+7)
        u_te, y_te = narma10(400, seed=seed*13+9991)
    elif task == "mackey17":
        u_tr, y_tr = mackey_glass(1500, seed=seed*13+7)
        u_te, y_te = mackey_glass(400, seed=seed*13+9991)
    elif task == "memcap":
        u_tr, y_tr = memory_capacity_task(1500, seed=seed*13+7)
        u_te, y_te = memory_capacity_task(400, seed=seed*13+9991)
    else:
        raise ValueError(task)
    W, Win = build_reservoir(struct, N, seed=seed)
    X_tr = run_reservoir(u_tr, W, Win, struct, N)
    k_tr = kappa(X_tr[WASHOUT:])
    Wout = ridge_fit(X_tr[WASHOUT:], y_tr[WASHOUT:])
    X_te = run_reservoir(u_te, W, Win, struct, N)
    y_hat = ridge_predict(X_te[WASHOUT:], Wout)
    if task == "memcap":
        mc = 0.0
        for k in range(y_te.shape[1]):
            r = np.corrcoef(y_hat[:, k], y_te[WASHOUT:, k])[0, 1]
            if np.isfinite(r): mc += r * r
        return float(mc), k_tr  # higher better
    return nrmse(y_te[WASHOUT:], y_hat), k_tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--tasks", default=",".join(TASK_SET))
    args = ap.parse_args()
    out_dir = OUT2 / "phase_f"; out_dir.mkdir(parents=True, exist_ok=True)
    vk = load_vec("ikaros")
    rng = np.random.default_rng(0xF00D)
    v_random = (rng.standard_normal(23) * np.std(vk) + np.mean(vk)).tolist()

    tasks = args.tasks.split(",")
    out = {"N": N, "seeds": args.seeds, "tasks": tasks, "results": {}}

    for task in tasks:
        per_cond = {"baseline_plain": [], "baseline_eq": [], "envelope_plain": [],
                    "envelope_eq": [], "random_envelope_eq": []}
        per_kappa = {k: [] for k in per_cond}
        for s in range(args.seeds):
            t0 = time.time()
            s_base = baseline_structure_v2(N, seed=0)
            s_base_eq = baseline_equalized(N, seed=42)
            s_env = derive_structure_v2(vk, N)
            s_env_eq = env_keyed_equalized(vk, N, seed=42)
            s_rnd_eq = env_keyed_equalized(v_random, N, seed=42)
            for label, struct in [("baseline_plain", s_base),
                                   ("baseline_eq", s_base_eq),
                                   ("envelope_plain", s_env),
                                   ("envelope_eq", s_env_eq),
                                   ("random_envelope_eq", s_rnd_eq)]:
                sc, kap = eval_struct(struct, N, s, task)
                per_cond[label].append(sc); per_kappa[label].append(kap)
            print(f"[F2][{task}] seed={s} (t={time.time()-t0:.1f}s)", flush=True)
        # report
        higher_better = (task == "memcap")
        agg = {}
        for label in per_cond:
            arr = np.array(per_cond[label])
            agg[label] = {"median": float(np.median(arr)),
                          "mean": float(np.mean(arr)),
                          "std": float(np.std(arr)),
                          "kappa_median": float(np.median(per_kappa[label])),
                          "per_seed": per_cond[label]}
        # compute improvements vs baseline_plain
        base = agg["baseline_plain"]["median"]
        for label in per_cond:
            v = agg[label]["median"]
            if higher_better:
                improvement = (v - base) / max(1e-9, abs(base)) * 100
            else:
                improvement = (base - v) / max(1e-9, abs(base)) * 100
            agg[label]["improvement_vs_baseline_plain_pct"] = improvement
        # C5 win check: env_eq beats baseline_eq by ≥10% AND env_eq beats random_env_eq
        env_eq = agg["envelope_eq"]["median"]; base_eq = agg["baseline_eq"]["median"]
        rnd_eq = agg["random_envelope_eq"]["median"]
        if higher_better:
            ratio = (env_eq - base_eq) / max(1e-9, abs(base_eq)) * 100
            beats_rnd = env_eq > rnd_eq
        else:
            ratio = (base_eq - env_eq) / max(1e-9, abs(base_eq)) * 100
            beats_rnd = env_eq < rnd_eq
        agg["C5_check"] = {"env_eq_vs_base_eq_pct": ratio,
                           "env_eq_beats_random_eq": bool(beats_rnd),
                           "C5_WIN": bool(ratio >= 10.0 and beats_rnd)}
        out["results"][task] = agg
        print(f"[F2][{task}] === SUMMARY ===", flush=True)
        for label in per_cond:
            print(f"  {label}: med={agg[label]['median']:.4f}  kappa={agg[label]['kappa_median']:.2e}  improvement={agg[label]['improvement_vs_baseline_plain_pct']:+.2f}%", flush=True)
        print(f"  C5_check: env_eq vs base_eq = {ratio:+.2f}%  beats_rnd={beats_rnd}  WIN={agg['C5_check']['C5_WIN']}", flush=True)

    # any wins?
    out["any_C5_WIN"] = any(out["results"][t]["C5_check"]["C5_WIN"] for t in tasks)
    (out_dir / "F2_result.json").write_text(json.dumps(out, indent=2))
    print(f"[F2] wrote F2_result.json  any_C5_WIN={out['any_C5_WIN']}", flush=True)


if __name__ == "__main__":
    main()
