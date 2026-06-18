"""Phase F — performance advantage hunt.

C5 baseline: vanilla deterministic mask, tanh activations, no scaling.
For each hypothesis, design an envelope-derived configuration that EXPLOITS
some property of the host substrate, and test whether it beats baseline.

Hypotheses tested here (numpy-tractable subset):
  H1: envelope-tuned sparsity matched to power-budget peak → MC efficiency
  H2: per-position weight scale from envelope = analog of CU heterogeneity
  H3: envelope-determined leak schedule
  H4: envelope-induced dropout pattern (free regularization)
  H5: envelope-driven ensemble of small reservoirs vs. one large

Constructive falsifier: each "win" must be accompanied by a "random-envelope
must be worse" check. If random envelope ties or beats the actual envelope,
the win is not envelope-attributed.

Pre-registered C5 WIN gate: envelope config beats baseline by ≥10% on at
least one task, AND random-envelope config is worse than actual envelope.
"""
from __future__ import annotations
import json, sys, time, argparse, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _lib import (load_vec, derive_structure_v2, baseline_structure_v2,
                  build_reservoir, run_reservoir, ridge_fit, ridge_predict,
                  nrmse, narma10, mackey_glass, memory_capacity_task,
                  apply_act, OUT2, WASHOUT, env_hash, ACT_CHOICES)

N = 128
TASKS = ["narma10", "mackey17", "memcap"]


def env_to_sparsity(vec, lo=0.10, hi=0.50):
    """H1: envelope→sparsity. Use median of vec mod 100 to pick target density."""
    h = hashlib.sha256(np.asarray(vec).tobytes()).digest()
    u = int.from_bytes(h[:4], "big") / (2**32)
    return lo + (hi - lo) * u


def env_struct_sparsity(vec, N, target_density):
    """Build mask with target density derived from envelope, rest deterministic."""
    h = env_hash(vec, n_bytes=N*N // 8 + 1024)
    bits = np.unpackbits(np.frombuffer(h[:N*N//8], dtype=np.uint8)).reshape(N, N)
    # threshold by target density
    if target_density >= 0.5:
        mask = bits.astype(bool)
    else:
        # use top-k bits
        flat_h = np.frombuffer(h[N*N//8:N*N//8 + N*N], dtype=np.uint8) if len(h) >= 2*N*N//8 + N*N else None
        # simpler: pick random sub-mask matching target
        rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
        mask = (rng.random((N, N)) < target_density)
    np.fill_diagonal(mask, False)
    return {"mask": mask, "acts": ["tanh"] * N,
            "perm": np.arange(N, dtype=np.int32),
            "weight_scale": np.ones(N),
            "leak": np.full(N, 0.3)}


def env_struct_scale_only(vec, N):
    """H2: per-position weight scale from envelope, rest deterministic baseline."""
    base = baseline_structure_v2(N, seed=0)
    s = derive_structure_v2(vec, N)
    base["weight_scale"] = s["weight_scale"]
    return base


def env_struct_leak_only(vec, N):
    """H3: per-neuron leak from envelope, rest baseline."""
    base = baseline_structure_v2(N, seed=0)
    s = derive_structure_v2(vec, N)
    base["leak"] = s["leak"]
    return base


def env_dropout_pattern(vec, N, dropout=0.2):
    """H4: envelope-derived fixed dropout pattern (kept neurons), rest baseline."""
    h = env_hash(vec, n_bytes=N + 256)
    bytes_n = np.frombuffer(h[:N], dtype=np.uint8)
    sorted_idx = np.argsort(bytes_n)
    keep = np.zeros(N, dtype=bool)
    keep[sorted_idx[int(N * dropout):]] = True
    base = baseline_structure_v2(N, seed=0)
    # zero out columns AND rows corresponding to dropped neurons
    drop_mask = np.outer(keep, keep)
    base["mask"] = base["mask"] & drop_mask
    return base


def env_ensemble_signature(vec, N_total=128, K=4):
    """H5: ensemble of K small reservoirs of size N_total/K, structures derived from
    different envelope hash slices."""
    Ns = [N_total // K] * K
    structs = []
    for i in range(K):
        h = env_hash(vec, n_bytes=4096)
        # offset slice
        sub_vec = np.concatenate([np.asarray(vec), [float(i)]])
        s = derive_structure_v2(sub_vec, Ns[i])
        structs.append(s)
    return structs, Ns


def task_data(task, seed, T_tr=1500, T_te=400):
    if task == "narma10":
        u_tr, y_tr = narma10(T_tr, seed=seed*13+7); u_te, y_te = narma10(T_te, seed=seed*13+9991)
        return u_tr, y_tr, u_te, y_te, False
    elif task == "mackey17":
        u_tr, y_tr = mackey_glass(T_tr, seed=seed*13+7); u_te, y_te = mackey_glass(T_te, seed=seed*13+9991)
        return u_tr, y_tr, u_te, y_te, False
    elif task == "memcap":
        u_tr, y_tr = memory_capacity_task(T_tr, seed=seed*13+7)
        u_te, y_te = memory_capacity_task(T_te, seed=seed*13+9991)
        return u_tr, y_tr, u_te, y_te, True
    raise ValueError(task)


def eval_score(struct, N, seed, task):
    u_tr, y_tr, u_te, y_te, is_multi = task_data(task, seed)
    W, Win = build_reservoir(struct, N, seed=seed)
    X_tr = run_reservoir(u_tr, W, Win, struct, N)
    Wout = ridge_fit(X_tr[WASHOUT:], y_tr[WASHOUT:])
    X_te = run_reservoir(u_te, W, Win, struct, N)
    y_hat = ridge_predict(X_te[WASHOUT:], Wout)
    if is_multi:  # memcap
        mc = 0.0
        for k in range(y_te.shape[1]):
            r = np.corrcoef(y_hat[:, k], y_te[WASHOUT:, k])[0, 1]
            if np.isfinite(r): mc += r * r
        return float(mc), "memory_capacity_higher_better"
    return nrmse(y_te[WASHOUT:], y_hat), "nrmse_lower_better"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--tasks", default=",".join(TASKS))
    args = ap.parse_args()
    out_dir = OUT2 / "phase_f"; out_dir.mkdir(parents=True, exist_ok=True)
    vk = load_vec("ikaros")
    rng = np.random.default_rng(0xFADE)
    v_random = (rng.standard_normal(23) * np.std(vk) + np.mean(vk)).tolist()
    s_base = baseline_structure_v2(N, seed=0)

    tasks = args.tasks.split(",")
    results = {"N": N, "seeds": args.seeds, "tasks": tasks, "hypotheses": {}}

    hypotheses = {
        "H1_sparsity": (
            lambda v: env_struct_sparsity(v, N, env_to_sparsity(v)),
            "envelope-tuned sparsity"),
        "H2_scale": (lambda v: env_struct_scale_only(v, N), "per-neuron weight scale from envelope"),
        "H3_leak": (lambda v: env_struct_leak_only(v, N), "per-neuron leak from envelope"),
        "H4_dropout": (lambda v: env_dropout_pattern(v, N, 0.2), "envelope-fixed dropout pattern"),
        "H5_full_keyed": (lambda v: derive_structure_v2(v, N), "full envelope-keyed (all axes, baseline comparison)"),
    }

    # baseline once per task
    baseline_scores = {}
    for task in tasks:
        scores = []
        for s in range(args.seeds):
            sc, conv = eval_score(s_base, N, s, task)
            scores.append(sc)
        baseline_scores[task] = {"median": float(np.median(scores)), "per_seed": scores, "conv": conv}
        print(f"[F][baseline][{task}] median={baseline_scores[task]['median']:.4f} ({conv})", flush=True)

    for hname, (builder, desc) in hypotheses.items():
        print(f"[F] === {hname}: {desc} ===", flush=True)
        s_env = builder(vk)
        s_rnd = builder(v_random)
        for task in tasks:
            env_scores, rnd_scores = [], []
            for s in range(args.seeds):
                t0 = time.time()
                try:
                    sc_e, conv = eval_score(s_env, N, s, task)
                    sc_r, _ = eval_score(s_rnd, N, s, task)
                    env_scores.append(sc_e); rnd_scores.append(sc_r)
                    print(f"[F][{hname}][{task}] seed={s} env={sc_e:.4f} rnd={sc_r:.4f} base={baseline_scores[task]['per_seed'][s]:.4f}  (t={time.time()-t0:.1f}s)", flush=True)
                except Exception as e:
                    print(f"[F][{hname}][{task}] seed={s} EXCEPTION: {e}", flush=True)
            if not env_scores:
                continue
            base_med = baseline_scores[task]["median"]
            env_med = float(np.median(env_scores)); rnd_med = float(np.median(rnd_scores))
            if "higher" in baseline_scores[task]["conv"]:
                # MC: higher better
                improvement_pct = (env_med - base_med) / max(1e-9, abs(base_med)) * 100
                env_beats_random = env_med > rnd_med
            else:
                # nrmse: lower better
                improvement_pct = (base_med - env_med) / max(1e-9, abs(base_med)) * 100
                env_beats_random = env_med < rnd_med
            win = improvement_pct >= 10.0 and env_beats_random
            results["hypotheses"].setdefault(hname, {})[task] = {
                "env_median": env_med, "random_env_median": rnd_med,
                "baseline_median": base_med,
                "improvement_pct_vs_baseline": improvement_pct,
                "env_beats_random": bool(env_beats_random),
                "C5_WIN": bool(win),
                "env_per_seed": env_scores, "random_per_seed": rnd_scores,
            }
            print(f"[F][{hname}][{task}] env_med={env_med:.4f} base={base_med:.4f} improvement={improvement_pct:.2f}%  beats_random={env_beats_random}  C5_WIN={win}", flush=True)

    # any wins?
    wins = []
    for hname, per_t in results["hypotheses"].items():
        for task, r in per_t.items():
            if r.get("C5_WIN"):
                wins.append({"hypothesis": hname, "task": task, "improvement_pct": r["improvement_pct_vs_baseline"]})
    results["wins"] = wins
    results["any_C5_WIN"] = bool(wins)
    print(f"[F] === SUMMARY === C5_wins={len(wins)}: {wins}", flush=True)
    (out_dir / "F_result.json").write_text(json.dumps(results, indent=2))
    print(f"[F] wrote F_result.json", flush=True)


if __name__ == "__main__":
    main()
