"""V4 Advantage hunt for embodiment3 — ONLY meaningful if V3 G1+G3 pass.

Tests whether the (now genuinely-chassi-bound) robust signature can give a
performance advantage. Per oracle O106: "Stop hashing. Start mapping."
The headline novel hypothesis here is **permutation-as-delay-line**:
- map per-core latency rank into the reservoir update permutation, so
  fast cores process early stages of the chain and slow cores process later
  stages. This may exploit real chassi-specific delay structure.

Comparison conditions (NARMA-10 by default):
  baseline   : random structure (seed=0)
  random_env : structure derived from a random "envelope" bitstring (control)
  hashed_env : structure from sha256(robust bitstring) — what V3 used
  mapped_env : permutation := per-core latency rank order (causal mapping)

Win criterion: mapped_env NRMSE ≤ 0.9 × min(baseline, random_env) for some task.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robust_signature import (load_signature, quantize_robust,
                                quantized_to_bitstring)
from v3_phase_c import (derive_structure, baseline_structure, train_eval,
                          N, narma10, run_reservoir, ridge_fit, ridge_predict,
                          nrmse, build_reservoir, WASHOUT, T_TRAIN, T_TEST,
                          ACT_CHOICES)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment3/phase_v4"
SIGS = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment3/signatures"
OUT.mkdir(parents=True, exist_ok=True)


def mackey_glass(T, tau=17, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(T + tau + 100)
    x[:tau + 1] = 1.2 + 0.01 * rng.standard_normal(tau + 1)
    for t in range(tau, T + tau + 99):
        x[t + 1] = x[t] + (0.2 * x[t - tau] / (1 + x[t - tau] ** 10) - 0.1 * x[t])
    series = x[tau + 100:]
    # Predict x[t+5] from x[t]
    u = series[:-5]
    y = series[5:]
    return u[:T], y[:T]


def memory_capacity(T, k_max=20, seed=0):
    """Standard MC task: predict delayed input u[t-k]. Reservoir-friendly."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1, 1, size=T + k_max)
    return u[k_max:], u  # u: full, predict u[t-k] for various k


def mapped_struct_from_signature(sig_path: Path, seed: int = 0):
    """Permutation := per-core latency rank. Mask + acts derived from
    quantized bitstring (so still chassi-bound, but perm has CAUSAL meaning)."""
    sig = load_signature(str(sig_path))
    q = quantize_robust(sig)
    bs = quantized_to_bitstring(q)
    # Mask + acts from bitstring (as in V3)
    mask, acts, _ = derive_structure(bs)
    # Permutation := per-core latency rank, tiled to N
    pc = q.get("per_core_bins", {})
    if pc:
        items = sorted(pc.items())
        ranks = np.array([v for _, v in items], dtype=np.int32)  # 0..k-1
        # Tile/repeat to N, breaking ties with a per-bitstring hash
        rng = np.random.default_rng(int(bs[:32], 2) % (2 ** 31))
        order = np.argsort(ranks)
        # Build N-length permutation by repeating tile + small perturbation
        tile = np.tile(order, (N // len(order)) + 1)[:N]
        # Add tiny rng-based perturbation to break ties
        keys = np.array([t * 10000 + rng.integers(10000) for t in tile])
        perm = np.argsort(keys).astype(np.int32)
    else:
        perm = np.arange(N, dtype=np.int32)
    return mask, acts, perm


def random_env_struct(seed):
    """Random 'envelope' bitstring (control for hashed_env)."""
    rng = np.random.default_rng(seed + 99991)
    bs = "".join(str(b) for b in rng.integers(0, 2, size=2048))
    return derive_structure(bs)


def eval_task(struct, seed, task="narma10"):
    """Returns NRMSE."""
    if task == "narma10":
        u_tr, y_tr = narma10(T_TRAIN, seed=seed * 13 + 7)
        u_te, y_te = narma10(T_TEST, seed=seed * 13 + 9991)
    elif task == "mackey":
        u_tr, y_tr = mackey_glass(T_TRAIN, seed=seed * 13 + 7)
        u_te, y_te = mackey_glass(T_TEST, seed=seed * 13 + 9991)
    else:
        raise ValueError(task)

    mask, acts, perm = struct
    W, Win = build_reservoir(mask, seed=seed)
    X_tr = run_reservoir(u_tr, W, Win, acts, perm, leak=0.3)
    Wout = ridge_fit(X_tr[WASHOUT:], y_tr[WASHOUT:])
    X_te = run_reservoir(u_te, W, Win, acts, perm, leak=0.3)
    y_hat = ridge_predict(X_te[WASHOUT:], Wout)
    return nrmse(y_te[WASHOUT:], y_hat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tasks", default="narma10,mackey")
    args = ap.parse_args()

    sig_path = SIGS / "ikaros_v2a_t0.json"
    if not sig_path.exists():
        print(f"[V4] missing signature {sig_path}", flush=True)
        return

    sig = load_signature(str(sig_path))
    q = quantize_robust(sig)
    bs = quantized_to_bitstring(q)

    struct_hashed = derive_structure(bs)
    struct_mapped = mapped_struct_from_signature(sig_path)
    print(f"[V4] hashed_env mask_density={struct_hashed[0].mean():.3f}", flush=True)
    print(f"[V4] mapped_env mask_density={struct_mapped[0].mean():.3f}", flush=True)
    print(f"[V4] per_core_n={len(q.get('per_core_bins', {}))}", flush=True)

    tasks = args.tasks.split(",")
    seeds = list(range(args.seeds))
    results = {"tasks": tasks, "seeds": seeds, "per_core_n": len(q.get("per_core_bins", {}))}

    for task in tasks:
        per_cond = {}
        for cond_name, struct_fn in [
            ("baseline", lambda s: baseline_structure(seed=s)),
            ("random_env", lambda s: random_env_struct(s)),
            ("hashed_env", lambda s: struct_hashed),
            ("mapped_env", lambda s: struct_mapped),
        ]:
            errs = []
            for s in seeds:
                try:
                    e = eval_task(struct_fn(s), s, task=task)
                except Exception as ex:
                    print(f"[V4][{task}][{cond_name}] seed={s} err={ex}", flush=True)
                    e = float("nan")
                errs.append(e)
                print(f"[V4][{task}][{cond_name}] seed={s} NRMSE={e:.4f}", flush=True)
            per_cond[cond_name] = {"per_seed": errs,
                                    "median": float(np.nanmedian(errs))}
        # win check: mapped_env median ≤ 0.9 × min(baseline, random)
        m_map = per_cond["mapped_env"]["median"]
        m_base = per_cond["baseline"]["median"]
        m_rand = per_cond["random_env"]["median"]
        m_hash = per_cond["hashed_env"]["median"]
        threshold = 0.9 * min(m_base, m_rand)
        win = m_map <= threshold and np.isfinite(m_map)
        results[task] = {
            "per_cond": per_cond,
            "win_threshold": threshold,
            "mapped_env_median": m_map,
            "WIN_mapped_vs_both": bool(win),
            "advantage_pct": (1 - m_map / min(m_base, m_rand)) * 100 if min(m_base, m_rand) > 0 else 0.0,
        }
        print(f"[V4][{task}] mapped={m_map:.4f} base={m_base:.4f} rand={m_rand:.4f} hash={m_hash:.4f} → WIN={win}", flush=True)

    any_win = any(results[t].get("WIN_mapped_vs_both", False) for t in tasks)
    results["ANY_WIN"] = bool(any_win)
    out = OUT / "v4_result.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"[V4] wrote {out}  ANY_WIN={any_win}", flush=True)


if __name__ == "__main__":
    main()
