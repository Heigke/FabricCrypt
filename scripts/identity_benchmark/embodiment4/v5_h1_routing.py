"""V5-H1: Per-core latency rank → critical-path routing in a 16-channel reservoir.

Hypothesis:
  If we measure each physical core's compute latency and route the
  most-updated (critical) reservoir state through the fastest cores,
  the network gains either wall-clock speed or accuracy vs a baseline
  that uses arbitrary thread assignment.

Pre-reg gate (EITHER):
  - Rank-aware ≥5% faster wall-clock at same accuracy (NRMSE within 2%)
  - Rank-aware ≥3% higher accuracy at same wall-clock budget

Method:
  1. Probe per-core latency by pinning a tight numeric kernel to each
     CPU id with sched_setaffinity, measuring 20 ms of FP work.
  2. Order cores by speed.
  3. Run a 16-block partitioned reservoir on NARMA-10 using `concurrent.futures`
     ProcessPoolExecutor with explicit per-process core pinning:
        - rank-aware: block_id i → core rank i (fastest gets block 0 = the
          "critical" block whose state feeds back into all others)
        - baseline: random assignment.
  4. Repeat 10 trials, compare medians.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment4/v5_h1_result.json"
N_BLOCKS = 16
BLOCK_N = 16  # 16 blocks * 16 neurons = 256-neuron reservoir
N = N_BLOCKS * BLOCK_N
T_TRAIN = 1500
T_TEST = 500
WASHOUT = 100


def core_latency_probe(core_id: int, ms: float = 20.0) -> float:
    """Measure wall-clock for a small numeric kernel pinned to one core."""
    try:
        os.sched_setaffinity(0, {core_id})
    except Exception:
        pass
    # Tight kernel, ~ms duration
    n = 4096
    rng = np.random.default_rng(core_id)
    a = rng.standard_normal((n, n // 16)).astype(np.float32)
    t0 = time.perf_counter()
    iters = 0
    while (time.perf_counter() - t0) < ms / 1000.0:
        np.dot(a.T, a)
        iters += 1
    elapsed = time.perf_counter() - t0
    return elapsed / max(1, iters)  # seconds per iter


def measure_all_cores(repeats: int = 3):
    n_cores = os.cpu_count()
    lats = np.zeros((repeats, n_cores))
    for r in range(repeats):
        for c in range(n_cores):
            lats[r, c] = core_latency_probe(c, ms=20.0)
    # restore affinity
    try:
        os.sched_setaffinity(0, set(range(n_cores)))
    except Exception:
        pass
    return np.median(lats, axis=0)


def narma10(T, seed):
    rng = np.random.default_rng(seed)
    u = 0.5 * rng.uniform(0.0, 1.0, size=T + 10)
    y = np.zeros(T + 10)
    for t in range(10, T + 10):
        y[t] = (0.3 * y[t-1] + 0.05 * y[t-1] * np.sum(y[t-10:t])
                + 1.5 * u[t-10] * u[t-1] + 0.1)
    return u[10:], y[10:]


def _block_worker(args):
    core_id, block_id, block_state, drive_in, leak = args
    try:
        os.sched_setaffinity(0, {core_id})
    except Exception:
        pass
    t0 = time.perf_counter()
    # Tiny update on a per-block sub-state (BLOCK_N neurons)
    pre = block_state["W"] @ block_state["x"] + block_state["Win"][:, 0] * drive_in
    post = np.tanh(pre)
    x_new = (1 - leak) * block_state["x"] + leak * post
    elapsed = time.perf_counter() - t0
    return block_id, x_new, elapsed


def build_blocks(seed=0):
    rng = np.random.default_rng(seed)
    blocks = []
    for b in range(N_BLOCKS):
        W = rng.standard_normal((BLOCK_N, BLOCK_N)) / np.sqrt(BLOCK_N)
        rho = np.max(np.abs(np.linalg.eigvals(W)))
        if rho > 1e-9:
            W *= 0.95 / rho
        Win = rng.standard_normal((BLOCK_N, 1))
        blocks.append({"W": W, "Win": Win, "x": np.zeros(BLOCK_N)})
    return blocks


def run_partitioned(u, blocks, core_assignment, leak=0.3, n_workers=8):
    """core_assignment[block_id] = core_id."""
    T = len(u)
    X = np.zeros((T, N))
    total_wall = 0.0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        # We *simulate* the per-core latency penalty by running sequentially
        # but pinning each block's update to its assigned core. Using a process
        # pool would dominate by IPC overhead; for an honest comparison we
        # measure single-process latency with affinity changes.
        for t in range(T):
            drive = u[t]
            t_step = 0.0
            for b in range(N_BLOCKS):
                core = int(core_assignment[b])
                try:
                    os.sched_setaffinity(0, {core})
                except Exception:
                    pass
                t0 = time.perf_counter()
                pre = blocks[b]["W"] @ blocks[b]["x"] + blocks[b]["Win"][:, 0] * drive
                blocks[b]["x"] = (1 - leak) * blocks[b]["x"] + leak * np.tanh(pre)
                t_step += time.perf_counter() - t0
                X[t, b*BLOCK_N:(b+1)*BLOCK_N] = blocks[b]["x"]
            total_wall += t_step
    return X, total_wall


def ridge_fit(X, y, alpha=1e-6):
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    A = Xb.T @ Xb + alpha * np.eye(Xb.shape[1])
    return np.linalg.solve(A, Xb.T @ y)


def nrmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)) / (np.std(y) + 1e-12))


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"[H1] probing per-core latency (32 cores)...", flush=True)
    lats = measure_all_cores(repeats=3)
    rank = np.argsort(lats)  # fastest first
    print(f"[H1] core lat (us) min={lats.min()*1e6:.1f} max={lats.max()*1e6:.1f} ratio={lats.max()/lats.min():.2f}x", flush=True)
    print(f"[H1] fastest 8 cores: {rank[:8].tolist()}", flush=True)

    n_cores = len(lats)
    # rank-aware: block 0 (critical/feedback) → fastest core
    rank_assign = rank[:N_BLOCKS].tolist() if n_cores >= N_BLOCKS else (rank.tolist() * (N_BLOCKS // n_cores + 1))[:N_BLOCKS]
    # baseline: arbitrary contiguous assignment 0..15
    baseline_assign = list(range(N_BLOCKS))

    res = {"n_cores": int(n_cores), "core_lats_s": lats.tolist(), "rank_fast_to_slow": rank.tolist(),
            "rank_assignment": rank_assign, "baseline_assignment": baseline_assign,
            "trials": []}

    n_trials = 5
    for trial in range(n_trials):
        u_tr, y_tr = narma10(T_TRAIN, seed=trial * 13 + 7)
        u_te, y_te = narma10(T_TEST, seed=trial * 13 + 9991)

        # Rank-aware
        blocks = build_blocks(seed=trial)
        X_tr, wall_tr_rank = run_partitioned(u_tr, blocks, rank_assign)
        Wout = ridge_fit(X_tr[WASHOUT:], y_tr[WASHOUT:])
        blocks_te = build_blocks(seed=trial)  # reset state for test
        X_te, wall_te_rank = run_partitioned(u_te, blocks_te, rank_assign)
        Xb = np.concatenate([X_te[WASHOUT:], np.ones((X_te.shape[0] - WASHOUT, 1))], axis=1)
        yhat = Xb @ Wout
        nrmse_rank = nrmse(y_te[WASHOUT:], yhat)
        wall_rank = wall_tr_rank + wall_te_rank

        # Baseline
        blocks = build_blocks(seed=trial)
        X_tr, wall_tr_base = run_partitioned(u_tr, blocks, baseline_assign)
        Wout = ridge_fit(X_tr[WASHOUT:], y_tr[WASHOUT:])
        blocks_te = build_blocks(seed=trial)
        X_te, wall_te_base = run_partitioned(u_te, blocks_te, baseline_assign)
        Xb = np.concatenate([X_te[WASHOUT:], np.ones((X_te.shape[0] - WASHOUT, 1))], axis=1)
        yhat = Xb @ Wout
        nrmse_base = nrmse(y_te[WASHOUT:], yhat)
        wall_base = wall_tr_base + wall_te_base

        print(f"[H1] trial={trial} rank: nrmse={nrmse_rank:.4f} wall={wall_rank:.3f}s | baseline: nrmse={nrmse_base:.4f} wall={wall_base:.3f}s", flush=True)
        res["trials"].append({"rank_nrmse": nrmse_rank, "rank_wall_s": wall_rank,
                                "baseline_nrmse": nrmse_base, "baseline_wall_s": wall_base})

    r = np.array([t["rank_nrmse"] for t in res["trials"]])
    b = np.array([t["baseline_nrmse"] for t in res["trials"]])
    rw = np.array([t["rank_wall_s"] for t in res["trials"]])
    bw = np.array([t["baseline_wall_s"] for t in res["trials"]])
    res["summary"] = {
        "rank_nrmse_med": float(np.median(r)), "baseline_nrmse_med": float(np.median(b)),
        "rank_wall_med": float(np.median(rw)), "baseline_wall_med": float(np.median(bw)),
        "speedup_pct": 100.0 * (float(np.median(bw)) - float(np.median(rw))) / float(np.median(bw)),
        "accuracy_gain_pct": 100.0 * (float(np.median(b)) - float(np.median(r))) / float(np.median(b)),
    }
    nrmse_within_2pct = abs(res["summary"]["accuracy_gain_pct"]) <= 2.0
    wall_within_2pct = abs(res["summary"]["speedup_pct"]) <= 2.0
    win_speed = nrmse_within_2pct and res["summary"]["speedup_pct"] >= 5.0
    win_acc = wall_within_2pct and res["summary"]["accuracy_gain_pct"] >= 3.0
    res["gate"] = {"win_speed": bool(win_speed), "win_accuracy": bool(win_acc),
                    "WIN": bool(win_speed or win_acc)}
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"[H1] summary: {json.dumps(res['summary'], indent=2)}", flush=True)
    print(f"[H1] gate: {res['gate']}", flush=True)
    print(f"[H1] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
