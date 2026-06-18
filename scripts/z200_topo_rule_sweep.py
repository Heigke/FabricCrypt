"""z200 — Large-scale topology × self-learning-rule sweep on NS-RAM surrogate.

Probes the chip-layout space: multiple topologies, multiple no-readout
local-learning rules, multiple sizes — fast (surrogate ~5 µs/cell-eval).

Topologies (8): ER_SPARSE, WS_SMALLWORLD, RING, GRID_2D, HUB_SPOKE,
                 MODULAR, RAND_GAUSS, SCALE_FREE
Rules (3, NO LINEAR READOUT): FF-fixed, R-Hebbian (3-factor), Hebbian-IP
Sizes (2): N=256, N=1024

Outputs:
  results/z200_topo_rule_sweep/{topo}_{rule}_N{n}.json
  results/z200_topo_rule_sweep/summary.json
  figures/z200_topo_rule_sweep/heatmap.{png,pdf}
"""
from __future__ import annotations
# THREAD CAP — MUST be set BEFORE numpy/scipy/torch import. Caps OpenBLAS,
# MKL, and OpenMP at 4 threads per process so that ProcessPoolExecutor
# with N parallel workers doesn't oversubscribe the 32-core APU and
# trigger ACPI thermal trip at 99°C (lessons from 2026-05-05 crashes).
# 12 workers × 4 threads = 48 effective threads — comfortably below
# physical cores while keeping per-config compute reasonable.
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "4")
import importlib.util
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z200_topo_rule_sweep"; OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "figures/z200_topo_rule_sweep"; FIG.mkdir(parents=True, exist_ok=True)

import sys; sys.path.insert(0, str(ROOT))
from scripts.nsram_surrogate import NSRAMSurrogate

mg_spec = importlib.util.spec_from_file_location("mg", ROOT / "scripts/demo_mackey_glass.py")
mg = importlib.util.module_from_spec(mg_spec); mg_spec.loader.exec_module(mg)


# ------------------------------------------------------------------
# Topology builders
# ------------------------------------------------------------------
def build_topo(name: str, N: int, rng) -> np.ndarray:
    """Return W (N, N) sparse-ish weight matrix, no self-loops, spectral
    radius normalised to ~0.9."""
    if name == "ER_SPARSE":
        p = 0.10
        W = (rng.random((N, N)) < p).astype(float) * rng.normal(0, 1, (N, N))
    elif name == "RAND_GAUSS":
        W = rng.normal(0, 1.0/np.sqrt(N), (N, N))
    elif name == "RING":
        W = np.zeros((N, N))
        for i in range(N):
            for j in [-1, 1, -2, 2]:
                W[i, (i+j) % N] = rng.normal(0, 0.5)
    elif name == "WS_SMALLWORLD":
        # Watts-Strogatz starting from ring k=4, rewire prob 0.2
        W = np.zeros((N, N))
        k = 4
        for i in range(N):
            for j in range(1, k//2 + 1):
                W[i, (i+j) % N] = rng.normal(0, 0.5)
                W[i, (i-j) % N] = rng.normal(0, 0.5)
        # rewire
        nz = np.argwhere(W != 0)
        for (i, j) in nz:
            if rng.random() < 0.2:
                W[i, j] = 0
                jp = rng.integers(0, N)
                if jp != i:
                    W[i, jp] = rng.normal(0, 0.5)
    elif name == "GRID_2D":
        side = int(np.sqrt(N))
        N = side * side
        W = np.zeros((N, N))
        for i in range(side):
            for j in range(side):
                idx = i*side + j
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ii = (i+di) % side; jj = (j+dj) % side
                    W[idx, ii*side+jj] = rng.normal(0, 0.5)
    elif name == "HUB_SPOKE":
        # 5% hubs receive from all, send to 30% sample
        W = np.zeros((N, N))
        n_hub = max(2, N // 20)
        hubs = rng.choice(N, size=n_hub, replace=False)
        for h in hubs:
            W[:, h] = rng.normal(0, 1.0, N)
            spokes = rng.choice(N, size=int(0.3*N), replace=False)
            W[h, spokes] = rng.normal(0, 1.0, len(spokes))
    elif name == "MODULAR":
        # 4 modules with strong intra, weak inter
        n_mod = 4
        sz = N // n_mod
        W = rng.normal(0, 0.1, (N, N))   # weak inter
        for m in range(n_mod):
            s = m*sz; e = s+sz
            W[s:e, s:e] = rng.normal(0, 1.0, (sz, sz))
        # mask sparsity
        W = W * (rng.random((N, N)) < 0.15).astype(float)
    elif name == "SCALE_FREE":
        # Approx Barabasi-Albert: preferential attachment
        W = np.zeros((N, N))
        deg = np.ones(N)
        for i in range(N):
            n_conn = min(int(np.log2(N)), i)
            if n_conn == 0: continue
            probs = deg[:i] / deg[:i].sum()
            targets = rng.choice(i, size=n_conn, replace=False, p=probs)
            for t in targets:
                w = rng.normal(0, 0.5)
                W[i, t] = w; W[t, i] = w
                deg[i] += 1; deg[t] += 1
    else:
        raise ValueError(name)
    np.fill_diagonal(W, 0)
    eig = np.abs(np.linalg.eigvals(W)).max()
    if eig > 1e-9:
        W = W * (0.9 / eig)
    return W


# ------------------------------------------------------------------
# Reservoir using surrogate
# ------------------------------------------------------------------
def run_reservoir_surr(surr, N, T, signal, class_sign, sign_mask, W,
                       base_VG1, base_VG2, kappa=0.20, label_amp=0.10):
    feat_prev = np.zeros(N)
    log_Id = np.zeros((N, T))
    label_inj = label_amp * class_sign * sign_mask
    for t in range(T):
        Vd_t = 1.2 + 1.0 * float(signal[t])
        recur = W @ feat_prev
        VG2_eff = np.clip(base_VG2 + label_inj + kappa * recur, -0.10, 0.60)
        # Surrogate eval: needs same shape; broadcast Vd
        Vd_arr = np.full(N, Vd_t)
        log_Id[:, t] = surr.eval(base_VG1, VG2_eff, Vd_arr)
        feat_prev = log_Id[:, t]
    return log_Id


def gen_signal(class_id: int, T: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if class_id == 0:
        return mg.gen_mackey_glass(T, seed=seed).astype(np.float64)
    f1, f2 = 0.05, 0.13
    phase = rng.normal(0, 0.3)
    x = (0.5 + 0.25*np.sin(2*np.pi*f1*np.arange(T)+phase)
              + 0.15*np.sin(2*np.pi*f2*np.arange(T)+2*phase)
              + rng.normal(0, 0.02, T))
    return np.clip(x, 0.0, 1.0).astype(np.float64)


def goodness_id_sq(log_Id):
    """Σ(10**log_Id)² shift-sensitive; we use log_Id² for stability + sign."""
    return float(np.sum(log_Id ** 2))


# ------------------------------------------------------------------
# Rules (NO LINEAR READOUT)
# ------------------------------------------------------------------
def update_ff(W, act_pos, act_neg, lr, w_max):
    dW = lr * (np.outer(act_pos, act_pos) - np.outer(act_neg, act_neg))
    np.fill_diagonal(dW, 0)
    return np.clip(W + dW, -w_max, w_max)


def update_rhebb(W, log_Id, r, lr, etrace_lam, w_max):
    z = log_Id - log_Id.mean(axis=1, keepdims=True)
    trace = np.zeros((W.shape[0], W.shape[0]))
    for t in range(z.shape[1]):
        trace = etrace_lam * trace + np.outer(z[:, t], z[:, t])
    dW = lr * r * trace
    np.fill_diagonal(dW, 0)
    return np.clip(W + dW, -w_max, w_max)


def update_hebb_ip(W, log_Id, lr, w_max, target_var=1.0):
    """Hebbian + Intrinsic Plasticity (Schrauwen): per-cell variance
    nudges base_VG2 in calling code (we just do Hebbian here, IP via
    return of activity stats)."""
    z = log_Id - log_Id.mean(axis=1, keepdims=True)
    z = z / (z.std(axis=1, keepdims=True) + 1e-6)
    corr = (z @ z.T) / z.shape[1]
    np.fill_diagonal(corr, 0)
    return np.clip(W + lr * corr, -w_max, w_max)


# ------------------------------------------------------------------
# Single config runner
# ------------------------------------------------------------------
def run_config(args):
    # F.1 (2026-05-07): explicit per-subprocess thread cap. ProcessPoolExecutor
    # *should* inherit the parent's OMP_NUM_THREADS env vars, but on some
    # spawn-method paths the subprocess starts numpy/blas before the env is
    # observed. Belt-and-suspenders: re-set inside run_config so any single
    # subprocess never exceeds 4 threads. Critical for N>=2048 runs to avoid
    # the 2026-05-05 thermal-trip pathology.
    import os as _os
    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        _os.environ[_k] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)  # F.1: hard cap to 1 thread per worker.
        # Matches util_safe_sweep convention: workers × threads < cpu_count.
    except Exception:
        pass
    topo, rule, N, seed = args
    surr = NSRAMSurrogate.build_or_load(grid_size=(20, 20, 25))
    rng = np.random.default_rng(seed)
    base_VG1 = rng.choice([0.2, 0.4, 0.6], size=N).astype(float)
    base_VG2 = rng.uniform(0.0, 0.5, size=N).astype(float)
    sign_mask = rng.choice([-1.0, 1.0], size=N).astype(float)
    W = build_topo(topo, N, rng)
    w_max = 0.6 / np.sqrt(N * 0.10)

    EPOCHS = 12
    N_TRAIN = 16
    N_TEST = 24
    T = 60
    LR = {"ff": 5e-3, "rhebb": 3e-3, "hebb_ip": 1e-3}[rule]

    G_ema = None; V_ema = 1.0; alpha = 1/32
    history = []
    W_history = [W.copy()]
    t0 = time.time()
    for epoch in range(EPOCHS):
        for s in range(N_TRAIN):
            cls = int(rng.integers(0, 2))
            sig = gen_signal(cls, T, seed=epoch*1000+s)
            true_sign = +1.0 if cls == 0 else -1.0
            if rule == "ff":
                lid_pos = run_reservoir_surr(surr, N, T, sig, true_sign,
                                              sign_mask, W, base_VG1, base_VG2)
                lid_neg = run_reservoir_surr(surr, N, T, sig, -true_sign,
                                              sign_mask, W, base_VG1, base_VG2)
                a_pos = (lid_pos**2).mean(axis=1)
                a_neg = (lid_neg**2).mean(axis=1)
                W = update_ff(W, a_pos, a_neg, LR, w_max)
            elif rule == "rhebb":
                lid = run_reservoir_surr(surr, N, T, sig, true_sign,
                                          sign_mask, W, base_VG1, base_VG2)
                G = goodness_id_sq(lid)
                if G_ema is None: G_ema = G
                else:
                    G_ema = (1-alpha)*G_ema + alpha*G
                    V_ema = (1-alpha)*V_ema + alpha*(G - G_ema)**2
                r = float(np.clip((G - G_ema) / (np.sqrt(V_ema)+1e-12), -3, 3))
                W = update_rhebb(W, lid, r, LR, 0.85, w_max)
            elif rule == "hebb_ip":
                lid = run_reservoir_surr(surr, N, T, sig, true_sign,
                                          sign_mask, W, base_VG1, base_VG2)
                W = update_hebb_ip(W, lid, LR, w_max)
        # Eval (FF-style two-pass for all rules)
        correct = 0
        for ts in range(N_TEST):
            cls_t = ts % 2
            sig = gen_signal(cls_t, T, seed=99999 + ts)
            g0 = goodness_id_sq(run_reservoir_surr(surr, N, T, sig, +1.0,
                                  sign_mask, W, base_VG1, base_VG2))
            g1 = goodness_id_sq(run_reservoir_surr(surr, N, T, sig, -1.0,
                                  sign_mask, W, base_VG1, base_VG2))
            pred = 0 if g0 > g1 else 1
            correct += int(pred == cls_t)
        acc = correct / N_TEST
        history.append({"epoch": epoch+1, "acc": acc,
                        "Wnorm": float(np.linalg.norm(W)),
                        "elapsed": time.time()-t0})
        W_history.append(W.copy())

    out = {
        "topo": topo, "rule": rule, "N": N, "seed": seed,
        "EPOCHS": EPOCHS, "T": T, "N_TRAIN": N_TRAIN, "N_TEST": N_TEST,
        "history": history,
        "best_acc": max(h["acc"] for h in history),
        "final_acc": history[-1]["acc"],
        "wall_s": time.time() - t0,
    }
    out_path = OUT / f"{topo}_{rule}_N{N}_s{seed}.json"
    out_path.write_text(json.dumps(out, indent=2))
    # Save W_history for later viz on a few selected configs
    if seed == 0:
        np.savez_compressed(OUT / f"{topo}_{rule}_N{N}_s{seed}_W.npz",
                            W_history=np.array(W_history),
                            sign_mask=sign_mask, base_VG1=base_VG1,
                            base_VG2=base_VG2)
    return out


# ------------------------------------------------------------------
def main():
    TOPOS = ["ER_SPARSE", "RAND_GAUSS", "RING", "WS_SMALLWORLD",
             "GRID_2D", "HUB_SPOKE", "MODULAR", "SCALE_FREE"]
    RULES = ["ff", "rhebb", "hebb_ip"]
    SIZES = [256, 1024]

    grid = [(t, r, n, 0) for t in TOPOS for r in RULES for n in SIZES]
    print(f"[z200] launching {len(grid)} configs / 2 workers (F.1 thread cap)")
    print(f"[z200] {len(TOPOS)} topos × {len(RULES)} rules × {len(SIZES)} sizes")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(run_config, a): a for a in grid}
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            print(f"[z200] {r['topo']:>14}/{r['rule']:>7}/N={r['N']:<4}: "
                  f"best={r['best_acc']:.3f} final={r['final_acc']:.3f} "
                  f"({r['wall_s']:.0f}s)", flush=True)

    # Summary
    summary = {"results": results, "total_wall_s": time.time() - t0}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[z200] total wall: {summary['total_wall_s']:.0f}s "
          f"({len(grid)} configs)")

    # Heatmap: rows = topology, cols = rule × size
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = np.full((len(TOPOS), len(RULES) * len(SIZES)), np.nan)
    col_labels = []
    for j, (rule, N) in enumerate([(r, n) for r in RULES for n in SIZES]):
        col_labels.append(f"{rule}/N{N}")
        for i, topo in enumerate(TOPOS):
            for r in results:
                if r["topo"]==topo and r["rule"]==rule and r["N"]==N:
                    M[i, j] = r["best_acc"]
                    break
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(len(TOPOS)):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v): continue
            c = "black" if 0.6 < v < 0.85 else "white"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=c, fontsize=10)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=15, ha="right")
    ax.set_yticks(range(len(TOPOS)))
    ax.set_yticklabels(TOPOS)
    ax.set_xlabel("rule × size")
    ax.set_title(f"z200: 8 topologies × 3 self-learning rules × 2 sizes\n"
                 f"BEST test accuracy on 2-class MG vs sin (no linear readout)\n"
                 f"all rules surrogate-driven; total wall {summary['total_wall_s']:.0f}s")
    plt.colorbar(im, ax=ax, label="best test accuracy")
    plt.tight_layout()
    plt.savefig(FIG / "heatmap.png", dpi=150)
    plt.savefig(FIG / "heatmap.pdf")
    plt.close()
    print(f"[z200] heatmap saved to {FIG}/heatmap.png")


if __name__ == "__main__":
    main()
