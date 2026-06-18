"""z57_cell_ablation_v2.py — fix z56 bugs.

z56 had two issues:
  1. G_bjt is the wrong proxy for "bistability strength" — it only
     pulls Vb DOWN above V_bjt_on=0.74 (an upper-bound stabilizer).
     The actual latch is set by V_latch.  Sweep V_latch instead, AND
     add a hard "no bistability at all" baseline using LinearCellArray.
  2. Retention window was too short (T=50, dt=0.05 → 2.5s) compared to
     τ ≈ 1/K_leak.  Use T=300 for the MC/XOR tasks.

Knobs (relative to CAL):
  bistability_V_latch  V_latch    sweep 0.30 -> 0.85, plus LINEAR baseline
  retention_K_leak     K_leak     T extended to 300 steps
  vg2_coupling         |K_back|   unchanged
  snapback_A_iii       A_iii      unchanged

Tasks: hopfield_K20, mem_capacity (lags 1..20), xor_tau2.
"""
from __future__ import annotations
import json, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from nsram.cell_fast import CellArray
from nsram.plasticity_net import NetSim, memory_capacity, topo_random
from scripts.z44_robustness import LinearCellArray

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z57_cell_ablation_v2")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {DEVICE}")

N_CELLS = 64
N_SEEDS = 3
P_LO, P_HI = -0.10, 0.40

CAL = dict(VTH0=0.43, K_back=-0.98, A_iii=4.71, G_bjt=1.00,
           V_bjt_on=0.74, V_latch=0.58, K_leak=0.021)


def hopfield_W(patterns):
    K, N = patterns.shape
    W = (patterns.T @ patterns) / N
    np.fill_diagonal(W, 0.0)
    return W


def make_cells(params, linear=False):
    VG2 = torch.full((N_CELLS,), -0.10, device=DEVICE)
    if linear:
        return LinearCellArray(N_CELLS, alpha=1.5, VG2=VG2,
                               K_leak=params.get("K_leak", 0.05),
                               device=DEVICE)
    return CellArray(N_CELLS, alpha=1.5, VG2=VG2, device=DEVICE, **params)


def make_net_hopfield(W_np, params, linear=False, fb=0.5):
    cells = make_cells(params, linear=linear)
    W = torch.from_numpy(W_np.astype(np.float32)).to(DEVICE)
    W_in = torch.zeros(N_CELLS, 1, device=DEVICE)
    return NetSim(cells=cells, W=W, W_in=W_in, feedback_gain=fb)


def make_net_reservoir(params, seed, linear=False, fb=0.5):
    cells = make_cells(params, linear=linear)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    W = topo_random(N_CELLS, p=0.1, seed=seed, device=DEVICE)
    W_in = torch.randn(N_CELLS, 1, generator=g, device=DEVICE) * 1.0
    return NetSim(cells=cells, W=W, W_in=W_in, feedback_gain=fb)


def task_hopfield(params, linear=False, K=20, noise=0.25, T_relax=300, seeds=N_SEEDS):
    accs = []
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        patterns = rng.choice([-1, 1], size=(K, N_CELLS)).astype(np.float32)
        W = hopfield_W(patterns)
        net = make_net_hopfield(W, params, linear=linear)
        correct = 0
        for k in range(K):
            p = patterns[k].copy()
            n_flip = int(noise * len(p))
            idx = rng.choice(len(p), n_flip, replace=False)
            p[idx] *= -1
            Vb_init = torch.from_numpy(np.where(p > 0, P_HI, P_LO).astype(np.float32)).to(DEVICE)
            net.cells.Vb = Vb_init.clone()
            u = torch.zeros(1, device=DEVICE)
            VG1 = torch.tensor(0.6, device=DEVICE)
            for _ in range(T_relax):
                net.step(u, VG1)
            final = net.cells.Vb.cpu().numpy()
            mid = 0.5 * (P_LO + P_HI)
            sign = np.where(final > mid, 1, -1).astype(np.float32)
            overlaps = patterns @ sign / len(sign)
            if int(np.argmax(overlaps)) == k:
                correct += 1
        accs.append(correct / K)
    return float(np.mean(accs)), float(np.std(accs))


def task_mc(params, linear=False, seeds=N_SEEDS):
    mcs = []
    for s in range(seeds):
        net = make_net_reservoir(params, seed=s, linear=linear, fb=0.5)
        mc = memory_capacity(net, T_train=600, T_test=300, n_lags=15,
                             VG1=0.6, washout=150, seed=s)
        mcs.append(mc)
    return float(np.mean(mcs)), float(np.std(mcs))


def task_xor_tau2(params, linear=False, T=800, seeds=N_SEEDS):
    accs = []
    for s in range(seeds):
        rng = np.random.default_rng(5000 + s)
        u_seq = rng.choice([-1, 1], T).astype(np.float32)
        target = np.zeros(T, dtype=np.float32)
        for t in range(2, T):
            target[t] = 1.0 if u_seq[t] != u_seq[t-2] else -1.0
        net = make_net_reservoir(params, seed=s, linear=linear, fb=0.5)
        U = torch.from_numpy(u_seq).unsqueeze(1).to(DEVICE)
        VG1 = torch.full((T,), 0.6, device=DEVICE)
        _, Vb = net.run(U, VG1)
        States = Vb.cpu().numpy()
        mu = States.mean(0, keepdims=True); sd = States.std(0, keepdims=True) + 1e-9
        States = (States - mu) / sd
        wash = 100
        Xtr = States[wash:T*3//4]; ytr = target[wash:T*3//4]
        Xte = States[T*3//4:];    yte = target[T*3//4:]
        Xt = Xtr.T @ Xtr + 1e-2 * np.eye(Xtr.shape[1])
        w = np.linalg.solve(Xt, Xtr.T @ ytr)
        pred = np.sign(Xte @ w)
        acc = float((pred == yte).mean())
        accs.append(acc)
    return float(np.mean(accs)), float(np.std(accs))


KNOBS = {
    "bistability_V_latch": ("V_latch",  [0.30, 0.45, 0.58, 0.75]),
    "retention_K_leak":    ("K_leak",   [0.005, 0.021, 0.10, 0.30]),
    "vg2_coupling":        ("K_back",   [0.0, -0.5, -0.98, -2.0]),
    "snapback_A_iii":      ("A_iii",    [1.0, 2.5, 4.71, 10.0]),
}

TASKS = [("hopfield_K20", task_hopfield),
         ("mem_capacity", task_mc),
         ("xor_tau2", task_xor_tau2)]


def main():
    t0 = time.time()
    results = {}
    print(f"\nN={N_CELLS}, seeds={N_SEEDS}\n")

    # ── linear-cell baseline (no bistability at all) ──────────────────
    print("── LINEAR baseline (no bistability) ──")
    lin_scores = {}
    for task_name, task_fn in TASKS:
        m, s = task_fn(dict(CAL), linear=True)
        lin_scores[task_name] = (m, s)
        print(f"  {task_name:14s}  LINEAR  {m:.3f}±{s:.3f}", flush=True)
    print()

    # ── bistable knob sweeps ──────────────────────────────────────────
    for knob_name, (param_name, levels) in KNOBS.items():
        print(f"── {knob_name} ({param_name}) ──")
        results[knob_name] = {"levels": levels, "param": param_name, "tasks": {}}
        for task_name, task_fn in TASKS:
            scores, stds = [], []
            for lvl in levels:
                p = dict(CAL)
                p[param_name] = lvl
                m, s = task_fn(p)
                scores.append(m); stds.append(s)
                print(f"  {task_name:14s}  {param_name}={lvl:+.3f}  "
                      f"{m:.3f}±{s:.3f}", flush=True)
            results[knob_name]["tasks"][task_name] = {
                "mean": scores, "std": stds}
        print()

    elapsed = time.time() - t0
    summary = {"results": results, "linear_baseline": lin_scores,
               "CAL": CAL, "N": N_CELLS, "seeds": N_SEEDS,
               "elapsed_s": elapsed}
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── plots ─────────────────────────────────────────────────────────
    knobs = list(KNOBS.keys())
    tasks = [t[0] for t in TASKS]

    sensitivity = np.zeros((len(knobs), len(tasks)))
    for i, k in enumerate(knobs):
        for j, t in enumerate(tasks):
            arr = np.array(results[k]["tasks"][t]["mean"])
            denom = max(abs((arr.max() + arr.min()) / 2), 1e-3)
            sensitivity[i, j] = (arr.max() - arr.min()) / denom

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # heatmap
    ax = axes[0, 0]
    im = ax.imshow(sensitivity, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(tasks))); ax.set_xticklabels(tasks, rotation=20)
    ax.set_yticks(range(len(knobs))); ax.set_yticklabels(knobs)
    for i in range(len(knobs)):
        for j in range(len(tasks)):
            v = sensitivity[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.5 * sensitivity.max() else "black",
                    fontsize=11, weight="bold")
    ax.set_title("Knob sensitivity per task")
    plt.colorbar(im, ax=ax, label="Δscore / mean")

    # per-task panels with linear baseline as horizontal line
    task_colors = {"hopfield_K20": "#3498db", "mem_capacity": "#e67e22",
                   "xor_tau2": "#27ae60"}
    for idx, t in enumerate(tasks):
        ax = axes[(idx+1) // 2, (idx+1) % 2]
        for k in knobs:
            levels = results[k]["levels"]
            arr = np.array(results[k]["tasks"][t]["mean"])
            std = np.array(results[k]["tasks"][t]["std"])
            ax.errorbar(range(len(levels)), arr, yerr=std,
                        marker="o", lw=2, label=k, capsize=3)
        # linear baseline
        lin_m, lin_s = lin_scores[t]
        ax.axhline(lin_m, color="black", ls="--", lw=2,
                   label=f"LINEAR (no bistability)={lin_m:.2f}")
        ax.fill_between([-0.3, 3.3], lin_m - lin_s, lin_m + lin_s,
                        color="black", alpha=0.1)
        ax.set_xlim(-0.3, 3.3)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["low", "med-lo", "BASELINE", "high"])
        ax.set_title(f"{t}")
        ax.set_ylabel("score"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"z57 — Cell-property ablation (fixed)  N={N_CELLS}, "
                 f"{N_SEEDS} seeds, {elapsed/60:.1f} min",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "ablation_v2.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {OUT/'ablation_v2.png'} ({elapsed/60:.1f} min)")

    # ── ranking ───────────────────────────────────────────────────────
    print("\n=== Sensitivity ranking (knob × task) ===")
    flat = []
    for i, k in enumerate(knobs):
        for j, t in enumerate(tasks):
            flat.append((sensitivity[i, j], k, t))
    flat.sort(reverse=True)
    for v, k, t in flat:
        print(f"  {v:.3f}  {k:24s}  {t}")

    print("\n=== Bistability gain over linear baseline ===")
    for t in tasks:
        baseline_score = float(np.mean([results[k]["tasks"][t]["mean"][2]
                                         for k in knobs]))
        lin_m = lin_scores[t][0]
        gain = baseline_score - lin_m
        rel = (baseline_score - lin_m) / max(abs(lin_m), 1e-3) * 100
        print(f"  {t:14s}  bistable_baseline={baseline_score:.3f}  "
              f"linear={lin_m:.3f}  Δ={gain:+.3f} ({rel:+.0f}%)")


if __name__ == "__main__":
    main()
