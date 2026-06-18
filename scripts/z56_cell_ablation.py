"""z56_cell_ablation.py — which cell properties drive which network tasks?

Vary 4 cell-level knobs one at a time around the calibrated baseline and
measure impact on three network tasks (Hopfield recall, memory capacity,
XOR-tau=2).  Output: a 4x3 heatmap "knob x task" giving relative ranking.

Goal: produce a *prioritization* for the next device iteration, robust
against the ~20-40% magnitude error in our cell_fast.  The ranking is
qualitatively the answer; magnitudes will refine when ported to Robert's
emulator.

Knobs (relative to CAL):
  bistability   G_bjt       0 -> no latch pull, 1.0 baseline, 2x stronger
  retention     K_leak      higher = faster decay = shorter retention
  vg2_coupling  |K_back|    0 = VG2 has no effect on Vth
  snapback      A_iii       impact-ionization gain

Tasks:
  hopfield_K20  fraction of K=20 patterns recalled with 25% bit-flip noise
  mem_capacity  Jaeger MC, sum over lags 1..20
  xor_tau2      classification accuracy of x[t] XOR x[t-2]
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

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z56_cell_ablation")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {DEVICE}")

N_CELLS = 64
N_SEEDS = 3
P_LO, P_HI = -0.10, 0.40

CAL = dict(VTH0=0.43, K_back=-0.98, A_iii=4.71, G_bjt=1.00,
           V_bjt_on=0.74, V_latch=0.58, K_leak=0.021)


# ── tasks ─────────────────────────────────────────────────────────────

def hopfield_W(patterns):
    K, N = patterns.shape
    W = (patterns.T @ patterns) / N
    np.fill_diagonal(W, 0.0)
    return W


def make_cells(params):
    VG2 = torch.full((N_CELLS,), -0.10, device=DEVICE)
    return CellArray(N_CELLS, alpha=1.5, VG2=VG2, device=DEVICE, **params)


def make_net_hopfield(W_np, params, fb=0.5):
    cells = make_cells(params)
    W = torch.from_numpy(W_np.astype(np.float32)).to(DEVICE)
    W_in = torch.zeros(N_CELLS, 1, device=DEVICE)
    return NetSim(cells=cells, W=W, W_in=W_in, feedback_gain=fb)


def make_net_reservoir(params, seed, fb=0.5):
    cells = make_cells(params)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    W = topo_random(N_CELLS, p=0.1, seed=seed, device=DEVICE)
    W_in = torch.randn(N_CELLS, 1, generator=g, device=DEVICE) * 1.0
    return NetSim(cells=cells, W=W, W_in=W_in, feedback_gain=fb)


def task_hopfield(params, K=20, noise=0.25, seeds=N_SEEDS):
    accs = []
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        patterns = rng.choice([-1, 1], size=(K, N_CELLS)).astype(np.float32)
        W = hopfield_W(patterns)
        net = make_net_hopfield(W, params)
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
            for _ in range(50):
                net.step(u, VG1)
            final = net.cells.Vb.cpu().numpy()
            mid = 0.5 * (P_LO + P_HI)
            sign = np.where(final > mid, 1, -1).astype(np.float32)
            overlaps = patterns @ sign / len(sign)
            if int(np.argmax(overlaps)) == k:
                correct += 1
        accs.append(correct / K)
    return float(np.mean(accs)), float(np.std(accs))


def task_mc(params, seeds=N_SEEDS):
    mcs = []
    for s in range(seeds):
        net = make_net_reservoir(params, seed=s, fb=0.5)
        mc = memory_capacity(net, T_train=600, T_test=300, n_lags=15,
                             VG1=0.6, washout=150, seed=s)
        mcs.append(mc)
    return float(np.mean(mcs)), float(np.std(mcs))


def task_xor_tau2(params, T=800, seeds=N_SEEDS):
    """Train ridge readout to compute x[t] XOR x[t-2] from reservoir state."""
    accs = []
    for s in range(seeds):
        rng = np.random.default_rng(5000 + s)
        u_seq = rng.choice([-1, 1], T).astype(np.float32)
        target = np.zeros(T, dtype=np.float32)
        for t in range(2, T):
            target[t] = 1.0 if u_seq[t] != u_seq[t-2] else -1.0
        net = make_net_reservoir(params, seed=s, fb=0.5)
        U = torch.from_numpy(u_seq).unsqueeze(1).to(DEVICE)
        VG1 = torch.full((T,), 0.6, device=DEVICE)
        _, Vb = net.run(U, VG1)
        States = Vb.cpu().numpy()
        # z-score
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


# ── ablation grid ─────────────────────────────────────────────────────

KNOBS = {
    "bistability_G_bjt": ("G_bjt",   [0.0, 0.5, 1.0, 2.0]),
    "retention_K_leak":  ("K_leak",  [0.005, 0.021, 0.10, 0.30]),
    "vg2_coupling":      ("K_back",  [0.0, -0.5, -0.98, -2.0]),
    "snapback_A_iii":    ("A_iii",   [1.0, 2.5, 4.71, 10.0]),
}

TASKS = [("hopfield_K20", task_hopfield),
         ("mem_capacity", task_mc),
         ("xor_tau2", task_xor_tau2)]


def main():
    t0 = time.time()
    results = {}
    print(f"\nN={N_CELLS}, seeds={N_SEEDS}\n")
    for knob_name, (param_name, levels) in KNOBS.items():
        print(f"── {knob_name} ({param_name}) ──")
        results[knob_name] = {"levels": levels, "param": param_name, "tasks": {}}
        for task_name, task_fn in TASKS:
            scores = []
            stds = []
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
    summary = {"results": results, "CAL": CAL, "N": N_CELLS,
               "seeds": N_SEEDS, "elapsed_s": elapsed}
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── heatmap: knob sensitivity per task ────────────────────────────
    knobs = list(KNOBS.keys())
    tasks = [t[0] for t in TASKS]
    sensitivity = np.zeros((len(knobs), len(tasks)))
    for i, k in enumerate(knobs):
        for j, t in enumerate(tasks):
            arr = np.array(results[k]["tasks"][t]["mean"])
            # sensitivity = (max - min) / |max+min|/2 — fractional dynamic range
            denom = max(abs((arr.max() + arr.min()) / 2), 1e-3)
            sensitivity[i, j] = (arr.max() - arr.min()) / denom

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    im = ax.imshow(sensitivity, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(tasks))); ax.set_xticklabels(tasks, rotation=20)
    ax.set_yticks(range(len(knobs))); ax.set_yticklabels(knobs)
    for i in range(len(knobs)):
        for j in range(len(tasks)):
            v = sensitivity[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.5 * sensitivity.max() else "black",
                    fontsize=11, weight="bold")
    ax.set_title("Cell-knob sensitivity per task\n(fractional dynamic range over swept levels)")
    plt.colorbar(im, ax=ax, label="Δscore / mean")

    # ── line plots: knob value vs task score ──────────────────────────
    ax = axes[1]
    colors = {"hopfield_K20": "#3498db", "mem_capacity": "#e67e22",
              "xor_tau2": "#27ae60"}
    for i, k in enumerate(knobs):
        levels = results[k]["levels"]
        for t in tasks:
            arr = np.array(results[k]["tasks"][t]["mean"])
            # normalize each curve to its baseline (idx 2 = baseline level)
            base = arr[2] if abs(arr[2]) > 1e-6 else 1.0
            ax.plot(np.arange(len(levels)) + i*0.05,
                    arr / base,
                    "o-", alpha=0.6,
                    label=f"{k}/{t}" if i == 0 else None,
                    color=colors[t])
    ax.axhline(1.0, color="black", ls="--", alpha=0.5)
    ax.set_xlabel("level index (0=low, 2=baseline, 3=high)")
    ax.set_ylabel("score / baseline")
    ax.set_title("Per-task response to each knob")
    ax.legend(fontsize=9, loc="best"); ax.grid(alpha=0.3)

    fig.suptitle(f"z56 — Cell-property ablation  N={N_CELLS}  "
                 f"({elapsed/60:.1f} min)",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "ablation.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {OUT/'ablation.png'} ({elapsed/60:.1f} min)")

    # ── ranking summary ───────────────────────────────────────────────
    print("\n=== Sensitivity ranking (knob × task) ===")
    flat = []
    for i, k in enumerate(knobs):
        for j, t in enumerate(tasks):
            flat.append((sensitivity[i, j], k, t))
    flat.sort(reverse=True)
    for v, k, t in flat:
        print(f"  {v:.3f}  {k:24s}  {t}")


if __name__ == "__main__":
    main()
