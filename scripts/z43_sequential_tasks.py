"""z43_sequential_tasks.py — sequential task learning + forgetting curves.

Brain-inspired benchmark: train on 5 distinct tasks in sequence, measure
how much each earlier task is retained as new ones come in.  This is the
direct analog of catastrophic forgetting in continual learning.

Three configurations compared:
  C1: Static homogeneous VG2 = +0.20 V (z41 sweet spot)
  C2: Heterogeneous VG2 — bimodal (some cells plastic, some stable)
       — like hippocampus + neocortex consolidation
  C3: Heterogeneous α — fast/slow weights (z40 winner)
       — Hinton dual-time-scale

Each "task" = drive network with a specific input pattern (random noise
seeded differently).  Readout = final-state vector.  Retention = correlation
of readout for task k after training on tasks k+1, k+2, ...

Plot: retention vs task-age (how many tasks ago) for each config.
The flatter the curve, the better the system holds long-term memory.
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
from nsram.plasticity_net import (NetSim, topo_small_world)

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z43_sequential_tasks")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {DEVICE}")
N_CELLS = 96
N_TASKS = 5
T_PER_TASK = 400
N_SEEDS = 3


def make_net(VG2, alpha, seed=0):
    cells = CellArray(N_CELLS, alpha=alpha,
                          VG2=VG2.to(DEVICE), device=DEVICE)
    W = topo_small_world(N_CELLS, k=4, p_rewire=0.1, seed=seed, device=DEVICE)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    W_in = torch.randn(N_CELLS, 1, generator=g, device=DEVICE)
    return NetSim(cells=cells, W=W, W_in=W_in, feedback_gain=0.27)


def task_input(task_id: int, T: int = T_PER_TASK):
    """Each task = different RNG seed driving the network."""
    rng = np.random.default_rng(1000 + task_id)
    u = rng.uniform(-1, 1, T).astype(np.float32)
    return torch.from_numpy(u).unsqueeze(1).to(DEVICE)


def measure_signature(net, task_id):
    """Drive net briefly with this task's input, return final-state vec."""
    U = task_input(task_id, T=200)
    VG1 = torch.full((U.shape[0],), 0.6).to(DEVICE)
    Id, Vb = net.run(U, VG1)
    return Vb[-50:].mean(dim=0).cpu().numpy()


def train_and_track(net, tasks):
    """Train on each task in sequence.  After each, compute & store
    signatures for ALL tasks (so we can see what's forgotten)."""
    n = len(tasks)
    sig_table = np.zeros((n, n, N_CELLS))    # (after-task, signature-of-task, cell)
    for k_train in range(n):
        # Drive with task k_train (this is the "training")
        U = task_input(tasks[k_train], T=T_PER_TASK)
        VG1 = torch.full((T_PER_TASK,), 0.6).to(DEVICE)
        net.run(U, VG1)
        # Snapshot net's signature on each task
        Vb_save = net.cells.Vb.clone()
        for k_query in range(n):
            net.cells.Vb = Vb_save.clone()
            sig_table[k_train, k_query] = measure_signature(net, tasks[k_query])
        net.cells.Vb = Vb_save
    return sig_table


def retention_curves(sig_table):
    """For each task k, compute retention(age) = correlation between
    sig_table[k, k] (its signature right after training) and
    sig_table[k+age, k] (signature later in time).

    Returns: retention[age] averaged over all task ks.
    """
    n = sig_table.shape[0]
    retention = {age: [] for age in range(n)}
    for k in range(n):
        baseline = sig_table[k, k]
        b_norm = (baseline - baseline.mean()) / (baseline.std() + 1e-9)
        for k_later in range(k, n):
            age = k_later - k
            later = sig_table[k_later, k]
            l_norm = (later - later.mean()) / (later.std() + 1e-9)
            r = float(np.dot(b_norm, l_norm) / len(b_norm))
            retention[age].append(r)
    return np.array([np.mean(retention[age]) for age in range(n)])


def run_config(label, VG2, alpha):
    print(f"\n=== {label} ===")
    all_curves = []
    tasks = list(range(N_TASKS))
    for s in range(N_SEEDS):
        net = make_net(VG2, alpha, seed=s)
        sig_table = train_and_track(net, tasks)
        curve = retention_curves(sig_table)
        all_curves.append(curve)
        print(f"  seed={s}  retention by age: " +
              "  ".join(f"a{i}={curve[i]:+.2f}" for i in range(N_TASKS)),
              flush=True)
    arr = np.stack(all_curves, axis=0)
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
    }


def main():
    t0 = time.time()
    # C1: homogeneous VG2 = 0.20
    VG2_homog = torch.full((N_CELLS,), 0.20)
    c1 = run_config("C1: homogeneous VG2=+0.20", VG2_homog, alpha=1.5)

    # C2: heterogeneous VG2 (bimodal)
    n_plastic = int(0.30 * N_CELLS)        # 30% plastic
    VG2_hetero = torch.cat([
        torch.full((n_plastic,), 0.40),
        torch.full((N_CELLS - n_plastic,), 0.0),
    ])
    torch.manual_seed(0)
    VG2_hetero = VG2_hetero[torch.randperm(N_CELLS)]
    c2 = run_config("C2: heterogeneous VG2 (bimodal)", VG2_hetero, alpha=1.5)

    # C3: heterogeneous α (fast/slow)
    n_fast = int(0.50 * N_CELLS)
    alpha_hetero = torch.cat([
        torch.full((n_fast,), 8.0),
        torch.full((N_CELLS - n_fast,), 0.1),
    ])
    torch.manual_seed(1)
    alpha_hetero = alpha_hetero[torch.randperm(N_CELLS)]
    c3 = run_config("C3: heterogeneous α (fast/slow)",
                     VG2_homog, alpha=alpha_hetero.to(DEVICE))

    elapsed = time.time() - t0
    with open(OUT / "summary.json", "w") as f:
        json.dump({"C1": c1, "C2": c2, "C3": c3, "elapsed_s": elapsed},
                    f, indent=2)

    fig, ax = plt.subplots(figsize=(10, 6))
    ages = np.arange(N_TASKS)
    cfgs = [
        ("C1: homogeneous VG2", c1, "#3498db"),
        ("C2: bimodal VG2 (30% plastic)", c2, "#e67e22"),
        ("C3: bimodal α (50/50 fast/slow)", c3, "#27ae60"),
    ]
    for label, c, color in cfgs:
        m = np.array(c["mean"]); s = np.array(c["std"])
        ax.errorbar(ages, m, yerr=s, fmt="o-", lw=2.5, capsize=4,
                      label=label, color=color)
    ax.axhline(1.0, color="black", ls="--", alpha=0.3, label="perfect retention")
    ax.axhline(0.0, color="red", ls="--", alpha=0.3, label="forgotten")
    ax.set_xlabel("age (tasks since training)", fontsize=12)
    ax.set_ylabel("retention (correlation)", fontsize=12)
    ax.set_title(f"Forgetting curves — {N_TASKS} sequential tasks, "
                  f"{N_SEEDS} seeds, {N_CELLS} cells", fontsize=13)
    ax.legend(fontsize=10, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "forgetting_curves.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {OUT/'forgetting_curves.png'}")
    print(f"Total: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
