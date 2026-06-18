"""z40_extended_plasticity.py — extend the z38 baseline with three
brain-inspired experiments:

  E1: Wider α range  (0.001 → 50) on small_world (z38 winner) — catch true
      bistability boundary and see if Lyap crosses zero
  E2: Heterogeneous plasticity  — split cells into "fast weights" (high α)
      and "slow weights" (low α).  Hinton 2016, dual-time-scale learning;
      hippocampus/neocortex consolidation.
  E3: Catastrophic forgetting  — train on task A, then task B, measure
      retention of A.  Plasticity-stability dilemma.
  E4: Criticality hunt — sweep feedback_gain to push the system toward
      edge-of-chaos.  Lyap should cross zero somewhere.

Output: z40_extended_plasticity/figures + summary.json
"""
from __future__ import annotations
import json, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from nsram.plasticity_net import (make_network, memory_capacity,
                                   lyapunov_proxy)
from nsram.cell_fast import CellArray

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z40_extended_plasticity")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {DEVICE}")
N_CELLS = 96
N_SEEDS = 3


# ─────────────────────────────────────────────────────────────────────
# E1: wider α
# ─────────────────────────────────────────────────────────────────────

def run_e1():
    print("\n=== E1: wide α range on small_world ===")
    alphas = np.logspace(-3, np.log10(50), 18)
    rs = {"alpha": [], "MC": [], "MC_std": [], "Lyap": [], "Lyap_std": []}
    for a in alphas:
        mcs, lys = [], []
        for s in range(N_SEEDS):
            net = make_network(N=N_CELLS, topology="small_world",
                                alpha=float(a), VG2_mean=0.20,
                                seed=s, device=DEVICE)
            mcs.append(memory_capacity(net, T_train=600, T_test=300, seed=s))
            net2 = make_network(N=N_CELLS, topology="small_world",
                                 alpha=float(a), VG2_mean=0.20,
                                 seed=s, device=DEVICE)
            lys.append(lyapunov_proxy(net2, T_warmup=200, T_meas=150, seed=s))
        rs["alpha"].append(float(a))
        rs["MC"].append(float(np.mean(mcs))); rs["MC_std"].append(float(np.std(mcs)))
        rs["Lyap"].append(float(np.mean(lys))); rs["Lyap_std"].append(float(np.std(lys)))
        print(f"  α={a:8.4f}  MC={rs['MC'][-1]:.2f}  Lyap={rs['Lyap'][-1]:+.3f}",
              flush=True)
    return rs


# ─────────────────────────────────────────────────────────────────────
# E2: heterogeneous α (fast/slow weights)
# ─────────────────────────────────────────────────────────────────────

def run_e2():
    """Mix fast cells (α_fast) and slow cells (α_slow) in fixed ratio.

    For a fixed mean α=1.5 (z38 sweet-spot), vary the *spread*: from
    homogeneous (all 1.5) to bimodal (some at 0.1, some at 15.0).
    """
    print("\n=== E2: heterogeneous α (fast/slow weights split) ===")
    rs = {"frac_fast": [], "MC": [], "MC_std": [], "Lyap": [], "Lyap_std": []}
    fracs = np.linspace(0.0, 1.0, 11)   # frac of "fast" cells
    A_FAST = 8.0; A_SLOW = 0.1
    for frac in fracs:
        mcs, lys = [], []
        for s in range(N_SEEDS):
            torch.manual_seed(s)
            n_fast = int(frac * N_CELLS)
            alphas = torch.cat([
                torch.full((n_fast,), A_FAST),
                torch.full((N_CELLS - n_fast,), A_SLOW),
            ])
            perm = torch.randperm(N_CELLS)
            alphas = alphas[perm]
            # build net manually with heterogeneous α
            VG2 = torch.full((N_CELLS,), 0.20, device=DEVICE)
            cells = CellArray(N_CELLS, alpha=alphas.to(DEVICE), VG2=VG2,
                                  device=DEVICE)
            from nsram.plasticity_net import topo_small_world, NetSim
            W = topo_small_world(N_CELLS, k=4, p_rewire=0.1, seed=s, device=DEVICE)
            g = torch.Generator(device=DEVICE).manual_seed(s)
            W_in = torch.randn(N_CELLS, 1, generator=g, device=DEVICE)
            net = NetSim(cells=cells, W=W, W_in=W_in, feedback_gain=0.8)
            mcs.append(memory_capacity(net, T_train=600, T_test=300, seed=s))
            cells2 = CellArray(N_CELLS, alpha=alphas.to(DEVICE), VG2=VG2,
                                   device=DEVICE)
            net2 = NetSim(cells=cells2, W=W, W_in=W_in, feedback_gain=0.8)
            lys.append(lyapunov_proxy(net2, T_warmup=200, T_meas=150, seed=s))
        rs["frac_fast"].append(float(frac))
        rs["MC"].append(float(np.mean(mcs))); rs["MC_std"].append(float(np.std(mcs)))
        rs["Lyap"].append(float(np.mean(lys))); rs["Lyap_std"].append(float(np.std(lys)))
        print(f"  frac_fast={frac:.2f}  MC={rs['MC'][-1]:.2f}  Lyap={rs['Lyap'][-1]:+.3f}",
              flush=True)
    return rs


# ─────────────────────────────────────────────────────────────────────
# E3: catastrophic forgetting
# ─────────────────────────────────────────────────────────────────────

def task_signature(net, T=400, seed=0):
    """Drive net with deterministic seed, return final-state vector."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1, 1, T).astype(np.float32)
    U = torch.from_numpy(u).unsqueeze(1).to(DEVICE)
    VG1 = torch.full((T,), 0.6).to(DEVICE)
    Id, Vb = net.run(U, VG1)
    return Vb[-100:].mean(dim=0).cpu().numpy()


def run_e3():
    """For each α: train on task A (seed 1), then task B (seed 2),
    measure how well the final state still represents task A."""
    print("\n=== E3: catastrophic forgetting ===")
    alphas = np.logspace(-2, np.log10(20), 10)
    rs = {"alpha": [], "retention_A": [], "retention_A_std": []}
    for a in alphas:
        retentions = []
        for s in range(N_SEEDS):
            netA = make_network(N=N_CELLS, topology="small_world",
                                  alpha=float(a), VG2_mean=0.20,
                                  seed=s, device=DEVICE)
            sigA = task_signature(netA, T=300, seed=10 + s)
            # Continue with task B (different driving seed)
            netA._task_b = task_signature(netA, T=300, seed=20 + s)
            sigA_after = netA.cells.Vb[:96].cpu().numpy()
            # Compute final state again with same seed as sigA but on PERTURBED net
            netA2 = make_network(N=N_CELLS, topology="small_world",
                                   alpha=float(a), VG2_mean=0.20,
                                   seed=s, device=DEVICE)
            netA2.cells.Vb = torch.from_numpy(sigA_after).to(DEVICE)
            sigA_replay = task_signature(netA2, T=300, seed=10 + s)
            sim = float(np.corrcoef(sigA, sigA_replay)[0, 1])
            if np.isnan(sim): sim = 0.0
            retentions.append(sim)
        rs["alpha"].append(float(a))
        rs["retention_A"].append(float(np.mean(retentions)))
        rs["retention_A_std"].append(float(np.std(retentions)))
        print(f"  α={a:8.4f}  retention(A)={rs['retention_A'][-1]:+.3f}",
              flush=True)
    return rs


# ─────────────────────────────────────────────────────────────────────
# E4: feedback_gain → criticality
# ─────────────────────────────────────────────────────────────────────

def run_e4():
    print("\n=== E4: criticality hunt via feedback_gain ===")
    fb_gains = np.linspace(0.0, 3.0, 12)
    rs = {"fb": [], "MC": [], "MC_std": [], "Lyap": [], "Lyap_std": []}
    for fb in fb_gains:
        mcs, lys = [], []
        for s in range(N_SEEDS):
            net = make_network(N=N_CELLS, topology="small_world",
                                alpha=1.5, VG2_mean=0.20,
                                feedback_gain=float(fb),
                                seed=s, device=DEVICE)
            mcs.append(memory_capacity(net, T_train=600, T_test=300, seed=s))
            net2 = make_network(N=N_CELLS, topology="small_world",
                                 alpha=1.5, VG2_mean=0.20,
                                 feedback_gain=float(fb),
                                 seed=s, device=DEVICE)
            lys.append(lyapunov_proxy(net2, T_warmup=200, T_meas=150, seed=s))
        rs["fb"].append(float(fb))
        rs["MC"].append(float(np.mean(mcs))); rs["MC_std"].append(float(np.std(mcs)))
        rs["Lyap"].append(float(np.mean(lys))); rs["Lyap_std"].append(float(np.std(lys)))
        print(f"  fb={fb:.2f}  MC={rs['MC'][-1]:.2f}  Lyap={rs['Lyap'][-1]:+.3f}",
              flush=True)
    return rs


# ─────────────────────────────────────────────────────────────────────
# Main + plotting
# ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    e1 = run_e1()
    e2 = run_e2()
    e3 = run_e3()
    e4 = run_e4()
    elapsed = time.time() - t0

    with open(OUT / "summary.json", "w") as f:
        json.dump({"E1": e1, "E2": e2, "E3": e3, "E4": e4,
                    "elapsed_s": elapsed}, f, indent=2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # E1
    ax = axes[0, 0]; r = e1
    ax2 = ax.twinx()
    ax.errorbar(r["alpha"], r["MC"], yerr=r["MC_std"], fmt="o-",
                  color="#27ae60", lw=2, label="MC", capsize=3)
    ax2.errorbar(r["alpha"], r["Lyap"], yerr=r["Lyap_std"], fmt="s-",
                   color="#c0392b", lw=2, label="Lyap", capsize=3)
    ax.set_xscale("log"); ax.set_xlabel("α", fontsize=11)
    ax.set_ylabel("MC", color="#27ae60", fontsize=11)
    ax2.set_ylabel("Lyap", color="#c0392b", fontsize=11)
    ax2.axhline(0, color="black", ls="--", alpha=0.5)
    ax.set_title("E1: wide α range — small_world topology")
    ax.grid(alpha=0.3)
    i = int(np.argmax(r["MC"]))
    ax.axvline(r["alpha"][i], color="green", ls=":", alpha=0.5)
    ax.text(r["alpha"][i], max(r["MC"])*0.95,
              f"  α*={r['alpha'][i]:.2f}\n  MC*={r['MC'][i]:.2f}",
              fontsize=9, color="#27ae60")

    # E2
    ax = axes[0, 1]; r = e2
    ax2 = ax.twinx()
    ax.errorbar(r["frac_fast"], r["MC"], yerr=r["MC_std"], fmt="o-",
                  color="#27ae60", lw=2, capsize=3)
    ax2.errorbar(r["frac_fast"], r["Lyap"], yerr=r["Lyap_std"], fmt="s-",
                   color="#c0392b", lw=2, capsize=3)
    ax.set_xlabel("fraction of fast cells (α=8)", fontsize=11)
    ax.set_ylabel("MC", color="#27ae60", fontsize=11)
    ax2.set_ylabel("Lyap", color="#c0392b", fontsize=11)
    ax.set_title("E2: heterogeneous α (fast/slow split)\n"
                  "α_fast=8.0, α_slow=0.1")
    ax.grid(alpha=0.3)
    i = int(np.argmax(r["MC"]))
    ax.axvline(r["frac_fast"][i], color="green", ls=":", alpha=0.5)
    ax.text(r["frac_fast"][i], max(r["MC"])*0.95,
              f"  best mix\n  frac={r['frac_fast'][i]:.2f}\n  MC={r['MC'][i]:.2f}",
              fontsize=9, color="#27ae60")

    # E3
    ax = axes[1, 0]; r = e3
    ax.errorbar(r["alpha"], r["retention_A"], yerr=r["retention_A_std"],
                  fmt="o-", color="#8e44ad", lw=2, capsize=3)
    ax.set_xscale("log"); ax.set_xlabel("α", fontsize=11)
    ax.set_ylabel("retention of task A (correlation)", fontsize=11)
    ax.axhline(1.0, color="black", ls="--", alpha=0.3, label="perfect retention")
    ax.axhline(0.0, color="red", ls="--", alpha=0.3, label="catastrophic forgetting")
    ax.set_title("E3: catastrophic forgetting after task B")
    ax.legend(); ax.grid(alpha=0.3)

    # E4
    ax = axes[1, 1]; r = e4
    ax2 = ax.twinx()
    ax.errorbar(r["fb"], r["MC"], yerr=r["MC_std"], fmt="o-",
                  color="#27ae60", lw=2, capsize=3)
    ax2.errorbar(r["fb"], r["Lyap"], yerr=r["Lyap_std"], fmt="s-",
                   color="#c0392b", lw=2, capsize=3)
    ax.set_xlabel("feedback_gain", fontsize=11)
    ax.set_ylabel("MC", color="#27ae60", fontsize=11)
    ax2.set_ylabel("Lyap", color="#c0392b", fontsize=11)
    ax2.axhline(0, color="black", ls="--", alpha=0.5, label="edge of chaos")
    ax.set_title("E4: criticality hunt (α=1.5 fixed)")
    ax.grid(alpha=0.3)

    fig.suptitle(f"Extended plasticity — N={N_CELLS} cells, "
                  f"{N_SEEDS} seeds.  Total {elapsed/60:.1f} min",
                  fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "extended.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {OUT/'extended.png'}")
    print(f"Wrote {OUT/'summary.json'}")


if __name__ == "__main__":
    main()
