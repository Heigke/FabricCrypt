"""z48_hebbian_learning.py — online Hebbian learning of recurrent weights W.

Until now our networks had FIXED random W. This script adds a learning
rule where W self-organizes from input statistics:

  ΔW[i,j] = η · (Vb_i − ⟨Vb⟩) · (Vb_j − ⟨Vb⟩) − decay · W[i,j]

This is Oja-like local Hebbian update with weight decay (prevents
runaway).  After settling, we freeze W and measure memory capacity, then
compare to fixed-random W.

Three experiments:
  H1: fixed-W vs Hebbian-W on random input — does self-organization help?
  H2: η sweep — find Goldilocks for Hebbian rate
  H3: Hebbian-W structure — analyze final W: clustered? scale-free?
       hierarchical?  (compares emergent structure to predefined topologies)
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
from nsram.plasticity_net import (NetSim, topo_random, topo_small_world,
                                   memory_capacity, lyapunov_proxy)

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z48_hebbian_learning")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {DEVICE}")
N_CELLS = 96
N_SEEDS = 3

CAL = dict(VTH0=0.43, K_back=-0.98, A_iii=4.71, G_bjt=1.00,
            V_bjt_on=0.74, V_latch=0.58, K_leak=0.021)


def make_cells(seed=0):
    VG2 = torch.full((N_CELLS,), -0.20, device=DEVICE)
    return CellArray(N_CELLS, alpha=1.5, VG2=VG2,
                          **CAL, device=DEVICE)


def make_W_in(seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.randn(N_CELLS, 1, generator=g, device=DEVICE)


def hebbian_update(W: torch.Tensor, Vb: torch.Tensor, mean_Vb: float,
                    eta: float, decay: float = 1e-3):
    """In-place: W += η · (Vb-mean) (Vb-mean)^T − decay · W"""
    delta = Vb - mean_Vb
    outer = torch.outer(delta, delta)
    W += eta * outer - decay * W
    # Zero diagonal (no self-loops)
    W.fill_diagonal_(0)
    # Row-normalize to keep dynamics bounded
    rs = W.abs().sum(dim=1, keepdim=True).clamp(min=1e-9)
    W /= rs


def train_hebbian(net: NetSim, T: int, eta: float, U: torch.Tensor,
                    VG1: float = 0.6):
    """Drive net while updating W via Hebbian rule."""
    VG1_seq = torch.full((T,), VG1, device=DEVICE)
    Vb_running = net.cells.Vb.mean().item()
    momentum = 0.95
    for t in range(T):
        net.step(U[t], VG1_seq[t])
        m = float(net.cells.Vb.mean().item())
        Vb_running = momentum * Vb_running + (1 - momentum) * m
        if eta > 0:
            hebbian_update(net.W, net.cells.Vb, Vb_running, eta=eta)


# ─────────────────────────────────────────────────────────────────────
# H1: fixed-W vs Hebbian-W
# ─────────────────────────────────────────────────────────────────────

def run_h1():
    print("\n=== H1: fixed-W vs Hebbian-W ===")
    rng = np.random.default_rng(0)
    T_train = 1500
    u = rng.uniform(-1, 1, T_train).astype(np.float32)
    U = torch.from_numpy(u).unsqueeze(1).to(DEVICE)

    rs = {"fixed_MC": [], "hebbian_MC": []}
    for s in range(N_SEEDS):
        # Fixed-W: random topology, no updates
        cells_f = make_cells(seed=s)
        W_f = topo_random(N_CELLS, p=0.1, seed=s, device=DEVICE) * 1.0
        net_f = NetSim(cells=cells_f, W=W_f.clone(),
                         W_in=make_W_in(seed=s), feedback_gain=0.27)
        train_hebbian(net_f, T_train, eta=0.0, U=U)
        net_f.cells.reset()
        mc_f = memory_capacity(net_f, T_train=600, T_test=300, seed=s)

        # Hebbian-W: same start, updates during training
        cells_h = make_cells(seed=s)
        W_h = topo_random(N_CELLS, p=0.1, seed=s, device=DEVICE) * 1.0
        net_h = NetSim(cells=cells_h, W=W_h.clone(),
                         W_in=make_W_in(seed=s), feedback_gain=0.27)
        train_hebbian(net_h, T_train, eta=2e-3, U=U)
        net_h.cells.reset()
        mc_h = memory_capacity(net_h, T_train=600, T_test=300, seed=s)

        rs["fixed_MC"].append(mc_f)
        rs["hebbian_MC"].append(mc_h)
        print(f"  seed={s}  fixed MC={mc_f:.2f}  hebbian MC={mc_h:.2f}  "
              f"Δ={mc_h-mc_f:+.2f}", flush=True)

    return {
        "fixed_MC_mean": float(np.mean(rs["fixed_MC"])),
        "fixed_MC_std": float(np.std(rs["fixed_MC"])),
        "hebbian_MC_mean": float(np.mean(rs["hebbian_MC"])),
        "hebbian_MC_std": float(np.std(rs["hebbian_MC"])),
    }


# ─────────────────────────────────────────────────────────────────────
# H2: η sweep
# ─────────────────────────────────────────────────────────────────────

def run_h2():
    print("\n=== H2: Hebbian rate η sweep ===")
    rng = np.random.default_rng(0)
    T_train = 1500
    u = rng.uniform(-1, 1, T_train).astype(np.float32)
    U = torch.from_numpy(u).unsqueeze(1).to(DEVICE)

    etas = np.logspace(-5, -1, 9)
    rs = {"eta": [], "MC": [], "MC_std": []}
    for eta in etas:
        mcs = []
        for s in range(N_SEEDS):
            cells = make_cells(seed=s)
            W = topo_random(N_CELLS, p=0.1, seed=s, device=DEVICE) * 1.0
            net = NetSim(cells=cells, W=W.clone(),
                           W_in=make_W_in(seed=s), feedback_gain=0.27)
            train_hebbian(net, T_train, eta=float(eta), U=U)
            net.cells.reset()
            mcs.append(memory_capacity(net, T_train=600, T_test=300, seed=s))
        rs["eta"].append(float(eta))
        rs["MC"].append(float(np.mean(mcs)))
        rs["MC_std"].append(float(np.std(mcs)))
        print(f"  η={eta:.1e}  MC={rs['MC'][-1]:.2f}±{rs['MC_std'][-1]:.2f}",
              flush=True)
    return rs


# ─────────────────────────────────────────────────────────────────────
# H3: emergent W structure
# ─────────────────────────────────────────────────────────────────────

def analyze_W(W: torch.Tensor):
    """Compute summary stats: degree distribution, clustering, eigvals."""
    Wn = W.cpu().numpy()
    degrees = (np.abs(Wn) > 0.01).sum(axis=1)
    sparsity = (np.abs(Wn) < 1e-6).mean()
    # Eigenvalue stats
    eigs = np.linalg.eigvals(Wn)
    spec_radius = float(np.abs(eigs).max())
    return {
        "mean_degree": float(degrees.mean()),
        "std_degree": float(degrees.std()),
        "max_degree": int(degrees.max()),
        "sparsity": float(sparsity),
        "spectral_radius": spec_radius,
    }


def run_h3():
    print("\n=== H3: emergent W structure (best η from H2) ===")
    rng = np.random.default_rng(0)
    T_train = 2000
    u = rng.uniform(-1, 1, T_train).astype(np.float32)
    U = torch.from_numpy(u).unsqueeze(1).to(DEVICE)

    eta_best = 2e-3   # tune from H2 result
    rs = {"initial": [], "final": [], "W_initial": None, "W_final": None}
    for s in range(N_SEEDS):
        cells = make_cells(seed=s)
        W_init = topo_random(N_CELLS, p=0.1, seed=s, device=DEVICE) * 1.0
        net = NetSim(cells=cells, W=W_init.clone(),
                       W_in=make_W_in(seed=s), feedback_gain=0.27)
        rs["initial"].append(analyze_W(W_init))
        train_hebbian(net, T_train, eta=eta_best, U=U)
        rs["final"].append(analyze_W(net.W))
        if s == 0:
            rs["W_initial"] = W_init.cpu().numpy().tolist()
            rs["W_final"] = net.W.cpu().numpy().tolist()
        print(f"  seed={s}  initial spec_rad={rs['initial'][-1]['spectral_radius']:.2f}  "
              f"final spec_rad={rs['final'][-1]['spectral_radius']:.2f}", flush=True)
    return rs


def main():
    t0 = time.time()
    h1 = run_h1()
    h2 = run_h2()
    h3 = run_h3()
    elapsed = time.time() - t0

    summary = {"H1": h1, "H2": h2, "H3": h3, "elapsed_s": elapsed}
    # Don't dump full W matrices into json
    summary_for_save = json.loads(json.dumps(summary))
    summary_for_save["H3"]["W_initial"] = "(N×N) — see plot"
    summary_for_save["H3"]["W_final"] = "(N×N) — see plot"
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary_for_save, f, indent=2)

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)

    # H1 bar
    ax = fig.add_subplot(gs[0, 0])
    bars = ax.bar(["fixed W", "Hebbian W"],
                    [h1["fixed_MC_mean"], h1["hebbian_MC_mean"]],
                    yerr=[h1["fixed_MC_std"], h1["hebbian_MC_std"]],
                    color=["#3498db", "#e67e22"], capsize=8)
    for b, v in zip(bars, [h1["fixed_MC_mean"], h1["hebbian_MC_mean"]]):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.2f}",
                  ha="center", va="bottom", fontsize=11, weight="bold")
    gain_pct = (h1["hebbian_MC_mean"] / h1["fixed_MC_mean"] - 1) * 100
    ax.set_ylabel("MC")
    ax.set_title(f"H1: Hebbian gain = {gain_pct:+.0f}%")
    ax.grid(alpha=0.3, axis="y")

    # H2 η sweep
    ax = fig.add_subplot(gs[0, 1])
    r = h2
    ax.errorbar(r["eta"], r["MC"], yerr=r["MC_std"], fmt="o-",
                  color="#27ae60", lw=2, capsize=4)
    i = int(np.argmax(r["MC"]))
    ax.axvline(r["eta"][i], color="green", ls=":", alpha=0.5)
    ax.text(r["eta"][i], max(r["MC"])*0.95,
              f"  η*={r['eta'][i]:.0e}\n  MC={r['MC'][i]:.2f}",
              fontsize=9, color="green")
    ax.set_xscale("log"); ax.set_xlabel("η  (Hebbian rate)")
    ax.set_ylabel("MC")
    ax.set_title("H2: η sweep")
    ax.grid(alpha=0.3)

    # H3 W matrices
    if h3["W_initial"] and h3["W_final"]:
        W_i = np.array(h3["W_initial"])
        W_f = np.array(h3["W_final"])
        vmax = max(np.abs(W_i).max(), np.abs(W_f).max())
        ax = fig.add_subplot(gs[0, 2])
        im = ax.imshow(W_i, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title("H3: initial W (random)")
        fig.colorbar(im, ax=ax)
        ax = fig.add_subplot(gs[1, 0])
        im = ax.imshow(W_f, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title("H3: final W (Hebbian-shaped)")
        fig.colorbar(im, ax=ax)

    # H3 stats
    ax = fig.add_subplot(gs[1, 1:])
    metrics = ["mean_degree", "std_degree", "max_degree", "spectral_radius"]
    init_means = [np.mean([s[m] for s in h3["initial"]]) for m in metrics]
    final_means = [np.mean([s[m] for s in h3["final"]]) for m in metrics]
    x = np.arange(len(metrics)); w = 0.35
    ax.bar(x - w/2, init_means, w, label="initial", color="#3498db")
    ax.bar(x + w/2, final_means, w, label="final", color="#e67e22")
    ax.set_xticks(x); ax.set_xticklabels(metrics, rotation=20)
    ax.set_title("H3: emergent W stats")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    for i, m in enumerate(metrics):
        ax.text(i - w/2, init_means[i], f"{init_means[i]:.2f}",
                  ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, final_means[i], f"{final_means[i]:.2f}",
                  ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"z48 — online Hebbian learning of recurrent W\n"
                  f"Total {elapsed/60:.1f} min, calibrated cell, N={N_CELLS}",
                  fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "hebbian.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {OUT/'hebbian.png'}")
    print(f"Total: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
