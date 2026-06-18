"""z49_bcm_learning.py — BCM (Bienenstock-Cooper-Munro) plasticity rule.

Z48 showed pure Hebbian destroys reservoir performance because it pushes
W toward spectral radius 1 (edge of chaos), but our bistable cells are
optimal SUBCRITICAL.  BCM has a self-stabilizing sliding threshold:

  ΔW[i,j] = η · y_pre · y_post · (y_post − θ_M[i]) − decay · W[i,j]
  θ_M[i] = ⟨y_post[i]²⟩  (running mean)

When a cell is too active (y² > θ), update becomes LTD (negative).
When too quiet (y² < θ), Hebbian (positive). Net effect: cells become
selective without saturating — exactly what cortex does.

Three experiments:
  B1: BCM vs fixed-W vs Hebbian (3-way comparison)
  B2: BCM η sweep
  B3: emergent W structure under BCM (does it stay subcritical?)
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
from nsram.plasticity_net import (NetSim, topo_random,
                                   memory_capacity, lyapunov_proxy)

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z49_bcm_learning")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {DEVICE}")
N_CELLS = 96
N_SEEDS = 3

CAL = dict(VTH0=0.43, K_back=-0.98, A_iii=4.71, G_bjt=1.00,
            V_bjt_on=0.74, V_latch=0.58, K_leak=0.021)


def make_net(seed=0):
    VG2 = torch.full((N_CELLS,), -0.20, device=DEVICE)
    cells = CellArray(N_CELLS, alpha=1.5, VG2=VG2,
                          **CAL, device=DEVICE)
    W = topo_random(N_CELLS, p=0.1, seed=seed, device=DEVICE) * 1.0
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    W_in = torch.randn(N_CELLS, 1, generator=g, device=DEVICE)
    return NetSim(cells=cells, W=W, W_in=W_in, feedback_gain=0.27)


def bcm_update(W, Vb, theta, eta, decay=1e-3, offset=0.3):
    """In-place BCM update."""
    y = Vb - offset
    factor = y * (y - theta)
    delta = eta * torch.outer(factor, y)
    W += delta - decay * W
    W.fill_diagonal_(0)
    rs = W.abs().sum(dim=1, keepdim=True).clamp(min=1e-9)
    W /= rs


def hebbian_update(W, Vb, mean_Vb, eta, decay=1e-3):
    delta = Vb - mean_Vb
    W += eta * torch.outer(delta, delta) - decay * W
    W.fill_diagonal_(0)
    rs = W.abs().sum(dim=1, keepdim=True).clamp(min=1e-9)
    W /= rs


def train_with_rule(net, T, U, rule="bcm", eta=2e-3, theta_momentum=0.95):
    """Drive net while updating W with the chosen rule.
    rule ∈ {fixed, hebbian, bcm}"""
    VG1 = torch.full((T,), 0.6, device=DEVICE)
    Vb_running = net.cells.Vb.mean().item()
    theta = torch.zeros(N_CELLS, device=DEVICE)
    momentum = 0.95
    offset = 0.3
    for t in range(T):
        net.step(U[t], VG1[t])
        Vb = net.cells.Vb
        m = float(Vb.mean().item())
        Vb_running = momentum * Vb_running + (1 - momentum) * m
        # update sliding threshold
        y = Vb - offset
        theta = theta_momentum * theta + (1 - theta_momentum) * (y * y)
        if rule == "fixed" or eta == 0:
            continue
        if rule == "hebbian":
            hebbian_update(net.W, Vb, Vb_running, eta=eta)
        elif rule == "bcm":
            bcm_update(net.W, Vb, theta, eta=eta, offset=offset)


def analyze_W(W):
    Wn = W.cpu().numpy()
    eigs = np.linalg.eigvals(Wn)
    return {"spectral_radius": float(np.abs(eigs).max()),
             "frobenius": float(np.linalg.norm(Wn))}


def run_b1():
    print("\n=== B1: fixed vs Hebbian vs BCM ===")
    rng = np.random.default_rng(0)
    T_train = 1500
    u = rng.uniform(-1, 1, T_train).astype(np.float32)
    U = torch.from_numpy(u).unsqueeze(1).to(DEVICE)

    rs = {"fixed": [], "hebbian": [], "bcm": []}
    for s in range(N_SEEDS):
        for rule in ["fixed", "hebbian", "bcm"]:
            net = make_net(seed=s)
            train_with_rule(net, T_train, U, rule=rule, eta=2e-3)
            net.cells.reset()
            mc = memory_capacity(net, T_train=600, T_test=300, seed=s)
            rs[rule].append(mc)
        print(f"  seed={s}  fixed={rs['fixed'][-1]:.2f}  "
              f"hebbian={rs['hebbian'][-1]:.2f}  bcm={rs['bcm'][-1]:.2f}",
              flush=True)
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in rs.items()}


def run_b2():
    print("\n=== B2: BCM η sweep ===")
    rng = np.random.default_rng(0)
    T_train = 1500
    u = rng.uniform(-1, 1, T_train).astype(np.float32)
    U = torch.from_numpy(u).unsqueeze(1).to(DEVICE)

    etas = np.logspace(-5, -1, 9)
    rs = {"eta": [], "MC": [], "MC_std": []}
    for eta in etas:
        mcs = []
        for s in range(N_SEEDS):
            net = make_net(seed=s)
            train_with_rule(net, T_train, U, rule="bcm", eta=float(eta))
            net.cells.reset()
            mcs.append(memory_capacity(net, T_train=600, T_test=300, seed=s))
        rs["eta"].append(float(eta))
        rs["MC"].append(float(np.mean(mcs)))
        rs["MC_std"].append(float(np.std(mcs)))
        print(f"  η={eta:.1e}  MC={rs['MC'][-1]:.2f}±{rs['MC_std'][-1]:.2f}",
              flush=True)
    return rs


def run_b3():
    print("\n=== B3: emergent W structure (BCM vs Hebbian) ===")
    rng = np.random.default_rng(0)
    T_train = 2000
    u = rng.uniform(-1, 1, T_train).astype(np.float32)
    U = torch.from_numpy(u).unsqueeze(1).to(DEVICE)

    rs = {"hebbian_specrad": [], "bcm_specrad": []}
    for s in range(N_SEEDS):
        # Hebbian
        net = make_net(seed=s)
        train_with_rule(net, T_train, U, rule="hebbian", eta=2e-3)
        rs["hebbian_specrad"].append(analyze_W(net.W)["spectral_radius"])
        # BCM
        net = make_net(seed=s)
        train_with_rule(net, T_train, U, rule="bcm", eta=2e-3)
        rs["bcm_specrad"].append(analyze_W(net.W)["spectral_radius"])
        print(f"  seed={s}  Hebbian spec_rad={rs['hebbian_specrad'][-1]:.2f}  "
              f"BCM spec_rad={rs['bcm_specrad'][-1]:.2f}", flush=True)
    return rs


def main():
    t0 = time.time()
    b1 = run_b1()
    b2 = run_b2()
    b3 = run_b3()
    elapsed = time.time() - t0

    with open(OUT / "summary.json", "w") as f:
        json.dump({"B1": b1, "B2": b2, "B3": b3, "elapsed_s": elapsed}, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # B1
    ax = axes[0]
    rules = ["fixed", "hebbian", "bcm"]
    means = [b1[r][0] for r in rules]
    stds = [b1[r][1] for r in rules]
    bars = ax.bar(rules, means, yerr=stds,
                    color=["#3498db", "#e74c3c", "#27ae60"], capsize=8)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.2f}",
                  ha="center", va="bottom", fontsize=11, weight="bold")
    ax.set_ylabel("MC")
    ax.set_title("B1: BCM vs Hebbian vs fixed")
    ax.grid(alpha=0.3, axis="y")

    # B2
    ax = axes[1]
    r = b2
    ax.errorbar(r["eta"], r["MC"], yerr=r["MC_std"], fmt="o-",
                  color="#27ae60", lw=2, capsize=4)
    i = int(np.argmax(r["MC"]))
    ax.axvline(r["eta"][i], color="green", ls=":", alpha=0.5)
    ax.text(r["eta"][i], max(r["MC"])*0.95,
              f"  η*={r['eta'][i]:.0e}\n  MC={r['MC'][i]:.2f}",
              fontsize=9, color="green")
    ax.set_xscale("log"); ax.set_xlabel("η  (BCM rate)")
    ax.set_ylabel("MC")
    ax.set_title("B2: BCM η sweep")
    ax.grid(alpha=0.3)

    # B3
    ax = axes[2]
    x = np.arange(N_SEEDS); w = 0.35
    ax.bar(x - w/2, b3["hebbian_specrad"], w, label="Hebbian", color="#e74c3c")
    ax.bar(x + w/2, b3["bcm_specrad"], w, label="BCM", color="#27ae60")
    ax.axhline(1.0, color="black", ls="--", alpha=0.5, label="edge of chaos")
    ax.set_xticks(x); ax.set_xticklabels([f"seed {i}" for i in range(N_SEEDS)])
    ax.set_ylabel("spectral radius")
    ax.set_title("B3: emergent spec_rad — BCM stays subcritical?")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"z49 — BCM plasticity (calibrated cell)  total {elapsed/60:.1f} min",
                  fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "bcm.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {OUT/'bcm.png'}")
    print(f"Total: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
