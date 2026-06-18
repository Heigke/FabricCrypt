"""z42_metaplasticity.py — homeostatic / adaptive VG2 (metaplasticity).

Inspired by Turrigiano-style homeostatic plasticity in cortex: if a cell
is firing too often, its excitability decreases (Vth raises, plasticity
dampens); if too quiet, it sensitizes.  In our NS-RAM cell, this maps to
adapting VG2 based on local activity.

Adaptation rule (per cell, per timestep):
  if Vb > V_active_high:          VG2[i] -= η · dt   (too plastic, dampen)
  elif Vb < V_active_low:          VG2[i] += η · dt   (too quiet, sensitize)

The system finds its own operating point.  Compare:
  - Static: fixed VG2 = 0.20 V
  - Adaptive: VG2 starts random in [-0.10, 0.40], adapts via η

Three sub-experiments:
  M1: Adaptive vs static — same task, measure MC + VG2 distribution
  M2: Adaptation rate η sweep — fast vs slow homeostasis
  M3: "Meditation effect" — does global VG2 boost (simulating ACh) help
       a previously-trained network learn novel patterns better?
"""
from __future__ import annotations
import json, time
from pathlib import Path
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from nsram.cell_fast import CellArray
from nsram.plasticity_net import (NetSim, topo_small_world,
                                   memory_capacity, lyapunov_proxy)

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z42_metaplasticity")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {DEVICE}")
N_CELLS = 96
N_SEEDS = 3


def make_net(VG2_init, alpha=1.5, seed=0):
    cells = CellArray(N_CELLS, alpha=alpha,
                          VG2=VG2_init.to(DEVICE), device=DEVICE)
    W = topo_small_world(N_CELLS, k=4, p_rewire=0.1, seed=seed, device=DEVICE)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    W_in = torch.randn(N_CELLS, 1, generator=g, device=DEVICE)
    return NetSim(cells=cells, W=W, W_in=W_in, feedback_gain=0.27)


def run_with_adaptation(net, U, VG1, eta, target_low=0.1, target_high=0.6,
                          vg2_clip=(-0.30, 0.60)):
    """Drive net with input + adapt VG2 each step.

    Returns Id_hist, Vb_hist, VG2_hist (all (T, N))
    """
    T = U.shape[0]
    Id_hist = torch.zeros(T, net.cells.N, device=DEVICE)
    Vb_hist = torch.zeros(T, net.cells.N, device=DEVICE)
    VG2_hist = torch.zeros(T, net.cells.N, device=DEVICE)
    for t in range(T):
        net.step(U[t], VG1[t])
        Id_hist[t] = net.cells.read(VG1[t])
        Vb_hist[t] = net.cells.Vb
        VG2_hist[t] = net.cells.VG2
        if eta > 0:
            # Homeostatic update
            high = (net.cells.Vb > target_high).float()
            low = (net.cells.Vb < target_low).float()
            d_vg2 = -eta * high + eta * low
            net.cells.VG2 = torch.clamp(net.cells.VG2 + d_vg2,
                                            vg2_clip[0], vg2_clip[1])
    return Id_hist, Vb_hist, VG2_hist


# ─────────────────────────────────────────────────────────────────────
# M1: Static vs Adaptive (single comparison)
# ─────────────────────────────────────────────────────────────────────

def run_m1():
    print("\n=== M1: static vs adaptive VG2 ===")
    rng = np.random.default_rng(0)
    T_train = 1500; T_test = 400
    u = rng.uniform(-1, 1, T_train + T_test).astype(np.float32)
    U = torch.from_numpy(u).unsqueeze(1).to(DEVICE)
    VG1 = torch.full((T_train + T_test,), 0.6).to(DEVICE)

    out = {"static": [], "adaptive": []}
    vg2_traj = None

    for s in range(N_SEEDS):
        # Static
        net_s = make_net(torch.full((N_CELLS,), 0.20), alpha=1.5, seed=s)
        Id_s, Vb_s, _ = run_with_adaptation(net_s, U, VG1, eta=0)
        # Adaptive — start from random VG2
        torch.manual_seed(s)
        VG2_init = torch.empty(N_CELLS).uniform_(-0.10, 0.40)
        net_a = make_net(VG2_init, alpha=1.5, seed=s)
        Id_a, Vb_a, VG2_t = run_with_adaptation(net_a, U, VG1, eta=2e-3)
        if s == 0:
            vg2_traj = VG2_t.cpu().numpy()

        # Compute MC on each separately
        for net, name in [(net_s, "static"), (net_a, "adaptive")]:
            net2 = NetSim(cells=CellArray(N_CELLS, alpha=net.cells.alpha,
                                                  VG2=net.cells.VG2,
                                                  device=DEVICE),
                            W=net.W, W_in=net.W_in,
                            feedback_gain=net.feedback_gain)
            mc = memory_capacity(net2, T_train=600, T_test=300, seed=s)
            out[name].append(mc)
        print(f"  seed={s}  static MC={out['static'][-1]:.2f}  "
              f"adaptive MC={out['adaptive'][-1]:.2f}", flush=True)

    return {
        "static_MC": float(np.mean(out["static"])),
        "static_MC_std": float(np.std(out["static"])),
        "adaptive_MC": float(np.mean(out["adaptive"])),
        "adaptive_MC_std": float(np.std(out["adaptive"])),
        "vg2_trajectory": vg2_traj.tolist() if vg2_traj is not None else None,
    }


# ─────────────────────────────────────────────────────────────────────
# M2: η sweep (homeostatic rate)
# ─────────────────────────────────────────────────────────────────────

def run_m2():
    print("\n=== M2: homeostatic rate η sweep ===")
    etas = np.logspace(-5, -1, 9)
    rs = {"eta": [], "MC": [], "MC_std": [],
           "VG2_mean": [], "VG2_std": []}
    rng = np.random.default_rng(0)
    T_settle = 1500
    u = rng.uniform(-1, 1, T_settle).astype(np.float32)
    U = torch.from_numpy(u).unsqueeze(1).to(DEVICE)
    VG1 = torch.full((T_settle,), 0.6).to(DEVICE)
    for eta in etas:
        mcs, vg2_means, vg2_stds = [], [], []
        for s in range(N_SEEDS):
            torch.manual_seed(s)
            VG2_init = torch.empty(N_CELLS).uniform_(-0.10, 0.40)
            net = make_net(VG2_init, alpha=1.5, seed=s)
            run_with_adaptation(net, U, VG1, eta=float(eta))
            # After settling, freeze VG2 and measure MC
            net2 = NetSim(cells=CellArray(N_CELLS, alpha=net.cells.alpha,
                                                  VG2=net.cells.VG2,
                                                  device=DEVICE),
                            W=net.W, W_in=net.W_in,
                            feedback_gain=net.feedback_gain)
            mcs.append(memory_capacity(net2, T_train=600, T_test=300, seed=s))
            vg2_means.append(float(net.cells.VG2.mean().item()))
            vg2_stds.append(float(net.cells.VG2.std().item()))
        rs["eta"].append(float(eta))
        rs["MC"].append(float(np.mean(mcs)))
        rs["MC_std"].append(float(np.std(mcs)))
        rs["VG2_mean"].append(float(np.mean(vg2_means)))
        rs["VG2_std"].append(float(np.mean(vg2_stds)))
        print(f"  η={eta:.1e}  MC={rs['MC'][-1]:.2f}±{rs['MC_std'][-1]:.2f}  "
              f"⟨VG2⟩={rs['VG2_mean'][-1]:+.3f}±{rs['VG2_std'][-1]:.3f}",
              flush=True)
    return rs


# ─────────────────────────────────────────────────────────────────────
# M3: "Meditation" — global VG2 boost on a pre-trained net
# ─────────────────────────────────────────────────────────────────────

def run_m3():
    """Train a network on task A, then perturb VG2 globally (boost or
    reduce) and measure how well it learns task B.  Models 'attentional
    state' / 'plasticity gating' (e.g. ACh release in meditation)."""
    print("\n=== M3: meditation effect (global VG2 boost) ===")
    boosts = np.linspace(-0.20, 0.30, 11)
    rs = {"boost": [], "MC_B": [], "MC_B_std": []}

    for boost in boosts:
        mcs = []
        for s in range(N_SEEDS):
            # Train (just drive) with task A input
            torch.manual_seed(s)
            net = make_net(torch.full((N_CELLS,), 0.10),
                              alpha=1.5, seed=s)
            rng = np.random.default_rng(100 + s)
            uA = rng.uniform(-1, 1, 800).astype(np.float32)
            U = torch.from_numpy(uA).unsqueeze(1).to(DEVICE)
            VG1 = torch.full((800,), 0.6).to(DEVICE)
            run_with_adaptation(net, U, VG1, eta=0)
            # Apply VG2 boost globally
            net.cells.VG2 = torch.clamp(net.cells.VG2 + float(boost),
                                            -0.30, 0.60)
            # Measure MC on task B (different rng seed)
            net2 = NetSim(cells=CellArray(N_CELLS, alpha=net.cells.alpha,
                                                  VG2=net.cells.VG2,
                                                  device=DEVICE),
                            W=net.W, W_in=net.W_in,
                            feedback_gain=net.feedback_gain)
            mc = memory_capacity(net2, T_train=600, T_test=300,
                                     seed=200 + s)
            mcs.append(mc)
        rs["boost"].append(float(boost))
        rs["MC_B"].append(float(np.mean(mcs)))
        rs["MC_B_std"].append(float(np.std(mcs)))
        print(f"  ΔVG2={boost:+.2f}  MC(task B)={rs['MC_B'][-1]:.2f}", flush=True)
    return rs


def main():
    t0 = time.time()
    m1 = run_m1()
    m2 = run_m2()
    m3 = run_m3()
    elapsed = time.time() - t0

    with open(OUT / "summary.json", "w") as f:
        json.dump({"M1": m1, "M2": m2, "M3": m3, "elapsed_s": elapsed},
                    f, indent=2)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3)

    # M1 bar comparison
    ax = fig.add_subplot(gs[0, 0])
    bars = ax.bar(["static\nVG2=0.20", "adaptive\nVG2 (η=2e-3)"],
                    [m1["static_MC"], m1["adaptive_MC"]],
                    yerr=[m1["static_MC_std"], m1["adaptive_MC_std"]],
                    color=["#3498db", "#e67e22"], capsize=8)
    for b, v in zip(bars, [m1["static_MC"], m1["adaptive_MC"]]):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.2f}",
                  ha="center", va="bottom", fontsize=11, weight="bold")
    ax.set_ylabel("Memory Capacity")
    ax.set_title("M1: static vs adaptive VG2")
    ax.grid(alpha=0.3, axis="y")

    # M1 trajectory
    ax = fig.add_subplot(gs[0, 1:])
    if m1["vg2_trajectory"] is not None:
        traj = np.array(m1["vg2_trajectory"])
        # Subset cells for clarity
        for i in range(0, traj.shape[1], traj.shape[1]//12):
            ax.plot(traj[:, i], lw=0.8, alpha=0.6)
        ax.plot(traj.mean(axis=1), color="black", lw=2.2, label="mean")
        ax.set_xlabel("step"); ax.set_ylabel("VG2 [V]")
        ax.set_title("M1: VG2 evolution under homeostasis (12 sample cells + mean)")
        ax.legend(); ax.grid(alpha=0.3)

    # M2 η sweep
    ax = fig.add_subplot(gs[1, 0])
    r = m2
    ax2 = ax.twinx()
    ax.errorbar(r["eta"], r["MC"], yerr=r["MC_std"], fmt="o-",
                  color="#27ae60", lw=2, capsize=3)
    ax2.errorbar(r["eta"], r["VG2_mean"], yerr=r["VG2_std"], fmt="s-",
                   color="#9b59b6", lw=2, capsize=3, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("η (homeostatic rate)")
    ax.set_ylabel("MC", color="#27ae60")
    ax2.set_ylabel("⟨VG2⟩ after settling [V]", color="#9b59b6")
    ax.set_title("M2: η sweep (too fast = saturates, too slow = no effect)")
    ax.grid(alpha=0.3)

    # M3 meditation effect
    ax = fig.add_subplot(gs[1, 1:])
    r = m3
    ax.errorbar(r["boost"], r["MC_B"], yerr=r["MC_B_std"], fmt="o-",
                  color="#e74c3c", lw=2, capsize=3)
    i = int(np.argmax(r["MC_B"]))
    ax.axvline(r["boost"][i], color="green", ls=":", alpha=0.5)
    ax.text(r["boost"][i], max(r["MC_B"])*0.95,
              f"  best boost = {r['boost'][i]:+.2f} V\n  MC = {r['MC_B'][i]:.2f}",
              fontsize=10, color="green")
    ax.set_xlabel("global VG2 boost ΔVG2 [V]  (after task A)")
    ax.set_ylabel("MC on novel task B")
    ax.set_title("M3: 'meditation effect' — does opening plasticity gate help?")
    ax.grid(alpha=0.3)

    fig.suptitle("z42 — metaplasticity (adaptive VG2 + homeostasis)",
                  fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "metaplasticity.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {OUT/'metaplasticity.png'}")
    print(f"Total: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
