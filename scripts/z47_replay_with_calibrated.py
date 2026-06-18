"""z47_replay_with_calibrated.py — replay the key plasticity experiments
with the *calibrated* cell_fast parameters (K_back=-0.98, data-empirical
sign).  Identify which previous findings survive the sign flip.

Sub-experiments (mirroring z38-z41):
  P1: topology × α sweep (replicates z38)
  P2: VG2 homogeneous sweep (replicates z41 V1) — should now show
      sweet-spot at NEGATIVE VG2 instead of positive
  P3: bistability advantage (replicates z44 R5)
  P4: feedback_gain criticality (replicates z40 E4)

Calibrated params from z45:
  VTH0=0.43, K_back=-0.98, A_iii=4.71, G_bjt=1.00,
  V_bjt_on=0.74, V_latch=0.58, K_leak=0.021
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
from nsram.plasticity_net import (NetSim, topo_random, topo_ring,
                                   topo_small_world, topo_scale_free,
                                   topo_hierarchical, topo_full,
                                   memory_capacity, lyapunov_proxy)

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z47_replay_with_calibrated")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {DEVICE}")
N_CELLS = 96
N_SEEDS = 3

# ── Calibrated parameters from z45 (data-empirical) ──
CAL = dict(
    VTH0=0.43, K_back=-0.98, A_iii=4.71, G_bjt=1.00,
    V_bjt_on=0.74, V_latch=0.58, K_leak=0.021,
)


def make_net(N, alpha, VG2, topology="small_world", fb=0.27, seed=0,
              ws_k=4, ws_p=0.1, **cal_override):
    cal = {**CAL, **cal_override}
    cells = CellArray(N, alpha=alpha, VG2=VG2.to(DEVICE),
                          VTH0=cal["VTH0"], K_back=cal["K_back"],
                          A_iii=cal["A_iii"], G_bjt=cal["G_bjt"],
                          V_bjt_on=cal["V_bjt_on"],
                          V_latch=cal["V_latch"], K_leak=cal["K_leak"],
                          device=DEVICE)
    if topology == "random":
        W = topo_random(N, p=0.1, seed=seed, device=DEVICE)
    elif topology == "ring":
        W = topo_ring(N, k=ws_k, device=DEVICE)
    elif topology == "small_world":
        W = topo_small_world(N, k=ws_k, p_rewire=ws_p, seed=seed, device=DEVICE)
    elif topology == "scale_free":
        W = topo_scale_free(N, m=3, seed=seed, device=DEVICE)
    elif topology == "hierarchical":
        W = topo_hierarchical(N, levels=3, branching=4,
                                seed=seed, device=DEVICE)
    elif topology == "full":
        W = topo_full(N, device=DEVICE)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    W_in = torch.randn(N, 1, generator=g, device=DEVICE)
    return NetSim(cells=cells, W=W, W_in=W_in, feedback_gain=fb)


# ─────────────────────────────────────────────────────────────────────
# P1: topology × α sweep (replicates z38)
# ─────────────────────────────────────────────────────────────────────

def run_p1():
    print("\n=== P1: topology × α sweep (calibrated) ===")
    topologies = ["random", "ring", "small_world", "scale_free",
                   "hierarchical", "full"]
    alphas = np.logspace(-1.5, 1.0, 8)
    rs = {topo: {"alpha": [], "MC": [], "MC_std": []} for topo in topologies}
    for topo in topologies:
        for a in alphas:
            mcs = []
            for s in range(N_SEEDS):
                # Use VG2 = -0.20 (where calibrated K_back negative says
                # the cell is most plastic — high body, low Vth via K_back<0).
                VG2 = torch.full((N_CELLS,), -0.20)
                net = make_net(N_CELLS, alpha=float(a), VG2=VG2,
                                 topology=topo, seed=s)
                mcs.append(memory_capacity(net, T_train=600, T_test=300, seed=s))
            rs[topo]["alpha"].append(float(a))
            rs[topo]["MC"].append(float(np.mean(mcs)))
            rs[topo]["MC_std"].append(float(np.std(mcs)))
        i = int(np.argmax(rs[topo]["MC"]))
        print(f"  {topo:14s}  α*={rs[topo]['alpha'][i]:.3f}  "
              f"MC*={rs[topo]['MC'][i]:.2f}±{rs[topo]['MC_std'][i]:.2f}",
              flush=True)
    return rs


# ─────────────────────────────────────────────────────────────────────
# P2: VG2 homogeneous sweep (replicates z41 V1) — sweet spot polarity
# ─────────────────────────────────────────────────────────────────────

def run_p2():
    print("\n=== P2: VG2 homogeneous sweep (small_world, α=1.5, calibrated) ===")
    vg2_grid = np.linspace(-0.40, 0.40, 17)
    rs = {"VG2": [], "MC": [], "MC_std": [], "Lyap": []}
    for vg2 in vg2_grid:
        mcs, lys = [], []
        for s in range(N_SEEDS):
            VG2 = torch.full((N_CELLS,), float(vg2))
            net = make_net(N_CELLS, alpha=1.5, VG2=VG2,
                             topology="small_world", seed=s)
            mcs.append(memory_capacity(net, T_train=600, T_test=300, seed=s))
            net2 = make_net(N_CELLS, alpha=1.5, VG2=VG2,
                              topology="small_world", seed=s)
            lys.append(lyapunov_proxy(net2, T_warmup=200, T_meas=150, seed=s))
        rs["VG2"].append(float(vg2))
        rs["MC"].append(float(np.mean(mcs)))
        rs["MC_std"].append(float(np.std(mcs)))
        rs["Lyap"].append(float(np.mean(lys)))
        print(f"  VG2={vg2:+.2f}  MC={rs['MC'][-1]:.2f}±{rs['MC_std'][-1]:.2f}  "
              f"Lyap={rs['Lyap'][-1]:+.3f}", flush=True)
    return rs


# ─────────────────────────────────────────────────────────────────────
# P3: bistability advantage replication
# ─────────────────────────────────────────────────────────────────────

class LinearCellArray:
    def __init__(self, N, alpha=1.5, VG2=None, K_leak=0.05, dt=0.05,
                  device="cpu"):
        self.N = N; self.dt = dt; self.device = device
        self.alpha = (alpha if isinstance(alpha, torch.Tensor)
                       else torch.full((N,), float(alpha), device=device))
        self.VG2 = (VG2.to(device) if VG2 is not None
                     else torch.full((N,), 0.0, device=device))
        self.K_leak = K_leak
        self.Vb = self.VG2.clone()
    def reset(self): self.Vb = self.VG2.clone()
    def channel_on(self, VG1): return torch.ones_like(self.VG2)
    def step(self, VG1, drive):
        if not isinstance(drive, torch.Tensor):
            drive = torch.full((self.N,), float(drive), device=self.device)
        self.Vb = (1 - self.K_leak * self.dt) * self.Vb + self.alpha * drive * self.dt
        self.Vb = torch.clamp(self.Vb, -2.0, 2.0)
        return self.read(VG1)
    def read(self, VG1): return torch.tanh(self.Vb)


def run_p3():
    print("\n=== P3: bistability vs linear-cell baseline (calibrated) ===")
    rs = {}
    bistable_mc = []
    linear_mc = []
    for s in range(5):
        torch.manual_seed(s)
        VG2 = torch.full((N_CELLS,), -0.20)   # calibrated optimum side
        # Bistable
        net = make_net(N_CELLS, alpha=1.5, VG2=VG2,
                         topology="small_world", seed=s)
        bistable_mc.append(memory_capacity(net, T_train=600, T_test=300, seed=s))
        # Linear
        cells = LinearCellArray(N_CELLS, alpha=1.5,
                                     VG2=VG2.to(DEVICE), device=DEVICE)
        W = topo_small_world(N_CELLS, k=4, p_rewire=0.1,
                                 seed=s, device=DEVICE)
        g = torch.Generator(device=DEVICE).manual_seed(s)
        W_in = torch.randn(N_CELLS, 1, generator=g, device=DEVICE)
        netL = NetSim(cells=cells, W=W, W_in=W_in, feedback_gain=0.27)
        linear_mc.append(memory_capacity(netL, T_train=600, T_test=300, seed=s))
    rs["bistable"] = (float(np.mean(bistable_mc)), float(np.std(bistable_mc)))
    rs["linear"] = (float(np.mean(linear_mc)), float(np.std(linear_mc)))
    rs["gain_pct"] = (rs["bistable"][0] / rs["linear"][0] - 1) * 100
    print(f"  bistable: {rs['bistable'][0]:.2f}±{rs['bistable'][1]:.2f}")
    print(f"  linear:   {rs['linear'][0]:.2f}±{rs['linear'][1]:.2f}")
    print(f"  → bistability gain = {rs['gain_pct']:+.0f}%")
    return rs


# ─────────────────────────────────────────────────────────────────────
# P4: feedback_gain criticality
# ─────────────────────────────────────────────────────────────────────

def run_p4():
    print("\n=== P4: feedback_gain criticality (calibrated) ===")
    fbs = np.linspace(0.0, 2.0, 9)
    rs = {"fb": [], "MC": [], "Lyap": []}
    for fb in fbs:
        mcs, lys = [], []
        for s in range(N_SEEDS):
            VG2 = torch.full((N_CELLS,), -0.20)
            net = make_net(N_CELLS, alpha=1.5, VG2=VG2,
                             topology="small_world", fb=float(fb), seed=s)
            mcs.append(memory_capacity(net, T_train=600, T_test=300, seed=s))
            net2 = make_net(N_CELLS, alpha=1.5, VG2=VG2,
                              topology="small_world", fb=float(fb), seed=s)
            lys.append(lyapunov_proxy(net2, T_warmup=200, T_meas=150, seed=s))
        rs["fb"].append(float(fb))
        rs["MC"].append(float(np.mean(mcs)))
        rs["Lyap"].append(float(np.mean(lys)))
        print(f"  fb={fb:.2f}  MC={rs['MC'][-1]:.2f}  Lyap={rs['Lyap'][-1]:+.3f}",
              flush=True)
    return rs


def main():
    t0 = time.time()
    p1 = run_p1()
    p2 = run_p2()
    p3 = run_p3()
    p4 = run_p4()
    elapsed = time.time() - t0

    with open(OUT / "summary.json", "w") as f:
        json.dump({"P1": p1, "P2": p2, "P3": p3, "P4": p4,
                    "elapsed_s": elapsed,
                    "calibrated_params": CAL}, f, indent=2)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # P1: topology × α
    ax = axes[0, 0]
    cmap = plt.cm.tab10
    for i, (topo, r) in enumerate(p1.items()):
        ax.errorbar(r["alpha"], r["MC"], yerr=r["MC_std"],
                      marker="o", lw=2, label=topo, capsize=3, color=cmap(i))
    ax.set_xscale("log"); ax.set_xlabel("α", fontsize=11)
    ax.set_ylabel("MC", fontsize=11)
    ax.set_title("P1: topology × α (calibrated, VG2=-0.20)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # P2: VG2 sweep
    ax = axes[0, 1]
    ax2 = ax.twinx()
    r = p2
    ax.errorbar(r["VG2"], r["MC"], yerr=r["MC_std"], fmt="o-",
                  color="#27ae60", lw=2, capsize=3, label="MC")
    ax2.plot(r["VG2"], r["Lyap"], "s-", color="#c0392b", lw=2,
                alpha=0.7, label="Lyap")
    ax2.axhline(0, color="black", ls="--", alpha=0.5)
    i = int(np.argmax(r["MC"]))
    ax.axvline(r["VG2"][i], color="green", ls=":", alpha=0.5)
    ax.text(r["VG2"][i], max(r["MC"])*0.9,
              f"  VG2*={r['VG2'][i]:+.2f}V\n  MC*={r['MC'][i]:.2f}",
              fontsize=10, color="green")
    ax.set_xlabel("VG2 [V]"); ax.set_ylabel("MC", color="#27ae60")
    ax2.set_ylabel("Lyap", color="#c0392b")
    ax.set_title("P2: VG2 sweet-spot (calibrated K_back=-0.98)")
    ax.grid(alpha=0.3)

    # P3: bistability vs linear bar
    ax = axes[1, 0]
    bs = ax.bar(["bistable\n(calibrated)", "linear\n(null model)"],
                  [p3["bistable"][0], p3["linear"][0]],
                  yerr=[p3["bistable"][1], p3["linear"][1]],
                  color=["#3498db", "#95a5a6"], capsize=8)
    for b, v in zip(bs, [p3["bistable"][0], p3["linear"][0]]):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.2f}",
                  ha="center", va="bottom", fontsize=11, weight="bold")
    ax.set_ylabel("MC")
    ax.set_title(f"P3: bistability gain = {p3['gain_pct']:+.0f}%")
    ax.grid(alpha=0.3, axis="y")

    # P4: fb_gain criticality
    ax = axes[1, 1]
    ax2 = ax.twinx()
    r = p4
    ax.plot(r["fb"], r["MC"], "o-", color="#27ae60", lw=2, label="MC")
    ax2.plot(r["fb"], r["Lyap"], "s-", color="#c0392b", lw=2,
                alpha=0.7, label="Lyap")
    ax2.axhline(0, color="black", ls="--", alpha=0.5,
                  label="edge of chaos")
    ax.set_xlabel("feedback_gain"); ax.set_ylabel("MC", color="#27ae60")
    ax2.set_ylabel("Lyap", color="#c0392b")
    ax.set_title("P4: criticality vs MC (calibrated)")
    ax.grid(alpha=0.3)

    fig.suptitle(f"z47 — replay with calibrated K_back=-0.98 (data-empirical)\n"
                  f"Total {elapsed/60:.1f} min, {N_CELLS} cells, {N_SEEDS} seeds",
                  fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "replay.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {OUT/'replay.png'}")
    print(f"Total: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
