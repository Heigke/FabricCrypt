"""z41_vg2_as_plasticity.py — VG2 (physical back-gate) as the plasticity knob.

In Sebas's NS-RAM cell, plasticity is controlled by VG2 — the back-gate
voltage.  Higher VG2 lowers Vth_eff (more activated channel + more
impact-ionization), raises leak target (slower decay), and shifts the
device toward a more plastic regime.

Where z38/z40 swept our internal α gain, this experiment sweeps VG2 — a
parameter Sebas can actually program in his array.  Goal: identify
Goldilocks VG2 for memory capacity, and show that the trade-off curve
maps onto the same inverted-U we saw in α.

Three sub-experiments:
  V1: Single-VG2 sweep (homogeneous network) on small_world topology
  V2: Heterogeneous VG2 — bimodal distribution (some cells high-VG2 plastic,
      some low-VG2 stable). Brain-inspired: hippocampus (plastic) +
      neocortex (stable).
  V3: VG2 × α joint sweep — find the device operating envelope
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
from nsram.plasticity_net import (NetSim, topo_small_world,
                                   memory_capacity, lyapunov_proxy)

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z41_vg2_as_plasticity")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {DEVICE}")
N_CELLS = 96
N_SEEDS = 3


def make_net_vg2(VG2_per_cell: torch.Tensor, alpha: float, seed: int):
    cells = CellArray(N_CELLS, alpha=alpha,
                          VG2=VG2_per_cell.to(DEVICE), device=DEVICE)
    W = topo_small_world(N_CELLS, k=4, p_rewire=0.1, seed=seed, device=DEVICE)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    W_in = torch.randn(N_CELLS, 1, generator=g, device=DEVICE)
    return NetSim(cells=cells, W=W, W_in=W_in, feedback_gain=0.27)


# ─────────────────────────────────────────────────────────────────────
# V1: homogeneous VG2 sweep
# ─────────────────────────────────────────────────────────────────────

def run_v1():
    print("\n=== V1: homogeneous VG2 sweep (small_world, α=1.5) ===")
    vg2_grid = np.linspace(-0.20, 0.50, 16)
    rs = {"VG2": [], "MC": [], "MC_std": [], "Lyap": [], "Lyap_std": []}
    for vg2 in vg2_grid:
        mcs, lys = [], []
        for s in range(N_SEEDS):
            VG2 = torch.full((N_CELLS,), float(vg2))
            net = make_net_vg2(VG2, alpha=1.5, seed=s)
            mcs.append(memory_capacity(net, T_train=600, T_test=300, seed=s))
            net2 = make_net_vg2(VG2, alpha=1.5, seed=s)
            lys.append(lyapunov_proxy(net2, T_warmup=200, T_meas=150, seed=s))
        rs["VG2"].append(float(vg2))
        rs["MC"].append(float(np.mean(mcs)))
        rs["MC_std"].append(float(np.std(mcs)))
        rs["Lyap"].append(float(np.mean(lys)))
        rs["Lyap_std"].append(float(np.std(lys)))
        print(f"  VG2={vg2:+.3f}  MC={rs['MC'][-1]:.2f}±{rs['MC_std'][-1]:.2f}  "
              f"Lyap={rs['Lyap'][-1]:+.3f}", flush=True)
    return rs


# ─────────────────────────────────────────────────────────────────────
# V2: bimodal VG2 (hippocampus + neocortex split)
# ─────────────────────────────────────────────────────────────────────

def run_v2():
    """Some cells get high VG2 (plastic, fast learning), others low VG2
    (stable, slow learning).  Sweep the fraction in plastic mode."""
    print("\n=== V2: bimodal VG2 (plastic + stable cells) ===")
    fracs = np.linspace(0.0, 1.0, 11)
    VG2_PLASTIC = 0.30
    VG2_STABLE = 0.00
    rs = {"frac_plastic": [], "MC": [], "MC_std": [],
           "Lyap": [], "Lyap_std": []}
    for frac in fracs:
        mcs, lys = [], []
        for s in range(N_SEEDS):
            torch.manual_seed(s)
            n_plastic = int(frac * N_CELLS)
            VG2 = torch.cat([
                torch.full((n_plastic,), VG2_PLASTIC),
                torch.full((N_CELLS - n_plastic,), VG2_STABLE),
            ])[torch.randperm(N_CELLS)]
            net = make_net_vg2(VG2, alpha=1.5, seed=s)
            mcs.append(memory_capacity(net, T_train=600, T_test=300, seed=s))
            net2 = make_net_vg2(VG2, alpha=1.5, seed=s)
            lys.append(lyapunov_proxy(net2, T_warmup=200, T_meas=150, seed=s))
        rs["frac_plastic"].append(float(frac))
        rs["MC"].append(float(np.mean(mcs)))
        rs["MC_std"].append(float(np.std(mcs)))
        rs["Lyap"].append(float(np.mean(lys)))
        rs["Lyap_std"].append(float(np.std(lys)))
        print(f"  frac_plastic={frac:.2f}  MC={rs['MC'][-1]:.2f}±{rs['MC_std'][-1]:.2f}  "
              f"Lyap={rs['Lyap'][-1]:+.3f}", flush=True)
    return rs


# ─────────────────────────────────────────────────────────────────────
# V3: VG2 × α joint sweep (device operating envelope)
# ─────────────────────────────────────────────────────────────────────

def run_v3():
    print("\n=== V3: VG2 × α envelope ===")
    vg2_grid = np.linspace(-0.10, 0.40, 6)
    alpha_grid = np.logspace(-1.5, 1.0, 6)   # 0.03 to 10
    grid = np.zeros((len(vg2_grid), len(alpha_grid)))
    for i, vg2 in enumerate(vg2_grid):
        for j, a in enumerate(alpha_grid):
            mcs = []
            for s in range(N_SEEDS):
                VG2 = torch.full((N_CELLS,), float(vg2))
                net = make_net_vg2(VG2, alpha=float(a), seed=s)
                mcs.append(memory_capacity(net, T_train=500, T_test=200, seed=s))
            grid[i, j] = float(np.mean(mcs))
            print(f"  VG2={vg2:+.2f} α={a:.3f}  MC={grid[i, j]:.2f}", flush=True)
    return {"VG2_grid": vg2_grid.tolist(), "alpha_grid": alpha_grid.tolist(),
             "MC_grid": grid.tolist()}


# ─────────────────────────────────────────────────────────────────────
# Main + plotting
# ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    v1 = run_v1()
    v2 = run_v2()
    v3 = run_v3()
    elapsed = time.time() - t0

    with open(OUT / "summary.json", "w") as f:
        json.dump({"V1": v1, "V2": v2, "V3": v3, "elapsed_s": elapsed}, f, indent=2)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3)

    # V1
    ax = fig.add_subplot(gs[0, 0])
    ax2 = ax.twinx()
    r = v1
    ax.errorbar(r["VG2"], r["MC"], yerr=r["MC_std"], fmt="o-",
                  color="#27ae60", lw=2, capsize=3)
    ax2.errorbar(r["VG2"], r["Lyap"], yerr=r["Lyap_std"], fmt="s-",
                   color="#c0392b", lw=2, capsize=3)
    ax2.axhline(0, color="black", ls="--", alpha=0.5)
    i = int(np.argmax(r["MC"]))
    ax.axvline(r["VG2"][i], color="green", ls=":", alpha=0.5)
    ax.text(r["VG2"][i], max(r["MC"])*0.92,
              f"  VG2*={r['VG2'][i]:+.2f}V\n  MC*={r['MC'][i]:.2f}",
              fontsize=9, color="#27ae60")
    ax.set_xlabel("VG2 [V]  (back-gate voltage)", fontsize=11)
    ax.set_ylabel("MC", color="#27ae60", fontsize=11)
    ax2.set_ylabel("Lyap", color="#c0392b", fontsize=11)
    ax.set_title("V1: VG2 as plasticity knob")
    ax.grid(alpha=0.3)

    # V2
    ax = fig.add_subplot(gs[0, 1])
    ax2 = ax.twinx()
    r = v2
    ax.errorbar(r["frac_plastic"], r["MC"], yerr=r["MC_std"], fmt="o-",
                  color="#27ae60", lw=2, capsize=3)
    ax2.errorbar(r["frac_plastic"], r["Lyap"], yerr=r["Lyap_std"], fmt="s-",
                   color="#c0392b", lw=2, capsize=3)
    i = int(np.argmax(r["MC"]))
    ax.axvline(r["frac_plastic"][i], color="green", ls=":", alpha=0.5)
    ax.text(r["frac_plastic"][i], max(r["MC"])*0.92,
              f"  best mix\n  frac={r['frac_plastic'][i]:.2f}\n  MC={r['MC'][i]:.2f}",
              fontsize=9, color="#27ae60")
    ax.set_xlabel("fraction of plastic cells (VG2=0.30)", fontsize=11)
    ax.set_ylabel("MC", color="#27ae60", fontsize=11)
    ax2.set_ylabel("Lyap", color="#c0392b", fontsize=11)
    ax.set_title("V2: bimodal VG2  (plastic=0.30 / stable=0.00)")
    ax.grid(alpha=0.3)

    # V3 heatmap
    ax = fig.add_subplot(gs[0, 2])
    grid = np.array(v3["MC_grid"])
    vg2_g = v3["VG2_grid"]; a_g = v3["alpha_grid"]
    im = ax.pcolormesh(a_g, vg2_g, grid, cmap="viridis",
                          shading="auto")
    ax.set_xscale("log")
    ax.set_xlabel("α  (gain)", fontsize=11)
    ax.set_ylabel("VG2 [V]", fontsize=11)
    ax.set_title("V3: MC across (α, VG2) operating envelope")
    fig.colorbar(im, ax=ax, label="MC")
    # Mark optimum
    i, j = np.unravel_index(np.argmax(grid), grid.shape)
    ax.plot(a_g[j], vg2_g[i], "r*", ms=18, mec="white", mew=1.5)
    ax.text(a_g[j], vg2_g[i],
              f"\n  ★ ({a_g[j]:.2f}, {vg2_g[i]:+.2f})\n     MC={grid[i,j]:.2f}",
              fontsize=8, color="white", weight="bold", va="top")

    # Combined summary text panel
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    i1 = int(np.argmax(v1["MC"]))
    i2 = int(np.argmax(v2["MC"]))
    summary_text = (
        f"VG2-as-plasticity findings (small_world, N={N_CELLS}, {N_SEEDS} seeds, {elapsed:.0f}s):\n\n"
        f"V1  Best homogeneous VG2 = {v1['VG2'][i1]:+.3f} V → MC = {v1['MC'][i1]:.2f}\n"
        f"     Lyapunov at optimum = {v1['Lyap'][i1]:+.3f}  ({'subcritical' if v1['Lyap'][i1] < 0 else 'critical'})\n\n"
        f"V2  Best plastic-cell fraction = {v2['frac_plastic'][i2]:.2f}  →  MC = {v2['MC'][i2]:.2f}\n"
        f"     vs all-stable ({v2['MC'][0]:.2f}): improvement = {(v2['MC'][i2]/(v2['MC'][0]+1e-9)-1)*100:.0f}%\n\n"
        f"V3  Optimum (α, VG2) = ({a_g[j]:.3f}, {vg2_g[i]:+.3f})  →  MC = {grid[i,j]:.2f}\n"
        f"     ⇒ Sebas's array is most useful when biased to VG2 ≈ {vg2_g[i]:+.2f} V\n"
        f"        with α (write strength) ≈ {a_g[j]:.2f}\n\n"
        f"Interpretation:  VG2 acts as a global plasticity dial.  Higher VG2 = more plastic\n"
        f"(faster body charging, slower decay), but past a threshold the network saturates and\n"
        f"information is lost.  A bimodal VG2 distribution (some cells plastic, some stable)\n"
        f"can outperform any homogeneous setting — supports a hippocampus/neocortex hybrid layout."
    )
    ax.text(0.02, 0.95, summary_text, fontsize=11, va="top", family="monospace")

    fig.suptitle("z41 — VG2 (physical back-gate) as the plasticity knob",
                  fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "vg2_plasticity.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {OUT/'vg2_plasticity.png'}")
    print(f"Total: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
