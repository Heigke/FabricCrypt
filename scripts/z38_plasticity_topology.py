"""z38_plasticity_topology.py — sweep plasticity α × topology.

For each (topology, α) combination, build an N-cell NS-RAM network and
measure:
  - Memory capacity (MC, Jaeger 2002)
  - Lyapunov exponent proxy (positive = chaotic, ≈0 = critical)
  - Avalanche power-law slope (-1.5 ≈ critical / Beggs-Plenz 2003)

Hypothesis (from criticality literature + meditation neuroscience):
  An "inverted-U" curve in MC vs α — too little plasticity = no memory,
  too much = chaotic / forgetful.  Optimum near Lyap ≈ 0.  Different
  topologies should peak at different α values.
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
                                   lyapunov_proxy, avalanche_stats)

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z38_plasticity_topology")
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {DEVICE}")

# Sweep grid
TOPOLOGIES = ["random", "ring", "small_world", "scale_free", "hierarchical", "full"]
ALPHAS = np.logspace(-2, np.log10(5.0), 12)   # 0.01 → 5.0
N_CELLS = 96
N_SEEDS = 3
T_TRAIN = 800
T_TEST  = 400


def measure_one(topology, alpha, seed):
    net = make_network(N=N_CELLS, topology=topology,
                        alpha=float(alpha), VG2_mean=0.20,
                        seed=seed, device=DEVICE)
    # Recording activity for avalanche analysis
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1, 1, T_TRAIN + T_TEST).astype(np.float32)
    U = torch.from_numpy(u).unsqueeze(1).to(DEVICE)
    VG1_seq = torch.full((T_TRAIN + T_TEST,), 0.6).to(DEVICE)
    Id, Vb = net.run(U, VG1_seq)
    # Reset for memory_capacity (it does its own driving)
    net2 = make_network(N=N_CELLS, topology=topology,
                         alpha=float(alpha), VG2_mean=0.20,
                         seed=seed, device=DEVICE)
    mc = memory_capacity(net2, T_train=T_TRAIN, T_test=T_TEST,
                            n_lags=20, washout=200, seed=seed)
    net3 = make_network(N=N_CELLS, topology=topology,
                         alpha=float(alpha), VG2_mean=0.20,
                         seed=seed, device=DEVICE)
    ly = lyapunov_proxy(net3, T_warmup=300, T_meas=200, seed=seed)
    av_slope, n_av = avalanche_stats(Vb, threshold_sigma=0.5)
    return {"MC": mc, "Lyap": ly, "av_slope": av_slope, "n_av": n_av}


def main():
    t0 = time.time()
    results = {topo: {"alpha": [], "MC": [], "Lyap": [],
                       "av_slope": [], "MC_std": [], "Lyap_std": []}
                for topo in TOPOLOGIES}
    for topo in TOPOLOGIES:
        print(f"\n=== topology: {topo} ===")
        for alpha in ALPHAS:
            mcs, lys, avs = [], [], []
            for s in range(N_SEEDS):
                m = measure_one(topo, alpha, seed=s)
                mcs.append(m["MC"]); lys.append(m["Lyap"]); avs.append(m["av_slope"])
            mc_mean, mc_std = float(np.mean(mcs)), float(np.std(mcs))
            ly_mean, ly_std = float(np.mean(lys)), float(np.std(lys))
            av_mean = float(np.mean(avs))
            print(f"  α={alpha:6.3f}  MC={mc_mean:.2f}±{mc_std:.2f}  "
                   f"Lyap={ly_mean:+.3f}±{ly_std:.3f}  av={av_mean:+.2f}  "
                   f"({time.time()-t0:.0f}s)", flush=True)
            results[topo]["alpha"].append(float(alpha))
            results[topo]["MC"].append(mc_mean)
            results[topo]["MC_std"].append(mc_std)
            results[topo]["Lyap"].append(ly_mean)
            results[topo]["Lyap_std"].append(ly_std)
            results[topo]["av_slope"].append(av_mean)

    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ─── Plots ───
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    cmap = plt.cm.tab10
    for i, topo in enumerate(TOPOLOGIES):
        r = results[topo]
        a = np.array(r["alpha"])
        mc = np.array(r["MC"]); mc_s = np.array(r["MC_std"])
        ly = np.array(r["Lyap"]); ly_s = np.array(r["Lyap_std"])
        av = np.array(r["av_slope"])
        c = cmap(i)
        axes[0].errorbar(a, mc, yerr=mc_s, marker="o", lw=2, label=topo,
                          color=c, capsize=3)
        axes[1].errorbar(a, ly, yerr=ly_s, marker="o", lw=2, label=topo,
                          color=c, capsize=3)
        axes[2].plot(a, av, marker="o", lw=2, label=topo, color=c)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("α  (plasticity gain)", fontsize=11)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9, loc="best")
    axes[0].set_ylabel("Memory Capacity", fontsize=11)
    axes[0].set_title("MC vs α  (higher = better)")
    axes[1].set_ylabel("Lyapunov exponent", fontsize=11)
    axes[1].set_title("Stability  (positive = chaotic)")
    axes[1].axhline(0, color="black", ls="--", alpha=0.5)
    axes[2].set_ylabel("avalanche slope", fontsize=11)
    axes[2].set_title("Criticality  (-1.5 ≈ critical)")
    axes[2].axhline(-1.5, color="red", ls="--", alpha=0.5,
                      label="critical (-1.5)")
    fig.suptitle(f"Plasticity sweep across topologies — N={N_CELLS} cells, "
                  f"{N_SEEDS} seeds each\n"
                  "Optimum α should sit near MC peak ∩ Lyap≈0 ∩ avalanche≈-1.5",
                  fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "plasticity_topology_sweep.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {OUT/'plasticity_topology_sweep.png'}")
    print(f"Wrote {OUT/'results.json'}")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")

    # Per-topology optima
    print("\nBest-α per topology (max MC):")
    for topo in TOPOLOGIES:
        r = results[topo]
        i = int(np.argmax(r["MC"]))
        print(f"  {topo:14s}  α*={r['alpha'][i]:.3f}  "
               f"MC*={r['MC'][i]:.2f}  Lyap*={r['Lyap'][i]:+.3f}")


if __name__ == "__main__":
    main()
