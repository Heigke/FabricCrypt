"""z53_hopfield_variability.py — does Hopfield bistable advantage
survive realistic device variability?

z52 showed bistable cells give +7pp recall over linear at K=25 patterns
under 25% noise.  But that was identical cells.  Sebas's chip will
have ~5-10% device variability (Vth, U0, geometry).  Test:

  Per-cell parameter spread σ ∈ {0%, 1%, 5%, 10%, 20%}
  Apply to K_back, A_iii, V_bjt_on (the calibrated knobs)
  K=25 patterns, noise=25%
  Compare bistable+variability vs linear+variability

If the bistable advantage survives 10% spread, it's relevant for
foundry parts.  If it collapses, we need to recommend tight tolerances.
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
from nsram.plasticity_net import NetSim

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z53_hopfield_variability")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {DEVICE}")
N_CELLS = 96
N_SEEDS = 8
P_LO, P_HI = -0.10, 0.40
K_TEST = 25
NOISE = 0.25

CAL = dict(VTH0=0.43, K_back=-0.98, A_iii=4.71, G_bjt=1.00,
            V_bjt_on=0.74, V_latch=0.58, K_leak=0.021)


def hopfield_W(patterns):
    K, N = patterns.shape
    W = (patterns.T @ patterns) / N
    np.fill_diagonal(W, 0.0)
    return W


from scripts.z44_robustness import LinearCellArray


def make_net_with_variability(W_np, spread_pct, seed, linear_cells=False):
    """Create network with per-cell Gaussian spread on key parameters.

    spread_pct: fraction (0.05 = 5% std on each param).
    """
    rng = np.random.default_rng(10000 + seed)
    if linear_cells:
        VG2 = torch.full((N_CELLS,), -0.10, device=DEVICE)
        cells = LinearCellArray(N_CELLS, alpha=1.5, VG2=VG2, device=DEVICE)
    else:
        VG2 = torch.full((N_CELLS,), -0.10, device=DEVICE)
        # Per-cell parameter draws
        K_back_arr = torch.tensor(
            CAL["K_back"] * (1 + spread_pct * rng.standard_normal(N_CELLS)),
            dtype=torch.float32, device=DEVICE)
        A_iii_arr = torch.tensor(
            CAL["A_iii"] * (1 + spread_pct * rng.standard_normal(N_CELLS)),
            dtype=torch.float32, device=DEVICE)
        V_bjt_on_arr = torch.tensor(
            CAL["V_bjt_on"] * (1 + spread_pct * rng.standard_normal(N_CELLS)),
            dtype=torch.float32, device=DEVICE)
        cells = CellArray(N_CELLS, alpha=1.5, VG2=VG2,
                              VTH0=CAL["VTH0"],
                              K_back=K_back_arr,
                              A_iii=A_iii_arr,
                              G_bjt=CAL["G_bjt"],
                              V_bjt_on=V_bjt_on_arr,
                              V_latch=CAL["V_latch"],
                              K_leak=CAL["K_leak"],
                              device=DEVICE)
    W = torch.from_numpy(W_np.astype(np.float32)).to(DEVICE)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    W_in = torch.zeros(N_CELLS, 1, device=DEVICE)
    return NetSim(cells=cells, W=W, W_in=W_in, feedback_gain=0.5)


def make_patterns(K, N, seed):
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=(K, N)).astype(np.float32)


def pattern_to_Vb(p):
    return np.where(p > 0, P_HI, P_LO).astype(np.float32)


def add_noise(p, noise_frac, rng):
    p_n = p.copy()
    n_flip = int(noise_frac * len(p))
    if n_flip > 0:
        idx = rng.choice(len(p), n_flip, replace=False)
        p_n[idx] *= -1
    return p_n


def recall_test(net, pattern_init, T_relax=50):
    Vb_init = torch.from_numpy(pattern_to_Vb(pattern_init)).to(DEVICE)
    net.cells.Vb = Vb_init.clone()
    u = torch.zeros(1, device=DEVICE)
    VG1 = torch.tensor(0.6, device=DEVICE)
    for _ in range(T_relax):
        net.step(u, VG1)
    return net.cells.Vb.cpu().numpy()


def closest_pattern(final_Vb, patterns):
    midpoint = 0.5 * (P_LO + P_HI)
    final_sign = np.where(final_Vb > midpoint, 1, -1).astype(np.float32)
    overlaps = patterns @ final_sign / len(final_sign)
    return int(np.argmax(overlaps)), float(overlaps.max())


def main():
    t0 = time.time()
    spreads = [0.0, 0.01, 0.05, 0.10, 0.20]
    rs = {"spread_pct": [], "bistable_acc": [], "bistable_std": [],
           "linear_acc": [], "linear_std": []}
    print(f"\nK={K_TEST}, noise={NOISE*100:.0f}%, N={N_CELLS}, {N_SEEDS} seeds")
    for sp in spreads:
        bist, lin = [], []
        for s in range(N_SEEDS):
            patterns = make_patterns(K_TEST, N_CELLS, seed=s)
            W = hopfield_W(patterns)
            rng_b = np.random.default_rng(2000 + s)
            rng_l = np.random.default_rng(2000 + s)

            net = make_net_with_variability(W, sp, seed=s, linear_cells=False)
            correct = 0
            for k in range(K_TEST):
                noisy = add_noise(patterns[k], NOISE, rng_b)
                final = recall_test(net, noisy)
                g, _ = closest_pattern(final, patterns)
                if g == k: correct += 1
            bist.append(correct / K_TEST)

            # For linear, no per-cell variability (it has fewer params anyway).
            net = make_net_with_variability(W, sp, seed=s, linear_cells=True)
            correct = 0
            for k in range(K_TEST):
                noisy = add_noise(patterns[k], NOISE, rng_l)
                final = recall_test(net, noisy)
                g, _ = closest_pattern(final, patterns)
                if g == k: correct += 1
            lin.append(correct / K_TEST)

        rs["spread_pct"].append(float(sp * 100))
        rs["bistable_acc"].append(float(np.mean(bist)))
        rs["bistable_std"].append(float(np.std(bist)))
        rs["linear_acc"].append(float(np.mean(lin)))
        rs["linear_std"].append(float(np.std(lin)))
        diff = rs["bistable_acc"][-1] - rs["linear_acc"][-1]
        print(f"  spread={sp*100:4.0f}%  bist={rs['bistable_acc'][-1]:.2f}±{rs['bistable_std'][-1]:.2f}  "
              f"lin={rs['linear_acc'][-1]:.2f}±{rs['linear_std'][-1]:.2f}  "
              f"Δ={diff*100:+.0f}pp", flush=True)

    elapsed = time.time() - t0
    with open(OUT / "summary.json", "w") as f:
        json.dump({**rs, "K": K_TEST, "noise": NOISE,
                    "N": N_CELLS, "seeds": N_SEEDS,
                    "elapsed_s": elapsed}, f, indent=2)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(rs["spread_pct"], [a*100 for a in rs["bistable_acc"]],
                  yerr=[s*100 for s in rs["bistable_std"]],
                  fmt="o-", lw=2.5, color="#3498db", capsize=5,
                  label="bistable (NS-RAM)")
    ax.errorbar(rs["spread_pct"], [a*100 for a in rs["linear_acc"]],
                  yerr=[s*100 for s in rs["linear_std"]],
                  fmt="s-", lw=2.5, color="#95a5a6", capsize=5,
                  label="linear cell (null model)")
    ax.set_xlabel("device parameter spread σ [%]", fontsize=12)
    ax.set_ylabel("recall accuracy [%]", fontsize=12)
    ax.set_title(f"Hopfield K={K_TEST} robustness to device variability\n"
                  f"N={N_CELLS}, noise={NOISE*100:.0f}% bit flips, {N_SEEDS} seeds",
                  fontsize=13)
    ax.legend(fontsize=11); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "variability.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {OUT/'variability.png'}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
