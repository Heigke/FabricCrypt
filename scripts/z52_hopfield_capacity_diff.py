"""z52_hopfield_capacity_diff.py — push K to find bistable vs linear
crossover on associative memory.

z51 showed Hopfield works perfectly on both cell types at K=8 + 20%
noise. To find where bistability actually matters, sweep K from 8 to
50 (capacity stress test), with realistic noise (25%).
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
           "results/z52_hopfield_capacity_diff")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {DEVICE}")
N_CELLS = 96
N_SEEDS = 5
P_LO, P_HI = -0.10, 0.40

CAL = dict(VTH0=0.43, K_back=-0.98, A_iii=4.71, G_bjt=1.00,
            V_bjt_on=0.74, V_latch=0.58, K_leak=0.021)


def hopfield_W(patterns):
    K, N = patterns.shape
    W = (patterns.T @ patterns) / N
    np.fill_diagonal(W, 0.0)
    return W


from scripts.z44_robustness import LinearCellArray


def make_net(W_np, alpha=1.5, fb=0.5, seed=0, linear_cells=False):
    if linear_cells:
        VG2 = torch.full((N_CELLS,), -0.10, device=DEVICE)
        cells = LinearCellArray(N_CELLS, alpha=alpha, VG2=VG2, device=DEVICE)
    else:
        VG2 = torch.full((N_CELLS,), -0.10, device=DEVICE)
        cells = CellArray(N_CELLS, alpha=alpha, VG2=VG2, **CAL, device=DEVICE)
    W = torch.from_numpy(W_np.astype(np.float32)).to(DEVICE)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    W_in = torch.zeros(N_CELLS, 1, device=DEVICE)
    return NetSim(cells=cells, W=W, W_in=W_in, feedback_gain=fb)


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
    Ks = list(range(5, 51, 5))
    noise = 0.25
    rs = {"K": [], "bistable_acc": [], "bistable_std": [],
           "linear_acc": [], "linear_std": []}
    print(f"\nSweep K with noise={noise*100:.0f}%, N={N_CELLS} cells")
    for K in Ks:
        bist, lin = [], []
        for s in range(N_SEEDS):
            patterns = make_patterns(K, N_CELLS, seed=s)
            W = hopfield_W(patterns)
            rng = np.random.default_rng(2000 + s)

            # Bistable
            net = make_net(W, seed=s, linear_cells=False)
            correct = 0
            for k in range(K):
                noisy = add_noise(patterns[k], noise, rng)
                final = recall_test(net, noisy)
                g, _ = closest_pattern(final, patterns)
                if g == k: correct += 1
            bist.append(correct / K)

            # Linear (same noise sequence)
            rng = np.random.default_rng(2000 + s)
            net = make_net(W, seed=s, linear_cells=True)
            correct = 0
            for k in range(K):
                noisy = add_noise(patterns[k], noise, rng)
                final = recall_test(net, noisy)
                g, _ = closest_pattern(final, patterns)
                if g == k: correct += 1
            lin.append(correct / K)

        rs["K"].append(K)
        rs["bistable_acc"].append(float(np.mean(bist)))
        rs["bistable_std"].append(float(np.std(bist)))
        rs["linear_acc"].append(float(np.mean(lin)))
        rs["linear_std"].append(float(np.std(lin)))
        diff = rs["bistable_acc"][-1] - rs["linear_acc"][-1]
        print(f"  K={K:3d}  bist={rs['bistable_acc'][-1]:.2f}  "
              f"lin={rs['linear_acc'][-1]:.2f}  Δ={diff:+.2f}", flush=True)

    elapsed = time.time() - t0
    with open(OUT / "summary.json", "w") as f:
        json.dump({**rs, "noise": noise, "N": N_CELLS, "seeds": N_SEEDS,
                    "elapsed_s": elapsed}, f, indent=2)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(rs["K"], rs["bistable_acc"], yerr=rs["bistable_std"],
                  fmt="o-", lw=2.5, color="#3498db", capsize=4,
                  label="bistable (NS-RAM)")
    ax.errorbar(rs["K"], rs["linear_acc"], yerr=rs["linear_std"],
                  fmt="s-", lw=2.5, color="#95a5a6", capsize=4,
                  label="linear cell (null model)")
    ax.axhline(1.0, color="green", ls="--", alpha=0.4, label="perfect recall")
    ax.axvline(0.14 * N_CELLS, color="red", ls=":", alpha=0.5,
                 label=f"Hopfield limit 0.14·N={0.14*N_CELLS:.0f}")
    ax.set_xlabel("# stored patterns K", fontsize=12)
    ax.set_ylabel("recall accuracy", fontsize=12)
    ax.set_title(f"Hopfield capacity — bistable vs linear cells\n"
                  f"N={N_CELLS}, noise={noise*100:.0f}% bit flips, {N_SEEDS} seeds",
                  fontsize=13)
    ax.legend(fontsize=11); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "capacity_diff.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {OUT/'capacity_diff.png'}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
