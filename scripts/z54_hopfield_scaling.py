"""z54_hopfield_scaling.py — does the bistable Hopfield advantage scale
with N?

z52 showed +7pp at N=96, K=25. z53 showed it survives 10% device
variability. Now: does it grow with array size? Test at N ∈ {64, 96,
128, 192, 256} with K matched to 0.25·N (above the 0.14 textbook limit
to stay in the saturation regime where bistability matters).
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
           "results/z54_hopfield_scaling")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {DEVICE}")
N_SEEDS = 5
P_LO, P_HI = -0.10, 0.40
NOISE = 0.25
K_RATIO = 0.25  # K = 0.25 * N (saturation regime)

CAL = dict(VTH0=0.43, K_back=-0.98, A_iii=4.71, G_bjt=1.00,
            V_bjt_on=0.74, V_latch=0.58, K_leak=0.021)


def hopfield_W(patterns):
    K, N = patterns.shape
    W = (patterns.T @ patterns) / N
    np.fill_diagonal(W, 0.0)
    return W


from scripts.z44_robustness import LinearCellArray


def make_net(W_np, N, seed, linear_cells=False):
    if linear_cells:
        VG2 = torch.full((N,), -0.10, device=DEVICE)
        cells = LinearCellArray(N, alpha=1.5, VG2=VG2, device=DEVICE)
    else:
        VG2 = torch.full((N,), -0.10, device=DEVICE)
        cells = CellArray(N, alpha=1.5, VG2=VG2, **CAL, device=DEVICE)
    W = torch.from_numpy(W_np.astype(np.float32)).to(DEVICE)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    W_in = torch.zeros(N, 1, device=DEVICE)
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
    Ns = [64, 96, 128, 192, 256, 384]
    rs = {"N": [], "K": [], "bistable": [], "bistable_std": [],
           "linear": [], "linear_std": []}
    print(f"\nScaling sweep — K = {K_RATIO} · N, noise = {NOISE*100:.0f}%, {N_SEEDS} seeds")
    for N in Ns:
        K = int(K_RATIO * N)
        bist, lin = [], []
        for s in range(N_SEEDS):
            patterns = make_patterns(K, N, seed=s)
            W = hopfield_W(patterns)

            rng = np.random.default_rng(2000 + s)
            net = make_net(W, N, seed=s, linear_cells=False)
            correct = 0
            for k in range(K):
                noisy = add_noise(patterns[k], NOISE, rng)
                final = recall_test(net, noisy)
                g, _ = closest_pattern(final, patterns)
                if g == k: correct += 1
            bist.append(correct / K)

            rng = np.random.default_rng(2000 + s)
            net = make_net(W, N, seed=s, linear_cells=True)
            correct = 0
            for k in range(K):
                noisy = add_noise(patterns[k], NOISE, rng)
                final = recall_test(net, noisy)
                g, _ = closest_pattern(final, patterns)
                if g == k: correct += 1
            lin.append(correct / K)

        rs["N"].append(N)
        rs["K"].append(K)
        rs["bistable"].append(float(np.mean(bist)))
        rs["bistable_std"].append(float(np.std(bist)))
        rs["linear"].append(float(np.mean(lin)))
        rs["linear_std"].append(float(np.std(lin)))
        diff = (rs["bistable"][-1] - rs["linear"][-1]) * 100
        print(f"  N={N:4d}  K={K:3d}  bist={rs['bistable'][-1]:.2f}±{rs['bistable_std'][-1]:.2f}  "
              f"lin={rs['linear'][-1]:.2f}±{rs['linear_std'][-1]:.2f}  Δ={diff:+.1f}pp",
              flush=True)

    elapsed = time.time() - t0
    with open(OUT / "summary.json", "w") as f:
        json.dump({**rs, "noise": NOISE, "K_ratio": K_RATIO,
                    "seeds": N_SEEDS, "elapsed_s": elapsed}, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.errorbar(rs["N"], [a*100 for a in rs["bistable"]],
                  yerr=[s*100 for s in rs["bistable_std"]],
                  fmt="o-", lw=2.5, color="#3498db", capsize=5,
                  label="bistable (NS-RAM)")
    ax.errorbar(rs["N"], [a*100 for a in rs["linear"]],
                  yerr=[s*100 for s in rs["linear_std"]],
                  fmt="s-", lw=2.5, color="#95a5a6", capsize=5,
                  label="linear cell")
    ax.set_xlabel("# cells N", fontsize=12)
    ax.set_ylabel("recall accuracy [%]", fontsize=12)
    ax.set_title(f"Hopfield scaling — K = {K_RATIO}·N, noise = {NOISE*100:.0f}%")
    ax.legend(fontsize=11); ax.grid(alpha=0.3)

    ax = axes[1]
    diffs = [(rs["bistable"][i] - rs["linear"][i]) * 100 for i in range(len(rs["N"]))]
    ax.plot(rs["N"], diffs, "o-", lw=2.5, color="#e67e22")
    ax.axhline(0, color="black", ls="--", alpha=0.5)
    ax.set_xlabel("# cells N"); ax.set_ylabel("bistable advantage [pp]")
    ax.set_title("Bistability advantage scaling")
    ax.grid(alpha=0.3)

    fig.suptitle(f"z54 — Hopfield scaling: bistable vs linear  ({elapsed:.0f}s)",
                  fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "scaling.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {OUT/'scaling.png'}")


if __name__ == "__main__":
    main()
